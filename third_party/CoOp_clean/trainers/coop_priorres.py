from __future__ import annotations

import os
import os.path as osp
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

# Make /workspace/meta_prompt_1/src importable without requiring repo-wide changes.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from meta_prompts.prior_residual_adapter import PriorResidualAdapter
from meta_prompts.shot_weighting import weighted_fewshot_loss

_tokenizer = _Tokenizer()


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, str(default))
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}



def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearnerPriorRes(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]

        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal clip_imsize ({clip_imsize})"

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        self.u_ctx = nn.Parameter(torch.zeros_like(ctx_vectors))
        nn.init.normal_(self.u_ctx, std=0.02)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
        self.use_context_gating = bool(cfg.TRAINER.COOP.USE_CONTEXT_GATING)

        # Residual formula switch:
        #   USE_LEGACY_RESIDUAL=True  -> old formula: ctx + lambda * (a - 1)  * u_ctx
        #   USE_LEGACY_RESIDUAL=False -> safe formula: ctx + lambda * (a - a0) * u_ctx
        self.use_legacy_residual = _env_flag("USE_LEGACY_RESIDUAL", default=False)
        mode = "legacy:(a-1)" if self.use_legacy_residual else "identity-centered:(a-a0)"
        print(f"[PriorRes] residual modulation mode = {mode}")

    def _apply_residual_modulation(
        self,
        context_gates: torch.Tensor,
        lambda_t: torch.Tensor,
        context_gates_base: torch.Tensor = None,
    ) -> torch.Tensor:
        ctx = self.ctx

        if not self.use_context_gating:
            return ctx

        if context_gates is None:
            return ctx

        if lambda_t is None or float(lambda_t.item()) <= 0:
            return ctx

        gate = context_gates.to(device=ctx.device, dtype=ctx.dtype)
        if gate.dim() == 2:
            gate = gate.squeeze(0)

        if self.use_legacy_residual:
            # Original formulation:
            #   ctx_eff = ctx + lambda_t * (a - 1) * u_ctx
            centered_gate = gate - 1.0
        else:
            # Identity-centered safe formulation:
            #   ctx_eff = ctx + lambda_t * (a - a0) * u_ctx
            #
            # If a0 is unavailable for any reason, fall back to gate.detach(),
            # making the shift zero instead of injecting an unsafe residual.
            if context_gates_base is None:
                gate_base = gate.detach()
            else:
                gate_base = context_gates_base.to(device=ctx.device, dtype=ctx.dtype).detach()
                if gate_base.dim() == 2:
                    gate_base = gate_base.squeeze(0)

            centered_gate = gate - gate_base

        lam = lambda_t.to(device=ctx.device, dtype=ctx.dtype).view(1)

        if ctx.dim() == 2:
            shift = centered_gate.unsqueeze(-1) * self.u_ctx
            return ctx + lam * shift

        centered_gate = centered_gate.unsqueeze(0).unsqueeze(-1)
        shift = centered_gate * self.u_ctx
        return ctx + lam * shift

    def forward(
        self,
        context_gates: torch.Tensor = None,
        lambda_t: torch.Tensor = None,
        context_gates_base: torch.Tensor = None,
    ):
        ctx = self._apply_residual_modulation(
            context_gates=context_gates,
            lambda_t=lambda_t,
            context_gates_base=context_gates_base,
        )
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i + 1, :, :]
                class_i = suffix[i:i + 1, :name_len, :]
                suffix_i = suffix[i:i + 1, name_len:, :]
                ctx_i_half1 = ctx[i:i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i:i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i + 1, :, :]
                class_i = suffix[i:i + 1, :name_len, :]
                suffix_i = suffix[i:i + 1, name_len:, :]
                ctx_i = ctx[i:i + 1, :, :]
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError(f"Unsupported class_token_position: {self.class_token_position}")

        return prompts


class CustomCLIPPriorRes(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearnerPriorRes(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        if not cfg.TRAINER.COOP.TASK_FEAT_PATH:
            raise ValueError("TRAINER.COOP.TASK_FEAT_PATH must be set for CoOpPriorRes")

        self.prior_adapter = PriorResidualAdapter(
            task_feat_path=cfg.TRAINER.COOP.TASK_FEAT_PATH,
            task_feat_mode=cfg.TRAINER.COOP.TASK_FEAT_MODE,
            n_ctx=self.prompt_learner.n_ctx,
            kmax=cfg.TRAINER.COOP.META_KMAX,
            hidden_dim=cfg.TRAINER.COOP.META_HIDDEN_DIM,
            gate_temperature=cfg.TRAINER.COOP.GATE_TEMPERATURE,
            init_gate_bias=cfg.TRAINER.COOP.INIT_GATE_BIAS,
            warmup_epochs=cfg.TRAINER.COOP.WARMUP_EPOCHS,
            ramp_epochs=cfg.TRAINER.COOP.RAMP_EPOCHS,
            lambda_max=cfg.TRAINER.COOP.LAMBDA_MAX,
            device=torch.device("cpu"),
        )

    def forward(self, image: torch.Tensor, epoch=None):
        image_features = self.image_encoder(image.type(self.dtype))
        adapter_out = self.prior_adapter(epoch=epoch)

        prompts = self.prompt_learner(
            context_gates=adapter_out["a"],
            lambda_t=adapter_out["lambda_t"],
            context_gates_base=adapter_out.get("a0", None),
        )
        tokenized_prompts = self.tokenized_prompts
        text_batch_size = int(os.environ.get("TEXT_BATCH_SIZE", "0"))
        if text_batch_size > 0 and prompts.shape[0] > text_batch_size:
            text_features_list = []
            for start in range(0, prompts.shape[0], text_batch_size):
                end = start + text_batch_size
                text_features_list.append(
                    self.text_encoder(prompts[start:end], tokenized_prompts[start:end])
                )
            text_features = torch.cat(text_features_list, dim=0)
        else:
            text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        if epoch is not None:
            return logits, adapter_out
        return logits


@TRAINER_REGISTRY.register()
class CoOpPriorRes(TrainerX):
    """Task-conditioned prior + residual joint adaptation on top of CoOp."""

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building custom CLIP with prior-residual adapter")
        self.model = CustomCLIPPriorRes(cfg, classnames, clip_model)

        print("Turning off gradients in image encoder and text encoder")
        for name, param in self.model.named_parameters():
            if ("prompt_learner" not in name) and ("prior_adapter" not in name):
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        self.optim_prompt = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched_prompt = build_lr_scheduler(self.optim_prompt, cfg.OPTIM)

        self.optim_meta = build_optimizer(self.model.prior_adapter, cfg.OPTIM)
        meta_lr_ratio = float(cfg.TRAINER.COOP.META_LR_RATIO)
        for group in self.optim_meta.param_groups:
            group["lr"] *= meta_lr_ratio
            if "initial_lr" in group:
                group["initial_lr"] *= meta_lr_ratio
        self.sched_meta = build_lr_scheduler(self.optim_meta, cfg.OPTIM)

        self.register_model("prompt_learner", self.model.prompt_learner, self.optim_prompt, self.sched_prompt)
        self.register_model("prior_adapter", self.model.prior_adapter, self.optim_meta, self.sched_meta)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label, slot_id = self.parse_batch_train(batch)
        cfg = self.cfg

        if self.epoch < cfg.TRAINER.COOP.WARMUP_EPOCHS:
            loss, output, aux = self._prompt_step(image, label, slot_id, epoch=self.epoch, warmup_only=True)
            summary = self._build_summary(loss, output, aux, prefix="")
            summary["stage"] = 0.0
        else:
            if cfg.TRAINER.COOP.ALTERNATE_OPT:
                loss_prompt, output_prompt, aux_prompt = self._prompt_step(
                    image, label, slot_id, epoch=self.epoch, warmup_only=False
                )
                loss_meta, output_meta, aux_meta = self._meta_step(image, label, slot_id, epoch=self.epoch)
                summary = self._build_summary(loss_meta, output_meta, aux_meta, prefix="")
                summary["loss_prompt_step"] = float(loss_prompt.item())
                summary["loss_meta_step"] = float(loss_meta.item())
                summary["acc_prompt_step"] = float(compute_accuracy(output_prompt, label)[0].item())
                summary["stage"] = 2.0
            else:
                loss, output, aux = self._joint_step(image, label, slot_id, epoch=self.epoch)
                summary = self._build_summary(loss, output, aux, prefix="")
                summary["stage"] = 1.0

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return summary

    def _joint_step(self, image, label, slot_id, epoch: int):
        self._set_trainable(prompt_on=True, meta_on=True)
        output, aux = self._model_forward(image, epoch)
        loss = self._compose_total_loss(output, label, aux, slot_id=slot_id, enable_b=True)

        self._zero_all_optimizers()
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim_prompt)
            self.scaler.step(self.optim_meta)
            self.scaler.update()
        else:
            loss.backward()
            self.optim_prompt.step()
            self.optim_meta.step()

        return loss.detach(), output.detach(), self._detach_aux(aux)

    def _prompt_step(self, image, label, slot_id, epoch: int, warmup_only: bool = False):
        self._set_trainable(prompt_on=True, meta_on=False)
        output, aux = self._model_forward(image, epoch)
        loss = self._compose_total_loss(
            output,
            label,
            aux,
            slot_id=slot_id,
            meta_regularize=not warmup_only,
            enable_b=not warmup_only,
        )
        self._zero_all_optimizers()
        self._backward_and_step(loss, self.optim_prompt, zero_all=False)
        return loss.detach(), output.detach(), self._detach_aux(aux)

    def _meta_step(self, image, label, slot_id, epoch: int):
        self._set_trainable(prompt_on=False, meta_on=True)
        output, aux = self._model_forward(image, epoch)
        loss = self._compose_total_loss(
            output,
            label,
            aux,
            slot_id=slot_id,
            meta_regularize=True,
            enable_b=True,
        )
        self._zero_all_optimizers()
        self._backward_and_step(loss, self.optim_meta, zero_all=False)
        return loss.detach(), output.detach(), self._detach_aux(aux)

    def _model_forward(self, image, epoch: int):
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output, aux = self.model(image, epoch=epoch)
        else:
            output, aux = self.model(image, epoch=epoch)
        return output, aux

    def _compose_total_loss(
        self,
        output: torch.Tensor,
        label: torch.Tensor,
        aux: Dict[str, torch.Tensor],
        slot_id: torch.Tensor = None,
        meta_regularize: bool = True,
        enable_b: bool = True,
    ) -> torch.Tensor:
        cfg = self.cfg

        loss_cls = F.cross_entropy(output, label)
        total = loss_cls

        use_b = bool(cfg.TRAINER.COOP.USE_B) and enable_b
        beta = float(cfg.TRAINER.COOP.B_LOSS_WEIGHT)

        b_loss = torch.zeros((), device=output.device, dtype=output.dtype)

        if use_b:
            if slot_id is None:
                raise ValueError("USE_B=True but slot_id is missing from the batch")

            b_loss, b_aux = weighted_fewshot_loss(
                logits=output,
                labels=label,
                b=aux["b_logits"],
                slot_ids=slot_id,
            )

            total = (1.0 - beta) * loss_cls + beta * b_loss

            aux["slot_weight_mean"] = b_aux["sample_weights"].mean()
            aux["slot_weight_max"] = b_aux["sample_weights"].max()
            aux["b_entropy"] = b_aux["b_entropy"]
            aux["top1_weight"] = b_aux["top1_weight"]
            aux["top4_weight_sum"] = b_aux["top4_weight_sum"]
            aux["keff"] = b_aux["keff_from_weights"]
        else:
            total = loss_cls
            aux["slot_weight_mean"] = torch.zeros((), device=output.device, dtype=output.dtype)
            aux["slot_weight_max"] = torch.zeros((), device=output.device, dtype=output.dtype)
            aux["b_entropy"] = aux.get("b_entropy", torch.zeros((), device=output.device, dtype=output.dtype))
            aux["top1_weight"] = aux.get("top1_weight", torch.zeros((), device=output.device, dtype=output.dtype))
            aux["top4_weight_sum"] = aux.get("top4_weight_sum", torch.zeros((), device=output.device, dtype=output.dtype))

        if meta_regularize:
            reg_a = float(cfg.TRAINER.COOP.DELTA_A_REG) * aux["delta_a"].pow(2).sum()
            reg_b = float(cfg.TRAINER.COOP.DELTA_B_REG) * aux["delta_b"].pow(2).sum()
            total = total + reg_a + reg_b
            aux["reg_a"] = reg_a.detach()
            aux["reg_b"] = reg_b.detach()
        else:
            aux["reg_a"] = torch.zeros((), device=output.device, dtype=output.dtype)
            aux["reg_b"] = torch.zeros((), device=output.device, dtype=output.dtype)

        aux["loss_cls"] = loss_cls.detach()
        aux["loss_b"] = b_loss.detach()
        aux["loss_total"] = total.detach()
        return total

    @staticmethod
    def _scalarize(x):
        if torch.is_tensor(x):
            if x.numel() == 1:
                return float(x.item())
            return float(x.float().mean().item())
        return float(x)

    def _build_summary(self, loss, output, aux, prefix=""):
        summary = {
            f"{prefix}loss": self._scalarize(loss),
            f"{prefix}acc": float(compute_accuracy(output, self._current_label)[0].item())
            if hasattr(self, "_current_label") else float("nan"),
            f"{prefix}loss_cls": self._scalarize(aux["loss_cls"]),
            f"{prefix}loss_b": self._scalarize(aux["loss_b"]),
            f"{prefix}reg_a": self._scalarize(aux["reg_a"]),
            f"{prefix}reg_b": self._scalarize(aux["reg_b"]),
            f"{prefix}lambda_t": self._scalarize(aux["lambda_t"]),
            f"{prefix}meff": self._scalarize(aux["meff"]),
            f"{prefix}keff": self._scalarize(aux["keff"]),
            f"{prefix}a0_mean": float(aux["a0"].float().mean().item()),
            f"{prefix}a_mean": float(aux["a"].float().mean().item()),
            f"{prefix}b0_mean": float(aux["b0"].float().mean().item()),
            f"{prefix}b_mean": float(aux["b"].float().mean().item()),
            f"{prefix}delta_a_norm": float(aux["delta_a"].float().norm().item()),
            f"{prefix}delta_b_norm": float(aux["delta_b"].float().norm().item()),
            f"{prefix}b_entropy": self._scalarize(aux["b_entropy"]),
            f"{prefix}top1_weight": self._scalarize(aux["top1_weight"]),
            f"{prefix}top4_weight_sum": self._scalarize(aux["top4_weight_sum"]),
        }
        if "slot_weight_mean" in aux:
            summary[f"{prefix}slot_weight_mean"] = self._scalarize(aux["slot_weight_mean"])
            summary[f"{prefix}slot_weight_max"] = self._scalarize(aux["slot_weight_max"])
        return summary

    def _set_trainable(self, prompt_on: bool, meta_on: bool):
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        for p in model.prompt_learner.parameters():
            p.requires_grad_(prompt_on)

        for p in model.prior_adapter.parameters():
            p.requires_grad_(meta_on)

        for name, p in model.named_parameters():
            if ("prompt_learner" not in name) and ("prior_adapter" not in name):
                p.requires_grad_(False)

    def _backward_and_step(self, loss, optimizer, zero_all: bool = False):
        if zero_all:
            self._zero_all_optimizers()

        optimizer.zero_grad()

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()

    def _zero_all_optimizers(self):
        self.optim_prompt.zero_grad()
        self.optim_meta.zero_grad()

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        slot_id = batch.get("slot_id", None)

        input = input.to(self.device)
        label = label.to(self.device)

        if slot_id is not None:
            slot_id = slot_id.to(self.device)

        self._current_label = label
        return input, label, slot_id

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            if name == "prompt_learner":
                if "token_prefix" in state_dict:
                    del state_dict["token_prefix"]
                if "token_suffix" in state_dict:
                    del state_dict["token_suffix"]

            print(f'Loading weights to {name} from "{model_path}" (epoch = {epoch})')
            self._models[name].load_state_dict(state_dict, strict=False)

    @staticmethod
    def _detach_aux(aux: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in aux.items():
            out[k] = v.detach() if torch.is_tensor(v) else v
        return out
