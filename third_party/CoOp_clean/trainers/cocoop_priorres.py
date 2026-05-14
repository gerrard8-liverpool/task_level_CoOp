from __future__ import annotations

import os
import os.path as osp
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict

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

# Make /workspace/meta_prompt_1/src importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from meta_prompts.prior_residual_adapter import PriorResidualAdapter

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


class PromptLearnerCoCoOpPriorRes(nn.Module):
    """CoCoOp prompt learner with dataset-prior safe residual on base context.

    Final context:
        ctx_eff(x, D) = ctx + lambda_t * (a_D - a_D0) * u_ctx + pi(x)

    At initialization, delta_a = 0, therefore a_D = a_D0 and the model starts
    exactly from vanilla CoCoOp:
        ctx_eff(x, D) = ctx + pi(x)
    """

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COCOOP.N_CTX
        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
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
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        self.u_ctx = nn.Parameter(torch.zeros_like(ctx_vectors))
        nn.init.normal_(self.u_ctx, std=0.02)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim)),
        ]))

        if dtype == torch.float16:
            self.meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim
        self.tokenized_prompts = tokenized_prompts
        self.use_context_gating = bool(cfg.TRAINER.COOP.USE_CONTEXT_GATING)

        self.use_legacy_residual = _env_flag("USE_LEGACY_RESIDUAL", default=False)
        mode = "legacy:(a-1)" if self.use_legacy_residual else "identity-centered:(a-a0)"
        print(f"[CoCoOpPriorRes] residual modulation mode = {mode}")

    def _apply_residual_modulation(
        self,
        context_gates: torch.Tensor = None,
        lambda_t: torch.Tensor = None,
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
            centered_gate = gate - 1.0
        else:
            if context_gates_base is None:
                gate_base = gate.detach()
            else:
                gate_base = context_gates_base.to(device=ctx.device, dtype=ctx.dtype).detach()
                if gate_base.dim() == 2:
                    gate_base = gate_base.squeeze(0)
            centered_gate = gate - gate_base

        lam = lambda_t.to(device=ctx.device, dtype=ctx.dtype).view(1)
        shift = centered_gate.unsqueeze(-1) * self.u_ctx
        return ctx + lam * shift

    def forward(
        self,
        im_features,
        context_gates: torch.Tensor = None,
        lambda_t: torch.Tensor = None,
        context_gates_base: torch.Tensor = None,
    ):
        # Dataset-level prior residual first: ctx + delta_D
        ctx = self._apply_residual_modulation(
            context_gates=context_gates,
            lambda_t=lambda_t,
            context_gates_base=context_gates_base,
        )  # [n_ctx, ctx_dim]

        # Instance-level CoCoOp bias: pi(x)
        bias = self.meta_net(im_features)  # [B, ctx_dim]
        bias = bias.unsqueeze(1)           # [B, 1, ctx_dim]
        ctx = ctx.unsqueeze(0)             # [1, n_ctx, ctx_dim]
        ctx_shifted = ctx + bias           # [B, n_ctx, ctx_dim]

        prompts = []
        prefix = self.token_prefix
        suffix = self.token_suffix

        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = torch.cat([prefix, ctx_i, suffix], dim=1)
            prompts.append(pts_i)

        prompts = torch.stack(prompts)
        return prompts


class CustomCLIPCoCoOpPriorRes(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        self.prompt_learner = PromptLearnerCoCoOpPriorRes(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        if not cfg.TRAINER.COOP.TASK_FEAT_PATH:
            raise ValueError("TRAINER.COOP.TASK_FEAT_PATH must be set for CoCoOpPriorRes")

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
            image_features,
            context_gates=adapter_out["a"],
            lambda_t=adapter_out["lambda_t"],
            context_gates_base=adapter_out.get("a0", None),
        )

        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            imf_i = imf_i / imf_i.norm()
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)

        logits = torch.stack(logits)

        if epoch is not None:
            return logits, adapter_out
        return logits


@TRAINER_REGISTRY.register()
class CoCoOpPriorRes(TrainerX):
    """Dataset-prior residual adapter on top of CoCoOp."""

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COCOOP.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building custom CLIP: CoCoOp + Safe PriorRes")
        self.model = CustomCLIPCoCoOpPriorRes(cfg, classnames, clip_model)

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

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        cfg = self.cfg

        prec = cfg.TRAINER.COCOOP.PREC
        if prec == "amp":
            with autocast():
                output, aux = self.model(image, epoch=self.epoch)
                loss = self._compose_total_loss(output, label, aux)
            self.optim_prompt.zero_grad()
            self.optim_meta.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim_prompt)
            self.scaler.step(self.optim_meta)
            self.scaler.update()
        else:
            output, aux = self.model(image, epoch=self.epoch)
            loss = self._compose_total_loss(output, label, aux)
            self.optim_prompt.zero_grad()
            self.optim_meta.zero_grad()
            loss.backward()
            self.optim_prompt.step()
            self.optim_meta.step()

        loss_summary = self._build_summary(loss.detach(), output.detach(), self._detach_aux(aux), label)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def _compose_total_loss(
        self,
        output: torch.Tensor,
        label: torch.Tensor,
        aux: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        cfg = self.cfg
        loss_cls = F.cross_entropy(output, label)

        reg_a = float(cfg.TRAINER.COOP.DELTA_A_REG) * aux["delta_a"].pow(2).sum()
        reg_b = float(cfg.TRAINER.COOP.DELTA_B_REG) * aux["delta_b"].pow(2).sum()

        total = loss_cls + reg_a + reg_b

        aux["loss_cls"] = loss_cls.detach()
        aux["loss_b"] = torch.zeros((), device=output.device, dtype=output.dtype)
        aux["reg_a"] = reg_a.detach()
        aux["reg_b"] = reg_b.detach()
        aux["loss_total"] = total.detach()
        return total

    @staticmethod
    def _scalarize(x):
        if torch.is_tensor(x):
            if x.numel() == 1:
                return float(x.item())
            return float(x.float().mean().item())
        return float(x)

    def _build_summary(self, loss, output, aux, label):
        return {
            "loss": self._scalarize(loss),
            "acc": float(compute_accuracy(output, label)[0].item()),
            "loss_cls": self._scalarize(aux["loss_cls"]),
            "reg_a": self._scalarize(aux["reg_a"]),
            "reg_b": self._scalarize(aux["reg_b"]),
            "lambda_t": self._scalarize(aux["lambda_t"]),
            "meff": self._scalarize(aux["meff"]),
            "keff": self._scalarize(aux["keff"]),
            "a0_mean": float(aux["a0"].float().mean().item()),
            "a_mean": float(aux["a"].float().mean().item()),
            "b0_mean": float(aux["b0"].float().mean().item()),
            "b_mean": float(aux["b"].float().mean().item()),
            "delta_a_norm": float(aux["delta_a"].float().norm().item()),
            "delta_b_norm": float(aux["delta_b"].float().norm().item()),
            "b_entropy": self._scalarize(aux["b_entropy"]),
            "top1_weight": self._scalarize(aux["top1_weight"]),
            "top4_weight_sum": self._scalarize(aux["top4_weight_sum"]),
        }

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

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
            ckpt_epoch = checkpoint["epoch"]

            if name == "prompt_learner":
                # Needed for base-to-new evaluation because classnames change.
                state_dict.pop("token_prefix", None)
                state_dict.pop("token_suffix", None)
                state_dict.pop("tokenized_prompts", None)

            print(f'Loading weights to {name} from "{model_path}" (epoch = {ckpt_epoch})')
            self._models[name].load_state_dict(state_dict, strict=False)

    @staticmethod
    def _detach_aux(aux: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in aux.items():
            out[k] = v.detach() if torch.is_tensor(v) else v
        return out
