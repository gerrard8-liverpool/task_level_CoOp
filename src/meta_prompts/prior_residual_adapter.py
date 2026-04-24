from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .meta_network import TaskMetaNetwork
from .task_feature_loader import TaskFeatureLoader


@dataclass
class PriorResidualOutput:
    """Container for prior-residual adapter outputs."""

    task_features: torch.Tensor
    a0: torch.Tensor
    b0: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    b_logits: torch.Tensor
    delta_a: torch.Tensor
    delta_b: torch.Tensor
    meff: torch.Tensor
    keff: torch.Tensor
    lambda_t: torch.Tensor
    slot_weights: torch.Tensor
    b_entropy: torch.Tensor
    top1_weight: torch.Tensor
    top4_weight_sum: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "task_features": self.task_features,
            "a0": self.a0,
            "b0": self.b0,
            "a": self.a,
            "b": self.b,
            "b_logits": self.b_logits,
            "delta_a": self.delta_a,
            "delta_b": self.delta_b,
            "meff": self.meff,
            "keff": self.keff,
            "lambda_t": self.lambda_t,
            "slot_weights": self.slot_weights,
            "b_entropy": self.b_entropy,
            "top1_weight": self.top1_weight,
            "top4_weight_sum": self.top4_weight_sum,
        }


class PriorResidualAdapter(nn.Module):
    """Dataset-conditioned prior + residual joint adapter for CoOp.

    This module:
      1. loads a dataset-level task feature vector z_D,
      2. predicts prior gates (a0, b0) with a light meta-network,
      3. refines them by learnable residual logits (delta_a, delta_b),
      4. exposes warm-up / ramp-up aware lambda_t.
    """

    def __init__(
        self,
        task_feat_path: str,
        task_feat_mode: str,
        n_ctx: int,
        kmax: int,
        hidden_dim: int,
        gate_temperature: float,
        init_gate_bias: float,
        warmup_epochs: int,
        ramp_epochs: int,
        lambda_max: float,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.n_ctx = int(n_ctx)
        self.kmax = int(kmax)
        self.warmup_epochs = int(warmup_epochs)
        self.ramp_epochs = int(ramp_epochs)
        self.lambda_max = float(lambda_max)
        self.logit_eps = 1e-6

        loader = TaskFeatureLoader(
            json_path=task_feat_path,
            mode=task_feat_mode,
            device=device or "cpu",
        )
        batch = loader.load()
        task_x = batch.x.float()
        if task_x.dim() != 2 or task_x.size(0) != 1:
            raise ValueError(f"Expected task feature tensor with shape [1, d], got {tuple(task_x.shape)}")

        self.feature_names = batch.feature_names
        self.feature_mode = batch.feature_mode
        self.task_feat_path = task_feat_path

        self.register_buffer("task_features", task_x, persistent=True)

        self.meta_net = TaskMetaNetwork(
            input_dim=task_x.size(-1),
            hidden_dim=hidden_dim,
            mmax=self.n_ctx,
            kmax=self.kmax,
            gate_temperature=gate_temperature,
            init_gate_bias=init_gate_bias,
        )

        self.delta_a = nn.Parameter(torch.zeros(self.n_ctx, dtype=torch.float32))
        self.delta_b = nn.Parameter(torch.zeros(self.kmax, dtype=torch.float32))

    def forward(self, epoch=None) -> Dict[str, torch.Tensor]:
        meta_out = self.meta_net(self.task_features)

        a0 = meta_out.context_gates.squeeze(0)
        b0 = meta_out.sample_gates.squeeze(0)

        a = torch.sigmoid(self._safe_logit(a0) + self.delta_a)
        b_logits = self._safe_logit(b0) + self.delta_b
        b = torch.sigmoid(b_logits)

        meff = a.sum()

        slot_weights = torch.softmax(b_logits, dim=0)
        keff = 1.0 / (slot_weights.pow(2).sum() + 1e-12)
        b_entropy = -(slot_weights * torch.log(slot_weights + 1e-12)).sum()
        top1_weight = slot_weights.max()
        top4_weight_sum = torch.topk(slot_weights, k=min(4, slot_weights.numel())).values.sum()

        lambda_t = torch.tensor(self.compute_lambda_t(epoch), device=a.device, dtype=a.dtype)

        out = PriorResidualOutput(
            task_features=self.task_features.squeeze(0),
            a0=a0,
            b0=b0,
            a=a,
            b=b,
            b_logits=b_logits,
            delta_a=self.delta_a,
            delta_b=self.delta_b,
            meff=meff.unsqueeze(0),
            keff=keff.unsqueeze(0),
            lambda_t=lambda_t.unsqueeze(0),
            slot_weights=slot_weights,
            b_entropy=b_entropy.unsqueeze(0),
            top1_weight=top1_weight.unsqueeze(0),
            top4_weight_sum=top4_weight_sum.unsqueeze(0),
        )
        return out.as_dict()

    def compute_lambda_t(self, epoch=None) -> float:
        """Warm-up + linear ramp-up schedule.

        Args:
            epoch: zero-based epoch index from Dassl trainer.
        """
        if epoch is None:
            return self.lambda_max

        one_based_epoch = int(epoch) + 1

        if one_based_epoch <= self.warmup_epochs:
            return 0.0

        if self.ramp_epochs <= 0:
            return self.lambda_max

        progress = (one_based_epoch - self.warmup_epochs) / float(self.ramp_epochs)
        progress = max(0.0, min(1.0, progress))
        return self.lambda_max * progress

    def extra_stats(self, epoch=None) -> Dict[str, float]:
        with torch.no_grad():
            out = self.forward(epoch)
            stats = {
                "lambda_t": float(out["lambda_t"].item()),
                "meff": float(out["meff"].item()),
                "keff": float(out["keff"].item()),
                "a0_mean": float(out["a0"].mean().item()),
                "a_mean": float(out["a"].mean().item()),
                "b0_mean": float(out["b0"].mean().item()),
                "b_mean": float(out["b"].mean().item()),
                "delta_a_norm": float(out["delta_a"].norm().item()),
                "delta_b_norm": float(out["delta_b"].norm().item()),
                "b_entropy": float(out["b_entropy"].item()),
                "top1_weight": float(out["top1_weight"].item()),
                "top4_weight_sum": float(out["top4_weight_sum"].item()),
            }
        return stats

    def _safe_logit(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(self.logit_eps, 1.0 - self.logit_eps)
        return torch.logit(x)
