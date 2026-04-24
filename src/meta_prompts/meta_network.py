from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class MetaNetworkOutput:
    """Container for meta-network outputs.

    Attributes:
        context_gates: Tensor of shape [B, mmax] in [0, 1].
        sample_gates: Tensor of shape [B, kmax] in [0, 1].
        meff: Effective context length, shape [B].
        keff: Effective sample count, shape [B].
        context_logits: Pre-sigmoid logits for context gates.
        sample_logits: Pre-sigmoid logits for sample gates.
        hidden_1: First hidden representation.
        hidden_2: Second hidden representation.
    """

    context_gates: torch.Tensor
    sample_gates: torch.Tensor
    meff: torch.Tensor
    keff: torch.Tensor
    context_logits: torch.Tensor
    sample_logits: torch.Tensor
    hidden_1: torch.Tensor
    hidden_2: torch.Tensor


class TaskMetaNetwork(nn.Module):
    """Two-layer MLP meta-network for continuous hyperparameter gating.

    Design aligned with the user's proposal:
      - input_dim = 4 task statistics
      - two shared hidden layers
      - dual heads for context gates a and sample gates b
      - sigmoid outputs in [0, 1]

    Default dimensions match the first-stage CoOp validation setup:
      input_dim=4, hidden_dim=64, mmax=16, kmax=16
    which yields 6560 parameters.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        mmax: int = 16,
        kmax: int = 16,
        gate_temperature: float = 1.0,
        init_gate_bias: float = 0.0,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if mmax <= 0:
            raise ValueError(f"mmax must be positive, got {mmax}")
        if kmax <= 0:
            raise ValueError(f"kmax must be positive, got {kmax}")
        if gate_temperature <= 0:
            raise ValueError(f"gate_temperature must be positive, got {gate_temperature}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mmax = mmax
        self.kmax = kmax
        self.gate_temperature = float(gate_temperature)
        self.init_gate_bias = float(init_gate_bias)

        # Shared trunk.
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ReLU(inplace=False)

        # Dual heads.
        self.context_head = nn.Linear(hidden_dim, mmax)
        self.sample_head = nn.Linear(hidden_dim, kmax)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights with Xavier and biases with zeros/head bias value."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        nn.init.xavier_uniform_(self.context_head.weight)
        nn.init.constant_(self.context_head.bias, self.init_gate_bias)

        nn.init.xavier_uniform_(self.sample_head.weight)
        nn.init.constant_(self.sample_head.bias, self.init_gate_bias)

    def forward(self, x: torch.Tensor) -> MetaNetworkOutput:
        """Forward pass.

        Args:
            x: Task feature tensor of shape [4] or [B, 4].

        Returns:
            MetaNetworkOutput containing gates and effective hyperparameters.
        """
        x = self._normalize_input_shape(x)

        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, but got tensor with last dim {x.size(-1)}"
            )

        hidden_1 = self.act(self.fc1(x))
        hidden_2 = self.act(self.fc2(hidden_1))

        context_logits = self.context_head(hidden_2)
        sample_logits = self.sample_head(hidden_2)

        context_gates = torch.sigmoid(context_logits / self.gate_temperature)
        sample_gates = torch.sigmoid(sample_logits / self.gate_temperature)

        meff = context_gates.sum(dim=-1)
        keff = sample_gates.sum(dim=-1)

        return MetaNetworkOutput(
            context_gates=context_gates,
            sample_gates=sample_gates,
            meff=meff,
            keff=keff,
            context_logits=context_logits,
            sample_logits=sample_logits,
            hidden_1=hidden_1,
            hidden_2=hidden_2,
        )

    def predict_gates(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience wrapper returning only (a, b)."""
        out = self.forward(x)
        return out.context_gates, out.sample_gates

    def parameter_count(self, trainable_only: bool = True) -> int:
        params = self.parameters() if not trainable_only else (p for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in params)

    def expected_parameter_count(self) -> int:
        """Closed-form parameter count from the architecture definition."""
        return (
            self.input_dim * self.hidden_dim + self.hidden_dim
            + self.hidden_dim * self.hidden_dim + self.hidden_dim
            + self.hidden_dim * self.mmax + self.mmax
            + self.hidden_dim * self.kmax + self.kmax
        )

    @staticmethod
    def _normalize_input_shape(x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input x must be a torch.Tensor, got {type(x)}")
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [4] -> [1, 4]
        elif x.dim() != 2:
            raise ValueError(f"Input x must have shape [4] or [B, 4], got {tuple(x.shape)}")
        return x

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"mmax={self.mmax}, kmax={self.kmax}, gate_temperature={self.gate_temperature}"
        )


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Smoke test for the task meta-network.")
    parser.add_argument("--json-path", type=str, default=None, help="Optional task feature JSON path")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "raw", "transformed", "normalized"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--mmax", type=int, default=16)
    parser.add_argument("--kmax", type=int, default=16)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.json_path is not None:
        # Local import to avoid a hard dependency when this file is used standalone.
        from task_feature_loader import TaskFeatureLoader

        loader = TaskFeatureLoader(
            json_path=args.json_path,
            mode=args.mode,  # type: ignore[arg-type]
            device=device,
        )
        batch = loader.load()
        x = batch.x
        input_summary = {
            "json_path": str(Path(args.json_path)),
            "feature_mode": batch.feature_mode,
            "feature_names": batch.feature_names,
            "input_tensor": x.detach().cpu().tolist(),
        }
    else:
        x = torch.randn(1, 4, device=device)
        input_summary = {
            "json_path": None,
            "feature_mode": "random",
            "feature_names": ["f1", "f2", "f3", "f4"],
            "input_tensor": x.detach().cpu().tolist(),
        }

    model = TaskMetaNetwork(
        input_dim=x.shape[-1],
        hidden_dim=args.hidden_dim,
        mmax=args.mmax,
        kmax=args.kmax,
        gate_temperature=args.gate_temperature,
    ).to(device)

    out = model(x)

    report = {
        "model": {
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "mmax": model.mmax,
            "kmax": model.kmax,
            "gate_temperature": model.gate_temperature,
            "parameter_count": model.parameter_count(),
            "expected_parameter_count": model.expected_parameter_count(),
        },
        "input": input_summary,
        "output": {
            "context_gates_shape": list(out.context_gates.shape),
            "sample_gates_shape": list(out.sample_gates.shape),
            "meff_shape": list(out.meff.shape),
            "keff_shape": list(out.keff.shape),
            "context_gates": out.context_gates.detach().cpu().tolist(),
            "sample_gates": out.sample_gates.detach().cpu().tolist(),
            "meff": out.meff.detach().cpu().tolist(),
            "keff": out.keff.detach().cpu().tolist(),
        },
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
