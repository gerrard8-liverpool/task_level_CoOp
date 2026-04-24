from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Union

import torch

FeatureMode = Literal["auto", "raw", "transformed", "normalized"]


@dataclass
class TaskFeatureBatch:
    json_path: str
    dataset_name: str
    split_name: str
    backbone: str
    feature_mode: str
    feature_names: List[str]
    x: torch.Tensor
    regime_k: Optional[int]
    base_feature_names: List[str]
    base_feature_values: List[float]
    regime_feature_names: List[str]
    regime_feature_values: List[float]


class TaskFeatureLoader:
    RAW_FEATURE_ORDER = ["C", "Ssemantic", "Dintra", "Dinter"]
    TRANSFORMED_FEATURE_ORDER = ["log_C", "Ssemantic", "log_Dintra", "log_Dinter"]

    def __init__(
        self,
        json_path: Union[str, Path],
        mode: FeatureMode = "auto",
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        regime_k: Optional[int] = None,
        regime_eps: float = 1e-8,
        append_regime_feature: bool = True,
    ) -> None:
        self.json_path = str(json_path)
        self.mode = mode
        self.device = torch.device(device)
        self.dtype = dtype
        self.regime_k = regime_k
        self.regime_eps = float(regime_eps)
        self.append_regime_feature = bool(append_regime_feature)

        if self.mode not in {"auto", "raw", "transformed", "normalized"}:
            raise ValueError(f"Unsupported mode: {self.mode}")
        if self.regime_k is not None and self.regime_k < 0:
            raise ValueError(f"regime_k must be non-negative, got {self.regime_k}")

    def load(self) -> TaskFeatureBatch:
        data = self._read_json(self.json_path)
        feature_mode, feature_dict = self._select_feature_dict(data)
        base_feature_names = self._ordered_feature_names(feature_mode, feature_dict)
        base_feature_values = [float(feature_dict[name]) for name in base_feature_names]

        regime_feature_names: List[str] = []
        regime_feature_values: List[float] = []

        final_feature_names = list(base_feature_names)
        final_feature_values = list(base_feature_values)

        if self.append_regime_feature and self.regime_k is not None:
            regime_value = math.log(float(self.regime_k) + self.regime_eps)
            regime_feature_names = ["log_k_shot"]
            regime_feature_values = [regime_value]
            final_feature_names.extend(regime_feature_names)
            final_feature_values.extend(regime_feature_values)
            resolved_mode = f"{feature_mode}+regime"
        else:
            resolved_mode = feature_mode

        x = torch.tensor(final_feature_values, dtype=self.dtype, device=self.device).unsqueeze(0)

        return TaskFeatureBatch(
            json_path=self.json_path,
            dataset_name=str(data.get("dataset_name", "")),
            split_name=str(data.get("split_name", "")),
            backbone=str(data.get("backbone", "")),
            feature_mode=resolved_mode,
            feature_names=final_feature_names,
            x=x,
            regime_k=self.regime_k,
            base_feature_names=base_feature_names,
            base_feature_values=base_feature_values,
            regime_feature_names=regime_feature_names,
            regime_feature_values=regime_feature_values,
        )

    @staticmethod
    def _read_json(path: Union[str, Path]) -> dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Task feature JSON not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _select_feature_dict(self, data: dict) -> tuple[str, dict]:
        raw = data.get("raw_features")
        transformed = data.get("transformed_features")
        normalized = data.get("normalized_features")

        if self.mode == "raw":
            if raw is None:
                raise ValueError(f"raw_features missing in {self.json_path}")
            return "raw", raw

        if self.mode == "transformed":
            if transformed is None:
                raise ValueError(f"transformed_features missing in {self.json_path}")
            return "transformed", transformed

        if self.mode == "normalized":
            if normalized is None:
                raise ValueError(f"normalized_features missing in {self.json_path}")
            return "normalized", normalized

        if normalized is not None:
            return "normalized", normalized
        if transformed is not None:
            return "transformed", transformed
        if raw is not None:
            return "raw", raw
        raise ValueError(f"No usable feature dict found in {self.json_path}")

    def _ordered_feature_names(self, feature_mode: str, feature_dict: dict) -> List[str]:
        if feature_mode == "raw":
            preferred_order: Sequence[str] = self.RAW_FEATURE_ORDER
        else:
            preferred_order = self.TRANSFORMED_FEATURE_ORDER

        if all(name in feature_dict for name in preferred_order):
            return list(preferred_order)

        return list(feature_dict.keys())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke test for task feature loading with optional regime feature.")
    parser.add_argument("json_path", type=str, help="Path to task feature JSON")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "raw", "transformed", "normalized"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--regime-k", type=int, default=None, help="If provided, append log(k) as regime feature")
    args = parser.parse_args()

    loader = TaskFeatureLoader(
        json_path=args.json_path,
        mode=args.mode,  # type: ignore[arg-type]
        device=args.device,
        regime_k=args.regime_k,
    )
    batch = loader.load()

    report = {
        "json_path": batch.json_path,
        "dataset_name": batch.dataset_name,
        "split_name": batch.split_name,
        "backbone": batch.backbone,
        "feature_mode": batch.feature_mode,
        "feature_names": batch.feature_names,
        "tensor_shape": list(batch.x.shape),
        "tensor": batch.x.detach().cpu().tolist(),
        "base_feature_names": batch.base_feature_names,
        "base_feature_values": batch.base_feature_values,
        "regime_k": batch.regime_k,
        "regime_feature_names": batch.regime_feature_names,
        "regime_feature_values": batch.regime_feature_values,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
