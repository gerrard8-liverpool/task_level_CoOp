from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import torch

FeatureMode = Literal["auto", "raw", "transformed", "normalized"]

RAW_ORDER = ["C", "Ssemantic", "Dintra", "Dinter"]
TRANSFORMED_ORDER = ["log_C", "Ssemantic", "log_Dintra", "log_Dinter"]
NORMALIZED_ORDER = ["norm_log_C", "norm_Ssemantic", "norm_log_Dintra", "norm_log_Dinter"]


@dataclass
class TaskFeatureBatch:
    """Container for a dataset-level task feature vector.

    Attributes:
        x: Feature tensor of shape [1, 4] by default.
        feature_names: Ordered feature names matching x.
        feature_mode: Which JSON section was used.
        dataset_name: Dataset name from the JSON.
        split_name: Split name from the JSON.
        backbone: Backbone name from the JSON.
        metadata: Full parsed JSON dictionary for optional downstream use.
    """

    x: torch.Tensor
    feature_names: List[str]
    feature_mode: str
    dataset_name: str
    split_name: str
    backbone: str
    metadata: Dict


class TaskFeatureLoader:
    """Load dataset-level task features extracted offline.

    Expected JSON structure is aligned with task_feature_extractor.py, e.g.

    {
      "dataset_name": "OxfordPets",
      "split_name": "train_x",
      "backbone": "RN50",
      "raw_features": {...},
      "transformed_features": {...},
      "normalized_features": null
    }

    Default behavior:
      - mode="auto": prefer normalized_features if available, otherwise transformed_features,
        otherwise raw_features.
      - return tensor shape [1, 4] for direct use in an MLP.
    """

    def __init__(
        self,
        json_path: str | Path,
        mode: FeatureMode = "auto",
        device: Optional[str | torch.device] = None,
        dtype: torch.dtype = torch.float32,
        add_batch_dim: bool = True,
        strict: bool = True,
    ) -> None:
        self.json_path = Path(json_path)
        self.mode = mode
        self.device = torch.device(device) if device is not None else None
        self.dtype = dtype
        self.add_batch_dim = add_batch_dim
        self.strict = strict

        self._cache: Optional[TaskFeatureBatch] = None

    def load(self, force_reload: bool = False) -> TaskFeatureBatch:
        if self._cache is not None and not force_reload:
            return self._cache

        data = self._read_json(self.json_path)
        resolved_mode, feature_names, values = self._select_features(data, self.mode)

        x = torch.tensor(values, dtype=self.dtype)
        if self.add_batch_dim:
            x = x.unsqueeze(0)  # [1, 4]
        if self.device is not None:
            x = x.to(self.device)

        batch = TaskFeatureBatch(
            x=x,
            feature_names=feature_names,
            feature_mode=resolved_mode,
            dataset_name=str(data.get("dataset_name", "")),
            split_name=str(data.get("split_name", "")),
            backbone=str(data.get("backbone", "")),
            metadata=data,
        )
        self._cache = batch
        return batch

    def get_tensor(self, force_reload: bool = False) -> torch.Tensor:
        return self.load(force_reload=force_reload).x

    def summary(self, force_reload: bool = False) -> Dict:
        batch = self.load(force_reload=force_reload)
        return {
            "json_path": str(self.json_path),
            "dataset_name": batch.dataset_name,
            "split_name": batch.split_name,
            "backbone": batch.backbone,
            "feature_mode": batch.feature_mode,
            "feature_names": batch.feature_names,
            "shape": list(batch.x.shape),
            "dtype": str(batch.x.dtype),
            "device": str(batch.x.device),
        }

    @staticmethod
    def _read_json(path: Path) -> Dict:
        if not path.is_file():
            raise FileNotFoundError(f"Task feature JSON not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Task feature JSON must decode to a dict, got {type(data)}")
        return data

    def _select_features(
        self,
        data: Dict,
        mode: FeatureMode,
    ) -> Tuple[str, List[str], List[float]]:
        if mode == "auto":
            if self._is_valid_feature_block(data.get("normalized_features"), NORMALIZED_ORDER):
                return "normalized", NORMALIZED_ORDER, self._ordered_values(data["normalized_features"], NORMALIZED_ORDER)
            if self._is_valid_feature_block(data.get("transformed_features"), TRANSFORMED_ORDER):
                return "transformed", TRANSFORMED_ORDER, self._ordered_values(data["transformed_features"], TRANSFORMED_ORDER)
            if self._is_valid_feature_block(data.get("raw_features"), RAW_ORDER):
                return "raw", RAW_ORDER, self._ordered_values(data["raw_features"], RAW_ORDER)
            raise ValueError(
                "No valid feature block found in JSON. Expected one of normalized_features, "
                "transformed_features, or raw_features."
            )

        if mode == "normalized":
            block = data.get("normalized_features")
            order = NORMALIZED_ORDER
        elif mode == "transformed":
            block = data.get("transformed_features")
            order = TRANSFORMED_ORDER
        elif mode == "raw":
            block = data.get("raw_features")
            order = RAW_ORDER
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if not self._is_valid_feature_block(block, order):
            msg = (
                f"Requested mode='{mode}', but JSON block is missing or invalid. "
                f"Expected keys: {order}"
            )
            if self.strict:
                raise ValueError(msg)
            # fallback only when strict=False
            return self._select_features(data, "auto")

        return mode, order, self._ordered_values(block, order)

    @staticmethod
    def _is_valid_feature_block(block: Optional[Dict], order: List[str]) -> bool:
        if block is None or not isinstance(block, dict):
            return False
        return all(k in block and block[k] is not None for k in order)

    @staticmethod
    def _ordered_values(block: Dict, order: List[str]) -> List[float]:
        values = []
        for key in order:
            val = block[key]
            if not isinstance(val, (int, float)):
                raise TypeError(f"Feature '{key}' must be numeric, got {type(val)}")
            values.append(float(val))
        return values


# Optional convenience function for one-line use.
def load_task_features(
    json_path: str | Path,
    mode: FeatureMode = "auto",
    device: Optional[str | torch.device] = None,
    dtype: torch.dtype = torch.float32,
    add_batch_dim: bool = True,
    strict: bool = True,
) -> TaskFeatureBatch:
    loader = TaskFeatureLoader(
        json_path=json_path,
        mode=mode,
        device=device,
        dtype=dtype,
        add_batch_dim=add_batch_dim,
        strict=strict,
    )
    return loader.load()


if __name__ == "__main__":
    import argparse
    import pprint

    parser = argparse.ArgumentParser(description="Inspect a task feature JSON and load it as a tensor.")
    parser.add_argument("json_path", type=str, help="Path to extracted task feature JSON")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "raw", "transformed", "normalized"])
    parser.add_argument("--no-batch-dim", action="store_true", help="Return shape [4] instead of [1, 4]")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    batch = load_task_features(
        json_path=args.json_path,
        mode=args.mode,  # type: ignore[arg-type]
        device=args.device,
        add_batch_dim=not args.no_batch_dim,
    )

    pprint.pprint({
        "dataset_name": batch.dataset_name,
        "split_name": batch.split_name,
        "backbone": batch.backbone,
        "feature_mode": batch.feature_mode,
        "feature_names": batch.feature_names,
        "tensor_shape": list(batch.x.shape),
        "tensor": batch.x.detach().cpu().tolist(),
    })
