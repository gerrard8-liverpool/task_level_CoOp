#!/usr/bin/env python3
"""Extract dataset-level task features for CoOp-style prompt learning.

This script follows the user's design document:
  f(D) = [C, Ssemantic, Dintra, Dinter]
with transformed features
  f_tilde(D) = [log(C), Ssemantic, log(Dintra + eps), log(Dinter + eps)]
computed from the training split only, using frozen CLIP encoders.

It is designed to work with the original KaiyangZhou/CoOp codebase (and its Dassl dataset
registry), while also supporting custom datasets that are registered in the same way.

Example:
    python task_feature_extractor.py \
        --coop-root /workspace/meta_prompt_1/third_party/CoOp \
        --root /workspace/datasets \
        --dataset-config-file /workspace/meta_prompt_1/third_party/CoOp/configs/datasets/oxford_pets.yaml \
        --dataset OxfordPets \
        --backbone RN50 \
        --output /workspace/meta_prompt_1/outputs/task_features/oxford_pets_train.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# -----------------------------
# CoOp/Dassl bootstrapping
# -----------------------------
def bootstrap_coop_imports(coop_root: str) -> None:
    """Add CoOp repo paths to sys.path so local modules can be imported.

    Supported layouts:
        A) <third_party>/CoOp/... and <third_party>/Dassl.pytorch/...
        B) <CoOp>/Dassl.pytorch/...
    """
    coop_root = os.path.abspath(coop_root)

    candidate_dassl_roots = [
        os.path.join(coop_root, "Dassl.pytorch"),
        os.path.join(os.path.dirname(coop_root), "Dassl.pytorch"),
    ]
    dassl_root = None
    for cand in candidate_dassl_roots:
        if os.path.isdir(cand):
            dassl_root = cand
            break

    if dassl_root is None:
        raise FileNotFoundError(
            "Cannot find Dassl.pytorch. Expected either '<coop_root>/Dassl.pytorch' "
            "or a sibling folder '<parent_of_coop_root>/Dassl.pytorch'."
        )

    for p in [coop_root, dassl_root]:
        if p not in sys.path:
            sys.path.insert(0, p)


# -----------------------------
# Config helpers (mirrors CoOp/train.py)
# -----------------------------
def extend_cfg(cfg) -> None:
    """Reproduce the CoOp-specific cfg extension from train.py."""
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16
    cfg.TRAINER.COOP.CSC = False
    cfg.TRAINER.COOP.CTX_INIT = ""
    cfg.TRAINER.COOP.PREC = "fp16"
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16
    cfg.TRAINER.COCOOP.CTX_INIT = ""
    cfg.TRAINER.COCOOP.PREC = "fp16"

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"


def build_cfg(args: argparse.Namespace):
    from dassl.config import get_cfg_default

    cfg = get_cfg_default()
    extend_cfg(cfg)

    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    if args.root:
        cfg.DATASET.ROOT = args.root
    if args.dataset:
        cfg.DATASET.NAME = args.dataset
    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone
    if args.num_shots is not None:
        cfg.DATASET.NUM_SHOTS = args.num_shots
    if args.seed is not None:
        cfg.SEED = args.seed
    if args.subsample_classes:
        cfg.DATASET.SUBSAMPLE_CLASSES = args.subsample_classes
    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    return cfg


# -----------------------------
# Dataset registry loading
# -----------------------------
def import_default_coop_modules() -> None:
    """Import the built-in CoOp datasets so they register into Dassl."""
    default_modules = [
        "datasets.oxford_pets",
        "datasets.oxford_flowers",
        "datasets.fgvc_aircraft",
        "datasets.dtd",
        "datasets.eurosat",
        "datasets.stanford_cars",
        "datasets.food101",
        "datasets.sun397",
        "datasets.caltech101",
        "datasets.ucf101",
        "datasets.imagenet",
        "datasets.imagenet_sketch",
        "datasets.imagenetv2",
        "datasets.imagenet_a",
        "datasets.imagenet_r",
    ]
    for mod in default_modules:
        try:
            importlib.import_module(mod)
        except Exception:
            # Some local forks may omit a subset of modules.
            pass


def import_user_modules(module_names: Sequence[str]) -> None:
    for mod in module_names:
        mod = mod.strip()
        if not mod:
            continue
        importlib.import_module(mod)


# -----------------------------
# CLIP loading
# -----------------------------
def load_clip_to_cpu(cfg):
    """Use CoOp's own CLIP loading path for exact compatibility."""
    from clip import clip

    backbone_name = cfg.MODEL.BACKBONE.NAME
    if backbone_name not in clip._MODELS:
        raise KeyError(
            f"Unknown CLIP backbone '{backbone_name}'. Available: {list(clip._MODELS.keys())}"
        )

    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
        model = None

    model = clip.build_model(state_dict or model.state_dict())
    return model


# -----------------------------
# Dataset wrapper for raw Datum list
# -----------------------------
class DatumImageDataset(Dataset):
    def __init__(self, data_source: Sequence, transform):
        self.data_source = list(data_source)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data_source)

    def __getitem__(self, index: int):
        item = self.data_source[index]
        impath = item.impath
        label = int(item.label)

        with Image.open(impath) as img:
            img = img.convert("RGB")
        img = self.transform(img)
        return {
            "img": img,
            "label": label,
            "impath": impath,
        }


@dataclass
class FeatureResult:
    dataset_name: str
    split_name: str
    backbone: str
    num_samples: int
    num_classes: int
    classnames: List[str]
    raw_features: Dict[str, float]
    transformed_features: Dict[str, float]
    normalized_features: Optional[Dict[str, float]]
    normalization_stats: Optional[Dict[str, List[float]]]
    eps: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_log(x: float, eps: float) -> float:
    return math.log(float(x) + eps)


def get_classnames(dataset) -> List[str]:
    if hasattr(dataset, "classnames") and dataset.classnames is not None:
        return [str(name).replace("_", " ") for name in dataset.classnames]

    if hasattr(dataset, "lab2cname") and dataset.lab2cname:
        max_lab = max(dataset.lab2cname.keys())
        return [str(dataset.lab2cname[i]).replace("_", " ") for i in range(max_lab + 1)]

    raise AttributeError(
        "Dataset does not expose 'classnames' or 'lab2cname'; cannot build text features."
    )


def select_split(dataset, split_name: str):
    if split_name not in {"train_x", "val", "test", "train_u"}:
        raise ValueError(f"Unsupported split '{split_name}'")
    data_source = getattr(dataset, split_name, None)
    if data_source is None:
        raise ValueError(f"Dataset has no split '{split_name}'")
    return data_source


def build_eval_transform(clip_model):
    from clip import clip

    resolution = int(clip_model.visual.input_resolution)
    return clip._transform(resolution)


@torch.no_grad()
def encode_text_features(
    classnames: Sequence[str],
    clip_model,
    device: torch.device,
    text_template: str,
) -> torch.Tensor:
    from clip import clip

    texts = [text_template.format(name) for name in classnames]
    tokenized = torch.cat([clip.tokenize(t) for t in texts]).to(device)
    text_features = clip_model.encode_text(tokenized)
    text_features = F.normalize(text_features.float(), dim=-1)
    return text_features


@torch.no_grad()
def encode_image_features(
    data_source: Sequence,
    clip_model,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    transform = build_eval_transform(clip_model)
    dataset = DatumImageDataset(data_source, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    features: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    impaths: List[str] = []

    for batch in tqdm(loader, desc="Extracting image features", leave=False):
        images = batch["img"].to(device)
        image_features = clip_model.encode_image(images)
        image_features = F.normalize(image_features.float(), dim=-1)

        features.append(image_features.cpu())
        labels.append(batch["label"].cpu())
        impaths.extend(batch["impath"])

    all_features = torch.cat(features, dim=0)
    all_labels = torch.cat(labels, dim=0)
    return all_features, all_labels, impaths


def compute_semantic_similarity(text_features: torch.Tensor) -> float:
    num_classes = int(text_features.shape[0])
    if num_classes <= 1:
        return 0.0

    sim_matrix = text_features @ text_features.t()
    triu_idx = torch.triu_indices(num_classes, num_classes, offset=1)
    return float(sim_matrix[triu_idx[0], triu_idx[1]].mean().item())


def compute_visual_statistics(
    image_features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    eps: float,
) -> Tuple[float, float, float, Dict[int, torch.Tensor], Dict[int, int]]:
    grouped: Dict[int, List[torch.Tensor]] = defaultdict(list)
    for feat, lab in zip(image_features, labels):
        grouped[int(lab.item())].append(feat)

    missing = [c for c in range(num_classes) if c not in grouped]
    if missing:
        raise ValueError(
            f"The selected split is missing classes {missing}. "
            "This usually means the split is not class-balanced or the few-shot construction is incomplete."
        )

    prototypes: Dict[int, torch.Tensor] = {}
    class_counts: Dict[int, int] = {}
    intra_values: List[torch.Tensor] = []

    for c in range(num_classes):
        class_feats = torch.stack(grouped[c], dim=0)
        mu_c = class_feats.mean(dim=0)
        prototypes[c] = mu_c
        class_counts[c] = int(class_feats.shape[0])

        diffs = class_feats - mu_c.unsqueeze(0)
        sq_dists = (diffs * diffs).sum(dim=-1)
        intra_c = sq_dists.mean()
        intra_values.append(intra_c)

    dintra = float(torch.stack(intra_values).mean().item())

    proto_tensor = torch.stack([prototypes[c] for c in range(num_classes)], dim=0)
    if num_classes <= 1:
        dinter = 0.0
    else:
        pairwise_sq = torch.cdist(proto_tensor, proto_tensor, p=2.0) ** 2
        triu_idx = torch.triu_indices(num_classes, num_classes, offset=1)
        dinter = float(pairwise_sq[triu_idx[0], triu_idx[1]].mean().item())

    rfisher = float(dinter / (dintra + eps))
    return dintra, dinter, rfisher, prototypes, class_counts


def normalize_transformed_feature(
    transformed: Dict[str, float],
    stats: Optional[Dict[str, List[float]]],
    eps: float,
) -> Optional[Dict[str, float]]:
    if stats is None:
        return None

    mu = stats["mean"]
    std = stats["std"]
    if len(mu) != 4 or len(std) != 4:
        raise ValueError("Normalization stats must contain 4-d mean/std vectors")

    values = [
        transformed["log_C"],
        transformed["Ssemantic"],
        transformed["log_Dintra"],
        transformed["log_Dinter"],
    ]
    normalized = [(v - m) / (s + eps) for v, m, s in zip(values, mu, std)]
    return {
        "norm_log_C": float(normalized[0]),
        "norm_Ssemantic": float(normalized[1]),
        "norm_log_Dintra": float(normalized[2]),
        "norm_log_Dinter": float(normalized[3]),
    }


def load_stats(stats_path: Optional[str]) -> Optional[Dict[str, List[float]]]:
    if not stats_path:
        return None
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    return stats


def maybe_mkdir_for_file(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def extract_features(args: argparse.Namespace) -> FeatureResult:
    bootstrap_coop_imports(args.coop_root)
    import_default_coop_modules()
    import_user_modules(args.extra_modules)

    if args.seed is not None and args.seed >= 0:
        seed_everything(args.seed)

    cfg = build_cfg(args)

    from dassl.data.datasets import build_dataset

    dataset = build_dataset(cfg)
    classnames = get_classnames(dataset)
    split_data = select_split(dataset, args.split)

    clip_model = load_clip_to_cpu(cfg)
    clip_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    clip_model = clip_model.to(device)

    text_features = encode_text_features(
        classnames=classnames,
        clip_model=clip_model,
        device=device,
        text_template=args.text_template,
    )
    semantic_similarity = compute_semantic_similarity(text_features.cpu())

    image_features, labels, _ = encode_image_features(
        data_source=split_data,
        clip_model=clip_model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    c = len(classnames)
    dintra, dinter, rfisher, _, class_counts = compute_visual_statistics(
        image_features=image_features,
        labels=labels,
        num_classes=c,
        eps=args.eps,
    )

    raw_features = {
        "C": float(c),
        "Ssemantic": float(semantic_similarity),
        "Dintra": float(dintra),
        "Dinter": float(dinter),
        "RFisher": float(rfisher),
    }
    transformed = {
        "log_C": float(safe_log(c, args.eps)),
        "Ssemantic": float(semantic_similarity),
        "log_Dintra": float(safe_log(dintra, args.eps)),
        "log_Dinter": float(safe_log(dinter, args.eps)),
    }

    stats = load_stats(args.stats_file)
    normalized = normalize_transformed_feature(transformed, stats, args.eps)

    result = FeatureResult(
        dataset_name=str(cfg.DATASET.NAME),
        split_name=args.split,
        backbone=str(cfg.MODEL.BACKBONE.NAME),
        num_samples=len(split_data),
        num_classes=c,
        classnames=list(classnames),
        raw_features=raw_features,
        transformed_features=transformed,
        normalized_features=normalized,
        normalization_stats=stats,
        eps=float(args.eps),
    )

    payload = asdict(result)
    payload["class_counts_in_selected_split"] = {str(k): int(v) for k, v in class_counts.items()}
    payload["config_summary"] = {
        "dataset_root": cfg.DATASET.ROOT,
        "dataset_name": cfg.DATASET.NAME,
        "num_shots": getattr(cfg.DATASET, "NUM_SHOTS", None),
        "subsample_classes": getattr(cfg.DATASET, "SUBSAMPLE_CLASSES", "all"),
        "backbone": cfg.MODEL.BACKBONE.NAME,
        "text_template": args.text_template,
    }

    if args.output:
        maybe_mkdir_for_file(args.output)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    if args.save_image_features:
        maybe_mkdir_for_file(args.save_image_features)
        torch.save(
            {
                "features": image_features,
                "labels": labels,
                "classnames": classnames,
                "dataset_name": str(cfg.DATASET.NAME),
                "split_name": args.split,
                "backbone": str(cfg.MODEL.BACKBONE.NAME),
            },
            args.save_image_features,
        )

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return result


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract task-level features for CoOp / meta-learned continuous hyperparameter optimization"
    )
    parser.add_argument("--coop-root", type=str, required=True, help="Path to the local CoOp repository root")
    parser.add_argument("--root", type=str, default="", help="Dataset root path (same meaning as CoOp --root)")
    parser.add_argument("--dataset", type=str, default="", help="Dataset registry name, e.g. OxfordPets")
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="Path to CoOp dataset yaml, e.g. configs/datasets/oxford_pets.yaml",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="",
        help="Optional method/config yaml if you want to reuse CoOp settings",
    )
    parser.add_argument("--backbone", type=str, default="RN50", help="CLIP backbone, e.g. RN50 / ViT-B/16")
    parser.add_argument(
        "--split",
        type=str,
        default="train_x",
        choices=["train_x", "val", "test", "train_u"],
        help="Which split to extract from. For your paper design, keep train_x.",
    )
    parser.add_argument("--num-shots", type=int, default=None, help="Few-shot setting if the dataset uses it")
    parser.add_argument("--subsample-classes", type=str, default="all", choices=["all", "base", "new"])
    parser.add_argument(
        "--text-template",
        type=str,
        default="{}",
        help="Text prompt for class names. Default '{}' matches the design doc's class-name-only setting.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument(
        "--extra-modules",
        nargs="*",
        default=[],
        help="Additional Python modules to import so custom datasets can register themselves",
    )
    parser.add_argument("--stats-file", type=str, default="", help="JSON file with mean/std for feature normalization")
    parser.add_argument("--output", type=str, default="", help="Where to save the extracted feature JSON")
    parser.add_argument(
        "--save-image-features",
        type=str,
        default="",
        help="Optional .pt file for cached per-image CLIP features",
    )
    parser.add_argument(
        "--transforms",
        type=str,
        nargs="+",
        default=None,
        help="Optional override for cfg.INPUT.TRANSFORMS (rarely needed here)",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Extra cfg overrides, same style as CoOp train.py",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_features(args)
