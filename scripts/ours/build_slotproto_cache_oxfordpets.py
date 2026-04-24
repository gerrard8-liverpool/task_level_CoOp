#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T

ROOT = Path("/workspace/meta_prompt_1")
THIRD_PARTY = ROOT / "third_party"
COOP_ROOT = THIRD_PARTY / "CoOp_clean"
DASSL_ROOT = THIRD_PARTY / "Dassl.pytorch"

sys.path.insert(0, str(COOP_ROOT))
sys.path.insert(0, str(DASSL_ROOT))

from clip import clip  # noqa: E402
from dassl.data.datasets import Datum  # noqa: E402


def load_clip_to_cpu(backbone_name: str):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model


def build_preprocess(input_resolution: int):
    return T.Compose([
        T.Resize(input_resolution, interpolation=Image.BICUBIC),
        T.CenterCrop(input_resolution),
        T.ToTensor(),
        T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


@torch.no_grad()
def extract_feature(visual, preprocess, impath: str, device: torch.device):
    img = Image.open(impath).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    feat = visual(x)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu()


def annotate_train_items(train_items, visual, preprocess, device):
    by_label = defaultdict(list)
    for idx, item in enumerate(train_items):
        by_label[item.label].append((idx, item))

    slot_meta = {}

    for label, pairs in sorted(by_label.items(), key=lambda kv: kv[0]):
        feats = []
        for _, item in pairs:
            feats.append(extract_feature(visual, preprocess, item.impath, device))
        feats = torch.stack(feats, dim=0)
        proto = feats.mean(dim=0, keepdim=True)
        proto = proto / proto.norm(dim=-1, keepdim=True)

        dists = torch.norm(feats - proto, dim=1)
        order = torch.argsort(dists, dim=0)

        for rank0, pos in enumerate(order.tolist()):
            idx, item = pairs[pos]
            slot_meta[idx] = {
                "slot_id": int(rank0),
                "slot_rank": int(rank0 + 1),
                "dist_to_proto": float(dists[pos].item()),
            }

    annotated = []
    for idx, item in enumerate(train_items):
        meta = slot_meta[idx]
        annotated.append(
            Datum(
                impath=item.impath,
                label=item.label,
                domain=item.domain,
                classname=item.classname,
                slot_id=meta["slot_id"],
                slot_rank=meta["slot_rank"],
                dist_to_proto=meta["dist_to_proto"],
            )
        )
    return annotated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default="/workspace/datasets")
    parser.add_argument("--dataset-name", type=str, default="oxford_pets")
    parser.add_argument("--backbone", type=str, default="RN50")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--kmax", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    split_dir = Path(args.dataset_root) / args.dataset_name / "split_fewshot"
    device = torch.device(args.device)

    print(f"Loading CLIP backbone={args.backbone} on device={device}")
    model = load_clip_to_cpu(args.backbone)
    model.float()
    model.to(device)
    model.eval()

    visual = model.visual
    preprocess = build_preprocess(model.visual.input_resolution)

    for seed in args.seeds:
        src = split_dir / f"shot_{args.kmax}-seed_{seed}.pkl"
        dst = split_dir / f"shot_{args.kmax}-seed_{seed}-slotproto.pkl"

        if not src.exists():
            raise FileNotFoundError(f"Missing source few-shot cache: {src}")

        if dst.exists() and not args.overwrite:
            print(f"[SKIP] {dst} already exists")
            continue

        print(f"[LOAD] {src}")
        with open(src, "rb") as f:
            data = pickle.load(f)

        train = data["train"]
        val = data["val"]

        print(f"[ANNOTATE] seed={seed} train_items={len(train)}")
        train_annot = annotate_train_items(train, visual, preprocess, device)

        out = {"train": train_annot, "val": val}
        with open(dst, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[SAVE] {dst}")

    print("Done.")


if __name__ == "__main__":
    main()
