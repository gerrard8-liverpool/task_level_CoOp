import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt


def load_ctx(path):
    ckpt = torch.load(path, map_location="cpu")
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt

    for k, v in sd.items():
        if torch.is_tensor(v) and (k == "ctx" or k.endswith(".ctx")):
            return v.float()

    raise KeyError(f"Cannot find ctx in {path}")


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def rel_diff(a, b):
    """Symmetric relative difference: ||a-b|| / mean(||a||, ||b||)."""
    diff = (a - b).norm().item()
    denom = 0.5 * (a.norm().item() + b.norm().item())
    return diff / max(denom, 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coop", required=True)
    parser.add_argument("--safe", required=True)
    parser.add_argument("--legacy", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    names = ["CoOp", "Safe", "Legacy"]
    ctxs = {
        "CoOp": load_ctx(args.coop),
        "Safe": load_ctx(args.safe),
        "Legacy": load_ctx(args.legacy),
    }

    cos_mat = np.zeros((3, 3), dtype=float)
    diff_mat = np.zeros((3, 3), dtype=float)

    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            cos_mat[i, j] = cosine(ctxs[ni], ctxs[nj])
            diff_mat[i, j] = rel_diff(ctxs[ni], ctxs[nj])

    cos_mat = np.clip(cos_mat, -1.0, 1.0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    im0 = axes[0].imshow(cos_mat, vmin=-1, vmax=1)
    axes[0].set_title("(a) Cosine similarity")
    axes[0].set_xticks(range(3), names)
    axes[0].set_yticks(range(3), names)
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, f"{cos_mat[i,j]:.2f}", ha="center", va="center")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(diff_mat)
    axes[1].set_title("(b) Relative difference norm")
    axes[1].set_xticks(range(3), names)
    axes[1].set_yticks(range(3), names)
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{diff_mat[i,j]:.2f}", ha="center", va="center")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle("Final Raw Prompt Context Geometry: Food101 -> OxfordPets, seed 3", y=1.03)
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)

    print("Saved:", out)
    print("Saved:", out.with_suffix(".png"))
    print("\nCosine matrix:")
    print(cos_mat)
    print("\nRelative difference matrix:")
    print(diff_mat)


if __name__ == "__main__":
    main()
