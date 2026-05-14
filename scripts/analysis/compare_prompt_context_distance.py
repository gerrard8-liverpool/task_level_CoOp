import argparse
from pathlib import Path
import torch


def load_prompt(path):
    ckpt = torch.load(path, map_location="cpu")

    # Dassl checkpoint usually stores model state in state_dict
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model" in ckpt:
            sd = ckpt["model"]
        else:
            sd = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

    print(f"\nLoaded: {path}")
    print("Available keys containing ctx/context/prompt:")
    keys = [k for k in sd.keys() if ("ctx" in k.lower() or "context" in k.lower() or "prompt" in k.lower())]
    for k in keys[:50]:
        v = sd[k]
        if torch.is_tensor(v):
            print(f"  {k}: {tuple(v.shape)}")

    # Common CoOp key
    candidates = [
        "ctx",
        "prompt_learner.ctx",
        "module.ctx",
        "module.prompt_learner.ctx",
    ]

    for k in candidates:
        if k in sd and torch.is_tensor(sd[k]):
            return sd[k].float(), k

    # fallback: choose tensor key named exactly or ending with .ctx
    for k, v in sd.items():
        if torch.is_tensor(v) and (k == "ctx" or k.endswith(".ctx")):
            return v.float(), k

    raise KeyError("Could not find context tensor key such as ctx or prompt_learner.ctx")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coop", required=True)
    parser.add_argument("--safe", required=True)
    parser.add_argument("--legacy", default=None)
    args = parser.parse_args()

    ctx_coop, key_coop = load_prompt(Path(args.coop))
    ctx_safe, key_safe = load_prompt(Path(args.safe))

    print("\n==== Context keys ====")
    print("CoOp key:", key_coop, "shape:", tuple(ctx_coop.shape))
    print("Safe key:", key_safe, "shape:", tuple(ctx_safe.shape))

    if ctx_coop.shape != ctx_safe.shape:
        raise ValueError(f"Shape mismatch: CoOp {ctx_coop.shape}, Safe {ctx_safe.shape}")

    diff = ctx_safe - ctx_coop
    coop_norm = ctx_coop.norm().item()
    safe_norm = ctx_safe.norm().item()
    diff_norm = diff.norm().item()
    rel = diff_norm / max(coop_norm, 1e-12)
    cos = torch.nn.functional.cosine_similarity(
        ctx_coop.flatten(), ctx_safe.flatten(), dim=0
    ).item()

    print("\n==== CoOp vs Safe final ctx ====")
    print("CoOp ctx norm:", coop_norm)
    print("Safe ctx norm:", safe_norm)
    print("Diff norm:", diff_norm)
    print("Relative diff norm:", rel)
    print("Cosine similarity:", cos)

    if args.legacy:
        ctx_legacy, key_legacy = load_prompt(Path(args.legacy))
        if ctx_legacy.shape != ctx_coop.shape:
            raise ValueError(f"Shape mismatch: Legacy {ctx_legacy.shape}, CoOp {ctx_coop.shape}")

        diff_l = ctx_legacy - ctx_coop
        legacy_norm = ctx_legacy.norm().item()
        diff_l_norm = diff_l.norm().item()
        rel_l = diff_l_norm / max(coop_norm, 1e-12)
        cos_l = torch.nn.functional.cosine_similarity(
            ctx_coop.flatten(), ctx_legacy.flatten(), dim=0
        ).item()

        print("\n==== CoOp vs Legacy final ctx ====")
        print("Legacy ctx norm:", legacy_norm)
        print("Diff norm:", diff_l_norm)
        print("Relative diff norm:", rel_l)
        print("Cosine similarity:", cos_l)

        diff_sl = ctx_safe - ctx_legacy
        print("\n==== Safe vs Legacy final ctx ====")
        print("Diff norm:", diff_sl.norm().item())
        print(
            "Cosine similarity:",
            torch.nn.functional.cosine_similarity(
                ctx_safe.flatten(), ctx_legacy.flatten(), dim=0
            ).item(),
        )


if __name__ == "__main__":
    main()
