from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def weighted_fewshot_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    b: torch.Tensor,
    slot_ids: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute weighted few-shot loss using fixed global slot ids.

    Args:
        logits: [B, C]
        labels: [B]
        b: [Kmax], slot scores after residual refinement
        slot_ids: [B], each in [0, Kmax-1]
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must have shape [B, C], got {tuple(logits.shape)}")
    if labels.dim() != 1:
        raise ValueError(f"labels must have shape [B], got {tuple(labels.shape)}")
    if slot_ids.dim() != 1:
        raise ValueError(f"slot_ids must have shape [B], got {tuple(slot_ids.shape)}")
    if logits.size(0) != labels.size(0) or labels.size(0) != slot_ids.size(0):
        raise ValueError("Batch size mismatch among logits, labels, and slot_ids")

    if (slot_ids < 0).any():
        raise ValueError("Found negative slot_id in training batch. Slot preprocessing is missing.")

    per_sample = F.cross_entropy(logits, labels, reduction="none")

    slot_weights = torch.softmax(b, dim=0)
    slot_ids = slot_ids.long().clamp_max(slot_weights.numel() - 1)
    sample_weights = slot_weights[slot_ids]

    weighted_loss = (sample_weights * per_sample).sum() / (sample_weights.sum() + 1e-12)

    keff = 1.0 / (slot_weights.pow(2).sum() + 1e-12)
    b_entropy = -(slot_weights * torch.log(slot_weights + 1e-12)).sum()
    top1_weight = slot_weights.max()
    top4_weight_sum = torch.topk(slot_weights, k=min(4, slot_weights.numel())).values.sum()

    aux = {
        "per_sample_loss": per_sample.detach(),
        "slot_ids": slot_ids.detach(),
        "slot_weights": slot_weights.detach(),
        "sample_weights": sample_weights.detach(),
        "weighted_loss_unscaled": weighted_loss.detach(),
        "keff_from_weights": keff.detach(),
        "b_entropy": b_entropy.detach(),
        "top1_weight": top1_weight.detach(),
        "top4_weight_sum": top4_weight_sum.detach(),
    }
    return weighted_loss, aux
