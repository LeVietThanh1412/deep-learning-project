from __future__ import annotations

import torch


def mean_iou(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    preds = logits.argmax(dim=1)
    ious = []
    for cls in range(num_classes):
        pred_mask = preds == cls
        target_mask = targets == cls
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        if union == 0:
            continue
        ious.append(intersection / union)
    if not ious:
        return torch.tensor(0.0, device=logits.device)
    return torch.stack(ious).mean()
