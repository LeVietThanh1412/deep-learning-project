from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2)

    intersection = (probs * targets_one_hot).sum(dim=(0, 2, 3))
    union = probs.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)

    return 1 - dice.mean()


def cce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    cce = F.cross_entropy(logits, targets)
    dsc = dice_loss(logits, targets)
    return alpha * cce + (1 - alpha) * dsc
