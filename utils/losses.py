# ==============================================================================
# utils/losses.py  —  19-class Cityscapes, ignore_index=255
# ==============================================================================

from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = 255  # Pixel không thuộc 19 lớp benchmark


# ──────────────────────────────────────────────────────────────────────────────
# 1. WEIGHTED CROSS-ENTROPY
# ──────────────────────────────────────────────────────────────────────────────
class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross-Entropy Loss có trọng số lớp để xử lý class imbalance.

    Cityscapes có sự mất cân bằng lớn:
      - "road" chiếm 30-40% pixel → trọng số nhỏ
      - "train", "rider" < 1% pixel → trọng số lớn

    Args:
        weight       : Tensor (19,) — trọng số từng lớp. None = CE thường.
        ignore_index : Pixel bị bỏ qua (255).
    """
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = IGNORE_INDEX,
    ):
        super().__init__()
        self.weight       = weight
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, 19, H, W) — raw logits (chưa softmax).
            targets : (B, H, W)     — trainId, dtype=int64, giá trị 0-18 hoặc 255.
        Returns:
            Scalar loss.
        """
        return F.cross_entropy(
            logits, targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="mean",
        )


# ──────────────────────────────────────────────────────────────────────────────
# 2. DICE LOSS
# DC_c = (2 * Σ p_c * y_c + ε) / (Σ p_c + Σ y_c + ε)
# L_dice = 1 - mean(DC_c)
# ──────────────────────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    """
    Soft Dice Loss cho multi-class segmentation.
    Tốt cho các lớp nhỏ (person, rider, train...) vì tối ưu trực tiếp overlap.

    Args:
        num_classes  : 19 lớp benchmark.
        ignore_index : 255 — pixel ignored không tham gia tính Dice.
        smooth       : Hằng số tránh chia 0.
    """
    def __init__(
        self,
        num_classes: int  = 19,
        ignore_index: int = IGNORE_INDEX,
        smooth: float     = 1e-5,
    ):
        super().__init__()
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.smooth       = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # One-hot targets: bỏ qua pixel ignore
        valid = (targets != self.ignore_index)                     # (B, H, W) bool
        tgt   = targets.clone()
        tgt[~valid] = 0                                            # Tạm gán 0
        tgt_oh = F.one_hot(tgt, num_classes=C).permute(0,3,1,2)   # (B,C,H,W)
        tgt_oh = tgt_oh.float()

        # Mask pixel ignore ra khỏi tính toán
        mask   = valid.unsqueeze(1).float()   # (B, 1, H, W)
        probs  = probs  * mask
        tgt_oh = tgt_oh * mask

        dice_losses = []
        for c in range(self.num_classes):
            p = probs[:,  c]    # (B, H, W)
            t = tgt_oh[:, c]   # (B, H, W)
            inter = (p * t).sum()
            union = p.sum() + t.sum()
            dc    = (2 * inter + self.smooth) / (union + self.smooth)
            dice_losses.append(1.0 - dc)

        return torch.stack(dice_losses).mean()


# ──────────────────────────────────────────────────────────────────────────────
# 3. COMBINED LOSS = α × WeightedCE + (1-α) × Dice
# ──────────────────────────────────────────────────────────────────────────────
class CombinedLoss(nn.Module):
    """
    Loss tổng hợp cho Cityscapes 19-class segmentation.

    Trả về (total, ce_val, dice_val) để tiện log riêng từng thành phần.

    Args:
        num_classes   : 19.
        ignore_index  : 255.
        alpha         : Trọng số CE. 0.5 = cân bằng. 0.7 = nghiêng về CE.
        class_weights : Tensor (19,) trọng số lớp. None = không dùng.
    """
    def __init__(
        self,
        num_classes: int                       = 19,
        ignore_index: int                      = IGNORE_INDEX,
        alpha: float                           = 0.5,
        class_weights: Optional[torch.Tensor]  = None,
    ):
        super().__init__()
        assert 0.0 <= alpha <= 1.0
        self.alpha = alpha

        self.ce   = WeightedCrossEntropyLoss(
            weight=class_weights, ignore_index=ignore_index
        )
        self.dice = DiceLoss(
            num_classes=num_classes, ignore_index=ignore_index
        )

    def forward(self, logits, targets):
        """
        Returns:
            (total_loss, ce_loss, dice_loss) — đều là scalar Tensor.
        """
        ce_val   = self.ce(logits, targets)
        dice_val = self.dice(logits, targets)
        total    = self.alpha * ce_val + (1.0 - self.alpha) * dice_val
        return total, ce_val, dice_val

    @staticmethod
    def compute_class_weights(
        class_pixel_counts: List[int],
        num_classes: int = 19,
        smooth: float    = 1.0,
    ) -> torch.Tensor:
        """
        Tính median-frequency balancing weights:
          w_c = median_freq / freq_c
          freq_c = count_c / total_pixels

        Phương pháp này ổn định hơn inverse-frequency thông thường.

        Args:
            class_pixel_counts : Số pixel của mỗi lớp (list 19 phần tử).

        Returns:
            weights : Tensor (19,) float32.
        """
        counts  = torch.tensor(class_pixel_counts, dtype=torch.float64)
        total   = counts.sum().clamp(min=1)
        freq    = counts / total                        # Tần suất mỗi lớp
        median  = freq[freq > 0].median()              # Median tần suất
        weights = median / freq.clamp(min=1e-6)        # Nghịch đảo chuẩn hóa
        weights = weights.clamp(max=10.0)              # Giới hạn trọng số tối đa
        return weights.float()


# ==============================================================================
# QUICK TEST
# ==============================================================================
if __name__ == "__main__":
    B, C, H, W = 2, 19, 64, 64
    logits  = torch.randn(B, C, H, W)
    targets = torch.randint(0, C, (B, H, W))
    # Thêm vài pixel ignore
    targets[0, :10, :10] = 255

    loss_fn = CombinedLoss(num_classes=C, alpha=0.5)
    total, ce, dice = loss_fn(logits, targets)
    print(f"Total={total.item():.4f}  CE={ce.item():.4f}  Dice={dice.item():.4f}")
    print("✓ CombinedLoss (19-class) hoạt động bình thường!")
