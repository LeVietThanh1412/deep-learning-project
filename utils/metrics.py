# ==============================================================================
# utils/metrics.py  —  19-class Cityscapes, ignore_index=255
# ==============================================================================

from typing import Dict, List
import torch
import numpy as np

NUM_CLASSES  = 19
IGNORE_INDEX = 255


class SegmentationMetrics:
    """
    Tính mIoU và các chỉ số khác bằng Confusion Matrix tích lũy.

    Cách dùng:
        metrics = SegmentationMetrics()
        for imgs, masks in loader:
            preds = torch.argmax(model(imgs), dim=1)
            metrics.update(preds, masks)
        result = metrics.compute()
        metrics.reset()

    Tất cả pixel có giá trị IGNORE_INDEX (255) đều bị bỏ qua.
    """

    def __init__(
        self,
        num_classes: int       = NUM_CLASSES,
        ignore_index: int      = IGNORE_INDEX,
        class_names: List[str] = None,
    ):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.class_names  = class_names or [f"cls_{i}" for i in range(num_classes)]
        # Ma trận nhầm lẫn (num_classes × num_classes)
        # confusion[i][j] = số pixel thật là i, dự đoán là j
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        """Reset về 0 đầu mỗi epoch."""
        self.cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Cập nhật confusion matrix.

        Args:
            preds   : (B, H, W) dtype long — nhãn dự đoán (0-18).
            targets : (B, H, W) dtype long — nhãn thật (0-18 hoặc 255).
        """
        if isinstance(preds,   torch.Tensor): preds   = preds.cpu().numpy()
        if isinstance(targets, torch.Tensor): targets = targets.cpu().numpy()

        preds   = preds.flatten().astype(np.int64)
        targets = targets.flatten().astype(np.int64)

        # Lọc bỏ pixel ignore
        valid   = targets != self.ignore_index
        preds   = preds[valid]
        targets = targets[valid]

        # Clip tránh index lỗi
        preds   = np.clip(preds,   0, self.num_classes - 1)
        targets = np.clip(targets, 0, self.num_classes - 1)

        # Cập nhật bằng bincount (O(N), rất nhanh)
        idx = targets * self.num_classes + preds
        self.cm += np.bincount(idx, minlength=self.num_classes**2)\
                     .reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict:
        """
        Tính toán các chỉ số từ confusion matrix.

        Returns:
            Dict:
              mIoU          : float — trung bình IoU 19 lớp.
              IoU_per_class : Dict[name → float].
              pixel_accuracy: float.
              mean_accuracy : float.
        """
        cm  = self.cm.astype(np.float64)
        iou = []
        acc = []

        for c in range(self.num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            denom_iou = tp + fp + fn
            denom_acc = tp + fn

            iou.append(tp / denom_iou if denom_iou > 0 else np.nan)
            acc.append(tp / denom_acc if denom_acc > 0 else np.nan)

        iou_arr = np.array(iou)
        acc_arr = np.array(acc)

        valid_iou = iou_arr[~np.isnan(iou_arr)]
        valid_acc = acc_arr[~np.isnan(acc_arr)]

        miou       = float(valid_iou.mean()) if len(valid_iou) > 0 else 0.0
        mean_acc   = float(valid_acc.mean()) if len(valid_acc) > 0 else 0.0
        pixel_acc  = float(np.diag(cm).sum() / cm.sum()) if cm.sum() > 0 else 0.0

        return {
            "mIoU"          : miou,
            "IoU_per_class" : {self.class_names[i]: float(iou[i])
                               for i in range(self.num_classes)},
            "pixel_accuracy": pixel_acc,
            "mean_accuracy" : mean_acc,
        }

    def print_report(self):
        """In báo cáo IoU từng lớp."""
        r = self.compute()
        print("\n" + "=" * 52)
        print(f"{'Cityscapes 19-class Metrics Report':^52}")
        print("=" * 52)
        print(f"  {'Lớp':<20} {'IoU':>8}")
        print("  " + "-" * 30)
        for name, iou in r["IoU_per_class"].items():
            val = f"{iou:.4f}" if not np.isnan(iou) else "  N/A"
            print(f"  {name:<20} {val:>8}")
        print("  " + "-" * 30)
        print(f"  {'mIoU':<20} {r['mIoU']:>8.4f}")
        print(f"  {'Pixel Accuracy':<20} {r['pixel_accuracy']:>8.4f}")
        print(f"  {'Mean Accuracy':<20} {r['mean_accuracy']:>8.4f}")
        print("=" * 52 + "\n")


def compute_miou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int  = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> float:
    """Tính nhanh mIoU cho một batch."""
    m = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index)
    m.update(preds, targets)
    return m.compute()["mIoU"]
