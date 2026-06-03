# ==============================================================================
# test.py  —  Đánh giá U-Net trên tập test Cityscapes 19-class
# ------------------------------------------------------------------------------
# python test.py --checkpoint checkpoints/best_model.pth
# python test.py --checkpoint checkpoints/best_model.pth --save_vis results/
# ==============================================================================

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from models import UNet
from utils  import (
    get_dataloader, CLASS_NAMES, NUM_CLASSES, IGNORE_INDEX,
    decode_mask, CombinedLoss, SegmentationMetrics,
)


def visualize_batch(images, gt_masks, pr_masks, save_dir, start_idx,
                    mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
    """Lưu ảnh ghép [Input | GT | Pred] cho từng ảnh trong batch."""
    mn = np.array(mean)[:,None,None]
    st = np.array(std )[:,None,None]
    for i in range(images.size(0)):
        img_np = ((images[i].cpu().numpy() * st + mn).clip(0,1) * 255).astype(np.uint8).transpose(1,2,0)
        gt_col = decode_mask(gt_masks[i].cpu().numpy())
        pr_col = decode_mask(pr_masks[i].cpu().numpy())
        combined = np.concatenate([img_np, gt_col, pr_col], axis=1)
        path = Path(save_dir) / f"vis_{start_idx+i:04d}.png"
        Image.fromarray(combined).save(path)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--data_root",   type=str, default="datasets/cityscapes")
    p.add_argument("--label_type",  type=str, default="labelIds",
                   choices=["labelIds", "trainIds"])
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--img_h",       type=int, default=256)
    p.add_argument("--img_w",       type=int, default=512)
    p.add_argument("--save_vis",    type=str, default=None)
    p.add_argument("--max_vis",     type=int, default=50)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg  = ckpt.get("config", {})

    model = UNet(
        in_channels=3, num_classes=NUM_CLASSES,
        bilinear=cfg.get("bilinear", True),
        base_channels=cfg.get("base_channels", 64),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[INFO] Loaded epoch={ckpt.get('epoch','?')} | "
          f"best_mIoU={ckpt.get('best_miou',0):.4f}")

    # DataLoader
    loader = get_dataloader(
        args.data_root, "test",
        batch_size=args.batch_size, img_size=(args.img_h, args.img_w),
        num_workers=args.num_workers, augment=False,
        label_type=args.label_type,
    )

    loss_fn = CombinedLoss(NUM_CLASSES, IGNORE_INDEX)
    metrics = SegmentationMetrics(NUM_CLASSES, IGNORE_INDEX, CLASS_NAMES)
    tot_loss = 0.0
    vis_count = 0

    if args.save_vis:
        Path(args.save_vis).mkdir(parents=True, exist_ok=True)

    print(f"\nĐánh giá {len(loader.dataset):,} ảnh test ...")
    for i, batch in enumerate(loader):
        if len(batch) == 2:
            imgs, masks = batch
        else:
            imgs  = batch
            masks = None

        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)

        if masks is not None:
            masks = masks.to(device, non_blocking=True)
            loss, _, _ = loss_fn(logits, masks)
            tot_loss  += loss.item()

        preds = torch.argmax(logits, dim=1)

        if masks is not None:
            metrics.update(preds, masks)

        # Visualize
        if args.save_vis and vis_count < args.max_vis and masks is not None:
            n = min(imgs.size(0), args.max_vis - vis_count)
            visualize_batch(imgs[:n], masks[:n], preds[:n],
                            args.save_vis, vis_count)
            vis_count += n

        if (i+1) % 20 == 0:
            print(f"  Batch {i+1}/{len(loader)}")

    # In kết quả
    metrics.print_report()
    r = metrics.compute()
    print(f"Test Loss     : {tot_loss/len(loader):.4f}")
    print(f"Test mIoU     : {r['mIoU']:.4f}")
    print(f"Pixel Accuracy: {r['pixel_accuracy']:.4f}")
    print(f"Mean Accuracy : {r['mean_accuracy']:.4f}")

    if args.save_vis:
        print(f"\nVisualization → {args.save_vis}/")


if __name__ == "__main__":
    main()
