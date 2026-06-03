# ==============================================================================
# predict.py  —  Dự đoán ảnh mới, xuất color mask + overlay
# python predict.py --checkpoint checkpoints/best_model.pth --input img.jpg
# python predict.py --checkpoint checkpoints/best_model.pth --input_dir imgs/
# ==============================================================================

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from models import UNet
from utils  import NUM_CLASSES, IGNORE_INDEX, CLASS_NAMES, decode_mask, DEFAULT_IMG_SIZE


MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def build_transform(img_size=DEFAULT_IMG_SIZE):
    H, W = img_size
    return T.Compose([
        T.Resize((H, W)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])


@torch.no_grad()
def predict_single(model, img_path, device, transform):
    """Dự đoán mask cho 1 ảnh. Trả về numpy (H_orig, W_orig) uint8."""
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size

    inp    = transform(img).unsqueeze(0).to(device)   # (1,3,H,W)
    logits = model(inp)                                # (1,19,H,W)
    pred   = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Resize về kích thước gốc
    pred_pil = Image.fromarray(pred, mode="L").resize((orig_w, orig_h), Image.NEAREST)
    return np.array(pred_pil, dtype=np.uint8)


def save_outputs(img_path, mask_idx, output_dir, save_raw=False):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(img_path).stem

    # Color mask
    color = decode_mask(mask_idx)
    Image.fromarray(color).save(out / f"color_{stem}.png")

    # Overlay (60% ảnh gốc + 40% mask)
    orig    = Image.open(img_path).convert("RGB")
    mask_pil = Image.fromarray(color).resize(orig.size, Image.NEAREST)
    overlay = Image.blend(orig, mask_pil, alpha=0.45)
    overlay.save(out / f"overlay_{stem}.png")

    # Raw (optional)
    if save_raw:
        np.save(str(out / f"raw_{stem}.npy"), mask_idx)

    print(f"  {stem} → color_{stem}.png  overlay_{stem}.png")


def print_stats(mask_idx, name):
    total = mask_idx.size
    print(f"\n  [{name}] Phân phối lớp:")
    for c in range(NUM_CLASSES):
        cnt = int((mask_idx == c).sum())
        if cnt > 0:
            print(f"    {CLASS_NAMES[c]:<20} {cnt:>8,}  ({cnt/total*100:.1f}%)")


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input",     type=str)
    g.add_argument("--input_dir", type=str)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="predictions")
    p.add_argument("--img_h",      type=int, default=DEFAULT_IMG_SIZE[0])
    p.add_argument("--img_w",      type=int, default=DEFAULT_IMG_SIZE[1])
    p.add_argument("--save_raw",   action="store_true")
    p.add_argument("--show_stats", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt  = torch.load(args.checkpoint, map_location=device)
    cfg   = ckpt.get("config", {})
    model = UNet(
        in_channels=3, num_classes=NUM_CLASSES,
        bilinear=cfg.get("bilinear", True),
        base_channels=cfg.get("base_channels", 64),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Model loaded | epoch={ckpt.get('epoch','?')}")

    transform = build_transform((args.img_h, args.img_w))

    EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
    imgs = [args.input] if args.input else \
           sorted([str(p) for p in Path(args.input_dir).iterdir()
                   if p.suffix.lower() in EXTS])

    print(f"\nDự đoán {len(imgs)} ảnh → {args.output_dir}/\n")
    for i, img_path in enumerate(imgs, 1):
        print(f"[{i}/{len(imgs)}] {Path(img_path).name}")
        mask = predict_single(model, img_path, device, transform)
        save_outputs(img_path, mask, args.output_dir, args.save_raw)
        if args.show_stats:
            print_stats(mask, Path(img_path).name)

    print(f"\nDone! → {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
