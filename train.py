# ==============================================================================
# train.py  —  U-Net Cityscapes 19-class | Resolution 512×256
# ------------------------------------------------------------------------------
# Cách chạy:
#   python train.py
#   python train.py --data_root /kaggle/input/cityscapes --epochs 60 --batch_size 8
#   python train.py --resume checkpoints/last.pth
# ==============================================================================

import os, argparse, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from models import UNet
from utils  import (
    get_dataloader, CLASS_NAMES, NUM_CLASSES, IGNORE_INDEX, DEFAULT_IMG_SIZE,
    CombinedLoss, SegmentationMetrics,
)


# ==============================================================================
# CẤU HÌNH MẶC ĐỊNH
# ==============================================================================
CFG = {
    # Dataset
    "data_root"    : "datasets/cityscapes",
    "img_size"     : DEFAULT_IMG_SIZE,   # (256, 512) = 512×256 pixels
    "label_type"   : "labelIds",         # "labelIds" hoặc "trainIds"

    # Model
    "base_channels": 64,
    "bilinear"     : True,

    # Training
    "epochs"       : 60,
    "batch_size"   : 8,
    "num_workers"  : 4,
    "lr"           : 1e-3,
    "weight_decay" : 1e-4,
    "alpha_loss"   : 0.5,   # α×CE + (1-α)×Dice

    # Paths
    "save_dir"     : "checkpoints",
    "log_dir"      : "runs",
    "resume"       : None,

    # Misc
    "seed"         : 42,
    "log_interval" : 20,   # Log mỗi N batch
}


# ==============================================================================
# HELPER
# ==============================================================================
def set_seed(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(state, path, is_best=False):
    torch.save(state, path)
    if is_best:
        best = Path(path).parent / "best_model.pth"
        torch.save(state, best)
        print(f"  ★ Best model → {best}  (mIoU={state['best_miou']:.4f})")


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    start = ckpt.get("epoch", 0) + 1
    best  = ckpt.get("best_miou", 0.0)
    print(f"[RESUME] epoch {start}, best mIoU = {best:.4f}")
    return start, best


# ==============================================================================
# TRAIN ONE EPOCH
# ==============================================================================
def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch,
                    log_interval, writer=None, scaler=None, use_amp=False):
    model.train()
    tot_loss = tot_ce = tot_dice = 0.0
    N = len(loader)

    for i, (imgs, masks) in enumerate(loader, 1):
        imgs  = imgs.to(device, non_blocking=True)   # (B, 3, H, W)
        masks = masks.to(device, non_blocking=True)  # (B, H, W) int64

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(imgs)               # (B, 19, H, W)
                loss, ce, dice = loss_fn(logits, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)               # (B, 19, H, W)
            loss, ce, dice = loss_fn(logits, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        tot_loss += loss.item(); tot_ce += ce.item(); tot_dice += dice.item()

        if i % log_interval == 0 or i == N:
            print(f"  [{i:4d}/{N}] loss={loss.item():.4f} "
                  f"ce={ce.item():.4f} dice={dice.item():.4f}")

    avg = tot_loss/N, tot_ce/N, tot_dice/N
    if writer:
        writer.add_scalar("Train/Loss",      avg[0], epoch)
        writer.add_scalar("Train/CE_Loss",   avg[1], epoch)
        writer.add_scalar("Train/Dice_Loss", avg[2], epoch)
    return {"loss": avg[0], "ce": avg[1], "dice": avg[2]}


# ==============================================================================
# VALIDATE
# ==============================================================================
@torch.no_grad()
def validate(model, loader, loss_fn, device, epoch, writer=None):
    model.eval()
    tot_loss = 0.0
    metrics  = SegmentationMetrics(NUM_CLASSES, IGNORE_INDEX, CLASS_NAMES)

    for imgs, masks in loader:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits       = model(imgs)
        loss, _, _   = loss_fn(logits, masks)
        tot_loss    += loss.item()
        preds        = torch.argmax(logits, dim=1)
        metrics.update(preds, masks)

    r    = metrics.compute()
    avg  = tot_loss / len(loader)

    if writer:
        writer.add_scalar("Val/Loss",      avg,              epoch)
        writer.add_scalar("Val/mIoU",      r["mIoU"],        epoch)
        writer.add_scalar("Val/PixelAcc",  r["pixel_accuracy"], epoch)
        import math
        for name, iou in r["IoU_per_class"].items():
            if not math.isnan(iou):
                writer.add_scalar(f"Val/IoU_{name}", iou, epoch)

    return {**r, "loss": avg}


# ==============================================================================
# MAIN
# ==============================================================================
def main(cfg):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device : {device}")
    if torch.cuda.is_available():
        print(f"       GPU    : {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    Path(cfg["save_dir"]).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(cfg["log_dir"])

    # DataLoaders
    print("\n[INFO] Tải dataset ...")
    train_loader = get_dataloader(
        cfg["data_root"], "train",
        batch_size=cfg["batch_size"], img_size=cfg["img_size"],
        num_workers=cfg["num_workers"], augment=True,
        label_type=cfg["label_type"],
    )
    val_loader = get_dataloader(
        cfg["data_root"], "val",
        batch_size=cfg["batch_size"], img_size=cfg["img_size"],
        num_workers=cfg["num_workers"], augment=False,
        label_type=cfg["label_type"],
    )

    # Model: 19 lớp đầu ra
    model = UNet(
        in_channels   = 3,
        num_classes   = NUM_CLASSES,   # 19
        bilinear      = cfg["bilinear"],
        base_channels = cfg["base_channels"],
    ).to(device)
    
    # Biên dịch mô hình để tăng tốc trên PyTorch 2.x
    if hasattr(torch, "compile"):
        print("[INFO] Đang biên dịch mô hình bằng torch.compile...")
        model = torch.compile(model)
        
    print(model)

    # Loss, Optimizer, Scheduler
    loss_fn   = CombinedLoss(NUM_CLASSES, IGNORE_INDEX, cfg["alpha_loss"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"]/100
    )

    # GradScaler for AMP
    use_amp = cfg.get("amp", False)
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    # Resume
    start_epoch, best_miou = 1, 0.0
    if cfg.get("resume"):
        start_epoch, best_miou = load_checkpoint(cfg["resume"], model, optimizer)

    # Training loop
    print(f"\n{'='*60}")
    print(f"Bắt đầu train: {cfg['epochs']} epochs | "
          f"img_size={cfg['img_size']} | batch={cfg['batch_size']} | amp={use_amp}")
    print(f"{'='*60}")

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        print(f"\n── Epoch {epoch}/{cfg['epochs']}  lr={lr:.6f}")

        tr  = train_one_epoch(model, train_loader, optimizer, loss_fn,
                               device, epoch, cfg["log_interval"], writer,
                               scaler=scaler, use_amp=use_amp)
        val = validate(model, val_loader, loss_fn, device, epoch, writer)
        scheduler.step()
        writer.add_scalar("Train/LR", lr, epoch)

        elapsed = time.time() - t0
        print(f"\n  Epoch {epoch:3d} | "
              f"TrLoss={tr['loss']:.4f} | "
              f"ValLoss={val['loss']:.4f} | "
              f"mIoU={val['mIoU']:.4f} | "
              f"PixAcc={val['pixel_accuracy']:.4f} | "
              f"{elapsed:.1f}s")

        # Save checkpoint
        is_best = val["mIoU"] > best_miou
        if is_best:
            best_miou = val["mIoU"]

        state = {
            "epoch": epoch, "best_miou": best_miou,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "config": cfg,
        }
        last_path = Path(cfg["save_dir"]) / "last.pth"
        save_checkpoint(state, last_path, is_best=is_best)

    writer.close()
    print(f"\n{'='*60}")
    print(f"Huấn luyện xong! Best mIoU = {best_miou:.4f}")
    print(f"Checkpoint: {cfg['save_dir']}/best_model.pth")
    print(f"{'='*60}")


# ==============================================================================
# ARG PARSER
# ==============================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",    type=str,   default=CFG["data_root"])
    p.add_argument("--save_dir",     type=str,   default=CFG["save_dir"])
    p.add_argument("--log_dir",      type=str,   default=CFG["log_dir"])
    p.add_argument("--img_h",        type=int,   default=CFG["img_size"][0])
    p.add_argument("--img_w",        type=int,   default=CFG["img_size"][1])
    p.add_argument("--batch_size",   type=int,   default=CFG["batch_size"])
    p.add_argument("--num_workers",  type=int,   default=CFG["num_workers"])
    p.add_argument("--epochs",       type=int,   default=CFG["epochs"])
    p.add_argument("--lr",           type=float, default=CFG["lr"])
    p.add_argument("--weight_decay", type=float, default=CFG["weight_decay"])
    p.add_argument("--alpha_loss",   type=float, default=CFG["alpha_loss"])
    p.add_argument("--base_channels",type=int,   default=CFG["base_channels"])
    p.add_argument("--no_bilinear",  action="store_true")
    p.add_argument("--label_type",   type=str,   default=CFG["label_type"],
                   choices=["labelIds", "trainIds"])
    p.add_argument("--log_interval", type=int,   default=CFG["log_interval"])
    p.add_argument("--resume",       type=str,   default=None)
    p.add_argument("--seed",         type=int,   default=CFG["seed"])
    p.add_argument("--amp",          action="store_true", help="Bật chế độ Automatic Mixed Precision (FP16)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = {
        "data_root"    : args.data_root,
        "save_dir"     : args.save_dir,
        "log_dir"      : args.log_dir,
        "img_size"     : (args.img_h, args.img_w),
        "label_type"   : args.label_type,
        "base_channels": args.base_channels,
        "bilinear"     : not args.no_bilinear,
        "epochs"       : args.epochs,
        "batch_size"   : args.batch_size,
        "num_workers"  : args.num_workers,
        "lr"           : args.lr,
        "weight_decay" : args.weight_decay,
        "alpha_loss"   : args.alpha_loss,
        "resume"       : args.resume,
        "seed"         : args.seed,
        "log_interval" : args.log_interval,
        "amp"          : args.amp,
    }
    main(cfg)
