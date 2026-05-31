from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.unet import UNet
from utils.dataloader import CityscapesSegDataset
from utils.losses import cce_dice_loss
from utils.metrics import mean_iou


def main() -> None:
    data_root = Path("datasets/cityscapes")
    train_images = data_root / "train" / "images"
    train_masks = data_root / "train" / "masks"
    val_images = data_root / "val" / "images"
    val_masks = data_root / "val" / "masks"

    train_ds = CityscapesSegDataset(train_images, train_masks)
    val_ds = CityscapesSegDataset(val_images, val_masks)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, num_classes=12).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1):
        model.train()
        for batch in train_loader:
            images = batch.image.to(device)
            masks = batch.mask.to(device)
            logits = model(images)
            loss = cce_dice_loss(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                images = batch.image.to(device)
                masks = batch.mask.to(device)
                logits = model(images)
                _ = mean_iou(logits, masks, num_classes=12)

    out_dir = Path("outputs/checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "unet_cityscapes.pt")


if __name__ == "__main__":
    main()
