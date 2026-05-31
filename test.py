from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.unet import UNet
from utils.dataloader import CityscapesSegDataset
from utils.metrics import mean_iou


def main() -> None:
    data_root = Path("datasets/cityscapes")
    test_images = data_root / "test" / "images"
    test_masks = data_root / "test" / "masks"

    test_ds = CityscapesSegDataset(test_images, test_masks)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, num_classes=12).to(device)

    ckpt = Path("outputs/checkpoints/unet_cityscapes.pt")
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images = batch.image.to(device)
            masks = batch.mask.to(device)
            logits = model(images)
            _ = mean_iou(logits, masks, num_classes=12)


if __name__ == "__main__":
    main()
