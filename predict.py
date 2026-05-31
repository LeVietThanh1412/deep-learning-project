from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from models.unet import UNet


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, num_classes=12).to(device)

    ckpt = Path("outputs/checkpoints/unet_cityscapes.pt")
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))

    image_path = Path("sample.png")
    if not image_path.exists():
        return

    image = Image.open(image_path).convert("RGB")
    image_tensor = torch.as_tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image_tensor = image_tensor.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        _ = logits.argmax(dim=1).squeeze(0).cpu().numpy()


if __name__ == "__main__":
    main()
