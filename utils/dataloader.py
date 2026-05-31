from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image


@dataclass(frozen=True)
class SegSample:
    image: torch.Tensor
    mask: torch.Tensor


class CityscapesSegDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, transform=None) -> None:
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted(images_dir.glob("*.png"))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> SegSample:
        image_path = self.images[idx]
        mask_path = self.masks_dir / image_path.name

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        image_tensor = torch.as_tensor(image)
        mask_tensor = torch.as_tensor(mask)

        return SegSample(image=image_tensor, mask=mask_tensor)
