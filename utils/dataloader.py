# ==============================================================================
# utils/dataloader.py  —  Cityscapes 19-class benchmark
# ------------------------------------------------------------------------------
# Hỗ trợ 2 cấu trúc thư mục:
#
# [A] Cấu trúc Cityscapes gốc (Kaggle standard):
#   root/
#     leftImg8bit/{train,val,test}/city/city_xxx_leftImg8bit.png
#     gtFine/{train,val,test}/city/city_xxx_gtFine_labelIds.png
#
# [B] Cấu trúc đơn giản (flat):
#   root/{train,val,test}/images/*.png
#   root/{train,val,test}/masks/*.png   ← gtFine_labelIds hoặc trainIds
#
# 19 lớp chuẩn Cityscapes benchmark (trainId 0-18):
#   0 road        5 pole       10 sky       15 bus
#   1 sidewalk    6 t.light    11 person    16 train
#   2 building    7 t.sign     12 rider     17 motorcycle
#   3 wall        8 vegetation 13 car       18 bicycle
#   4 fence       9 terrain    14 truck
#
# ignore_index = 255  (tất cả lớp không thuộc 19 lớp benchmark)
# ==============================================================================

import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as T


# ==============================================================================
# MAPPING: Cityscapes labelId (0-33) → trainId (0-18, 255)
# Nguồn: https://github.com/mcordts/cityscapesScripts
# ==============================================================================
LABEL_TO_TRAIN: Dict[int, int] = {
    0: 255,  # unlabeled
    1: 255,  # egoVehicle
    2: 255,  # rectificationBorder
    3: 255,  # outOfRoi
    4: 255,  # static
    5: 255,  # dynamic
    6: 255,  # ground
    7:   0,  # road
    8:   1,  # sidewalk
    9: 255,  # parking
    10: 255, # railTrack
    11:  2,  # building
    12:  3,  # wall
    13:  4,  # fence
    14: 255, # guardRail
    15: 255, # bridge
    16: 255, # tunnel
    17:  5,  # pole
    18: 255, # poleGroup
    19:  6,  # trafficLight
    20:  7,  # trafficSign
    21:  8,  # vegetation
    22:  9,  # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    29: 255, # caravan
    30: 255, # trailer
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
    -1: 255, # licencePlate
}

# Bảng tra cứu nhanh dạng array (index 0-255)
# np.vectorize(LABEL_TO_TRAIN.get) chậm → dùng lookup table
_LUT = np.full(256, 255, dtype=np.uint8)
for src, tgt in LABEL_TO_TRAIN.items():
    if 0 <= src <= 255:
        _LUT[src] = tgt

# Tên 19 lớp theo thứ tự trainId
CLASS_NAMES: List[str] = [
    "road",          # 0
    "sidewalk",      # 1
    "building",      # 2
    "wall",          # 3
    "fence",         # 4
    "pole",          # 5
    "traffic light", # 6
    "traffic sign",  # 7
    "vegetation",    # 8
    "terrain",       # 9
    "sky",           # 10
    "person",        # 11
    "rider",         # 12
    "car",           # 13
    "truck",         # 14
    "bus",           # 15
    "train",         # 16
    "motorcycle",    # 17
    "bicycle",       # 18
]

# Bảng màu RGB cho 19 lớp (chuẩn Cityscapes)
COLOR_MAP = np.array([
    [128,  64, 128],  # 0  road
    [244,  35, 232],  # 1  sidewalk
    [ 70,  70,  70],  # 2  building
    [102, 102, 156],  # 3  wall
    [190, 153, 153],  # 4  fence
    [153, 153, 153],  # 5  pole
    [250, 170,  30],  # 6  traffic light
    [220, 220,   0],  # 7  traffic sign
    [107, 142,  35],  # 8  vegetation
    [152, 251, 152],  # 9  terrain
    [ 70, 130, 180],  # 10 sky
    [220,  20,  60],  # 11 person
    [255,   0,   0],  # 12 rider
    [  0,   0, 142],  # 13 car
    [  0,   0,  70],  # 14 truck
    [  0,  60, 100],  # 15 bus
    [  0,  80, 100],  # 16 train
    [  0,   0, 230],  # 17 motorcycle
    [119,  11,  32],  # 18 bicycle
], dtype=np.uint8)

NUM_CLASSES  = 19   # Số lớp benchmark
IGNORE_INDEX = 255  # Pixel bị bỏ qua khi tính loss & metric

# Kích thước ảnh mặc định (H, W)
DEFAULT_IMG_SIZE = (256, 512)   # 512×256 pixels


def convert_label_to_train(mask_arr: np.ndarray) -> np.ndarray:
    """
    Chuyển mask labelId (0-33) → trainId (0-18, 255).
    Dùng lookup table để tốc độ nhanh nhất.

    Args:
        mask_arr: numpy (H, W) dtype uint8, giá trị 0-33.

    Returns:
        numpy (H, W) dtype uint8, giá trị 0-18 hoặc 255.
    """
    return _LUT[mask_arr.astype(np.uint8)]


def decode_mask(mask: np.ndarray) -> np.ndarray:
    """
    Chuyển mask trainId → ảnh màu RGB để visualize.

    Args:
        mask: numpy (H, W) hoặc Tensor (H, W), giá trị 0-18 (255 → đen).

    Returns:
        numpy (H, W, 3) uint8.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask = mask.astype(np.int64)
    # Pixel ignore (255) → lớp 0 của bảng màu (sẽ gán đen riêng)
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        color[mask == c] = COLOR_MAP[c]
    # Pixel ignore → đen
    color[mask == IGNORE_INDEX] = [0, 0, 0]
    return color


# ==============================================================================
# DATASET
# ==============================================================================
class CityscapesDataset(Dataset):
    """
    Dataset Cityscapes 19-class cho Semantic Segmentation.

    Hỗ trợ 2 chế độ (tự động phát hiện):
      mode='cityscapes' — Cấu trúc gốc Cityscapes (leftImg8bit / gtFine)
      mode='flat'       — Cấu trúc đơn giản (images/ masks/)

    Args:
        root         : Thư mục gốc dataset.
        split        : "train", "val", hoặc "test".
        img_size     : (H, W) để resize. Mặc định (256, 512).
        augment      : Bật augmentation (chỉ dùng cho train).
        label_type   : "labelIds"  → cần convert sang trainId.
                       "trainIds"  → đã là trainId (0-18, 255), dùng trực tiếp.
    """

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        root: str,
        split: str        = "train",
        img_size: Tuple   = DEFAULT_IMG_SIZE,
        augment: bool     = True,
        label_type: str   = "labelIds",   # hoặc "trainIds"
    ):
        super().__init__()
        assert split in ("train", "val", "test")
        assert label_type in ("labelIds", "trainIds")

        self.root       = Path(root)
        self.split      = split
        self.img_size   = img_size  # (H, W)
        self.augment    = augment and (split == "train")
        self.label_type = label_type

        # Phát hiện cấu trúc thư mục
        self.mode, self.img_paths, self.mask_paths = self._discover()

        self.normalize = T.Normalize(mean=self.MEAN, std=self.STD)
        print(
            f"[Dataset] split={split} | mode={self.mode} | "
            f"samples={len(self.img_paths)} | img_size={img_size}"
        )

    # ---------------------------------------------------------------------- #
    def _discover(self):
        """Tự động phát hiện cấu trúc thư mục và thu thập đường dẫn file."""

        # 1. Thử cấu trúc Cityscapes gốc
        img_dir  = self.root / "leftImg8bit" / self.split
        mask_dir = self.root / "gtFine"      / self.split
        if img_dir.exists() and mask_dir.exists():
            return self._collect_cityscapes(img_dir, mask_dir)

        # 2. Thử cấu trúc của Kaggle Dataset (như trong ảnh bạn gửi)
        img_dir_k  = self.root / "Cityscape Dataset" / "leftImg8bit" / self.split
        mask_dir_k = self.root / "Fine Annotations" / "gtFine" / self.split
        if not mask_dir_k.exists():
             # Đôi khi nó nằm trực tiếp trong Fine Annotations
             mask_dir_k = self.root / "Fine Annotations" / self.split
        if img_dir_k.exists() and mask_dir_k.exists():
            return self._collect_cityscapes(img_dir_k, mask_dir_k)

        # 3. Thử cấu trúc flat
        img_dir_f  = self.root / self.split / "images"
        mask_dir_f = self.root / self.split / "masks"

        if img_dir_f.exists():
            return self._collect_flat(img_dir_f, mask_dir_f)

        raise FileNotFoundError(
            f"Không tìm thấy dataset tại: {self.root}\n"
            "Hỗ trợ 3 cấu trúc:\n"
            "  [A] root/leftImg8bit/{split}/city/*.png  &  root/gtFine/{split}/city/*.png\n"
            "  [B] root/Cityscape Dataset/leftImg8bit/{split}/... & root/Fine Annotations/...\n"
            "  [C] root/{split}/images/*.png            &  root/{split}/masks/*.png"
        )

    def _collect_cityscapes(self, img_dir, mask_dir):
        """Thu thập file theo cấu trúc Cityscapes gốc (per-city subfolder)."""
        suffix = "_gtFine_labelIds.png" if self.label_type == "labelIds" \
                 else "_gtFine_labelTrainIds.png"

        imgs, masks = [], []
        for city_dir in sorted(img_dir.iterdir()):
            if not city_dir.is_dir():
                continue
            for img_p in sorted(city_dir.glob("*_leftImg8bit.png")):
                # Tên mask tương ứng
                stem     = img_p.stem.replace("_leftImg8bit", "")
                mask_p   = mask_dir / city_dir.name / (stem + suffix)
                if mask_p.exists():
                    imgs.append(img_p)
                    masks.append(mask_p)
                elif self.split == "test":
                    imgs.append(img_p)

        return "cityscapes", imgs, masks

    def _collect_flat(self, img_dir, mask_dir):
        """Thu thập file theo cấu trúc flat (images/ masks/)."""
        EXTS = {".png", ".jpg", ".jpeg"}
        imgs, masks = [], []

        for img_p in sorted(img_dir.iterdir()):
            if img_p.suffix.lower() not in EXTS:
                continue
            mask_p = mask_dir / (img_p.stem + ".png")
            if mask_dir.exists() and mask_p.exists():
                imgs.append(img_p)
                masks.append(mask_p)
            elif self.split == "test":
                imgs.append(img_p)

        return "flat", imgs, masks

    # ---------------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        # Load ảnh RGB
        img = Image.open(self.img_paths[idx]).convert("RGB")

        # Load mask
        has_mask = idx < len(self.mask_paths)
        if has_mask:
            mask_arr = np.array(
                Image.open(self.mask_paths[idx]), dtype=np.uint8
            )
            # Chuyển labelId → trainId nếu cần
            if self.label_type == "labelIds":
                mask_arr = convert_label_to_train(mask_arr)
            mask = Image.fromarray(mask_arr)
        else:
            mask = None

        # Augmentation (chỉ train)
        if self.augment and mask is not None:
            img, mask = self._augment(img, mask)

        # Resize
        H, W = self.img_size
        img  = img.resize((W, H), Image.BILINEAR)
        if mask is not None:
            mask = mask.resize((W, H), Image.NEAREST)  # NEAREST giữ đúng nhãn

        # → Tensor
        img_t = TF.to_tensor(img)           # (3, H, W) float [0,1]
        img_t = self.normalize(img_t)       # Chuẩn hóa ImageNet

        if mask is not None:
            mask_t = torch.from_numpy(
                np.array(mask, dtype=np.int64)
            )                               # (H, W) int64
            return img_t, mask_t
        return img_t

    def _augment(self, img, mask):
        """
        Augmentation đồng bộ ảnh + mask:
          1. Random Horizontal Flip
          2. Random Scale Crop (80%–120%)
          3. Color Jitter (chỉ ảnh)
        """
        # 1. Horizontal Flip
        if torch.rand(1).item() > 0.5:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)

        # 2. Random Scale Crop
        W, H = img.size
        scale = 0.8 + 0.4 * torch.rand(1).item()  # [0.8, 1.2]
        cH    = int(H * scale)
        cW    = int(W * scale)
        # Pad nếu crop lớn hơn ảnh gốc
        if cH > H or cW > W:
            pad_h = max(0, cH - H)
            pad_w = max(0, cW - W)
            img   = TF.pad(img,  [pad_w//2, pad_h//2, pad_w-pad_w//2, pad_h-pad_h//2])
            mask  = TF.pad(mask, [pad_w//2, pad_h//2, pad_w-pad_w//2, pad_h-pad_h//2],
                           fill=IGNORE_INDEX)
            W, H  = img.size
            cH, cW = min(cH, H), min(cW, W)

        i, j, th, tw = T.RandomCrop.get_params(img, (cH, cW))
        img  = TF.crop(img,  i, j, th, tw)
        mask = TF.crop(mask, i, j, th, tw)

        # 3. Color Jitter (chỉ ảnh)
        img = T.ColorJitter(brightness=0.3, contrast=0.3,
                            saturation=0.3, hue=0.1)(img)
        return img, mask


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================
def get_dataloader(
    root: str,
    split: str,
    batch_size: int  = 8,
    img_size: Tuple  = DEFAULT_IMG_SIZE,
    num_workers: int = 2,
    augment: bool    = True,
    label_type: str  = "labelIds",
    pin_memory: bool = True,
) -> DataLoader:
    """
    Tạo DataLoader cho một split.

    Args:
        root        : Thư mục gốc dataset.
        split       : "train", "val", "test".
        batch_size  : Số ảnh mỗi batch.
        img_size    : (H, W).
        num_workers : Worker song song (Kaggle dùng 2).
        augment     : Bật augmentation (tự động tắt nếu split≠train).
        label_type  : "labelIds" hoặc "trainIds".
        pin_memory  : True nếu dùng GPU.
    """
    dataset = CityscapesDataset(
        root=root, split=split, img_size=img_size,
        augment=augment, label_type=label_type,
    )
    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = (split == "train"),
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = (split == "train"),
        persistent_workers = (num_workers > 0),
    )
    return loader
