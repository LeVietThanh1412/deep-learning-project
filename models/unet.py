# ==============================================================================
# models/unet.py  —  U-Net tự xây dựng (không dùng thư viện kiến trúc ngoài)
# Tham khảo: Ronneberger et al., MICCAI 2015  https://arxiv.org/abs/1505.04597
#
# Sơ đồ kiến trúc:
#   Input (B,3,H,W)
#      │
#   [stem]──────────────────────────────────────────────────────skip4─┐
#      │                                                               │
#   [down1]─────────────────────────────────────────────skip3─┐       │
#      │                                                       │       │
#   [down2]──────────────────────────────────skip2─┐           │       │
#      │                                           │           │       │
#   [down3]──────────────────────skip1─┐           │           │       │
#      │                               │           │           │       │
#   [down4] ← Bottleneck               │           │           │       │
#      │                               │           │           │       │
#   [up1] ←────────────────────────────┘           │           │       │
#      │                                           │           │       │
#   [up2] ←────────────────────────────────────────┘           │       │
#      │                                                       │       │
#   [up3] ←────────────────────────────────────────────────────┘       │
#      │                                                               │
#   [up4] ←────────────────────────────────────────────────────────────┘
#      │
#   [out_conv] → (B, num_classes, H, W)  logits
#
# Số lớp: 11 lớp Cityscapes + 1 void = 12 kênh đầu ra
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# KHỐI CƠ BẢN: DoubleConv
# Dùng ở cả encoder lẫn decoder.
# Chuỗi: Conv(3x3) → BN → ReLU → Conv(3x3) → BN → ReLU
# padding=1 để giữ nguyên kích thước không gian (H, W).
# ──────────────────────────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    """
    2 lớp Conv+BN+ReLU liên tiếp.

    Args:
        in_channels  : Số kênh đầu vào.
        out_channels : Số kênh đầu ra.
        mid_channels : Số kênh trung gian (mặc định = out_channels).
    """
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            # Conv thứ nhất: in_ch → mid_ch
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),   # Chuẩn hóa batch
            nn.ReLU(inplace=True),          # inplace tiết kiệm bộ nhớ

            # Conv thứ hai: mid_ch → out_ch
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


# ──────────────────────────────────────────────────────────────────────────────
# ENCODER BLOCK: DownBlock
# MaxPool2d(2×2) giảm HxW xuống 1/2, sau đó DoubleConv tăng số kênh.
# Feature map trả về được dùng làm skip connection cho decoder tương ứng.
# ──────────────────────────────────────────────────────────────────────────────
class DownBlock(nn.Module):
    """
    Khối encoder: MaxPool(2×2) → DoubleConv.
    Spatial: H,W → H/2, W/2  |  Channels: in_ch → out_ch
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),   # Giảm không gian ÷ 2
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


# ──────────────────────────────────────────────────────────────────────────────
# DECODER BLOCK: UpBlock
# 1) Upsample ×2 (Bilinear hoặc ConvTranspose2d)
# 2) Ghép (concat) với skip connection từ encoder
# 3) DoubleConv để tích hợp thông tin
#
# Skip connection là điểm mấu chốt của U-Net:
#   Thông tin vị trí chi tiết từ encoder bù đắp cho thông tin bị mất khi pool.
# ──────────────────────────────────────────────────────────────────────────────
class UpBlock(nn.Module):
    """
    Khối decoder: Upsample → Concat(skip) → DoubleConv.

    Args:
        in_channels  : Kênh đầu vào (từ tầng decoder trước).
        out_channels : Kênh đầu ra.
        bilinear     : True → Bilinear+Conv; False → ConvTranspose2d.
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        if bilinear:
            # Bilinear: không tham số học, nhẹ hơn
            self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            # ConvTranspose2d: học được pattern upsample, hiệu quả hơn
            self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_dec: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_dec  : Feature map decoder (cần upsample).
            x_skip : Feature map encoder tương ứng (skip connection).
        """
        # Bước 1: Upsample
        x_dec = self.up(x_dec)

        # Bước 2: Căn chỉnh kích thước (xử lý chênh lệch do phép pool ceil/floor)
        dh = x_skip.size(2) - x_dec.size(2)
        dw = x_skip.size(3) - x_dec.size(3)
        # F.pad thêm pixel vào x_dec: (left, right, top, bottom)
        x_dec = F.pad(x_dec, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])

        # Bước 3: Ghép skip connection theo chiều kênh (dim=1)
        x_cat = torch.cat([x_skip, x_dec], dim=1)

        # Bước 4: DoubleConv tích hợp đặc trưng
        return self.conv(x_cat)


# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT HEAD: OutConv
# Conv 1×1 chiếu số kênh → số lớp phân đoạn (logits, chưa qua Softmax).
# ──────────────────────────────────────────────────────────────────────────────
class OutConv(nn.Module):
    """Conv 1×1: chiếu đặc trưng → logit cho từng lớp phân đoạn."""
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ──────────────────────────────────────────────────────────────────────────────
# MÔ HÌNH CHÍNH: UNet
# Tổng hợp Encoder + Bottleneck + Decoder + Output Head
#
# Số kênh encoder mặc định (base_channels=64):
#   stem   :   3 →  64   (không pool)
#   down1  :  64 → 128   (H/2)
#   down2  : 128 → 256   (H/4)
#   down3  : 256 → 512   (H/8)
#   down4  : 512 → 1024  (H/16) ← Bottleneck
#
# Decoder (mirror):
#   up1: 1024 → 512  (H/8)
#   up2:  512 → 256  (H/4)
#   up3:  256 → 128  (H/2)
#   up4:  128 →  64  (H)
#
# out_conv: 64 → num_classes  (H, W)
# ──────────────────────────────────────────────────────────────────────────────
class UNet(nn.Module):
    """
    U-Net hoàn chỉnh cho Semantic Segmentation.

    Args:
        in_channels   : Số kênh ảnh vào (mặc định 3 = RGB).
        num_classes   : Số lớp phân đoạn (mặc định 12 = 11 + void).
        bilinear      : Phương pháp upsample (True=Bilinear, False=ConvTranspose).
        base_channels : Số kênh tầng encoder đầu (mặc định 64).
    """

    def __init__(
        self,
        in_channels: int   = 3,
        num_classes: int   = 12,
        bilinear: bool     = True,
        base_channels: int = 64,
    ):
        super().__init__()
        self.in_channels   = in_channels
        self.num_classes   = num_classes
        self.bilinear      = bilinear
        self.base_channels = base_channels

        C      = base_channels          # C = 64
        factor = 2 if bilinear else 1   # Bilinear giảm kênh bottleneck sớm hơn

        # ── ENCODER ──────────────────────────────────────────────────────── #
        self.stem  = DoubleConv(in_channels, C)           # (B, C,    H,    W)
        self.down1 = DownBlock(C,     C * 2)              # (B, 2C,   H/2,  W/2)
        self.down2 = DownBlock(C * 2, C * 4)              # (B, 4C,   H/4,  W/4)
        self.down3 = DownBlock(C * 4, C * 8)              # (B, 8C,   H/8,  W/8)
        self.down4 = DownBlock(C * 8, C * 16 // factor)  # (B, 16C*, H/16, W/16)

        # ── DECODER ──────────────────────────────────────────────────────── #
        # in_channels của UpBlock = kênh decoder trước + kênh skip (concat)
        self.up1 = UpBlock(C * 16,           C * 8  // factor, bilinear)
        self.up2 = UpBlock(C * 8,            C * 4  // factor, bilinear)
        self.up3 = UpBlock(C * 4,            C * 2  // factor, bilinear)
        self.up4 = UpBlock(C * 2,            C,                bilinear)

        # ── OUTPUT HEAD ───────────────────────────────────────────────────── #
        self.out_conv = OutConv(C, num_classes)

        # Khởi tạo trọng số
        self._init_weights()

    # ── FORWARD ──────────────────────────────────────────────────────────── #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Ảnh đầu vào (B, 3, H, W). H,W nên chia hết cho 16.

        Returns:
            logits : (B, num_classes, H, W) — CHƯA qua Softmax.
                     nn.CrossEntropyLoss tự apply log-softmax bên trong.
        """
        # Encoder — lưu feature map làm skip connection
        x1 = self.stem(x)    # skip 4 — chi tiết nhất, kênh ít nhất
        x2 = self.down1(x1)  # skip 3
        x3 = self.down2(x2)  # skip 2
        x4 = self.down3(x3)  # skip 1
        x5 = self.down4(x4)  # Bottleneck — trừu tượng nhất, kênh nhiều nhất

        # Decoder — upsample + hợp nhất skip connection
        x = self.up1(x5, x4)  # dùng skip 1
        x = self.up2(x,  x3)  # dùng skip 2
        x = self.up3(x,  x2)  # dùng skip 3
        x = self.up4(x,  x1)  # dùng skip 4

        # Output
        return self.out_conv(x)  # (B, num_classes, H, W)

    # ── INFERENCE ────────────────────────────────────────────────────────── #
    def predict_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Trả về nhãn lớp dự đoán cho từng pixel (dùng khi inference).
        Áp dụng Softmax → argmax (không tính gradient).

        Args:
            x : (B, 3, H, W)

        Returns:
            mask : (B, H, W)  dtype=torch.long, giá trị [0, num_classes-1]
        """
        with torch.no_grad():
            logits = self.forward(x)            # (B, C, H, W)
            probs  = F.softmax(logits, dim=1)   # Xác suất từng lớp
            mask   = torch.argmax(probs, dim=1) # Lấy lớp có xác suất cao nhất
        return mask

    # ── INIT WEIGHTS ─────────────────────────────────────────────────────── #
    def _init_weights(self):
        """
        Khởi tạo trọng số:
          - Conv/ConvTranspose: Kaiming uniform (phù hợp ReLU)
          - BatchNorm         : weight=1, bias=0
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── UTILS ─────────────────────────────────────────────────────────────── #
    def count_parameters(self) -> int:
        """Tổng số tham số có thể học."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"UNet(in_channels={self.in_channels}, "
            f"num_classes={self.num_classes}, "
            f"base_channels={self.base_channels}, "
            f"bilinear={self.bilinear}, "
            f"params={self.count_parameters():,})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# QUICK TEST  (python models/unet.py)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = UNet(in_channels=3, num_classes=12, bilinear=True)
    print(model)

    dummy = torch.randn(2, 3, 256, 256)
    out   = model(dummy)
    mask  = model.predict_mask(dummy)

    print(f"Input  : {dummy.shape}")
    print(f"Output : {out.shape}")   # (2, 12, 256, 256)
    print(f"Mask   : {mask.shape}")  # (2, 256, 256)
    assert out.shape  == (2, 12, 256, 256)
    assert mask.shape == (2, 256, 256)
    print("✓ Tất cả kiểm tra đã qua thành công!")
