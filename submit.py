# ==============================================================================
# submit.py  —  Tạo kết quả dự đoán trên tập test để nộp lên Cityscapes Benchmark
# ==============================================================================

import os
import argparse
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp
from utils import get_dataloader

# Khai báo cứng 19 class cho Cityscapes
NUM_CLASSES = 19

# Bảng ánh xạ ngược từ trainId (0-18) -> labelId (0-33) của Cityscapes
TRAIN_TO_LABEL = np.array([
    7,   # 0: road
    8,   # 1: sidewalk
    11,  # 2: building
    12,  # 3: wall
    13,  # 4: fence
    17,  # 5: pole
    19,  # 6: traffic light
    20,  # 7: traffic sign
    21,  # 8: vegetation
    22,  # 9: terrain
    23,  # 10: sky
    24,  # 11: person
    25,  # 12: rider
    26,  # 13: car
    27,  # 14: truck
    28,  # 15: bus
    31,  # 16: train
    32,  # 17: motorcycle
    33,  # 18: bicycle
], dtype=np.uint8)

def main():
    parser = argparse.ArgumentParser(description="Tạo dự đoán chuẩn submit Cityscapes Benchmark")
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn file checkpoint .pth")
    parser.add_argument("--data_root", type=str, default="datasets/cityscapes", help="Thư mục dataset")
    parser.add_argument("--output_dir", type=str, default="submission", help="Thư mục lưu kết quả dự đoán")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size khi test")
    parser.add_argument("--num_workers", type=int, default=2, help="Số workers")
    parser.add_argument("--img_h", type=int, default=256, help="Kích thước H")
    parser.add_argument("--img_w", type=int, default=512, help="Kích thước W")
    parser.add_argument("--arch", type=str, default="unet", choices=["unet", "resnet34"], help="Kiến trúc mạng sử dụng")
    parser.add_argument("--no_zip", action="store_true", help="Không tự động nén file zip")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Sử dụng thiết bị: {device}")

    # ==========================================
    # 1. LOAD MODEL & TRỌNG SỐ
    # ==========================================
    print(f"[INFO] Đang tải mô hình kiến trúc: {args.arch.upper()}...")
    if args.arch == "resnet34":
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
        )
    else:
        # Import mạng U-Net tự code của bạn (Sửa đường dẫn import nếu cần)
        from models.unet import UNet
        model = UNet(in_channels=3, num_classes=NUM_CLASSES, base_channels=64, bilinear=True)
    
    model = model.to(device)

    print(f"[INFO] Đang nạp trọng số từ: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    # Xử lý trích xuất weights (tương thích cả lưu checkpoint lẫn lưu thuần)
    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
        epoch = ckpt.get("epoch", "?")
    else:
        state_dict = ckpt
        epoch = "?"

    # Xử lý lỗi _orig_mod (do torch.compile)
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict)
    model.eval()
    print(f"[OK] Tải mô hình thành công! (Epoch: {epoch})")

    # ==========================================
    # 2. KHỞI TẠO DATALOADER
    # ==========================================
    loader = get_dataloader(
        root=args.data_root,
        split="test",
        batch_size=args.batch_size,
        img_size=(args.img_h, args.img_w),
        num_workers=args.num_workers,
        augment=False,
        label_type="labelIds"
    )
    
    dataset = loader.dataset
    if len(dataset) == 0:
        print("[ERROR] Không tìm thấy ảnh trong tập test! Hãy check lại --data_root")
        return

    # Thư mục lưu xuất file
    sub_dir = Path(args.output_dir) / "test"
    sub_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Đang xử lý {len(dataset):,} ảnh tập test...")
    
    img_idx = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            imgs = imgs.to(device, non_blocking=True)
            
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            for i in range(imgs.size(0)):
                # Lấy tên file gốc
                img_path = Path(dataset.images[img_idx]) if hasattr(dataset, 'images') else Path(dataset.img_paths[img_idx])
                img_idx += 1

                # Resize mặt nạ về kích thước gốc 2048x1024
                orig_w, orig_h = 2048, 1024
                pred_train_ids = preds[i]
                pred_label_ids = TRAIN_TO_LABEL[pred_train_ids]

                pred_pil = Image.fromarray(pred_label_ids)
                pred_pil = pred_pil.resize((orig_w, orig_h), Image.NEAREST)

                # Xác định tên thành phố để gom nhóm (VD: berlin, munich)
                filename = img_path.name
                city_name = filename.split('_')[0]
                city_out_dir = sub_dir / city_name
                city_out_dir.mkdir(parents=True, exist_ok=True)

                if "_leftImg8bit" in filename:
                    pred_filename = filename.replace("_leftImg8bit", "_pred")
                else:
                    pred_filename = img_path.stem + "_pred.png"
                
                pred_pil.save(city_out_dir / pred_filename)

    print(f"\n[INFO] Hoàn thành xuất {img_idx} ảnh dự đoán tại: {sub_dir.resolve()}")

    # ==========================================
    # 3. NÉN THÀNH FILE ZIP
    # ==========================================
    if not args.no_zip:
        # Tên file zip (VD: /kaggle/working/submission.zip)
        zip_path_no_ext = str(Path(args.output_dir).resolve())
        print(f"[INFO] Đang đóng gói kết quả thành file zip...")
        
        # Hàm shutil này sẽ nén toàn bộ thư mục 'submission' thành 'submission.zip'
        shutil.make_archive(zip_path_no_ext, 'zip', root_dir=args.output_dir)
        
        print(f"★ File nén đã sẵn sàng: {zip_path_no_ext}.zip")
        print("Tải file này về và nộp thẳng lên Cityscapes Benchmark nhé!")

if __name__ == "__main__":
    main()