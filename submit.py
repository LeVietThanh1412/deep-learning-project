# ==============================================================================
# submit.py  —  Tạo kết quả dự đoán trên tập test để nộp lên Cityscapes Benchmark
# ------------------------------------------------------------------------------
# Cách chạy:
#   python submit.py --checkpoint checkpoints/best_model.pth
#   python submit.py --checkpoint checkpoints/best_model.pth --data_root datasets/cityscapes
# ==============================================================================

import os
import argparse
import shutil
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from models import UNet
from utils import get_dataloader, NUM_CLASSES

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
    parser = argparse.ArgumentParser(description="Tạo kết quả dự đoán chuẩn để submit Cityscapes Benchmark")
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn file checkpoint .pth")
    parser.add_argument("--data_root", type=str, default="datasets/cityscapes", help="Thư mục chứa dataset")
    parser.add_argument("--output_dir", type=str, default="submission", help="Thư mục lưu kết quả dự đoán")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size khi test")
    parser.add_argument("--num_workers", type=int, default=2, help="Số workers cho dataloader")
    parser.add_argument("--img_h", type=int, default=256, help="Kích thước H mô hình nhận")
    parser.add_argument("--img_w", type=int, default=512, help="Kích thước W mô hình nhận")
    parser.add_argument("--no_zip", action="store_true", help="Không tự động nén file zip")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Sử dụng thiết bị: {device}")

    # 1. Load Model & Cấu hình từ checkpoint
    print(f"[INFO] Đang tải checkpoint từ: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", {})

    model = UNet(
        in_channels=3,
        num_classes=NUM_CLASSES,
        bilinear=cfg.get("bilinear", True),
        base_channels=cfg.get("base_channels", 64),
    ).to(device)
    
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[INFO] Tải mô hình thành công! Epoch huấn luyện trước đó: {ckpt.get('epoch', '?')}")

    # 2. Khởi tạo Dataset & DataLoader
    # Lấy DataLoader cho tập test
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
        print("[LỖI] Không tìm thấy ảnh nào trong tập test! Kiểm tra lại --data_root")
        return

    # Đường dẫn thư mục lưu ảnh (dưới dạng subfolder của các thành phố giống tập test)
    # Ví dụ: submission/test/berlin/berlin_000000_000019_pred.png
    sub_dir = Path(args.output_dir) / "test"
    sub_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Đang xử lý {len(dataset):,} ảnh tập test...")
    
    img_idx = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            # Đối với tập test không có nhãn, dataloader trả về Tensor ảnh trực tiếp
            if isinstance(batch, (list, tuple)):
                imgs = batch[0]
            else:
                imgs = batch
            
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)  # (B, 19, H, W)
            preds = torch.argmax(logits, dim=1).cpu().numpy()  # (B, H, W) các giá trị trainId (0-18)

            for i in range(imgs.size(0)):
                img_path = dataset.img_paths[img_idx]
                img_idx += 1

                # 1. Đọc kích thước gốc của ảnh
                with Image.open(img_path) as orig_img:
                    orig_w, orig_h = orig_img.size

                # 2. Ánh xạ ngược trainId (0-18) sang labelId gốc (0-33)
                pred_train_ids = preds[i]
                pred_label_ids = TRAIN_TO_LABEL[pred_train_ids]

                # 3. Resize mask dự đoán về kích thước gốc bằng NEAREST
                pred_pil = Image.fromarray(pred_label_ids)
                pred_pil = pred_pil.resize((orig_w, orig_h), Image.NEAREST)

                # 4. Xác định tên thành phố và tạo thư mục con tương ứng
                filename = img_path.name
                city_name = filename.split('_')[0]
                city_out_dir = sub_dir / city_name
                city_out_dir.mkdir(parents=True, exist_ok=True)

                # 5. Lưu ảnh dự đoán với hậu tố _pred.png theo chuẩn Cityscapes
                if "_leftImg8bit" in filename:
                    pred_filename = filename.replace("_leftImg8bit", "_pred")
                else:
                    pred_filename = img_path.stem + "_pred.png"
                
                pred_pil.save(city_out_dir / pred_filename)

    print(f"\n[INFO] Hoàn thành xuất {img_idx} ảnh dự đoán tại: {sub_dir.resolve()}")

    # 3. Tự động nén thành file zip
    if not args.no_zip:
        zip_path = Path(args.output_dir) / "submission.zip"
        print(f"[INFO] Đang đóng gói kết quả thành file zip: {zip_path.resolve()}")
        
        # Nén thư mục test/ bên trong output_dir vào file zip
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(sub_dir):
                for file in files:
                    file_path = Path(root) / file
                    # Giữ cấu trúc tương đối bắt đầu từ folder "test/..."
                    arcname = file_path.relative_to(Path(args.output_dir))
                    zipf.write(file_path, arcname)
                    
        print(f"★ File nén đã sẵn sàng: {zip_path.resolve()}")
        print("Bạn chỉ cần tải file 'submission.zip' này và nộp lên Cityscapes Benchmark!")

if __name__ == "__main__":
    main()
