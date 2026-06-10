import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ---> QUAN TRỌNG: Import mạng UNet tự code của bạn vào đây <---
from models.unet import UNet  

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN 
# ==========================================
MODEL_PATH = os.path.join('outputs', 'checkpoints', 'model1_05CE.pth') 

# THAY ĐỔI: Chuyển từ 1 file sang đọc nguyên 1 thư mục
INPUT_DIR = os.path.join('datasets', 'data', 'input_viet_nam')
OUTPUT_DIR = os.path.join('datasets', 'data', 'output_viet_nam')

NUM_CLASSES = 19 
IMG_HEIGHT = 256 
IMG_WIDTH = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. KHỞI TẠO MÔ HÌNH U-NET
# ==========================================
print("[INFO] Đang khởi tạo và load trọng số mô hình...")
model = UNet(
    in_channels=3, 
    num_classes=NUM_CLASSES, 
    base_channels=64, 
    bilinear=True     
)

checkpoint = torch.load(MODEL_PATH, map_location=device)

if "model_state" in checkpoint:
    state_dict = checkpoint["model_state"]
else:
    state_dict = checkpoint 

clean_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("_orig_mod.", "")
    clean_state_dict[new_key] = v

model.load_state_dict(clean_state_dict)
model.to(device)
model.eval()
print("[OK] Tải mô hình thành công!\n")

# Bảng 19 màu chuẩn Cityscapes
COLOR_MAP = np.array([
    [128, 64, 128],   # 0: road
    [244, 35, 232],   # 1: sidewalk
    [70, 70, 70],     # 2: building
    [102, 102, 156],  # 3: wall
    [190, 153, 153],  # 4: fence
    [153, 153, 153],  # 5: pole
    [250, 170, 30],   # 6: traffic light
    [220, 220, 0],    # 7: traffic sign
    [107, 142, 35],   # 8: vegetation
    [152, 251, 152],  # 9: terrain
    [70, 130, 180],   # 10: sky
    [220, 20, 60],    # 11: person
    [255, 0, 0],      # 12: rider
    [0, 0, 142],      # 13: car
    [0, 0, 70],       # 14: truck
    [0, 60, 100],     # 15: bus
    [0, 80, 100],     # 16: train
    [0, 0, 230],      # 17: motorcycle
    [119, 11, 32],    # 18: bicycle
], dtype=np.uint8)

# Tạo sẵn thư mục output nếu chưa có
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Kiểm tra thư mục input
if not os.path.exists(INPUT_DIR):
    raise FileNotFoundError(f"Thư mục không tồn tại: {INPUT_DIR}. Hãy tạo thư mục và bỏ ảnh vào!")

# ==========================================
# 3. VÒNG LẶP XỬ LÝ HÀNG LOẠT ẢNH
# ==========================================
valid_extensions = ('.jpg', '.jpeg', '.png')
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)]

if len(image_files) == 0:
    print(f"[CẢNH BÁO] Không tìm thấy ảnh nào trong thư mục {INPUT_DIR}")
else:
    print(f"[INFO] Bắt đầu xử lý {len(image_files)} ảnh...\n" + "="*50)

for idx, filename in enumerate(image_files, 1):
    image_path = os.path.join(INPUT_DIR, filename)
    print(f"[{idx}/{len(image_files)}] Đang xử lý: {filename}")
    
    # --- Đọc và Tiền xử lý ---
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"  -> Lỗi: Không thể đọc ảnh {filename}")
        continue
        
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (IMG_WIDTH, IMG_HEIGHT))
    
    # ĐÃ THÊM: Chuẩn hóa theo chuẩn ImageNet (Giống hệt lúc train)
    image_norm = image_resized / 255.0 
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_norm = (image_norm - mean) / std
    
    image_tensor = torch.tensor(image_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    # --- Dự đoán ---
    with torch.no_grad():
        output = model(image_tensor) 
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy() 

    # --- Hiển thị và Lưu kết quả ---
    pred_color_mask = COLOR_MAP[pred_mask]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Ảnh gốc (Input)")
    plt.imshow(image_resized)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Mặt nạ dự đoán (Mask)")
    plt.imshow(pred_color_mask)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    overlay = cv2.addWeighted(image_resized, 0.6, pred_color_mask, 0.4, 0)
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()

    # Lưu ảnh với tên tương ứng
    output_path = os.path.join(OUTPUT_DIR, f"ketqua_{filename}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Đóng biểu đồ để giải phóng bộ nhớ RAM (Quan trọng khi chạy vòng lặp)
    plt.close()
    
    print(f"  -> Đã lưu tại: {output_path}")

print("="*50 + "\n[HOÀN THÀNH] Xử lý toàn bộ dữ liệu Việt Nam thành công!")