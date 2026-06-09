import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import os

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN (Đã chuẩn hóa theo Tree của bạn)
# ==========================================
MODEL_PATH = os.path.join('outputs', 'checkpoints', 'model3_Resnet34Lr13.pth')
IMAGE_PATH = os.path.join('datasets', 'data', 'input', 'test_image_3.png')
OUTPUT_DIR = os.path.join('datasets', 'data', 'output')

NUM_CLASSES = 19
IMG_HEIGHT = 256 
IMG_WIDTH = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. KHỞI TẠO MÔ HÌNH VÀ LOAD TRỌNG SỐ
# ==========================================
print("[INFO] Đang tải mô hình...")
model = smp.Unet(
    encoder_name="resnet34", 
    encoder_weights=None, 
    in_channels=3, 
    classes=NUM_CLASSES
)

# Đọc checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=device)
state_dict = checkpoint["model_state"]

# Xử lý lỗi tiền tố "_orig_mod." do torch.compile() sinh ra lúc train
clean_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("_orig_mod.", "") # Gọt bỏ tiền tố
    clean_state_dict[new_key] = v

# Load trọng số đã làm sạch vào model
model.load_state_dict(clean_state_dict)
model.to(device)
model.eval()
print("[OK] Tải mô hình thành công!")

# ==========================================
# 3. ĐỌC VÀ TIỀN XỬ LÝ ẢNH
# ==========================================
print(f"[INFO] Đọc ảnh từ: {IMAGE_PATH}")
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Không tìm thấy ảnh tại: {IMAGE_PATH}")

image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Dòng code cũ của bạn
image_resized = cv2.resize(image_rgb, (IMG_WIDTH, IMG_HEIGHT))

# ---> CHÈN THÊM ĐOẠN NÀY VÀO ĐỂ XEM ẢNH RESIZE <---
plt.figure(figsize=(8, 4))
plt.imshow(image_resized)
plt.title(f"Ảnh sau khi resize - Kích thước: {image_resized.shape}")
plt.axis("off")
plt.show()
# ---------------------------------------------------

# Chuẩn hóa ảnh (Normalization)
image_norm = image_resized / 255.0 
image_tensor = torch.tensor(image_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

# ==========================================
# 4. CHẠY MÔ HÌNH DỰ ĐOÁN
# ==========================================
print("[INFO] Đang phân tích ảnh...")
with torch.no_grad():
    output = model(image_tensor) 
    pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy() 

# ==========================================
# 5. HIỂN THỊ VÀ LƯU KẾT QUẢ
# ==========================================
# ==========================================
# 5. TẠO BẢNG MÀU VÀ HIỂN THỊ KẾT QUẢ
# ==========================================
# Bảng màu đại diện cho 19 class chuẩn của Cityscapes (R, G, B)
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

# Chuyển mask dạng số (0-18) thành mask dạng màu RGB
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

# Tạo folder output và lưu file
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, 'ketqua_demo_rs.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')

print(f"[OK] Đã lưu ảnh kết quả tại: {output_path}")
plt.show()