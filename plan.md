# Kế hoạch dự án Phân đoạn ảnh

Ngày: 2026-05-27

## 1. Phát biểu bài toán (dùng cho mở đầu báo cáo)
- Phân đoạn ảnh gán nhãn lớp cho từng pixel trong ảnh đầu vào. Khác với phân loại ảnh (mỗi ảnh một nhãn) hoặc phát hiện đối tượng (dùng bounding box), phân đoạn ảnh cung cấp hiểu biết ở mức điểm ảnh (pixel-level). 
- Mục tiêu là xây dựng, huấn luyện và đánh giá mô hình học sâu có khả năng tách vùng quan tâm (ROI) khỏi nền (background). 
- Đầu vào là ảnh RGB; đầu ra là mặt nạ (mask) có cùng kích thước không gian với ảnh gốc.

## 2. Tổng quan nghiên cứu (literature review)
- FCN (2015): Thay lớp fully connected bằng lớp tích chập để giữ thông tin không gian và tạo dự đoán dày (dense).
- U-Net (2015): Encoder-decoder với skip connection; mạnh trong dữ liệu nhỏ, phổ biến trong y tế.
- DeepLab v1-v3+ (2017-2018): Atrous convolution + ASPP để lấy ngữ cảnh đa tỉ lệ.
- SOTA gần đây (bonus): Transformer như SegFormer hoặc foundation model như SAM.

## 3. Hướng đề tài + gợi ý backbone
Chọn một hướng và khớp dataset với backbone:

| Hướng | Dataset ví dụ | Ghi chú | Backbone đề xuất |
| --- | --- | --- | --- |
| Y tế | U não, phân đoạn tế bào | Dữ liệu nhỏ, mask tương phản cao | U-Net + ResNet34 hoặc EfficientNet-B0 |
| Lái xe tự động | Cityscapes | Nhiều lớp, đa tỉ lệ | DeepLabv3+ + ResNet50 hoặc SegFormer-B0 |
| Vệ tinh | Đường, mái nhà | Cấu trúc dài, mảnh | DeepLabv3+ + ResNet101 hoặc U-Net++ + EfficientNet |
| Ablation | Bất kỳ bộ nhị phân | So sánh loss | U-Net + ResNet34 hoặc EfficientNet-B0 |

## 4. Pipeline dữ liệu
1) Thu thập dữ liệu
- Cần ảnh RGB (H, W, 3) và mask (H, W, 1).
- Xác định nhãn lớp và cách mã hóa mask (nhị phân hoặc đa lớp).

2) Tiền xử lý
- Resize về kích thước cố định (vd: 256x256 hoặc 512x512) theo GPU.
- Chuẩn hóa theo mean/std ImageNet nếu dùng backbone pretrained.
- Chia train/val/test.

3) Tăng cường dữ liệu (phải đồng bộ ảnh + mask)
- Xoay, lật, scale, đổi sáng/tương phản, thêm nhiễu.
- Dùng Albumentations để đảm bảo ảnh và mask biến đổi giống nhau.

## 5. Kiến trúc mô hình
Phương án A: U-Net
- Encoder: backbone pretrained (ResNet/EfficientNet) qua segmentation_models_pytorch.
- Decoder: upsampling đối xứng + skip connection.

Phương án B: DeepLabv3+
- Atrous convolution + ASPP cho đa tỉ lệ.
- Tốt khi vật thể có kích thước thay đổi lớn.

## 6. Hàm loss
- Tránh MSE.
- Dùng BCEWithLogitsLoss + Dice Loss (hoặc Focal Loss nếu mất cân bằng nặng).
- Theo dõi tỉ lệ mất cân bằng, cân nhắc class weight nếu cần.

## 7. Huấn luyện và đánh giá
- Optimizer: Adam hoặc AdamW.
- LR scheduler: Cosine hoặc ReduceLROnPlateau.
- Metrics: IoU và Dice coefficient.
- Mục tiêu: IoU hoặc Dice >= 0.75.

## 8. Phân tích lỗi (bắt buộc)
- Trực quan hóa: chồng mask dự đoán (đỏ) lên mask thật (xanh).
- Phân tích lỗi: biên mờ, ánh sáng chói, che khuất, vật thể nhỏ.

## 9. Kế hoạch thí nghiệm (ablation)
- Baseline: U-Net + BCEWithLogitsLoss.
- Biến thể 1: U-Net + BCE + Dice.
- Biến thể 2: DeepLabv3+ + BCE + Dice.
- Theo dõi cải thiện Dice/IoU và chất lượng biên.

## 10. Triển khai (tùy chọn lấy điểm cộng)
- Lưu checkpoint (.pth).
- Dựng API FastAPI: upload ảnh -> trả mask + ảnh overlay.
- Frontend đơn giản: Vue/Vite.

## 11. Phân công nhóm
- Thành viên A: thu thập dữ liệu + pipeline tiền xử lý.
- Thành viên B: mô hình + vòng lặp train.
- Thành viên C: đánh giá + trực quan + phân tích lỗi.
- Thành viên D: viết báo cáo + triển khai (nếu làm).

## 12. Timeline gợi ý
Tuần 1:
- Chọn dataset + hướng đề tài.
- Xây pipeline dữ liệu + baseline.

Tuần 2:
- Train baseline + đánh giá.
- Thử loss variants + ablation.

Tuần 3:
- Phân tích lỗi + hoàn thiện báo cáo + demo triển khai.

## 13. Hạng mục trong báo cáo
- Phát biểu bài toán.
- Tổng quan nghiên cứu.
- Mô tả dữ liệu + tiền xử lý.
- Sơ đồ kiến trúc mô hình.
- Thiết lập huấn luyện + metric.
- Kết quả định lượng (IoU/Dice).
- Kết quả định tính (overlay).
- Phân tích lỗi + thảo luận.

---
Ghi chú:
- Nếu nhóm chốt dataset cụ thể, cập nhật lại Mục 3.
- Nêu rõ giới hạn GPU/CPU để chọn kích thước ảnh và backbone.
