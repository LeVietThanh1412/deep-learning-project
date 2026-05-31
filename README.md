# U-Net Cityscapes Segmentation

## Giới thiệu

Dự án triển khai bài toán phân đoạn ảnh theo mức pixel (semantic segmentation) với dữ liệu Cityscapes. Mục tiêu là huấn luyện mô hình U-Net để dự đoán mặt nạ phân lớp cho từng điểm ảnh, đánh giá bằng mIoU/Dice và phân tích lỗi dự đoán.

## Mục tiêu

- Huấn luyện mô hình phân đoạn ảnh để dự đoán vùng quan tâm theo pixel-level.
- Đầu ra gồm mặt nạ phân đoạn và ảnh overlay mask lên ảnh gốc.
- Đạt tiêu chí hiệu năng tối thiểu (ví dụ Dice >= 0.75).
- Phân tích các vùng mô hình dự đoán sai.

## Dữ liệu

- Bộ dữ liệu: Cityscapes (đã tiền xử lý theo cấu trúc thư mục bên dưới).
- Đầu vào: ảnh RGB.
- Đầu ra: mask phân lớp (mặc định 11 lớp + 1 lớp void).

## Cấu trúc thư mục

```
U-Net_Cityscapes_Segmentation/
├── datasets/
│   └── cityscapes/            # Thư mục chứa dữ liệu đã tiền xử lý
│       ├── train/             # Tập huấn luyện
│       ├── val/               # Tập tinh chỉnh siêu tham số
│       └── test/              # Tập đánh giá độc lập
├── models/
│   ├── __init__.py
│   └── unet.py                # Định nghĩa kiến trúc U-Net (Encoder, Decoder, Skip Connections, Softmax)
├── utils/
│   ├── __init__.py
│   ├── dataloader.py          # Xử lý input (ảnh RGB) và output (mask 11 lớp + 1 void)
│   ├── losses.py              # Cài đặt CCE kết hợp Dice Loss/Weighted Loss
│   └── metrics.py             # Tính mIoU cho 11 lớp
├── train.py                   # Chạy vòng lặp huấn luyện (training loop)
├── test.py                    # Đánh giá mô hình trên tập test
├── predict.py                 # Dự đoán ảnh mới và xuất mask
├── notebooks/
│   └── train_kaggle.ipynb      # Notebook khung để chạy trên Kaggle
└── requirements.txt           # Danh sách thư viện cần thiết
```

## Cài đặt môi trường

Tạo môi trường Python và cài đặt thư viện:

```bash
pip install -r requirements.txt
```

## Hướng dẫn chuẩn bị dữ liệu

Cấu trúc dữ liệu trong `datasets/cityscapes/`:

```
datasets/cityscapes/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Lưu ý:
- Tên file ảnh và mask phải khớp nhau.
- Mask nên là ảnh nhãn với giá trị lớp nguyên (0..num_classes-1).

## Huấn luyện

Chạy huấn luyện cơ bản:

```bash
python train.py
```

Điểm cần điều chỉnh:
- Số lớp (`num_classes`) trong [models/unet.py](models/unet.py).
- Đường dẫn dữ liệu trong [train.py](train.py).
- Các siêu tham số (batch size, learning rate).

## Đánh giá

Chạy đánh giá trên tập test:

```bash
python test.py
```

Kết quả đánh giá sử dụng mIoU làm thước đo chính.

## Dự đoán

Chạy dự đoán cho ảnh đơn:

```bash
python predict.py
```

`predict.py` lưu kết quả dự đoán (mask) theo cấu hình trong file.

## Chạy trên Kaggle

Notebook mẫu để chạy trên Kaggle đã có sẵn:

- [notebooks/train_kaggle.ipynb](notebooks/train_kaggle.ipynb)

Các bước gợi ý:
1. Upload dataset lên Kaggle và đảm bảo đường dẫn `/kaggle/input/<ten_dataset>`.
2. Mở notebook và cập nhật biến `city_root` theo tên dataset.
3. Hoàn thiện phần Dataset/Dataloader trong notebook.
4. Chạy toàn bộ notebook và lưu model vào `/kaggle/working`.

## Yêu cầu hiệu năng

- IoU hoặc Dice >= mức tối thiểu (ví dụ Dice >= 0.75).
- Phân tích các vùng mô hình dự đoán sai.

## Gợi ý mô hình

- U-Net: triển khai nhanh, dễ điều chỉnh.
- DeepLabv3+: phù hợp nếu dữ liệu đa dạng và cần hiệu năng cao hơn.

## Thành viên

- Lê Việt Thành
- Nguyễn Gia Phát
