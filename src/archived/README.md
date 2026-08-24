# Thư mục archived/

Chứa các file đã được xây dựng nhưng **chưa cần thiết** cho giai đoạn thực nghiệm hiện tại.

## Nội dung

| File | Mô tả | Khi nào dùng lại |
|:---|:---|:---|
| `train_fpn.py` | Training script cho FPN Multi-scale Decoder | Sau khi có kết quả E0–E4, nếu muốn cải thiện boundary quality |
| `extract_features.py` | Pre-compute và cache đặc trưng DINOv2 để tăng tốc training | Khi dataset lớn và cần tăng tốc inference |

## Lý do archive

Theo tài liệu thực nghiệm (Mục 16):
> *"Chưa cần thêm FPN, 2.5D, SAM hoặc architecture mới.  
> Trước mắt chỉ cần chạy sạch: Frozen vs Full FT vs Partial FT vs LoRA."*

## Cách khôi phục

```bash
# Di chuyển file trở lại src/ khi cần
mv src/archived/train_fpn.py src/
mv src/archived/extract_features.py src/
```
