# 🫀 Left Atrium Segmentation với DINOv2

Đồ án nghiên cứu ứng dụng mô hình nền tảng **DINOv2** (Vision Transformer) vào bài toán phân vùng tự động tâm nhĩ trái (Left Atrium) từ ảnh MRI 3D. Dự án tái hiện và mở rộng 5 kỹ thuật cải tiến từ bài báo gốc [[2411.09598]](docs/2411.09598v1.pdf).

---

## 📊 Kết quả thực nghiệm

| Mục | Kỹ thuật cải tiến | Vấn đề giải quyết | Val Dice | Test Dice |
|:---:|:---|:---|:---:|:---:|
| **1** | BCE + Dice Loss | Class Imbalance | 0.8343 | **0.7376** |
| **2** | LoRA Fine-tuning Encoder | Domain Shift | 0.8494 | **0.7819** |
| **3** | FPN Multi-scale Decoder | Mất chi tiết đường viền | 0.8784 | **0.8151** |
| **4** | Hậu xử lý LCC | Đốm nhiễu rời rạc | — | 0.7311 |
| **5** | Đầu vào 2.5D [z-1, z, z+1] | Thiếu ngữ cảnh 3D | 0.7965 | 0.7102 |

> **FPN Multi-scale Decoder (Mục 3) đạt kết quả tốt nhất: Test Dice = 0.8151**, tăng **+10.5%** so với baseline.

---

## 🗂️ Cấu trúc thư mục

```
Left_atrium_segmentation/
├── docs/
│   └── 2411.09598v1.pdf          # Bài báo tham khảo gốc
│
├── notebooks/
│   ├── colab_train.ipynb          # Notebook huấn luyện & đánh giá trên Google Colab
│   └── eda.ipynb                  # Khám phá dữ liệu (EDA)
│
├── src/
│   ├── model.py                   # Kiến trúc mô hình:
│   │                              #   · DINOv2Segmenter (Mục 1, 4, 5)
│   │                              #   · DINOv2FPNSegmenter (Mục 3)
│   ├── dataset.py                 # Dataset loader 2D & 2.5D
│   ├── evaluate.py                # Đánh giá + trực quan hóa + LCC
│   ├── lora.py                    # Mục 2: LoRALinear + inject_lora_into_backbone
│   ├── train.py                   # Hàm tiện ích: dice_score, iou_score, v.v.
│   ├── train_fast.py              # Script train Mục 1, 4, 5 (AMP FP16)
│   ├── train_lora.py              # Script train Mục 2: LoRA Fine-tuning
│   ├── train_fpn.py               # Script train Mục 3: FPN Decoder
│   ├── prepare_data.py            # Tiền xử lý: NIfTI 3D → lát cắt 2D PNG
│   └── extract_features.py        # Trích xuất features (tùy chọn, tăng tốc)
│
├── data/                          # ⚠️ Gitignored — dữ liệu MRI gốc
│   ├── train_2d/
│   ├── val_2d/
│   └── test_2d/
│
├── results/                       # ⚠️ Gitignored — checkpoint + kết quả
│
├── requirements.txt
└── README.md
```

---

## 🧠 Các kỹ thuật cải tiến

### Mục 1 — BCE + Dice Loss (Class Imbalance)
Tâm nhĩ trái chỉ chiếm ~5% diện tích ảnh MRI. Cross-Entropy đơn thuần bị "ưu tiên" tối ưu nền đen (95%) → Giải pháp kết hợp BCE + Dice Loss.

### Mục 2 — LoRA Fine-tuning (Domain Shift)
DINOv2 pre-trained trên ImageNet (ảnh tự nhiên), đặc trưng không phù hợp với ảnh MRI. Giải pháp: inject ma trận LoRA (rank=4) vào 2 block Transformer cuối với differential learning rate (decoder: 1e-3, LoRA: 1e-5).

### Mục 3 — FPN Multi-scale Decoder (Mất chi tiết đường viền)
DINOv2 chia ảnh thành patch 14×14, decoder gốc chỉ dùng layer cuối → đường viền bị vuông vức. Giải pháp FPN:
- Trích xuất đặc trưng từ **4 layer Transformer cuối** (lớp 9, 10, 11, 12)
- **Top-down merge**: ngữ cảnh sâu → bổ sung cho đặc trưng nông (chi tiết cạnh)
- Concat 4×128 = 512 channels → upsample → dự đoán mask mượt mà hơn

### Mục 4 — Hậu xử lý LCC (False Positives)
Tâm nhĩ là một vùng liên thông duy nhất. Giữ lại Largest Connected Component và xóa bỏ các đốm nhiễu rời rạc.

### Mục 5 — Đầu vào 2.5D (Thiếu ngữ cảnh 3D)
Thay vì 1 lát cắt, đưa 3 lát cắt liên tiếp [z-1, z, z+1] vào 3 kênh RGB để mô hình nhận biết sự liên tục không gian 3D mà không cần 3D CNN nặng nề.

---

## 🚀 Hướng dẫn chạy trên Google Colab

> **Yêu cầu:** Google Colab GPU T4 (miễn phí)

### 1. Chuẩn bị dữ liệu (chạy 1 lần)
```bash
python src/prepare_data.py --data_dir data/raw --output_dir data
```

### 2. Huấn luyện

**Mục 1 — Baseline BCE+Dice:**
```bash
python src/train_fast.py --data_root data --batch_size 64 --epochs 35 --lr 1e-3 --loss bce_dice --save_dir results
```

**Mục 2 — LoRA Fine-tuning:**
```bash
python src/train_lora.py --data_root data --batch_size 32 --epochs 30 --lr_decoder 1e-3 --lr_lora 1e-5 --lora_blocks 2 --lora_rank 4 --save_dir results
```

**Mục 3 — FPN Decoder:**
```bash
python src/train_fpn.py --data_root data --batch_size 32 --epochs 35 --lr 1e-3 --n_levels 4 --save_dir results
```

### 3. Đánh giá
```bash
python src/evaluate.py --model vit_small --checkpoint results/best_decoder_vit_small_bce_dice_fpn.pth --data_root data
```

> Xem đầy đủ trong `notebooks/colab_train.ipynb` để chạy từng bước trực quan.

---

## 📦 Cài đặt

```bash
git clone https://github.com/TrungNGD195/Left_atrium_segmentation.git
cd Left_atrium_segmentation
pip install -r requirements.txt
```

---

## 📚 Tài liệu tham khảo

- **Bài báo gốc:** [DINOv2-based Left Atrium Segmentation (arXiv:2411.09598)](https://arxiv.org/abs/2411.09598) — file PDF trong [`docs/`](docs/2411.09598v1.pdf)
- **DINOv2:** Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023
- **Dataset:** [Left Atrium Segmentation Challenge](http://atriaseg2018.cardiacatlas.org/)
