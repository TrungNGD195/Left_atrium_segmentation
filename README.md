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

## 🚀 Workflow ViT-Large: local → GitHub → server

GitHub origin/main là nguồn mã chuẩn. Chỉ sửa code tại máy local, commit/push lên GitHub; server chỉ pull fast-forward và chạy thí nghiệm. Không push trực tiếp vào server hoặc sửa code trong worktree server.

### 1. Đồng bộ mã

Tại local, trước khi push:

~~~bash
git fetch origin
git rebase origin/main
git push origin main
~~~

Trên server, trong project:

~~~bash
bash scripts/sync_server.sh
~~~

data/, results/, logs/, checkpoint .pth/.pt đều chỉ lưu trên server và bị Git ignore.

### 2. Cài môi trường server (một lần)

~~~bash
sudo apt update && sudo apt install -y tmux
cd /mnt/data/users/trungptit/Left_atrium_segmentation
bash scripts/bootstrap_server.sh
~~~

Script tạo .venv, cài PyTorch 2.11 + CUDA 12.8, kiểm tra GPU CUDA và tạo logs/, results/vit_large/e1, results/vit_large/e2.

### 3. Smoke test qua tmux

Frozen BCE+Dice (E1, batch 8):

~~~bash
tmux new -s la-e1
.venv/bin/python src/train_baseline.py --model vit_large --loss bce_dice --epochs 1 --batch_size 8 --num_workers 4 --data_root data --save_dir results/vit_large/e1 2>&1 | tee logs/la-e1-smoke.log
~~~

Full fine-tuning (E2, batch 1) chỉ chạy sau khi E1 thành công:

~~~bash
tmux new -s la-e2
.venv/bin/python src/train_full_ft.py --model vit_large --epochs 1 --batch_size 1 --num_workers 4 --data_root data --save_dir results/vit_large/e2 2>&1 | tee logs/la-e2-smoke.log
~~~

Detach tmux bằng Ctrl+B, sau đó D; quay lại bằng tmux attach -t la-e1 hoặc tmux attach -t la-e2.

Checkpoint ViT-Small không tương thích với ViT-Large; chỉ evaluate checkpoint được train bằng --model vit_large.

### 4. Đánh giá

~~~bash
.venv/bin/python src/evaluate.py --model vit_large --checkpoint results/vit_large/e1/<checkpoint>.pth --data_root data --save_dir results/vit_large/e1
~~~

---

## 📚 Tài liệu tham khảo

- **Bài báo gốc:** [DINOv2-based Left Atrium Segmentation (arXiv:2411.09598)](https://arxiv.org/abs/2411.09598) — file PDF trong [`docs/`](docs/2411.09598v1.pdf)
- **DINOv2:** Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023
- **Dataset:** [Left Atrium Segmentation Challenge](http://atriaseg2018.cardiacatlas.org/)
