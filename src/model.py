"""
model.py
--------
Kiến trúc mô hình DINOv2 + Decoder cho phân vùng tâm nhĩ trái.

Encoder: DINOv2 ViT (đóng băng - frozen)
Decoder: Các lớp Conv2d + Upsample để khôi phục độ phân giải

Theo bài báo:
  - DINOv2 chia ảnh 448×448 thành 32×32 = 1024 patches (patch_size=14)
  - Đầu ra encoder: (B, 1024, embed_dim) → reshape → (B, embed_dim, 32, 32)
  - Decoder upscale từ 32×32 → 448×448 qua các lớp conv + upsample
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2Encoder(nn.Module):
    """
    Bộ mã hoá DINOv2 (frozen).
    Tải mô hình pre-trained từ torch.hub và đóng băng trọng số.
    """

    # Mapping tên mô hình → (tên trên hub, embed_dim)
    MODEL_CONFIGS = {
        "vit_small": ("dinov2_vits14", 384),
        "vit_base": ("dinov2_vitb14", 768),
        "vit_large": ("dinov2_vitl14", 1024),
        # "vit_giant": ("dinov2_vitg14", 1536),  # Cần nhiều VRAM
    }

    def __init__(self, model_name="vit_small"):
        super().__init__()

        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Model '{model_name}' không được hỗ trợ. "
                f"Chọn từ: {list(self.MODEL_CONFIGS.keys())}"
            )

        hub_name, self.embed_dim = self.MODEL_CONFIGS[model_name]

        # Tải mô hình DINOv2 từ facebookresearch
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            hub_name,
            pretrained=True,
        )

        # Đóng băng toàn bộ trọng số encoder
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.eval()

    def forward(self, x):
        """
        Args:
            x: tensor (B, 3, 448, 448)
        Returns:
            features: tensor (B, embed_dim, 32, 32)
        """
        # DINOv2 forward_features trả về dict chứa patch tokens
        with torch.no_grad():
            output = self.backbone.forward_features(x)

        # Lấy patch tokens (bỏ class token)
        # output["x_norm_patchtokens"]: (B, num_patches, embed_dim)
        patch_tokens = output["x_norm_patchtokens"]

        B, N, D = patch_tokens.shape
        # N = 32*32 = 1024 patches cho ảnh 448×448
        h = w = int(N ** 0.5)

        # Reshape: (B, N, D) → (B, D, h, w)
        features = patch_tokens.permute(0, 2, 1).reshape(B, D, h, w)

        return features


class SegmentationDecoder(nn.Module):
    """
    Bộ giải mã để upscale feature map từ 32×32 → 448×448.

    Kiến trúc gồm:
      1. Conv 1×1 để giảm số kênh (LinearClassifier)
      2. Chuỗi các khối UpBlock: Upsample + Conv + BN + ReLU
      3. Conv 1×1 cuối cùng cho đầu ra 1 kênh (binary segmentation)
    """

    def __init__(self, in_channels, target_size=448):
        super().__init__()
        self.target_size = target_size

        # 1×1 conv để giảm chiều đặc trưng
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Chuỗi upsampling: 32 → 64 → 128 → 256 → 448
        self.up1 = self._up_block(256, 128)   # 32  → 64
        self.up2 = self._up_block(128, 64)    # 64  → 128
        self.up3 = self._up_block(64, 32)     # 128 → 256

        # Lớp cuối: Conv 1×1 để ra 1 kênh
        self.head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    @staticmethod
    def _up_block(in_ch, out_ch):
        """Khối Upsample ×2 + Conv + BN + ReLU."""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Args:
            x: tensor (B, embed_dim, 32, 32)
        Returns:
            logits: tensor (B, 1, 448, 448)
        """
        x = self.reduce(x)   # (B, 256, 32, 32)
        x = self.up1(x)      # (B, 128, 64, 64)
        x = self.up2(x)      # (B, 64, 128, 128)
        x = self.up3(x)      # (B, 32, 256, 256)

        # Interpolate về kích thước chính xác 448×448
        x = F.interpolate(x, size=(self.target_size, self.target_size),
                          mode="bilinear", align_corners=False)

        logits = self.head(x)  # (B, 1, 448, 448)
        return logits


class DINOv2Segmenter(nn.Module):
    """
    Mô hình hoàn chỉnh: DINOv2 Encoder (frozen) + Decoder (trainable).
    """

    def __init__(self, model_name="vit_small"):
        super().__init__()
        self.encoder = DINOv2Encoder(model_name=model_name)
        self.decoder = SegmentationDecoder(
            in_channels=self.encoder.embed_dim,
            target_size=448,
        )

    def forward(self, x):
        """
        Args:
            x: tensor (B, 3, 448, 448)
        Returns:
            logits: tensor (B, 1, 448, 448) — chưa qua sigmoid
        """
        features = self.encoder(x)  # (B, embed_dim, 32, 32)
        logits = self.decoder(features)  # (B, 1, 448, 448)
        return logits

    def get_trainable_params(self):
        """Chỉ trả về các tham số của decoder (có thể huấn luyện)."""
        return self.decoder.parameters()

    def num_trainable_params(self):
        """Đếm số tham số có thể huấn luyện."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_params(self):
        """Đếm tổng số tham số."""
        return sum(p.numel() for p in self.parameters())


# ────────────────────────────────────────────────────────────
# MỤC 3: Feature Pyramid Network (FPN) Decoder
# Giải quyết vấn đề mất chi tiết đường viền do patch_size=14
# ────────────────────────────────────────────────────────────

class DINOv2FPNEncoder(nn.Module):
    """
    Encoder lấy đặc trưng từ nhiều lớp Transformer của DINOv2.

    Vấn đề của encoder gốc:
      - Chỉ lấy đặc trưng từ lớp CUỐI CÙNG (lớp 12 của ViT-S)
      - Lớp cuối rất "trừu tượng" (biết TÂM NHĨ là gì) nhưng mất chi tiết cạnh
      - Lớp đầu còn giữ thông tin vị trí cạnh (texture, edges) nhưng không được dùng

    Giải pháp FPN:
      - Lấy đặc trưng từ N lớp cuối (mặc định 4 lớp: 9, 10, 11, 12)
      - Mỗi lớp cung cấp "góc nhìn" khác nhau về ảnh MRI
      - Decoder kết hợp tất cả → vừa có ngữ cảnh (deep), vừa có chi tiết (shallow)
    """

    MODEL_CONFIGS = {
        "vit_small": ("dinov2_vits14", 384, 12),
        "vit_base":  ("dinov2_vitb14", 768, 12),
        "vit_large": ("dinov2_vitl14", 1024, 24),
    }

    def __init__(self, model_name="vit_small", n_levels=4):
        super().__init__()
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Model '{model_name}' không hợp lệ.")

        hub_name, self.embed_dim, _ = self.MODEL_CONFIGS[model_name]
        self.n_levels = n_levels

        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", hub_name, pretrained=True
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def forward(self, x):
        """
        Args:
            x: tensor (B, 3, 448, 448)
        Returns:
            features: list of n_levels tensors, mỗi cái (B, embed_dim, 32, 32)
                      từ nông đến sâu (early → late blocks)
        """
        with torch.no_grad():
            # Trả về tuple của n_levels tensor, mỗi cái (B, N, D)
            outputs = self.backbone.get_intermediate_layers(
                x, n=self.n_levels, return_class_token=False
            )

        B = x.shape[0]
        features = []
        for out in outputs:
            N, D = out.shape[1], out.shape[2]
            H = W = int(N ** 0.5)   # 32 × 32 = 1024 patches
            feat = out.permute(0, 2, 1).reshape(B, D, H, W)  # (B, D, 32, 32)
            features.append(feat)

        return features  # list: [f_shallow, ..., f_deep]


class FPNDecoder(nn.Module):
    """
    FPN Decoder: kết hợp đặc trưng từ nhiều lớp ViT để cải thiện chi tiết đường viền.

    Cơ chế:
      1. Lateral conv: chiếu mỗi lớp về cùng số kênh (fpn_ch)
      2. Top-down merge: cộng thêm đặc trưng ngữ cảnh (deep) vào đặc trưng chi tiết (shallow)
      3. Ghép tất cả lại: concat → upsample → dự đoán mask
    """

    def __init__(self, in_channels, n_levels=4, target_size=448, fpn_ch=128):
        super().__init__()
        self.target_size = target_size
        self.n_levels = n_levels

        # Lateral convolutions: project mỗi level về fpn_ch channels
        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, fpn_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_ch),
                nn.ReLU(inplace=True),
            )
            for _ in range(n_levels)
        ])

        # Top-down smooth conv (sau khi cộng top-down)
        self.smooths = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_ch, fpn_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_ch),
                nn.ReLU(inplace=True),
            )
            for _ in range(n_levels)
        ])

        # Upsampling decoder sau khi ghép tất cả các level
        concat_ch = n_levels * fpn_ch   # 4 × 128 = 512
        self.up1 = self._up_block(concat_ch, 256)   # 32 → 64
        self.up2 = self._up_block(256, 128)          # 64 → 128
        self.up3 = self._up_block(128, 64)           # 128 → 256

        self.head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    @staticmethod
    def _up_block(in_ch, out_ch):
        """Khối Upsample ×2 + Conv + BN + ReLU."""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, feature_list):
        """
        Args:
            feature_list: list of n_levels tensors, mỗi cái (B, embed_dim, 32, 32)
                          từ nông đến sâu
        Returns:
            logits: (B, 1, 448, 448)
        """
        # Bước 1: Lateral projection
        fpn_feats = [lat(f) for lat, f in zip(self.laterals, feature_list)]
        # fpn_feats: list of (B, fpn_ch, 32, 32)

        # Bước 2: Top-down merge (từ deep → shallow)
        # Vì tất cả cùng kích thước 32×32, chỉ cần cộng thẳng
        merged = [None] * self.n_levels
        merged[-1] = fpn_feats[-1]   # Lớp sâu nhất, giữ nguyên
        for i in range(self.n_levels - 2, -1, -1):
            merged[i] = fpn_feats[i] + merged[i + 1]   # Cộng top-down

        # Bước 3: Smooth conv sau mỗi merge
        smoothed = [s(m) for s, m in zip(self.smooths, merged)]

        # Bước 4: Concat tất cả levels
        x = torch.cat(smoothed, dim=1)  # (B, n_levels*fpn_ch, 32, 32)

        # Bước 5: Upsample → predict
        x = self.up1(x)   # (B, 256, 64, 64)
        x = self.up2(x)   # (B, 128, 128, 128)
        x = self.up3(x)   # (B, 64, 256, 256)
        x = F.interpolate(x, size=(self.target_size, self.target_size),
                          mode="bilinear", align_corners=False)
        return self.head(x)   # (B, 1, 448, 448)


class DINOv2FPNSegmenter(nn.Module):
    """
    Mô hình phân vùng hoàn chỉnh: DINOv2 FPN Encoder + FPN Decoder.

    Ưu điểm so với DINOv2Segmenter gốc:
      - Khai thác đặc trưng từ nhiều lớp → không bỏ mất chi tiết cạnh
      - Top-down merge → ngữ cảnh từ lớp sâu được truyền về lớp nông
      - Đường viền tâm nhĩ mượt hơn và chính xác hơn
    """

    def __init__(self, model_name="vit_small", n_levels=4):
        super().__init__()
        self.encoder = DINOv2FPNEncoder(model_name=model_name, n_levels=n_levels)
        self.decoder = FPNDecoder(
            in_channels=self.encoder.embed_dim,
            n_levels=n_levels,
            target_size=448,
        )

    def forward(self, x):
        """
        Args:
            x: tensor (B, 3, 448, 448)
        Returns:
            logits: tensor (B, 1, 448, 448)
        """
        features = self.encoder(x)    # list of n_levels × (B, D, 32, 32)
        logits = self.decoder(features)  # (B, 1, 448, 448)
        return logits

    def get_trainable_params(self):
        """Chỉ train decoder (encoder DINOv2 vẫn frozen)."""
        return self.decoder.parameters()

    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    print("Kiểm tra kiến trúc mô hình...")

    model = DINOv2Segmenter(model_name="vit_small")
    print(f"\nEncoder: DINOv2 ViT-Small (embed_dim={model.encoder.embed_dim})")
    print(f"Tổng tham số:          {model.num_total_params():>12,}")
    print(f"Tham số huấn luyện:    {model.num_trainable_params():>12,}")

    dummy = torch.randn(2, 3, 448, 448)
    with torch.no_grad():
        output = model(dummy)
    print(f"\nInput  shape: {dummy.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Forward pass DINOv2Segmenter thành công!")

    print("\n--- FPN Segmenter ---")
    fpn_model = DINOv2FPNSegmenter(model_name="vit_small", n_levels=4)
    print(f"Tham số huấn luyện: {fpn_model.num_trainable_params():>12,}")
    with torch.no_grad():
        out_fpn = fpn_model(dummy)
    print(f"Output FPN shape:   {out_fpn.shape}")
    print("✓ FPN forward pass thành công!")

