"""DINOv2 encoder and the basic decoder shared by experiments E0--E4.

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
        "vit_giant": ("dinov2_vitg14", 1536),
    }

    def __init__(self, model_name="vit_base"):
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

    def has_trainable_backbone_parameters(self):
        """Whether full fine-tuning (E2) requires autograd through the backbone."""
        return any(param.requires_grad for param in self.backbone.parameters())

    def train(self, mode=True):
        """Keep E0/E1's frozen DINOv2 backbone in eval mode."""
        super().train(mode)
        if mode and not self.has_trainable_backbone_parameters():
            self.backbone.eval()
        else:
            self.backbone.train(mode)
        return self

    def forward(self, x):
        """
        Args:
            x: tensor (B, 3, 448, 448)
        Returns:
            features: tensor (B, embed_dim, 32, 32)
        """
        # Frozen E0/E1 can skip autograd.  E2 must not: otherwise
        # their encoder parameters receive no gradient and the experiment is invalid.
        if self.has_trainable_backbone_parameters():
            output = self.backbone.forward_features(x)
        else:
            with torch.no_grad():
                output = self.backbone.forward_features(x)

        # Lấy patch tokens (bỏ class token)
        # output["x_norm_patchtokens"]: (B, num_patches, embed_dim)
        patch_tokens = output["x_norm_patchtokens"]

        B, N, D = patch_tokens.shape
        # N = 32*32 = 1024 patches cho ảnh 448×448
        h, w = x.shape[-2] // 14, x.shape[-1] // 14
        if N != h * w:
            raise RuntimeError(
                f"Expected {h * w} patch tokens for input {tuple(x.shape)}, got {N}."
            )

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

    def __init__(self, model_name="vit_base"):
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
    print("✓ Forward pass thành công!")
