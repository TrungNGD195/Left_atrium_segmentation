"""
lora.py
-------
Triển khai LoRA (Low-Rank Adaptation) cho DINOv2 ViT encoder.

Vấn đề giải quyết (Mục 2):
  DINOv2 được pre-train hoàn toàn trên ảnh tự nhiên (ImageNet),
  trong khi ảnh MRI tim có đặc trưng rất khác biệt (grayscale, low contrast).
  Đóng băng hoàn toàn encoder → Domain Shift nghiêm trọng.

Giải pháp LoRA:
  Thay vì fine-tune toàn bộ encoder (hàng triệu tham số, dễ overfitting),
  LoRA chèn thêm 2 ma trận thấp hạng A, B vào các attention projection:
      W_new = W_frozen + (B @ A) * (alpha / rank)
  → Chỉ cần train ~0.5% tham số so với full fine-tune!
  → Mô hình học được domain ảnh MRI mà không quên kiến thức ImageNet.

Tham khảo: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
"""

import torch
import torch.nn as nn
import math


# ────────────────────────────────────────────────────────────
# LoRA Linear Layer
# ────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """
    Thay thế nn.Linear bằng phiên bản LoRA:
        output = x @ W.T + x @ A.T @ B.T * (alpha / rank)
    Trong đó W bị đóng băng, chỉ A và B được huấn luyện.

    Args:
        linear:  nn.Linear gốc (trọng số bị đóng băng)
        rank:    Hạng của ma trận thấp hạng (thường 4 hoặc 8)
        alpha:   Hệ số scale (thường bằng rank để scale = 1.0)
    """

    def __init__(self, linear: nn.Linear, rank: int = 4, alpha: float = 4.0):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = linear.in_features
        out_features = linear.out_features

        # Đóng băng trọng số gốc
        linear.weight.requires_grad = False
        if linear.bias is not None:
            linear.bias.requires_grad = False

        # Ma trận LoRA A: khởi tạo theo Kaiming để tránh vanishing gradient
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Ma trận LoRA B: khởi tạo 0 → output ban đầu = output gốc (không thay đổi)
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Kết quả gốc (frozen)
        result = self.linear(x)
        # Cộng thêm LoRA delta: x → A → B, scale by alpha/rank
        lora_delta = (x @ self.lora_A.T) @ self.lora_B.T
        return result + lora_delta * self.scaling

    def extra_repr(self) -> str:
        return (
            f"in={self.linear.in_features}, out={self.linear.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}"
        )


# ────────────────────────────────────────────────────────────
# Inject LoRA vào DINOv2 transformer blocks
# ────────────────────────────────────────────────────────────
def inject_lora_into_backbone(
    backbone,
    num_last_blocks: int = 2,
    rank: int = 4,
    alpha: float = 4.0,
) -> int:
    """
    Inject LoRA vào các attention QKV projection của N block cuối cùng
    trong DINOv2 ViT backbone.

    Chiến lược:
      - N - num_last_blocks block đầu: đóng băng hoàn toàn (không thay đổi)
      - num_last_blocks block cuối: unfreeze + chèn LoRA vào attn.qkv

    Args:
        backbone:         DINOv2 backbone (backbone.blocks là danh sách ViT blocks)
        num_last_blocks:  Số block cuối cần áp dụng LoRA (thường 2–4)
        rank:             Hạng LoRA (4 hoặc 8)
        alpha:            Scale factor (thường bằng rank)

    Returns:
        num_lora_params:  Số tham số LoRA được thêm vào (trainable)
    """
    total_blocks = len(backbone.blocks)
    lora_start = total_blocks - num_last_blocks
    num_lora_params = 0

    for i, block in enumerate(backbone.blocks):
        if i < lora_start:
            # Block đầu: đóng băng hoàn toàn
            for param in block.parameters():
                param.requires_grad = False
        else:
            # Block cuối: đóng băng trọng số gốc nhưng unfreeze LoRA
            for param in block.parameters():
                param.requires_grad = False

            # Inject LoRA vào attention QKV projection
            if hasattr(block, "attn") and hasattr(block.attn, "qkv"):
                qkv_original = block.attn.qkv
                if isinstance(qkv_original, nn.Linear):
                    lora_qkv = LoRALinear(qkv_original, rank=rank, alpha=alpha)
                    block.attn.qkv = lora_qkv
                    num_lora_params += lora_qkv.lora_A.numel() + lora_qkv.lora_B.numel()

            # Inject LoRA vào projection sau attention (proj)
            if hasattr(block, "attn") and hasattr(block.attn, "proj"):
                proj_original = block.attn.proj
                if isinstance(proj_original, nn.Linear):
                    lora_proj = LoRALinear(proj_original, rank=rank, alpha=alpha)
                    block.attn.proj = lora_proj
                    num_lora_params += lora_proj.lora_A.numel() + lora_proj.lora_B.numel()

    return num_lora_params


def get_lora_parameters(model):
    """
    Trả về chỉ các tham số LoRA (lora_A và lora_B) cần huấn luyện
    từ toàn bộ mô hình.
    """
    lora_params = []
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_params.append(param)
    return lora_params
