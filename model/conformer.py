"""Conformer encoder building blocks.

Implements the Conformer block from:
  'Conformer: Convolution-augmented Transformer for Speech Recognition'
  (Gulati et al., 2020)

Changes from a vanilla Transformer block:
  - Macaron-style half-step FFN pair sandwiching attention + convolution
  - Rotary positional embeddings (RoPE) in self-attention
  - Depthwise separable convolution with GLU gating and BatchNorm
  - RMSNorm instead of LayerNorm
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# ---------------------------------------------------------------------------
# Rotary Positional Embedding
# ---------------------------------------------------------------------------


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_len: int = 2048, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(
                f"RoPE requires even head_dim, got {head_dim}. "
                "Choose d_model/num_heads with an even quotient."
            )
        self.head_dim = int(head_dim)
        self.max_len = int(max_len)

        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        positions = torch.arange(self.max_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # (max_len, head_dim/2)
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    def forward(self, seq_len: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_len:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_len={self.max_len}. "
                "Increase --max-len or reduce input length."
            )
        cos = self.cos_cached[:seq_len].to(device=device, dtype=dtype)[None, None, :, :]
        sin = self.sin_cached[:seq_len].to(device=device, dtype=dtype)[None, None, :, :]
        return cos, sin


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_even * sin + x_odd * cos
    out = torch.empty_like(x)
    out[..., 0::2] = x_rot_even
    out[..., 1::2] = x_rot_odd
    return out


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention with RoPE
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, max_len: int = 2048):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = nn.Dropout(dropout)
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"RoPE requires even head_dim, got d_model/num_heads={d_model}/{num_heads}={self.head_dim}."
            )

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_len=max_len)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.size()
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, L, Hd = x.size()
        return x.transpose(1, 2).contiguous().view(B, L, H * Hd)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None) -> torch.Tensor:
        B, L, _ = q.size()

        Q = self.split_heads(self.W_q(q))
        K = self.split_heads(self.W_k(k))
        V = self.split_heads(self.W_v(v))
        cos, sin = self.rope(L, device=Q.device, dtype=Q.dtype)
        Q = _apply_rope(Q, cos, sin)
        K = _apply_rope(K, cos, sin)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask[:, None, None, :]      # (B, L) -> (B, 1, 1, L)
            elif mask.dim() == 3:
                mask = mask[:, None, :, :]          # (B, L, L) -> (B, 1, L, L)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = self.combine_heads(out)
        out = self.W_o(out)
        out = self.dropout(out)
        return out


# ---------------------------------------------------------------------------
# Feed-Forward Network (Macaron half-step)
# ---------------------------------------------------------------------------

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1, residual_scale: float = 0.5):
        super().__init__()
        d_ff = d_model * expansion
        self.ln = RMSNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop2 = nn.Dropout(dropout)
        self.residual_scale = residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop1(y)
        y = self.fc2(y)
        y = self.drop2(y)
        return x + self.residual_scale * y


# ---------------------------------------------------------------------------
# Convolution Module
# ---------------------------------------------------------------------------

class ConvolutionNetwork(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 32, dropout: float = 0.1, causal: bool = False):
        super().__init__()
        self.ln = RMSNorm(d_model)
        self.pw_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        # Manual padding to avoid CUDA issues with padding="same"
        self.kernel_size = int(kernel_size)
        self.causal = bool(causal)
        if self.causal:
            self.pad_left = self.kernel_size - 1
            self.pad_right = 0
        else:
            self.pad_left = (self.kernel_size - 1) // 2
            self.pad_right = self.kernel_size // 2
        self.dw_conv = nn.Conv1d(
            d_model, d_model, kernel_size, padding=0, groups=d_model
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.pw_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln(x)
        y = y.transpose(1, 2)
        y = self.pw_conv1(y)
        y = self.glu(y)
        if self.pad_left or self.pad_right:
            y = F.pad(y, (self.pad_left, self.pad_right))
        y = self.dw_conv(y)
        y = self.bn(y)
        y = self.act(y)
        y = self.pw_conv2(y)
        y = self.drop(y)
        y = y.transpose(1, 2)
        return x + y


# ---------------------------------------------------------------------------
# Conformer Block
# ---------------------------------------------------------------------------

class ConformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_expansion: int = 4,
        ffn_dropout: float = 0.1,
        attn_dropout: float = 0.1,
        conv_kernel_size: int = 32,
        conv_dropout: float = 0.1,
        max_len: int = 2048,
        causal_conv: bool = False,
    ):
        super().__init__()
        self.ffn1 = FeedForwardNetwork(d_model, ffn_expansion, ffn_dropout, residual_scale=0.5)
        self.mhsa_ln = RMSNorm(d_model)
        self.mhsa = MultiHeadSelfAttention(d_model, num_heads, attn_dropout, max_len)
        self.conv = ConvolutionNetwork(d_model, conv_kernel_size, conv_dropout, causal=causal_conv)
        self.ffn2 = FeedForwardNetwork(d_model, ffn_expansion, ffn_dropout, residual_scale=0.5)
        self.final_ln = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        x = self.ffn1(x)
        x_norm = self.mhsa_ln(x)
        x = x + self.mhsa(x_norm, x_norm, x_norm, mask=mask)
        x = self.conv(x)
        x = self.ffn2(x)
        return self.final_ln(x)
