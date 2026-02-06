import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RelativeSinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        positions = torch.arange(-(max_len - 1), max_len).float()
        dim_indices = torch.arange(0, d_model, 2).float()
        frequencies = 1.0 / (10000 ** (dim_indices / d_model))

        angles = positions.unsqueeze(1) * frequencies.unsqueeze(0)
        pe = torch.zeros(2 * max_len - 1, d_model)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)

        self.register_buffer("pe", pe)

    def forward(self, seq_len: int) -> torch.Tensor:
        center = self.max_len - 1
        start = center - (seq_len - 1)
        end = center + seq_len
        return self.pe[start:end]  # (2L-1, d_model)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0, max_len: int = 2048):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = nn.Dropout(dropout)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_r = nn.Linear(d_model, d_model, bias=False)

        self.rel_pe = RelativeSinusoidalPE(d_model, max_len=max_len)

        # Global content and position biases (Transformer-XL style)
        self.u = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.v = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.u.unsqueeze(0))
        nn.init.xavier_uniform_(self.v.unsqueeze(0))

    def split_heads(self, x):
        B, L, D = x.size()
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x):
        B, H, L, Hd = x.size()
        return x.transpose(1, 2).contiguous().view(B, L, H * Hd)

    def _relative_gather(self, QR, L):
        """
        Convert raw position scores (B, H, L, 2L-1) to properly indexed
        relative position scores (B, H, L, L).

        For query position i and key position j, the relative position
        embedding index is (i - j + L - 1).
        """
        B, H = QR.shape[:2]
        i_idx = torch.arange(L, device=QR.device).unsqueeze(1)  # (L, 1)
        j_idx = torch.arange(L, device=QR.device).unsqueeze(0)  # (1, L)
        rel_idx = (i_idx - j_idx + (L - 1)).long()  # (L, L)
        rel_idx = rel_idx[None, None].expand(B, H, L, L)
        return QR.gather(dim=-1, index=rel_idx)

    def forward(self, q, k, v, mask=None):
        B, L, _ = q.size()

        Q = self.split_heads(self.W_q(q))  # (B, H, L, d)
        K = self.split_heads(self.W_k(k))  # (B, H, L, d)
        V = self.split_heads(self.W_v(v))  # (B, H, L, d)

        # Relative position embeddings
        R = self.rel_pe(L)  # (2L-1, d_model)
        R = self.W_r(R)  # (2L-1, d_model)
        R = R.view(-1, self.num_heads, self.head_dim).permute(1, 0, 2)  # (H, 2L-1, d)

        # Content-to-content + global content bias: (Q + u) @ K^T
        content_score = torch.matmul(
            Q + self.u[None, :, None, :], K.transpose(-2, -1)
        )  # (B, H, L, L)

        # Content-to-position + global position bias: (Q + v) @ R^T
        QR = torch.matmul(
            Q + self.v[None, :, None, :], R.transpose(-2, -1)
        )  # (B, H, L, 2L-1)
        position_score = self._relative_gather(QR, L)  # (B, H, L, L)

        # Combine and scale
        scores = (content_score + position_score) / math.sqrt(self.head_dim)

        # Apply mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask[:, None, None, :]  # (B, L) -> (B, 1, 1, L)
            elif mask.dim() == 3:
                mask = mask[:, None, :, :]  # (B, L, L) -> (B, 1, L, L)
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = self.combine_heads(out)
        out = self.W_o(out)
        out = self.dropout(out)
        return out


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, expansion=4, dropout=0.1, residual_scale=0.5):
        super().__init__()
        d_ff = d_model * expansion
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop2 = nn.Dropout(dropout)
        self.residual_scale = residual_scale

    def forward(self, x):
        y = self.ln(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop1(y)
        y = self.fc2(y)
        y = self.drop2(y)
        return x + self.residual_scale * y


class ConvolutionNetwork(nn.Module):
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.pw_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        pad = (kernel_size - 1) // 2
        self.dw_conv = nn.Conv1d(
            d_model, d_model, kernel_size, padding=pad, groups=d_model
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
        y = self.dw_conv(y)
        y = self.bn(y)
        y = self.act(y)
        y = self.pw_conv2(y)
        y = self.drop(y)
        y = y.transpose(1, 2)
        return x + y


class ConformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        ffn_expansion=4,
        ffn_dropout=0.1,
        attn_dropout=0.1,
        conv_kernel_size=31,
        conv_dropout=0.1,
        max_len=2048,
    ):
        super().__init__()
        self.ffn1 = FeedForwardNetwork(d_model, ffn_expansion, ffn_dropout, 0.5)
        self.mhsa_ln = nn.LayerNorm(d_model)
        self.mhsa = MultiHeadSelfAttention(d_model, num_heads, attn_dropout, max_len)
        self.conv = ConvolutionNetwork(d_model, conv_kernel_size, conv_dropout)
        self.ffn2 = FeedForwardNetwork(d_model, ffn_expansion, ffn_dropout, 0.5)
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        x = self.ffn1(x)
        x_norm = self.mhsa_ln(x)
        x = x + self.mhsa(x_norm, x_norm, x_norm, mask=mask)
        x = self.conv(x)
        x = self.ffn2(x)
        return self.final_ln(x)