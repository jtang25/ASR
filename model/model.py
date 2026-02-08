"""Conformer ASR models: CTC and RNN-Transducer variants.

Architecture from:
  'Conformer: Convolution-augmented Transformer for Speech Recognition'
  (Gulati et al., 2020)

Conformer(L) config: 17 encoder layers, d_model=512, 8 heads, kernel=32,
                      1-layer LSTM decoder (dim=640), joint_dim=640, ~118M params.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conformer import ConformerBlock


# ---------------------------------------------------------------------------
# Conv Subsampling (shared by both CTC and Transducer)
# ---------------------------------------------------------------------------

class ConvSubsampling(nn.Module):
    """Two Conv2d layers with stride 2 each -> 4x time reduction."""

    def __init__(self, d_model: int, n_mels: int = 80):
        super().__init__()
        self.conv1 = nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1)

        freq_out = self._conv_out_size(self._conv_out_size(n_mels))
        self.linear = nn.Linear(d_model * freq_out, d_model)
        self.dropout = nn.Dropout(0.1)

    @staticmethod
    def _conv_out_size(size: int, kernel: int = 3, stride: int = 2, pad: int = 1) -> int:
        return (size + 2 * pad - kernel) // stride + 1

    def get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        lengths = input_lengths
        for _ in range(2):
            lengths = (lengths + 2 * 1 - 3) // 2 + 1
        return lengths

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        B, C, T, Fr = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * Fr)
        x = self.linear(x)
        x = self.dropout(x)

        output_lengths = self.get_output_lengths(input_lengths)
        return x, output_lengths


# ---------------------------------------------------------------------------
# Conformer Encoder (shared)
# ---------------------------------------------------------------------------

class ConformerEncoder(nn.Module):
    """Conv subsampling followed by a stack of Conformer blocks."""

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 17,
        ffn_expansion: int = 4,
        ffn_dropout: float = 0.1,
        attn_dropout: float = 0.1,
        conv_kernel_size: int = 32,
        conv_dropout: float = 0.1,
        max_len: int = 2048,
    ):
        super().__init__()
        self.subsampling = ConvSubsampling(d_model, n_mels)
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_expansion=ffn_expansion,
                    ffn_dropout=ffn_dropout,
                    attn_dropout=attn_dropout,
                    conv_kernel_size=conv_kernel_size,
                    conv_dropout=conv_dropout,
                    max_len=max_len,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, mel: torch.Tensor, mel_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, output_lengths = self.subsampling(mel, mel_lengths)
        T = x.size(1)
        mask = torch.arange(T, device=x.device)[None, :] < output_lengths[:, None]
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x, output_lengths

    def get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        return self.subsampling.get_output_lengths(input_lengths)


# ---------------------------------------------------------------------------
# CTC Model (encoder-only)
# ---------------------------------------------------------------------------

class ConformerASR(nn.Module):
    """Conformer encoder + linear CTC head."""

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 17,
        vocab_size: int = 29,
        ffn_expansion: int = 4,
        ffn_dropout: float = 0.1,
        attn_dropout: float = 0.1,
        conv_kernel_size: int = 32,
        conv_dropout: float = 0.1,
        max_len: int = 2048,
    ):
        super().__init__()
        self.encoder = ConformerEncoder(
            n_mels=n_mels,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_expansion=ffn_expansion,
            ffn_dropout=ffn_dropout,
            attn_dropout=attn_dropout,
            conv_kernel_size=conv_kernel_size,
            conv_dropout=conv_dropout,
            max_len=max_len,
        )
        self.ctc_head = nn.Linear(d_model, vocab_size)

    # Convenience alias so checkpoint-loading code can access subsampling
    @property
    def subsampling(self):
        return self.encoder.subsampling

    def forward(self, mel: torch.Tensor, mel_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, output_lengths = self.encoder(mel, mel_lengths)
        logits = self.ctc_head(x)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, output_lengths


# ---------------------------------------------------------------------------
# Prediction Network (RNN-T decoder)
# ---------------------------------------------------------------------------

class PredictionNetwork(nn.Module):
    """Single-layer LSTM prediction network for RNN-T.

    Matches paper config: 1 LSTM layer, hidden_dim=640.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden_dim: int = 640, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, y: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None):
        """
        Args:
            y: (B, U) token indices
            state: optional (h, c) LSTM state, each (num_layers, B, hidden_dim)
        Returns:
            out: (B, U, hidden_dim)
            state: (h, c)
        """
        embedded = self.embedding(y)
        out, state = self.lstm(embedded, state)
        return out, state


# ---------------------------------------------------------------------------
# Joint Network
# ---------------------------------------------------------------------------

class JointNetwork(nn.Module):
    """Combines encoder and predictor outputs to produce logits.

    Training:  enc (B,T,D_e), pred (B,U+1,D_p) -> (B,T,U+1,V)
    Decoding:  enc (B,D_e),   pred (B,D_p)      -> (B,V)
    """

    def __init__(self, encoder_dim: int, predictor_dim: int, joint_dim: int, vocab_size: int):
        super().__init__()
        self.enc_proj = nn.Linear(encoder_dim, joint_dim)
        self.pred_proj = nn.Linear(predictor_dim, joint_dim)
        self.out_proj = nn.Linear(joint_dim, vocab_size)

    def forward(self, enc_out: torch.Tensor, pred_out: torch.Tensor) -> torch.Tensor:
        """Full joint for training: broadcast over T and U dimensions."""
        enc = self.enc_proj(enc_out).unsqueeze(2)   # (B, T, 1, J)
        pred = self.pred_proj(pred_out).unsqueeze(1)  # (B, 1, U+1, J)
        return self.out_proj(torch.tanh(enc + pred))  # (B, T, U+1, V)

    def forward_step(self, enc_step: torch.Tensor, pred_step: torch.Tensor) -> torch.Tensor:
        """Single-step joint for decoding.

        Args:
            enc_step: (B, D_enc) encoder output at one time step
            pred_step: (B, D_pred) predictor output at one label step
        Returns:
            logits: (B, V)
        """
        enc = self.enc_proj(enc_step)
        pred = self.pred_proj(pred_step)
        return self.out_proj(torch.tanh(enc + pred))


# ---------------------------------------------------------------------------
# Conformer Transducer
# ---------------------------------------------------------------------------

class ConformerTransducer(nn.Module):
    """Full Conformer Transducer as described in the paper.

    Default hyperparams match Conformer(L): 118.8M params.
    """

    def __init__(
        self,
        n_mels: int = 80,
        encoder_dim: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 17,
        vocab_size: int = 1024,
        ffn_expansion: int = 4,
        ffn_dropout: float = 0.1,
        attn_dropout: float = 0.1,
        conv_kernel_size: int = 32,
        conv_dropout: float = 0.1,
        max_len: int = 2048,
        pred_embed_dim: int = 256,
        pred_hidden_dim: int = 640,
        pred_num_layers: int = 1,
        joint_dim: int = 640,
        blank_idx: int = 0,
    ):
        super().__init__()
        self.blank_idx = blank_idx
        self.vocab_size = vocab_size

        self.encoder = ConformerEncoder(
            n_mels=n_mels,
            d_model=encoder_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            ffn_expansion=ffn_expansion,
            ffn_dropout=ffn_dropout,
            attn_dropout=attn_dropout,
            conv_kernel_size=conv_kernel_size,
            conv_dropout=conv_dropout,
            max_len=max_len,
        )
        self.predictor = PredictionNetwork(
            vocab_size=vocab_size,
            embed_dim=pred_embed_dim,
            hidden_dim=pred_hidden_dim,
            num_layers=pred_num_layers,
        )
        self.joint = JointNetwork(
            encoder_dim=encoder_dim,
            predictor_dim=pred_hidden_dim,
            joint_dim=joint_dim,
            vocab_size=vocab_size,
        )

    @property
    def subsampling(self):
        return self.encoder.subsampling

    def forward(
        self,
        mel: torch.Tensor,
        mel_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mel: (B, T, n_mels)
            mel_lengths: (B,)
            targets: (B, U_max) target token IDs (no blank prepended)
            target_lengths: (B,)
        Returns:
            logits: (B, T', U_max+1, vocab_size) raw logits for rnnt_loss
            encoder_lengths: (B,)
        """
        enc_out, enc_lengths = self.encoder(mel, mel_lengths)

        # Prepend blank/SOS to targets for the prediction network
        B = targets.size(0)
        sos = targets.new_full((B, 1), self.blank_idx)
        pred_input = torch.cat([sos, targets], dim=1)  # (B, U_max + 1)
        pred_out, _ = self.predictor(pred_input)        # (B, U_max + 1, D_pred)

        logits = self.joint(enc_out, pred_out)  # (B, T', U_max + 1, V)
        return logits, enc_lengths

    def encode(self, mel: torch.Tensor, mel_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode only (for decoding)."""
        return self.encoder(mel, mel_lengths)