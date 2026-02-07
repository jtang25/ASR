import torch
import torch.nn as nn
import torch.nn.functional as F

from .conformer import ConformerBlock

class ConvSubsampling(nn.Module):
    """
    Two Conv2d layers with stride 2 each -> 4x time reduction.
    Treats mel spectrogram as a 1-channel image: (B, 1, T, n_mels).
    """

    def __init__(self, d_model, n_mels=80):
        super().__init__()
        self.conv1 = nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1)

        # After two stride-2 convs, freq dim becomes n_mels // 4
        freq_out = self._conv_out_size(self._conv_out_size(n_mels))
        self.linear = nn.Linear(d_model * freq_out, d_model)
        self.dropout = nn.Dropout(0.1)

    @staticmethod
    def _conv_out_size(size, kernel=3, stride=2, pad=1):
        return (size + 2 * pad - kernel) // stride + 1

    def get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Compute output sequence lengths after two stride-2 convolutions."""
        lengths = input_lengths
        for _ in range(2):
            lengths = (lengths + 2 * 1 - 3) // 2 + 1
        return lengths

    def forward(self, x, input_lengths):
        """
        Args:
            x: (B, T, n_mels) mel spectrogram
            input_lengths: (B,) original frame counts
        Returns:
            out: (B, T//4, d_model)
            output_lengths: (B,) subsampled lengths
        """
        x = x.unsqueeze(1)  # (B, 1, T, n_mels)
        x = F.relu(self.conv1(x))  # (B, C, T//2, F//2)
        x = F.relu(self.conv2(x))  # (B, C, T//4, F//4)

        B, C, T, Fr = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * Fr)  # (B, T//4, C * F//4)
        x = self.linear(x)  # (B, T//4, d_model)
        x = self.dropout(x)

        output_lengths = self.get_output_lengths(input_lengths)
        return x, output_lengths


class ConformerASR(nn.Module):
    """
    Full CTC-based ASR model:
        Mel spectrogram -> Conv subsampling (4x) -> N x Conformer blocks -> Linear CTC head
    """

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 12,
        vocab_size: int = 29,  # blank + space + ' + a-z
        ffn_expansion: int = 4,
        ffn_dropout: float = 0.1,
        attn_dropout: float = 0.1,
        conv_kernel_size: int = 31,
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

        self.ctc_head = nn.Linear(d_model, vocab_size)

    def forward(self, mel, mel_lengths):
        """
        Args:
            mel: (B, T, n_mels) log-mel spectrogram
            mel_lengths: (B,) number of valid frames per sample
        Returns:
            log_probs: (B, T', vocab_size) log probabilities
            output_lengths: (B,) output sequence lengths
        """
        x, output_lengths = self.subsampling(mel, mel_lengths)

        # Build padding mask: (B, T') where True = valid, False = padding
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device)[None, :] < output_lengths[:, None]
        mask = mask.float()  # (B, T')

        for layer in self.layers:
            x = layer(x, mask=mask)

        logits = self.ctc_head(x)  # (B, T', vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, output_lengths
