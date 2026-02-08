"""Data loading and feature extraction for Conformer ASR.

Supports both character-level and SentencePiece tokenization.
SpecAugment follows the paper: F=27, 2 freq masks, 10 time masks with pS=0.05.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from tokenizer import CharTokenizer, SentencePieceTokenizer

# Re-export for backward compatibility
from tokenizer import BLANK_IDX, CHAR_VOCAB as VOCAB, CHAR_VOCAB_SIZE as VOCAB_SIZE


def text_to_tokens(text: str) -> List[int]:
    """Character-level tokenization (backward compat)."""
    return CharTokenizer().encode(text)


def tokens_to_text(tokens: List[int]) -> str:
    """Character-level decoding (backward compat)."""
    return CharTokenizer().decode(tokens)


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

class LogMelSpectrogram(nn.Module):
    """80-dim log-mel spectrogram (25 ms window, 10 ms hop)."""

    def __init__(self, sample_rate: int = 16000, n_mels: int = 80, n_fft: int = 400, hop_length: int = 160):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            norm="slaney",
            mel_scale="slaney",
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.mel_spec(waveform)          # (n_mels, T)
        log_mel = torch.log(mel + 1e-9)
        return log_mel.transpose(0, 1)         # (T, n_mels)


# ---------------------------------------------------------------------------
# SpecAugment (paper spec: F=27, 10 time masks with pS=0.05)
# ---------------------------------------------------------------------------

class SpecAugment(nn.Module):
    def __init__(
        self,
        freq_mask_param: int = 27,
        num_freq_masks: int = 2,
        pS: float = 0.05,
        num_time_masks: int = 10,
    ):
        super().__init__()
        self.freq_mask_param = int(freq_mask_param)
        self.num_freq_masks = int(num_freq_masks)
        self.pS = float(pS)
        self.num_time_masks = int(num_time_masks)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (T, n_mels)
        Returns:
            (T, n_mels)
        """
        x = mel.transpose(0, 1).unsqueeze(0)  # (1, n_mels, T)

        for _ in range(self.num_freq_masks):
            x = torchaudio.transforms.FrequencyMasking(self.freq_mask_param)(x)

        T = x.size(-1)
        time_mask_param = max(1, int(self.pS * T))
        for _ in range(self.num_time_masks):
            x = torchaudio.transforms.TimeMasking(time_mask_param)(x)

        return x.squeeze(0).transpose(0, 1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LibriSpeechASR(Dataset):
    """Wraps torchaudio.datasets.LIBRISPEECH with feature extraction.

    Accepts either a CharTokenizer or SentencePieceTokenizer.
    """

    def __init__(
        self,
        root: str,
        split: str = "train-clean-100",
        n_mels: int = 80,
        augment: bool = False,
        download: bool = True,
        tokenizer: CharTokenizer | SentencePieceTokenizer | None = None,
    ):
        self.dataset = torchaudio.datasets.LIBRISPEECH(root=root, url=split, download=download)
        self.mel_extractor = LogMelSpectrogram(n_mels=n_mels)
        self.augment = SpecAugment() if augment else None
        self.tokenizer = tokenizer or CharTokenizer()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        waveform, sample_rate, transcript, _, _, _ = self.dataset[idx]

        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        mel = self.mel_extractor(waveform.squeeze(0))  # (T, n_mels)

        # Per-feature-bin normalization (zero mean, unit variance over time)
        mel = (mel - mel.mean(dim=0, keepdim=True)) / (mel.std(dim=0, keepdim=True) + 1e-9)

        if self.augment is not None:
            mel = self.augment(mel)

        tokens = torch.tensor(self.tokenizer.encode(transcript), dtype=torch.long)

        return mel, tokens, int(mel.size(0)), int(tokens.size(0))


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mels, tokens, mel_lengths, token_lengths = zip(*batch)

    mel_lengths_t = torch.tensor(mel_lengths, dtype=torch.long)
    token_lengths_t = torch.tensor(token_lengths, dtype=torch.long)

    max_mel_len = int(mel_lengths_t.max().item())
    n_mels = int(mels[0].size(1))
    mel_padded = torch.zeros(len(mels), max_mel_len, n_mels, dtype=mels[0].dtype)
    for i, mel in enumerate(mels):
        mel_padded[i, : mel.size(0)] = mel

    max_token_len = int(token_lengths_t.max().item())
    tokens_padded = torch.zeros(len(tokens), max_token_len, dtype=torch.long)
    for i, tok in enumerate(tokens):
        tokens_padded[i, : tok.size(0)] = tok

    return mel_padded, tokens_padded, mel_lengths_t, token_lengths_t


# ---------------------------------------------------------------------------
# DataLoader helper
# ---------------------------------------------------------------------------

def get_dataloader(
    root: str,
    split: str,
    batch_size: int,
    n_mels: int = 80,
    augment: bool = False,
    num_workers: int = 4,
    download: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    return_sampler: bool = False,
    tokenizer: CharTokenizer | SentencePieceTokenizer | None = None,
) -> DataLoader | tuple[DataLoader, Optional[DistributedSampler]]:

    dataset = LibriSpeechASR(
        root=root,
        split=split,
        n_mels=n_mels,
        augment=augment,
        download=download,
        tokenizer=tokenizer,
    )

    is_train = "train" in split
    sampler: Optional[DistributedSampler] = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=is_train,
            drop_last=is_train,
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(is_train and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=is_train,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    if return_sampler:
        return loader, sampler
    return loader
