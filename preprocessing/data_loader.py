import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import List, Tuple, Optional
import os


# ---------------------------------------------------------------------------
# Character Tokenizer
# ---------------------------------------------------------------------------
# Vocab: 0=<blank> (CTC), 1=<space>, 2=', 3-28=a-z
# Total: 29 tokens

BLANK_IDX = 0
VOCAB = ["<blank>", " ", "'"] + list("abcdefghijklmnopqrstuvwxyz")
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}
IDX_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)


def text_to_tokens(text: str) -> List[int]:
    """Convert transcript string to list of token indices."""
    text = text.lower().strip()
    tokens = []
    for ch in text:
        if ch in CHAR_TO_IDX:
            tokens.append(CHAR_TO_IDX[ch])
        # Skip unknown characters (punctuation, etc.)
    return tokens


def tokens_to_text(tokens: List[int]) -> str:
    """Convert token indices back to string."""
    return "".join(IDX_TO_CHAR.get(t, "") for t in tokens if t != BLANK_IDX)


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

class LogMelSpectrogram(nn.Module):
    """Compute 80-dim log-mel spectrogram from raw waveform."""

    def __init__(self, sample_rate=16000, n_mels=80, n_fft=400, hop_length=160):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,          # 25ms window at 16kHz
            hop_length=hop_length, # 10ms hop at 16kHz
            n_mels=n_mels,
            norm="slaney",
            mel_scale="slaney",
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (num_samples,) raw audio
        Returns:
            features: (T, n_mels) log-mel spectrogram
        """
        mel = self.mel_spec(waveform)  # (n_mels, T)
        log_mel = torch.log(mel + 1e-9)
        return log_mel.transpose(0, 1)  # (T, n_mels)


# ---------------------------------------------------------------------------
# SpecAugment
# ---------------------------------------------------------------------------

class SpecAugment(nn.Module):
    """SpecAugment: frequency and time masking for data augmentation."""

    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ):
        super().__init__()
        self.freq_masks = nn.ModuleList(
            [torchaudio.transforms.FrequencyMasking(freq_mask_param) for _ in range(num_freq_masks)]
        )
        self.time_masks = nn.ModuleList(
            [torchaudio.transforms.TimeMasking(time_mask_param) for _ in range(num_time_masks)]
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (T, n_mels) or (n_mels, T) spectrogram
        Returns:
            augmented spectrogram, same shape
        """
        # torchaudio masks expect (channel, freq, time) or (freq, time)
        x = mel.transpose(0, 1).unsqueeze(0)  # (1, n_mels, T)
        for mask in self.freq_masks:
            x = mask(x)
        for mask in self.time_masks:
            x = mask(x)
        return x.squeeze(0).transpose(0, 1)  # (T, n_mels)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LibriSpeechASR(Dataset):
    """Wraps torchaudio.datasets.LIBRISPEECH with feature extraction."""

    def __init__(
        self,
        root: str,
        split: str = "train-clean-100",
        n_mels: int = 80,
        augment: bool = False,
        download: bool = True,
    ):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=root, url=split, download=download
        )
        self.mel_extractor = LogMelSpectrogram(n_mels=n_mels)
        self.augment = SpecAugment() if augment else None

        # For global CMVN (computed lazily or skipped — per-utterance norm used here)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        waveform, sample_rate, transcript, _, _, _ = self.dataset[idx]

        # Resample if needed (LibriSpeech should already be 16kHz)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        # Extract features
        mel = self.mel_extractor(waveform.squeeze(0))  # (T, n_mels)

        # Per-utterance normalization (zero mean, unit variance)
        mel = (mel - mel.mean()) / (mel.std() + 1e-9)

        # SpecAugment (training only)
        if self.augment is not None:
            mel = self.augment(mel)

        # Tokenize transcript
        tokens = torch.tensor(text_to_tokens(transcript), dtype=torch.long)

        mel_length = mel.size(0)
        token_length = tokens.size(0)

        return mel, tokens, mel_length, token_length


# ---------------------------------------------------------------------------
# Collate Function
# ---------------------------------------------------------------------------

def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad mel spectrograms and token sequences to batch max lengths.

    Returns:
        mel_padded: (B, T_max, n_mels)
        tokens_padded: (B, S_max)
        mel_lengths: (B,)
        token_lengths: (B,)
    """
    mels, tokens, mel_lengths, token_lengths = zip(*batch)

    mel_lengths = torch.tensor(mel_lengths, dtype=torch.long)
    token_lengths = torch.tensor(token_lengths, dtype=torch.long)

    # Pad mel spectrograms
    max_mel_len = mel_lengths.max().item()
    n_mels = mels[0].size(1)
    mel_padded = torch.zeros(len(mels), max_mel_len, n_mels)
    for i, mel in enumerate(mels):
        mel_padded[i, : mel.size(0)] = mel

    # Pad token sequences
    max_token_len = token_lengths.max().item()
    tokens_padded = torch.zeros(len(tokens), max_token_len, dtype=torch.long)
    for i, tok in enumerate(tokens):
        tokens_padded[i, : tok.size(0)] = tok

    return mel_padded, tokens_padded, mel_lengths, token_lengths


# ---------------------------------------------------------------------------
# DataLoader Helpers
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
) -> DataLoader | tuple[DataLoader, Optional[DistributedSampler]]:
    dataset = LibriSpeechASR(
        root=root, split=split, n_mels=n_mels, augment=augment, download=download
    )
    shuffle = "train" in split
    sampler: Optional[DistributedSampler] = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=shuffle,
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(shuffle and sampler is None),
    )
    if return_sampler:
        return loader, sampler
    return loader
