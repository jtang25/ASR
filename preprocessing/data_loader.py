"""Data loading and feature extraction for Conformer ASR.

Supports both character-level and SentencePiece tokenization.
SpecAugment follows the paper: F=27, 2 freq masks, 10 time masks with pS=0.05.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torchaudio
from torch.nn.utils.rnn import pad_sequence
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
        self.freq_masks = nn.ModuleList(
            [torchaudio.transforms.FrequencyMasking(self.freq_mask_param) for _ in range(self.num_freq_masks)]
        )
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (T, n_mels)
        Returns:
            (T, n_mels)
        """
        x = mel.transpose(0, 1).unsqueeze(0)  # (1, n_mels, T)

        for freq_mask in self.freq_masks:
            x = freq_mask(x)

        T = x.size(-1)
        self.time_mask.time_mask_param = max(1, int(self.pS * T))
        for _ in range(self.num_time_masks):
            x = self.time_mask(x)

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
        split_root = os.path.join(root, "LibriSpeech", split)
        nested_split_root = os.path.join(split_root, split)

        if os.path.isdir(nested_split_root):
            # Some extractions create .../LibriSpeech/<split>/<split>/...
            # Point LIBRISPEECH to the parent and disable archive prefix.
            ds_root = split_root
            folder_in_archive = ""
        else:
            ds_root = root
            folder_in_archive = "LibriSpeech"

        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=ds_root,
            url=split,
            folder_in_archive=folder_in_archive,
            download=download,
        )
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


class JeromePowellASR(Dataset):
    """Direct loader for Jerome Powell chapter folders at <root>/9999/<chapter>."""

    _SPLIT_CHAPTERS = {
        "train-clean-100": ["001", "002", "003", "004", "005", "006", "007"],
        "dev-clean": ["008"],
        "test-clean": ["008"],
    }

    def __init__(
        self,
        root: str,
        split: str = "train-clean-100",
        n_mels: int = 80,
        augment: bool = False,
        tokenizer: CharTokenizer | SentencePieceTokenizer | None = None,
    ):
        speaker_root = os.path.join(root, "9999")
        if not os.path.isdir(speaker_root):
            raise FileNotFoundError(f"JeromePowell root missing expected directory: {speaker_root}")

        chapters = self._resolve_chapters(speaker_root, split)
        if not chapters:
            raise RuntimeError(
                f"No chapters selected for split={split!r} under {speaker_root}."
            )

        entries: list[tuple[str, str, str]] = []
        for chapter in chapters:
            chapter_dir = os.path.join(speaker_root, chapter)
            if not os.path.isdir(chapter_dir):
                continue

            trans_path = os.path.join(chapter_dir, f"9999-{chapter}.trans.txt")
            if not os.path.isfile(trans_path):
                continue

            with open(trans_path, "r", encoding="utf-8") as f:
                for line in f:
                    row = line.strip().split(None, 1)
                    if len(row) != 2:
                        continue
                    utt_id, transcript = row
                    flac_path = os.path.join(chapter_dir, f"{utt_id}.flac")
                    if not os.path.isfile(flac_path):
                        continue
                    entries.append((utt_id, flac_path, transcript))

        if not entries:
            raise RuntimeError(
                "No usable JeromePowell samples found. "
                f"root={root}, split={split}, chapters={chapters}"
            )

        # Keep deterministic ordering by utterance ID.
        entries.sort(key=lambda x: x[0])
        self.entries = entries
        self.mel_extractor = LogMelSpectrogram(n_mels=n_mels)
        self.augment = SpecAugment() if augment else None
        self.tokenizer = tokenizer or CharTokenizer()

    @classmethod
    def _resolve_chapters(cls, speaker_root: str, split: str) -> list[str]:
        if split in cls._SPLIT_CHAPTERS:
            return list(cls._SPLIT_CHAPTERS[split])

        # Allow explicit chapter-like split names: "001", "8", "jp-003", "jp-8".
        chapter_raw = split
        if split.startswith("jp-"):
            chapter_raw = split.split("-", 1)[1]
        if chapter_raw.isdigit():
            chapter = f"{int(chapter_raw):03d}"
            if os.path.isdir(os.path.join(speaker_root, chapter)):
                return [chapter]

        # Fallback: all available chapter directories.
        chapters = []
        for name in sorted(os.listdir(speaker_root)):
            chapter_dir = os.path.join(speaker_root, name)
            if os.path.isdir(chapter_dir) and name.isdigit() and len(name) == 3:
                chapters.append(name)
        return chapters

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        _utt_id, flac_path, transcript = self.entries[idx]
        waveform, sample_rate = torchaudio.load(flac_path)

        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

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

    mel_padded = pad_sequence(mels, batch_first=True)
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=0)

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
    prefetch_factor: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    download: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    return_sampler: bool = False,
    tokenizer: CharTokenizer | SentencePieceTokenizer | None = None,
) -> DataLoader | tuple[DataLoader, Optional[DistributedSampler]]:
    ls_split_root = os.path.join(root, "LibriSpeech", split)
    ls_nested_split_root = os.path.join(ls_split_root, split)
    has_librispeech_split = os.path.isdir(ls_split_root) or os.path.isdir(ls_nested_split_root)
    has_jp_layout = os.path.isdir(os.path.join(root, "9999"))

    if has_jp_layout and not has_librispeech_split:
        dataset = JeromePowellASR(
            root=root,
            split=split,
            n_mels=n_mels,
            augment=augment,
            tokenizer=tokenizer,
        )
    else:
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
        pin_memory=pin_memory,
        drop_last=is_train,
        persistent_workers=(persistent_workers and num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    if return_sampler:
        return loader, sampler
    return loader
