"""SentencePiece tokenizer wrapper for Conformer ASR.

Vocabulary layout:
    Index 0        = <blank>  (RNN-T / CTC blank)
    Index 1..V     = SentencePiece tokens (shifted by +1 from raw SP IDs)

Train a tokenizer with prepare_tokenizer.py before using this module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import sentencepiece as spm

BLANK_IDX = 0


class SentencePieceTokenizer:
    """Thin wrapper that reserves index 0 for <blank>."""

    def __init__(self, model_path: str):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"SentencePiece model not found: {model_path}")
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self._blank_idx = BLANK_IDX

    # ----- properties -----

    @property
    def blank_idx(self) -> int:
        return self._blank_idx

    @property
    def sp_vocab_size(self) -> int:
        """Number of tokens in the raw SentencePiece model."""
        return self.sp.GetPieceSize()

    @property
    def vocab_size(self) -> int:
        """Total vocabulary including <blank> at index 0."""
        return self.sp.GetPieceSize() + 1

    # ----- encode / decode -----

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs (blank-aware, 1-indexed SP tokens)."""
        ids = self.sp.EncodeAsIds(text.strip().lower())
        return [i + 1 for i in ids]  # shift so blank stays at 0

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text, skipping blanks."""
        sp_ids = [i - 1 for i in ids if i > self._blank_idx]
        return self.sp.DecodeIds(sp_ids)

    # ----- vocab inspection -----

    def id_to_piece(self, token_id: int) -> str:
        if token_id == self._blank_idx:
            return "<blank>"
        return self.sp.IdToPiece(token_id - 1)

    def piece_to_id(self, piece: str) -> int:
        if piece == "<blank>":
            return self._blank_idx
        return self.sp.PieceToId(piece) + 1


# ---------------------------------------------------------------------------
# Character-level tokenizer (kept for backward compat / CTC experiments)
# ---------------------------------------------------------------------------

CHAR_VOCAB = ["<blank>", " ", "'"] + list("abcdefghijklmnopqrstuvwxyz")
CHAR_TO_IDX = {c: i for i, c in enumerate(CHAR_VOCAB)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CHAR_VOCAB)}
CHAR_VOCAB_SIZE = len(CHAR_VOCAB)


class CharTokenizer:
    """Simple character tokenizer: blank + space + apostrophe + a-z = 29 tokens."""

    def __init__(self):
        self._blank_idx = BLANK_IDX

    @property
    def blank_idx(self) -> int:
        return self._blank_idx

    @property
    def vocab_size(self) -> int:
        return CHAR_VOCAB_SIZE

    def encode(self, text: str) -> List[int]:
        text = text.lower().strip()
        return [CHAR_TO_IDX[ch] for ch in text if ch in CHAR_TO_IDX]

    def decode(self, ids: List[int]) -> str:
        return "".join(IDX_TO_CHAR.get(t, "") for t in ids if t != self._blank_idx)

    def id_to_piece(self, token_id: int) -> str:
        return IDX_TO_CHAR.get(token_id, "")

    def piece_to_id(self, piece: str) -> int:
        return CHAR_TO_IDX.get(piece, 0)    