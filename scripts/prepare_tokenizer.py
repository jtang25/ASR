"""Train a SentencePiece tokenizer on LibriSpeech transcripts.

Usage:
    python prepare_tokenizer.py --data-root ./data --vocab-size 1023

This produces ./tokenizer/sp_1k.model which is loaded by SentencePieceTokenizer.
The total ASR vocabulary will be vocab_size + 1 (index 0 reserved for <blank>).
"""

from __future__ import annotations

import argparse
import os

import sentencepiece as spm


def extract_transcripts(data_root: str, splits: list[str]) -> list[str]:
    """Walk LibriSpeech directory tree and collect all transcripts."""
    transcripts: list[str] = []
    for split in splits:
        split_dir = os.path.join(data_root, "LibriSpeech", split)
        if not os.path.isdir(split_dir):
            print(f"  [WARN] Directory not found, skipping: {split_dir}")
            continue
        for root, _dirs, files in os.walk(split_dir):
            for fname in files:
                if not fname.endswith(".trans.txt"):
                    continue
                with open(os.path.join(root, fname), "r", encoding="utf-8") as fh:
                    for line in fh:
                        parts = line.strip().split(None, 1)
                        if len(parts) == 2:
                            transcripts.append(parts[1].lower())
    return transcripts


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SentencePiece on LibriSpeech.")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train-clean-100", "train-clean-360", "train-other-500"],
        help="LibriSpeech splits whose transcripts are used for training.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=1023,
        help="SentencePiece vocab size. Total model vocab = this + 1 (for blank).",
    )
    parser.add_argument("--model-prefix", default="./tokenizer/sp_1k")
    parser.add_argument("--model-type", default="unigram", choices=["unigram", "bpe"])
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.model_prefix) or ".", exist_ok=True)

    print(f"Extracting transcripts from {args.splits} ...")
    transcripts = extract_transcripts(args.data_root, args.splits)
    if not transcripts:
        raise RuntimeError(
            "No transcripts found. Make sure the data has been downloaded "
            "and --data-root points to the correct location."
        )
    print(f"  Found {len(transcripts):,} utterances.")

    txt_path = args.model_prefix + "_corpus.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for t in transcripts:
            f.write(t + "\n")
    print(f"  Wrote corpus to {txt_path}")

    print(f"Training SentencePiece ({args.model_type}, vocab={args.vocab_size}) ...")
    spm.SentencePieceTrainer.Train(
        input=txt_path,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=1.0,
        # Disable special tokens we do not need; blank is handled externally
        pad_id=-1,
        eos_id=-1,
        bos_id=-1,
        unk_id=0,
        normalization_rule_name="identity",  # keep text as-is (already lowered)
    )
    print(f"Done. Tokenizer saved to {args.model_prefix}.model")
    print(f"Total ASR vocab (with blank) = {args.vocab_size + 1}")


if __name__ == "__main__":
    main()