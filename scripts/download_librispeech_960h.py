#!/usr/bin/env python3
"""Download LibriSpeech splits needed for 960h Conformer training."""

from __future__ import annotations

import argparse

import torchaudio


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LibriSpeech 960h splits via torchaudio.")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train-clean-100", "train-clean-360", "train-other-500", "dev-other"],
        help="LibriSpeech split names accepted by torchaudio.datasets.LIBRISPEECH(url=...).",
    )
    args = parser.parse_args()

    for split in args.splits:
        print(f"Downloading {split} into {args.data_root} ...")
        torchaudio.datasets.LIBRISPEECH(root=args.data_root, url=split, download=True)

    print("Done.")
    print(f"Expected dataset root: {args.data_root}/LibriSpeech")


if __name__ == "__main__":
    main()
