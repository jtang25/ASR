#!/usr/bin/env python3
"""Batch-transcribe Jerome Powell FLAC clips with Faster-Whisper.

Outputs LibriSpeech-style chapter transcript files:
  <chapter_dir>/9999-<chapter>.trans.txt
with lines:
  <utt_id> <UPPERCASE_NORMALIZED_TEXT>
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path
from typing import Callable

import ctranslate2
from faster_whisper import WhisperModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-root",
        default="dataset/JeromePowell/9999",
        help="Root containing chapter directories (001..008).",
    )
    p.add_argument(
        "--chapters",
        nargs="*",
        default=None,
        help="Optional chapter ids to process (e.g. 001 002). Defaults to all numeric dirs.",
    )
    p.add_argument("--model", default="large-v3-turbo", help="Whisper model name/path.")
    p.add_argument("--model-cache-dir", default=None, help="Optional model cache/download directory.")
    p.add_argument("--language", default="en", help="Whisper language code.")
    p.add_argument("--beam-size", type=int, default=5, help="Beam size for Whisper decoding.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument(
        "--compute-type",
        default="auto",
        choices=["auto", "float16", "float32", "int8", "int8_float16", "int8_float32"],
    )
    p.add_argument("--condition-on-previous-text", action="store_true")
    p.add_argument("--vad-filter", action="store_true")
    p.add_argument("--log-every", type=int, default=50, help="Progress log interval per chapter.")
    p.add_argument(
        "--max-files-per-chapter",
        type=int,
        default=0,
        help="Optional cap for quick testing (0 = all files).",
    )
    return p.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"


def resolve_compute_type(compute_type: str, device: str) -> str:
    if compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"


def load_normalizer() -> Callable[[str], str]:
    script_path = Path(__file__).resolve().with_name("convert_to_librispeech.py")
    spec = importlib.util.spec_from_file_location("convert_to_librispeech", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load normalizer from: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    normalize_text = getattr(module, "normalize_text", None)
    if normalize_text is None:
        raise RuntimeError(f"normalize_text not found in: {script_path}")
    return normalize_text


def normalize_for_librispeech(text: str, normalizer: Callable[[str], str]) -> str:
    normalized = normalizer(text).upper()
    if normalized:
        return normalized
    fallback = re.sub(r"[^A-Za-z\s]", " ", text).upper()
    fallback = re.sub(r"\s+", " ", fallback).strip()
    return fallback or "UNTRANSCRIBED"


def iter_chapters(root: Path, selected: list[str] | None) -> list[Path]:
    if selected:
        chapter_ids = [c.zfill(3) for c in selected]
        return [root / c for c in chapter_ids]
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit())


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    chapters = iter_chapters(dataset_root, args.chapters)
    if not chapters:
        raise RuntimeError(f"No chapter directories found under: {dataset_root}")

    device = resolve_device(args.device)
    compute_type = resolve_compute_type(args.compute_type, device)
    normalizer = load_normalizer()

    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Compute type: {compute_type}")
    print(f"Dataset root: {dataset_root}")
    print(f"Chapters: {', '.join(p.name for p in chapters)}")

    model = WhisperModel(
        args.model,
        device=device,
        compute_type=compute_type,
        download_root=args.model_cache_dir,
    )

    total_files = 0
    total_failures = 0

    for chapter_dir in chapters:
        if not chapter_dir.is_dir():
            print(f"[WARN] Missing chapter directory: {chapter_dir}", file=sys.stderr)
            continue

        chapter = chapter_dir.name
        flacs = sorted(chapter_dir.glob("*.flac"))
        if not flacs:
            print(f"[WARN] No FLAC files found in: {chapter_dir}", file=sys.stderr)
            continue
        if args.max_files_per_chapter > 0:
            flacs = flacs[: args.max_files_per_chapter]

        out_path = chapter_dir / f"9999-{chapter}.trans.txt"
        lines: list[str] = []
        chapter_failures = 0

        print(f"[INFO] Chapter {chapter}: transcribing {len(flacs)} files...")
        for idx, flac_path in enumerate(flacs, start=1):
            try:
                segments, _info = model.transcribe(
                    str(flac_path),
                    language=args.language,
                    task="transcribe",
                    beam_size=args.beam_size,
                    condition_on_previous_text=args.condition_on_previous_text,
                    vad_filter=args.vad_filter,
                    temperature=0.0,
                    word_timestamps=False,
                    without_timestamps=True,
                )
                raw_text = " ".join(
                    seg.text.strip() for seg in segments if seg.text and seg.text.strip()
                ).strip()
                text = normalize_for_librispeech(raw_text, normalizer)
            except Exception as exc:  # pragma: no cover - best effort per utterance
                chapter_failures += 1
                total_failures += 1
                text = "UNTRANSCRIBED"
                print(f"[WARN] {flac_path.name}: {exc}", file=sys.stderr)

            lines.append(f"{flac_path.stem} {text}")
            if idx % max(1, args.log_every) == 0 or idx == len(flacs):
                print(f"[INFO] Chapter {chapter}: {idx}/{len(flacs)} done")

        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp_path.replace(out_path)

        total_files += len(flacs)
        print(
            f"[DONE] Chapter {chapter}: wrote {out_path} "
            f"({len(lines)} lines, failures={chapter_failures})"
        )

    print(f"[SUMMARY] Files processed: {total_files}, failures: {total_failures}")


if __name__ == "__main__":
    main()
