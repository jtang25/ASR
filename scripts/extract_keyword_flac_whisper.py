#!/usr/bin/env python3
"""Extract keyword-only FLAC clips with Faster-Whisper word timestamps.

Input:
  - JSON mapping: term -> [utt_id, ...]
    (for example: dataset/JeromePowell_asr/keyword_occurrences_001_008.json)
  - Source audio root with files at: <audio_root>/<chapter>/<utt_id>.flac

Output:
  - Per-keyword FLAC clips under <output_dir>/<term_slug>/
  - CSV manifest with start/end times and output paths
  - JSON listing utterances where expected keywords were not found by alignment
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from faster_whisper import WhisperModel


NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_token(text: str) -> str:
    return NON_ALNUM_RE.sub("", text.lower())


def phrase_to_tokens(phrase: str) -> list[str]:
    out: list[str] = []
    for tok in phrase.split():
        norm = normalize_token(tok)
        if norm:
            out.append(norm)
    return out


def slugify(text: str) -> str:
    slug = NON_ALNUM_RE.sub("_", text.lower()).strip("_")
    return slug or "term"


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def ffmpeg_crop_to_flac(
    audio_in: Path,
    audio_out: Path,
    start_sec: float,
    end_sec: float,
    overwrite: bool,
) -> None:
    audio_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(audio_in),
        "-ss",
        f"{start_sec:.3f}",
        "-to",
        f"{end_sec:.3f}",
        "-c:a",
        "flac",
        str(audio_out),
    ]
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--keywords-json",
        default="dataset/JeromePowell_asr/keyword_occurrences_001_008.json",
        help="JSON file mapping term -> utterance IDs",
    )
    p.add_argument(
        "--audio-root",
        default="dataset/JeromePowell_asr/9999",
        help="Root folder containing chapter dirs (001..008)",
    )
    p.add_argument(
        "--output-dir",
        default="dataset/JeromePowell_asr/keyword_segments_whisper",
        help="Where extracted keyword FLAC clips are written",
    )
    p.add_argument("--model", default="large-v3", help="Faster-Whisper model name/path")
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Whisper inference device",
    )
    p.add_argument(
        "--compute-type",
        default="float16",
        help="Faster-Whisper compute_type (e.g., float16, int8_float16, int8)",
    )
    p.add_argument("--language", default="en")
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument(
        "--pad-ms",
        type=int,
        default=300,
        help="Padding applied before/after each keyword segment",
    )
    p.add_argument(
        "--min-clip-ms",
        type=int,
        default=700,
        help="Minimum output clip duration; expanded symmetrically when needed",
    )
    p.add_argument(
        "--terms",
        nargs="+",
        default=None,
        help="Optional subset of terms to process (exact term text)",
    )
    p.add_argument(
        "--max-utts",
        type=int,
        default=0,
        help="Limit unique utterances for quick testing (0 = all)",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    keywords_json = Path(args.keywords_json)
    audio_root = Path(args.audio_root)
    output_dir = Path(args.output_dir)
    manifest_csv = output_dir / "manifest.csv"
    missing_json = output_dir / "missing_alignment.json"

    if not keywords_json.exists():
        raise FileNotFoundError(f"keywords JSON not found: {keywords_json}")
    if not audio_root.exists():
        raise FileNotFoundError(f"audio root not found: {audio_root}")

    term_to_utts: dict[str, list[str]] = json.loads(keywords_json.read_text(encoding="utf-8"))
    if args.terms:
        keep = set(args.terms)
        term_to_utts = {k: v for k, v in term_to_utts.items() if k in keep}

    utt_to_terms: dict[str, set[str]] = defaultdict(set)
    for term, utts in term_to_utts.items():
        for utt in utts:
            utt_to_terms[utt].add(term)

    utt_ids = sorted(utt_to_terms.keys())
    if args.max_utts > 0:
        utt_ids = utt_ids[: args.max_utts]

    device = resolve_device(args.device)
    print(f"Loading model: {args.model} (device={device}, compute_type={args.compute_type})")
    model = WhisperModel(args.model, device=device, compute_type=args.compute_type)

    output_dir.mkdir(parents=True, exist_ok=True)
    pad_sec = max(0.0, float(args.pad_ms) / 1000.0)
    min_clip_sec = max(0.1, float(args.min_clip_ms) / 1000.0)

    missing: dict[str, list[str]] = defaultdict(list)
    rows: list[dict[str, str]] = []
    created = 0
    skipped_existing = 0
    failed = 0

    for idx, utt_id in enumerate(utt_ids, start=1):
        parts = utt_id.split("-")
        if len(parts) != 3:
            continue
        chapter = parts[1]
        audio_in = audio_root / chapter / f"{utt_id}.flac"
        if not audio_in.exists():
            failed += 1
            for term in sorted(utt_to_terms[utt_id]):
                rows.append(
                    {
                        "term": term,
                        "utt_id": utt_id,
                        "occurrence": "",
                        "start_sec": "",
                        "end_sec": "",
                        "audio_in": str(audio_in),
                        "audio_out": "",
                        "matched_text": "",
                        "status": "missing_audio",
                    }
                )
            continue

        segments, _info = model.transcribe(
            str(audio_in),
            language=args.language,
            beam_size=args.beam_size,
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=False,
        )

        words: list[tuple[str, float, float, str]] = []
        for seg in segments:
            if not seg.words:
                continue
            for w in seg.words:
                if w.start is None or w.end is None:
                    continue
                raw = (w.word or "").strip()
                token = normalize_token(raw)
                if not token:
                    continue
                words.append((token, float(w.start), float(w.end), raw))

        # Term token cache for this utterance only.
        term_tokens: dict[str, list[str]] = {
            term: phrase_to_tokens(term) for term in sorted(utt_to_terms[utt_id])
        }

        found_any_for_term: dict[str, bool] = {term: False for term in term_tokens}

        for term, toks in term_tokens.items():
            if not toks:
                continue
            k = len(toks)
            occ = 0
            for wi in range(0, len(words) - k + 1):
                slice_tokens = [words[wi + j][0] for j in range(k)]
                if slice_tokens != toks:
                    continue

                occ += 1
                found_any_for_term[term] = True
                start = max(0.0, words[wi][1] - pad_sec)
                end = max(start + 0.01, words[wi + k - 1][2] + pad_sec)
                # Avoid overly tight clips by enforcing a minimum duration.
                duration = end - start
                if duration < min_clip_sec:
                    mid = 0.5 * (start + end)
                    half = 0.5 * min_clip_sec
                    start = max(0.0, mid - half)
                    end = max(start + 0.01, mid + half)
                matched_text = " ".join(words[wi + j][3] for j in range(k)).strip()

                term_slug = slugify(term)
                audio_out = output_dir / term_slug / f"{utt_id}__{term_slug}__{occ:02d}.flac"
                if audio_out.exists() and not args.overwrite:
                    skipped_existing += 1
                    status = "exists"
                else:
                    try:
                        ffmpeg_crop_to_flac(
                            audio_in=audio_in,
                            audio_out=audio_out,
                            start_sec=start,
                            end_sec=end,
                            overwrite=args.overwrite,
                        )
                        created += 1
                        status = "ok"
                    except subprocess.CalledProcessError:
                        failed += 1
                        status = "ffmpeg_failed"

                rows.append(
                    {
                        "term": term,
                        "utt_id": utt_id,
                        "occurrence": str(occ),
                        "start_sec": f"{start:.3f}",
                        "end_sec": f"{end:.3f}",
                        "audio_in": str(audio_in),
                        "audio_out": str(audio_out),
                        "matched_text": matched_text,
                        "status": status,
                    }
                )

        for term, found in found_any_for_term.items():
            if not found:
                missing[term].append(utt_id)
                rows.append(
                    {
                        "term": term,
                        "utt_id": utt_id,
                        "occurrence": "",
                        "start_sec": "",
                        "end_sec": "",
                        "audio_in": str(audio_in),
                        "audio_out": "",
                        "matched_text": "",
                        "status": "not_found_in_word_timestamps",
                    }
                )

        if idx % 25 == 0 or idx == len(utt_ids):
            print(f"processed {idx}/{len(utt_ids)} utterances | created={created} skipped={skipped_existing} failed={failed}")

    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "term",
                "utt_id",
                "occurrence",
                "start_sec",
                "end_sec",
                "audio_in",
                "audio_out",
                "matched_text",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    missing_json.write_text(json.dumps(missing, indent=2), encoding="utf-8")

    print()
    print(f"manifest: {manifest_csv}")
    print(f"missing:  {missing_json}")
    print(f"created clips: {created}")
    print(f"existing skipped: {skipped_existing}")
    print(f"failed: {failed}")


if __name__ == "__main__":
    main()
