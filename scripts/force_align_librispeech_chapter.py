"""
Forced-align an existing LibriSpeech-style chapter using CTC alignment.

This script refines utterance boundaries from an existing chapter's
`split_audio.sh` and `.trans.txt` by running CTC forced alignment per utterance
on a local search window around each current segment.

It writes a new chapter directory with:
  - speaker-chapter.trans.txt (copied transcript text)
  - split_audio.sh            (refined split commands)
  - alignment_report.json     (old/new timing + scores + fallback reasons)

Example:
  python scripts/force_align_librispeech_chapter.py \
    --transcript dataset/JeromePowell_retimed/9999/003/9999-003.trans.txt \
    --split-script dataset/JeromePowell_retimed/9999/003/split_audio.sh \
    --output-dir dataset/JeromePowell_forced_aligned \
    --search-before 3.0 \
    --search-after 3.0 \
    --buffer-ms 150
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import torch
import torchaudio


SPLIT_RE = re.compile(
    r'-i\s+"([^"]+)"\s+-ss\s+([0-9.]+)\s+-t\s+([0-9.]+).*"(.*(\\|/))?((\d+)-(\d+)-(\d{4}))\.flac"$'
)


def parse_transcript(transcript_path: Path) -> tuple[list[str], dict[str, str]]:
    utt_order = []
    utt_text = {}
    with transcript_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            utt_id, text = parts
            utt_order.append(utt_id)
            utt_text[utt_id] = text
    return utt_order, utt_text


def parse_split_script(split_script_path: Path) -> list[dict]:
    rows = []
    with split_script_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line.startswith("ffmpeg "):
                continue
            m = SPLIT_RE.search(line)
            if not m:
                continue
            audio_path = m.group(1)
            start_s = float(m.group(2))
            duration_s = float(m.group(3))
            utt_id = m.group(6)
            speaker_id = m.group(7)
            chapter_id = m.group(8)
            rows.append(
                {
                    "utt_id": utt_id,
                    "speaker_id": speaker_id,
                    "chapter_id": chapter_id,
                    "audio_path": audio_path,
                    "start": start_s,
                    "duration": duration_s,
                }
            )
    if not rows:
        raise ValueError(f"No ffmpeg split commands parsed from: {split_script_path}")
    return rows


def load_audio_window(
    ffmpeg_path: str,
    audio_path: str,
    start_s: float,
    duration_s: float,
    sample_rate: int,
) -> torch.Tensor:
    cmd = [
        ffmpeg_path,
        "-nostdin",
        "-loglevel",
        "error",
        "-ss",
        f"{start_s:.3f}",
        "-t",
        f"{duration_s:.3f}",
        "-i",
        audio_path,
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-",
    ]
    raw = subprocess.check_output(cmd)
    arr = np.frombuffer(raw, dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(0)


def normalize_for_ctc(text: str) -> str:
    # Keep only A-Z and spaces. Replace spaces with "|" for wav2vec CTC labels.
    cleaned = re.sub(r"[^A-Za-z\s]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().upper()
    return cleaned.replace(" ", "|")


def make_token_ids(text: str, label_to_idx: dict[str, int]) -> list[int]:
    token_str = normalize_for_ctc(text)
    ids = []
    for ch in token_str:
        idx = label_to_idx.get(ch)
        if idx is not None:
            ids.append(idx)
    return ids


def forced_align_utterance(
    model: torch.nn.Module,
    sample_rate: int,
    blank_idx: int,
    label_to_idx: dict[str, int],
    waveform: torch.Tensor,
    text: str,
) -> tuple[float, float, float] | None:
    target_ids = make_token_ids(text, label_to_idx)
    if not target_ids:
        return None

    with torch.inference_mode():
        emissions, _ = model(waveform)  # [B, T, C]
        log_probs = torch.log_softmax(emissions, dim=-1)

    num_frames = int(log_probs.shape[1])
    if num_frames <= len(target_ids):
        return None

    targets = torch.tensor([target_ids], dtype=torch.int64)
    alignments, scores = torchaudio.functional.forced_align(
        log_probs,
        targets,
        blank=blank_idx,
    )
    alignment = alignments[0]
    score_vec = scores[0]

    non_blank = torch.nonzero(alignment != blank_idx, as_tuple=False).squeeze(-1)
    if non_blank.numel() == 0:
        return None

    start_frame = int(non_blank[0].item())
    end_frame = int(non_blank[-1].item()) + 1

    sec_per_frame = (waveform.shape[1] / sample_rate) / num_frames
    utt_start_local = start_frame * sec_per_frame
    utt_end_local = end_frame * sec_per_frame

    # Mean frame score over non-blank aligned frames.
    mean_score = float(score_vec[non_blank].mean().item())
    return utt_start_local, utt_end_local, mean_score


def build_ffmpeg_cmd(
    audio_path: str,
    out_path: Path,
    start_s: float,
    duration_s: float,
) -> str:
    return (
        f'ffmpeg -y -loglevel error -i "{audio_path}" '
        f"-ss {start_s:.3f} -t {duration_s:.3f} "
        f'-ar 16000 -ac 1 "{out_path.as_posix()}"'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Forced-align a LibriSpeech chapter.")
    parser.add_argument("--transcript", required=True, help="Path to speaker-chapter.trans.txt")
    parser.add_argument("--split-script", required=True, help="Path to split_audio.sh")
    parser.add_argument("--output-dir", required=True, help="Root output directory")
    parser.add_argument("--ffmpeg-path", default="ffmpeg", help="Path to ffmpeg executable")
    parser.add_argument("--search-before", type=float, default=3.0, help="Search window lead-in (seconds)")
    parser.add_argument("--search-after", type=float, default=3.0, help="Search window tail (seconds)")
    parser.add_argument("--buffer-ms", type=int, default=150, help="Output segment buffer in milliseconds")
    parser.add_argument("--min-segment-seconds", type=float, default=0.5, help="Minimum final segment duration")
    parser.add_argument(
        "--utt-ids",
        default="",
        help="Optional comma-separated utterance IDs to align (e.g. 9999-003-0001,9999-003-0002)",
    )
    args = parser.parse_args()

    transcript_path = Path(args.transcript)
    split_script_path = Path(args.split_script)
    output_root = Path(args.output_dir)

    utt_order, utt_text = parse_transcript(transcript_path)
    rows = parse_split_script(split_script_path)

    selected_ids = {x.strip() for x in args.utt_ids.split(",") if x.strip()}
    if selected_ids:
        rows = [r for r in rows if r["utt_id"] in selected_ids]
        if not rows:
            raise ValueError("No matching utterances found for --utt-ids in split script.")
        # Keep transcript lines consistent with filtered rows order.
        utt_order = [r["utt_id"] for r in rows]

    speaker_id = rows[0]["speaker_id"]
    chapter_id = rows[0]["chapter_id"]
    chapter_dir = output_root / speaker_id / chapter_id
    chapter_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CTC model for forced alignment...")
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model()
    model.eval()
    labels = bundle.get_labels()
    blank_idx = 0

    label_to_idx = {lbl.upper(): i for i, lbl in enumerate(labels)}
    sample_rate = 16000
    buffer_s = args.buffer_ms / 1000.0

    report = []
    ffmpeg_cmds = []
    trans_lines = []

    total = len(rows)
    print(f"Aligning {total} utterances...")
    for i, row in enumerate(rows, 1):
        utt_id = row["utt_id"]
        audio_path = row["audio_path"]
        old_start = float(row["start"])
        old_duration = float(row["duration"])
        text = utt_text.get(utt_id, "")

        if not text:
            # Keep old timing if transcript text is missing.
            new_start = old_start
            new_duration = old_duration
            report.append(
                {
                    "utt_id": utt_id,
                    "old_start": old_start,
                    "old_duration": old_duration,
                    "new_start": new_start,
                    "new_duration": new_duration,
                    "status": "fallback_missing_text",
                    "score": None,
                }
            )
        else:
            window_start = max(0.0, old_start - args.search_before)
            window_duration = old_duration + args.search_before + args.search_after
            aligned = None
            status = "aligned"
            score = None

            try:
                wave = load_audio_window(
                    args.ffmpeg_path,
                    audio_path,
                    window_start,
                    window_duration,
                    sample_rate,
                )
                aligned = forced_align_utterance(
                    model=model,
                    sample_rate=sample_rate,
                    blank_idx=blank_idx,
                    label_to_idx=label_to_idx,
                    waveform=wave,
                    text=text,
                )
            except Exception as exc:  # noqa: BLE001
                status = f"fallback_exception:{type(exc).__name__}"
                aligned = None

            if aligned is None:
                new_start = old_start
                new_duration = old_duration
                if status == "aligned":
                    status = "fallback_align_failed"
            else:
                local_start, local_end, score = aligned
                utt_start = window_start + local_start
                utt_end = window_start + local_end
                if utt_end <= utt_start:
                    new_start = old_start
                    new_duration = old_duration
                    status = "fallback_invalid_bounds"
                else:
                    new_start = max(0.0, utt_start - buffer_s)
                    new_duration = max(args.min_segment_seconds, (utt_end - utt_start) + 2 * buffer_s)

            report.append(
                {
                    "utt_id": utt_id,
                    "old_start": old_start,
                    "old_duration": old_duration,
                    "new_start": new_start,
                    "new_duration": new_duration,
                    "status": status,
                    "score": score,
                }
            )

        out_flac = chapter_dir / f"{utt_id}.flac"
        ffmpeg_cmds.append(build_ffmpeg_cmd(audio_path, out_flac, new_start, new_duration))
        trans_lines.append(f"{utt_id} {text}")

        if i % 25 == 0 or i == total:
            print(f"  {i}/{total} aligned")

    trans_out = chapter_dir / f"{speaker_id}-{chapter_id}.trans.txt"
    split_out = chapter_dir / "split_audio.sh"
    report_out = chapter_dir / "alignment_report.json"

    with trans_out.open("w", encoding="utf-8") as f:
        f.write("\n".join(trans_lines) + "\n")

    with split_out.open("w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        f.write("set -e\n\n")
        for cmd in ffmpeg_cmds:
            f.write(cmd + "\n")

    with report_out.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_transcript": str(transcript_path),
                "source_split_script": str(split_script_path),
                "speaker_id": speaker_id,
                "chapter_id": chapter_id,
                "search_before": args.search_before,
                "search_after": args.search_after,
                "buffer_ms": args.buffer_ms,
                "num_utterances": total,
                "num_fallbacks": sum(1 for r in report if r["status"].startswith("fallback")),
                "rows": report,
            },
            f,
            indent=2,
        )

    print("Done.")
    print(f"  Transcript: {trans_out}")
    print(f"  Split script: {split_out}")
    print(f"  Report: {report_out}")


if __name__ == "__main__":
    main()
