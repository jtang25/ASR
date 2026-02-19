#!/usr/bin/env python3
"""Burn right-aligned, cropped live captions onto a video via streaming RNN-T."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SPACE_RE = re.compile(r"\s+")


@dataclass
class CaptionUpdate:
    time_sec: float
    text: str


@dataclass
class CaptionSegment:
    start_sec: float
    end_sec: float
    text: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run streaming RNN-T ASR on a video and burn right-aligned captions "
            "that show only the trailing words (left-cropped)."
        )
    )
    p.add_argument("--checkpoint", required=True, help="Path to RNN-T checkpoint.")
    p.add_argument("--video", required=True, help="Input video path.")
    p.add_argument("--output", default="", help="Output captioned video path.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--amp-dtype", choices=["auto", "off", "fp16", "bf16"], default="off")
    p.add_argument(
        "--compile",
        choices=["off", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        default="off",
    )
    p.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate for ASR path.")
    p.add_argument("--chunk-size-enc", type=int, default=0)
    p.add_argument("--left-context-chunks", type=int, default=-2)
    p.add_argument("--right-context", type=int, default=-1)
    p.add_argument("--max-symbols-per-step", type=int, default=5)

    p.add_argument(
        "--context-words",
        type=int,
        default=18,
        help="Internal context tail size before cropping for display.",
    )
    p.add_argument(
        "--display-words",
        type=int,
        default=6,
        help="How many trailing words to render (right-aligned).",
    )
    p.add_argument(
        "--tail-hold-sec",
        type=float,
        default=0.8,
        help="How long to hold the final caption after last update.",
    )
    p.add_argument("--min-segment-sec", type=float, default=0.12, help="Drop very short caption segments.")
    p.add_argument("--font-name", default="Arial", help="ASS font name.")
    p.add_argument("--font-size", type=int, default=46, help="Caption font size.")
    p.add_argument("--right-margin", type=int, default=72, help="Pixels from right edge.")
    p.add_argument("--bottom-margin", type=int, default=58, help="Pixels from bottom edge.")
    p.add_argument(
        "--audio-mode",
        choices=["copy", "aac"],
        default="copy",
        help="Output audio strategy. 'copy' keeps source track when possible.",
    )
    p.add_argument("--crf", type=int, default=18, help="Video quality for libx264 encode (lower is higher quality).")
    p.add_argument("--preset", default="medium", help="libx264 preset.")
    p.add_argument("--ass-out", default="", help="Optional path to keep generated ASS subtitle file.")
    return p.parse_args()


def _normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


def _tail_words(text: str, n_words: int) -> str:
    if n_words <= 0:
        return ""
    words = text.split()
    if len(words) <= n_words:
        return text
    return " ".join(words[-n_words:])


def _run_checked(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=True,
    )


def _tool_exists(tool: str) -> bool:
    from shutil import which

    return which(tool) is not None


def _probe_video(video_path: Path) -> tuple[int, int, float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=width,height",
        "-show_entries",
        "format=duration",
        "-select_streams",
        "v:0",
        "-of",
        "json",
        str(video_path),
    ]
    result = _run_checked(cmd)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    width = int(streams[0].get("width", 1920)) if streams else 1920
    height = int(streams[0].get("height", 1080)) if streams else 1080
    duration = float(payload.get("format", {}).get("duration", "0.0") or 0.0)
    return width, height, duration


def _sec_to_ass_time(t_sec: float) -> str:
    t_sec = max(0.0, float(t_sec))
    hours = int(t_sec // 3600)
    minutes = int((t_sec % 3600) // 60)
    seconds = int(t_sec % 60)
    centis = int(round((t_sec - int(t_sec)) * 100))
    if centis == 100:
        centis = 0
        seconds += 1
        if seconds == 60:
            seconds = 0
            minutes += 1
            if minutes == 60:
                minutes = 0
                hours += 1
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centis:02d}"


def _escape_ass_text(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def _updates_to_segments(
    updates: list[CaptionUpdate],
    total_sec: float,
    min_segment_sec: float,
    tail_hold_sec: float,
) -> list[CaptionSegment]:
    if not updates:
        return []

    ordered: list[CaptionUpdate] = []
    for u in updates:
        if ordered and u.text == ordered[-1].text:
            ordered[-1] = CaptionUpdate(time_sec=u.time_sec, text=u.text)
        elif ordered and u.time_sec <= ordered[-1].time_sec:
            continue
        else:
            ordered.append(u)

    segments: list[CaptionSegment] = []
    for i, u in enumerate(ordered):
        start = max(0.0, u.time_sec)
        if i + 1 < len(ordered):
            end = min(total_sec, ordered[i + 1].time_sec)
        else:
            end = min(total_sec, start + max(0.0, tail_hold_sec))
            if end <= start:
                end = total_sec
        if end - start < min_segment_sec:
            continue
        segments.append(CaptionSegment(start_sec=start, end_sec=end, text=u.text))

    return segments


def _write_ass(
    out_path: Path,
    segments: list[CaptionSegment],
    width: int,
    height: int,
    font_name: str,
    font_size: int,
    right_margin: int,
    bottom_margin: int,
) -> None:
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "Collisions: Normal\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
        "Alignment,MarginL,MarginR,MarginV,Encoding\n"
        f"Style: Caption,{font_name},{font_size},&H00FFFFFF,&H000000FF,&H00141414,&H66000000,"
        f"0,0,0,0,100,100,0,0,1,2,0,3,24,{right_margin},{bottom_margin},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )
    lines = [header]
    for seg in segments:
        start = _sec_to_ass_time(seg.start_sec)
        end = _sec_to_ass_time(seg.end_sec)
        text = _escape_ass_text(seg.text)
        lines.append(f"Dialogue: 0,{start},{end},Caption,,0,0,0,,{text}\n")
    out_path.write_text("".join(lines), encoding="utf-8")


def _build_caption_segments(
    tokenizer,
    hyp_ids: list[int],
    updates_raw: list[CaptionUpdate],
    audio_sec: float,
    context_words: int,
    display_words: int,
    min_segment_sec: float,
    tail_hold_sec: float,
) -> list[CaptionSegment]:
    updates = updates_raw[:]
    if not updates:
        final_text = _normalize_text(tokenizer.decode(hyp_ids))
        if final_text:
            display = _tail_words(_tail_words(final_text, context_words), display_words)
            updates.append(CaptionUpdate(time_sec=0.0, text=display))
    return _updates_to_segments(
        updates=updates,
        total_sec=audio_sec,
        min_segment_sec=min_segment_sec,
        tail_hold_sec=tail_hold_sec,
    )


def _extract_audio_ffmpeg(video_path: Path, wav_out: Path, sample_rate: int) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(wav_out),
    ]
    _run_checked(cmd)


def _burn_ass_ffmpeg(
    video_path: Path,
    output_path: Path,
    ass_path: Path,
    crf: int,
    preset: str,
    audio_mode: str,
) -> None:
    def _cmd(mode: str) -> list[str]:
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"ass={ass_path.name}",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
        ]
        if mode == "copy":
            cmd += ["-c:a", "copy"]
        else:
            cmd += ["-c:a", "aac", "-b:a", "192k"]
        cmd.append(str(output_path))
        return cmd

    try:
        _run_checked(_cmd(audio_mode), cwd=ass_path.parent)
    except subprocess.CalledProcessError:
        if audio_mode != "copy":
            raise
        # Some source codecs cannot be copied into target container.
        _run_checked(_cmd("aac"), cwd=ass_path.parent)


def main() -> None:
    args = parse_args()
    if args.context_words < 1:
        raise ValueError("--context-words must be >= 1")
    if args.display_words < 1:
        raise ValueError("--display-words must be >= 1")
    if args.display_words > args.context_words:
        raise ValueError("--display-words must be <= --context-words")

    if not _tool_exists("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH.")
    if not _tool_exists("ffprobe"):
        raise RuntimeError("ffprobe not found on PATH.")

    try:
        from preprocessing import LogMelSpectrogram
        from transcribe_streaming_rnnt import (
            _build_model_from_config,
            _enable_fast_cuda_backends,
            _load_audio,
            _load_tokenizer,
            _maybe_compile_for_infer,
            _resolve_amp_dtype,
            resolve_device,
            streaming_rnnt_greedy_decode,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Missing ASR dependencies. Install project requirements first "
            "(for example: pip install -r requirements.txt)."
        ) from exc

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else video_path.with_name(f"{video_path.stem}.captioned.mp4")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    _enable_fast_cuda_backends(device)
    amp_dtype = _resolve_amp_dtype(args.amp_dtype, device)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}) or ckpt.get("args", {})
    if cfg.get("loss_type", "rnnt") != "rnnt":
        raise ValueError("Checkpoint is not RNN-T. This script currently supports RNN-T checkpoints.")

    tokenizer = _load_tokenizer(cfg)
    model = _build_model_from_config(cfg, device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print(
            f"[WARN] Checkpoint/model mismatch "
            f"(missing={len(missing)}, unexpected={len(unexpected)})."
        )
    model.eval()
    compiled = _maybe_compile_for_infer(model, args.compile)

    with tempfile.TemporaryDirectory(prefix="asr_caption_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        wav_path = tmp_dir / "audio.wav"
        ass_tmp_path = tmp_dir / "captions.ass"

        _extract_audio_ffmpeg(video_path=video_path, wav_out=wav_path, sample_rate=args.sample_rate)
        waveform = _load_audio(str(wav_path), args.sample_rate)

        mel_extractor = LogMelSpectrogram(sample_rate=args.sample_rate, n_mels=int(cfg.get("n_mels", 80)))
        mel = mel_extractor(waveform)
        mel = (mel - mel.mean(dim=0, keepdim=True)) / (mel.std(dim=0, keepdim=True) + 1e-9)

        ckpt_chunk = int(cfg.get("streaming_chunk_size", 0))
        ckpt_left = int(cfg.get("streaming_left_context_chunks", -1))
        ckpt_right = int(cfg.get("streaming_right_context", 0))
        chunk_size_enc = int(args.chunk_size_enc) if args.chunk_size_enc > 0 else max(1, ckpt_chunk or 8)
        left_context_chunks = ckpt_left if args.left_context_chunks == -2 else int(args.left_context_chunks)
        right_context = ckpt_right if args.right_context == -1 else int(args.right_context)

        updates: list[CaptionUpdate] = []
        last_display = ""

        def on_progress(chunk_end_sec: float, ids: list[int], emitted_this_chunk: int) -> None:
            nonlocal last_display
            if emitted_this_chunk <= 0:
                return
            full_text = _normalize_text(tokenizer.decode(ids))
            if not full_text:
                return
            # Keep a bigger tail internally, then render only the trailing few words.
            context_tail = _tail_words(full_text, args.context_words)
            display_tail = _tail_words(context_tail, args.display_words)
            if not display_tail:
                return
            if display_tail != last_display:
                updates.append(CaptionUpdate(time_sec=chunk_end_sec, text=display_tail))
                last_display = display_tail

        hyp_ids, metrics = streaming_rnnt_greedy_decode(
            model=model,
            mel=mel,
            chunk_size_enc=chunk_size_enc,
            left_context_chunks=left_context_chunks,
            right_context=right_context,
            max_symbols_per_step=args.max_symbols_per_step,
            amp_dtype=amp_dtype,
            progress_hook=on_progress,
        )
        hyp_text = _normalize_text(tokenizer.decode(hyp_ids))

        width, height, video_sec = _probe_video(video_path)
        total_sec = max(float(metrics["audio_sec"]), video_sec, 0.0)
        segments = _build_caption_segments(
            tokenizer=tokenizer,
            hyp_ids=hyp_ids,
            updates_raw=updates,
            audio_sec=total_sec,
            context_words=args.context_words,
            display_words=args.display_words,
            min_segment_sec=float(args.min_segment_sec),
            tail_hold_sec=float(args.tail_hold_sec),
        )
        if not segments:
            raise RuntimeError("No caption segments generated (no decoded text emitted).")

        _write_ass(
            out_path=ass_tmp_path,
            segments=segments,
            width=width,
            height=height,
            font_name=args.font_name,
            font_size=int(args.font_size),
            right_margin=int(args.right_margin),
            bottom_margin=int(args.bottom_margin),
        )
        _burn_ass_ffmpeg(
            video_path=video_path,
            output_path=output_path,
            ass_path=ass_tmp_path,
            crf=int(args.crf),
            preset=args.preset,
            audio_mode=args.audio_mode,
        )

        if args.ass_out:
            ass_out = Path(args.ass_out)
            ass_out.parent.mkdir(parents=True, exist_ok=True)
            ass_out.write_text(ass_tmp_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"checkpoint               : {ckpt_path}")
    print(f"video                    : {video_path}")
    print(f"output                   : {output_path}")
    print(f"device                   : {device}")
    print(f"amp_dtype                : {str(amp_dtype).replace('torch.', '') if amp_dtype else 'off'}")
    print(f"torch_compile            : {args.compile if compiled else 'off'}")
    print(f"chunk_size_enc           : {chunk_size_enc}")
    print(f"left_context_chunks      : {left_context_chunks}")
    print(f"right_context_enc        : {right_context}")
    print(f"context_words            : {args.context_words}")
    print(f"display_words            : {args.display_words}")
    print(f"caption_segments         : {len(segments)}")
    print(f"audio_sec                : {metrics['audio_sec']:.3f}")
    print(f"decode_sec               : {metrics['decode_sec']:.3f}")
    print(f"rtf                      : {metrics['rtf']:.4f}")
    print(f"hyp                      : {hyp_text}")


if __name__ == "__main__":
    main()
