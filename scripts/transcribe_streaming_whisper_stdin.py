#!/usr/bin/env python3
"""Sliding-window faster-whisper transcription from raw PCM stdin.

Reads 16-bit mono PCM from stdin, keeps a recent audio window, and emits
incremental transcript updates at a fixed interval.
"""

from __future__ import annotations

import argparse
from array import array
import os
import sys
import time

import numpy as np
import torch
from faster_whisper import WhisperModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sliding-window faster-whisper transcription from PCM stdin.")
    p.add_argument("--model", default="large-v3-turbo", help="Whisper model size or local path.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument(
        "--compute-type",
        default="auto",
        choices=["auto", "float16", "float32", "int8", "int8_float16", "int8_float32"],
    )
    p.add_argument("--model-cache-dir", default=None, help="Optional model download/cache directory.")

    p.add_argument("--sample-rate", type=int, default=16000, help="PCM sample rate (Hz).")
    p.add_argument("--chunk-ms", type=float, default=200.0, help="stdin read chunk size (ms).")
    p.add_argument("--window-sec", type=float, default=12.0, help="Sliding window size (seconds).")
    p.add_argument("--emit-interval-sec", type=float, default=0.5, help="Emit partials every N seconds.")
    p.add_argument("--min-audio-sec", type=float, default=0.5, help="Minimum received audio before decoding.")
    p.add_argument("--print-unchanged", action="store_true", help="Print every emit tick even if text unchanged.")
    p.add_argument("--text-only", action="store_true", help="Emit only transcript text lines.")

    p.add_argument("--language", default="en", help="Language code (e.g. en). Use 'auto' for auto-detect.")
    p.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    p.add_argument("--beam-size", type=int, default=1, help="Beam size (1 is fastest).")
    p.add_argument("--condition-on-previous-text", action="store_true", help="Enable Whisper previous-text conditioning.")
    p.add_argument("--vad-filter", action="store_true", help="Enable VAD filtering in Whisper.")
    return p.parse_args()


def _resolve_device(arg: str) -> str:
    if arg == "cpu":
        return "cpu"
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_compute_type(compute_type: str, device: str) -> str:
    if compute_type != "auto":
        return compute_type
    if device == "cuda":
        # Good latency/quality default for GPU.
        return "float16"
    return "int8"


def _decode_window(
    model: WhisperModel,
    audio: np.ndarray,
    language: str,
    task: str,
    beam_size: int,
    condition_on_previous_text: bool,
    vad_filter: bool,
) -> str:
    lang = None if language == "auto" else language
    segments, _info = model.transcribe(
        audio,
        language=lang,
        task=task,
        beam_size=beam_size,
        condition_on_previous_text=condition_on_previous_text,
        vad_filter=vad_filter,
        temperature=0.0,
        word_timestamps=False,
        without_timestamps=True,
    )
    return " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()


def main() -> None:
    args = parse_args()
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be > 0")
    if args.chunk_ms <= 0:
        raise ValueError("--chunk-ms must be > 0")
    if args.window_sec <= 0:
        raise ValueError("--window-sec must be > 0")
    if args.emit_interval_sec <= 0:
        raise ValueError("--emit-interval-sec must be > 0")
    if args.min_audio_sec < 0:
        raise ValueError("--min-audio-sec must be >= 0")
    if args.beam_size < 1:
        raise ValueError("--beam-size must be >= 1")

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    device = _resolve_device(args.device)
    compute_type = _resolve_compute_type(args.compute_type, device)

    model = WhisperModel(
        args.model,
        device=device,
        compute_type=compute_type,
        download_root=args.model_cache_dir,
    )

    chunk_samples = max(1, int(args.sample_rate * args.chunk_ms / 1000.0))
    chunk_bytes = chunk_samples * 2  # int16 mono
    max_window_samples = max(1, int(args.window_sec * args.sample_rate))
    emit_every_samples = max(1, int(args.emit_interval_sec * args.sample_rate))
    min_audio_samples = max(0, int(args.min_audio_sec * args.sample_rate))

    if not args.text_only:
        print(f"model     : {args.model}")
        print(f"device    : {device}")
        print(f"compute   : {compute_type}")
        print(
            f"mode      : sliding-window whisper "
            f"(window={args.window_sec:.1f}s emit_every={args.emit_interval_sec:.2f}s)"
        )
        print(
            "decode    : "
            f"task={args.task} language={args.language} beam={args.beam_size} "
            f"cond_prev={args.condition_on_previous_text} vad={args.vad_filter}"
        )
        print("Listening on stdin (16kHz, 16-bit, mono PCM)...", flush=True)

    buffer = np.empty((0,), dtype=np.float32)
    total_samples = 0
    next_emit_at = min_audio_samples
    last_hyp = ""
    t0 = time.time()

    try:
        while True:
            chunk = sys.stdin.buffer.read(chunk_bytes)
            if not chunk:
                break

            pcm = array("h")
            pcm.frombytes(chunk)
            if not pcm:
                continue
            if sys.byteorder != "little":
                pcm.byteswap()

            wav_chunk = np.asarray(pcm, dtype=np.float32) / 32768.0
            total_samples += int(wav_chunk.shape[0])
            buffer = np.concatenate((buffer, wav_chunk))
            if buffer.shape[0] > max_window_samples:
                buffer = buffer[-max_window_samples:]

            if total_samples < next_emit_at:
                continue

            hyp = _decode_window(
                model=model,
                audio=buffer,
                language=args.language,
                task=args.task,
                beam_size=args.beam_size,
                condition_on_previous_text=args.condition_on_previous_text,
                vad_filter=args.vad_filter,
            )
            now_sec = total_samples / args.sample_rate
            if args.print_unchanged or hyp != last_hyp:
                if args.text_only:
                    print(hyp, flush=True)
                else:
                    elapsed = time.time() - t0
                    print(f"[{now_sec:7.2f}s | wall {elapsed:6.2f}s] {hyp}", flush=True)
                last_hyp = hyp
            next_emit_at += emit_every_samples

    except KeyboardInterrupt:
        pass

    if buffer.shape[0] >= max(1, min_audio_samples):
        final_hyp = _decode_window(
            model=model,
            audio=buffer,
            language=args.language,
            task=args.task,
            beam_size=args.beam_size,
            condition_on_previous_text=args.condition_on_previous_text,
            vad_filter=args.vad_filter,
        )
        now_sec = total_samples / args.sample_rate
        if args.print_unchanged or final_hyp != last_hyp:
            if args.text_only:
                print(final_hyp, flush=True)
            else:
                elapsed = time.time() - t0
                print(f"[{now_sec:7.2f}s | wall {elapsed:6.2f}s] {final_hyp}", flush=True)
        if not args.text_only:
            print(f"Final: {final_hyp}", flush=True)


if __name__ == "__main__":
    main()

