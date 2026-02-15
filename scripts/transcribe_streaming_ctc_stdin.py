#!/usr/bin/env python3
"""Sliding-window CTC transcription from raw PCM stdin.

Designed for low-latency "streaming-like" inference with non-streaming CTC checkpoints.
It reads 16-bit mono PCM from stdin, keeps a recent audio window, and emits partial
transcripts at a fixed interval.

Example (Linux local mic -> remote server over SSH):
  arecord -q -f S16_LE -r 16000 -c 1 -t raw \\
    | ssh user@server "cd /workspace/asr && .venv/bin/python scripts/transcribe_streaming_ctc_stdin.py \\
        --checkpoint checkpoints_jp_only_ctc_no_vn/best.pt \\
        --lm-path lm/3-gram.pruned.1e-7.lower.arpa"
"""

from __future__ import annotations

import argparse
from array import array
import contextlib
import io
import os
from pathlib import Path
import sys
import time

import torch

# Allow running as: python3 scripts/transcribe_streaming_ctc_stdin.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from decoding import beam_search_decode, build_lm_decoder, lm_beam_search_decode
from model import ConformerASR
from preprocessing import LogMelSpectrogram
from tokenizer import BLANK_IDX, CharTokenizer, SentencePieceTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sliding-window CTC transcription from PCM stdin.")
    p.add_argument("--checkpoint", required=True, help="Path to CTC checkpoint.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")

    p.add_argument("--sample-rate", type=int, default=16000, help="PCM sample rate (Hz).")
    p.add_argument("--chunk-ms", type=float, default=200.0, help="stdin read chunk size (ms).")
    p.add_argument("--window-sec", type=float, default=12.0, help="Sliding window size (seconds).")
    p.add_argument("--emit-interval-sec", type=float, default=0.5, help="Emit partials every N seconds.")
    p.add_argument("--min-audio-sec", type=float, default=0.5, help="Minimum received audio before decoding.")
    p.add_argument("--print-unchanged", action="store_true", help="Print every emit tick even if text unchanged.")

    p.add_argument("--beam-size", type=int, default=20)
    p.add_argument("--beam-token-prune", type=int, default=0)
    p.add_argument("--lm-path", default=None)
    p.add_argument("--lm-alpha", type=float, default=0.5)
    p.add_argument("--lm-beta", type=float, default=1.0)
    p.add_argument("--lm-beam-width", type=int, default=128)
    p.add_argument("--text-only", action="store_true", help="Emit only transcript text lines.")
    return p.parse_args()


def _resolve_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast_dtype(precision: str, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if precision == "fp32":
        return None
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    # auto
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _load_tokenizer(cfg: dict):
    tok_type = cfg.get("tokenizer", "sp")
    if tok_type == "sp":
        sp_model = cfg.get("sp_model")
        if not sp_model:
            raise ValueError("Checkpoint config has tokenizer='sp' but no sp_model path.")
        return SentencePieceTokenizer(sp_model)
    return CharTokenizer()


def _build_model(cfg: dict, vocab_size: int, device: torch.device) -> ConformerASR:
    return ConformerASR(
        n_mels=int(cfg.get("n_mels", 80)),
        d_model=int(cfg.get("d_model", 256)),
        num_heads=int(cfg.get("num_heads", 4)),
        num_layers=int(cfg.get("num_layers", 12)),
        vocab_size=int(vocab_size),
        conv_kernel_size=int(cfg.get("conv_kernel", 32)),
        max_len=int(cfg.get("max_len", 2048)),
        ffn_dropout=float(cfg.get("dropout", 0.1)),
        attn_dropout=float(cfg.get("dropout", 0.1)),
        conv_dropout=float(cfg.get("dropout", 0.1)),
        streaming_chunk_size=int(cfg.get("streaming_chunk_size", 0)),
        streaming_left_context_chunks=int(cfg.get("streaming_left_context_chunks", -1)),
        streaming_right_context=int(cfg.get("streaming_right_context", 0)),
        streaming_causal_conv=bool(cfg.get("streaming_causal_conv", False)),
    ).to(device)


def _ctc_decoder_vocab(tokenizer) -> list[str]:
    return [str(tokenizer.id_to_piece(i)) for i in range(int(tokenizer.vocab_size))]


@contextlib.contextmanager
def _silence_process_stdio(enabled: bool):
    """Temporarily silence OS-level stdout/stderr (catches C/C++ extension prints)."""
    if not enabled:
        yield
        return

    saved_out = os.dup(1)
    saved_err = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull)


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
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
    if args.beam_token_prune < 0:
        raise ValueError("--beam-token-prune must be >= 0")
    if args.lm_beam_width < 1:
        raise ValueError("--lm-beam-width must be >= 1")

    device = _resolve_device(args.device)
    autocast_dtype = _autocast_dtype(args.precision, device)
    amp_enabled = device.type == "cuda" and autocast_dtype is not None

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}) or ckpt.get("args", {}) or {}
    if cfg.get("loss_type", "ctc") != "ctc":
        raise ValueError("Checkpoint is not CTC. This script supports only CTC checkpoints.")

    tokenizer = _load_tokenizer(cfg)
    model = _build_model(cfg, int(cfg.get("vocab_size", tokenizer.vocab_size)), device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    mel_extractor = LogMelSpectrogram(n_mels=int(cfg.get("n_mels", 80)))

    lm_decoder = None
    if args.lm_path:
        if args.text_only:
            # Suppress verbose LM loader output so only transcript text is emitted.
            with _silence_process_stdio(True):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    lm_decoder = build_lm_decoder(
                        vocab=_ctc_decoder_vocab(tokenizer),
                        lm_path=args.lm_path,
                        blank_idx=BLANK_IDX,
                        alpha=float(args.lm_alpha),
                        beta=float(args.lm_beta),
                    )
        else:
            lm_decoder = build_lm_decoder(
                vocab=_ctc_decoder_vocab(tokenizer),
                lm_path=args.lm_path,
                blank_idx=BLANK_IDX,
                alpha=float(args.lm_alpha),
                beta=float(args.lm_beta),
            )

    chunk_samples = max(1, int(args.sample_rate * args.chunk_ms / 1000.0))
    chunk_bytes = chunk_samples * 2  # int16 mono
    max_window_samples = max(1, int(args.window_sec * args.sample_rate))
    emit_every_samples = max(1, int(args.emit_interval_sec * args.sample_rate))
    min_audio_samples = max(0, int(args.min_audio_sec * args.sample_rate))
    token_prune = args.beam_token_prune if args.beam_token_prune > 0 else None

    if not args.text_only:
        print(f"checkpoint: {ckpt_path}")
        print(f"device    : {device}")
        print(
            f"mode      : sliding-window ctc "
            f"(window={args.window_sec:.1f}s emit_every={args.emit_interval_sec:.2f}s)"
        )
        print(
            "decode    : "
            + (
                f"ctc+lm ({args.lm_path}, alpha={args.lm_alpha}, beta={args.lm_beta}, width={args.lm_beam_width})"
                if lm_decoder is not None
                else f"ctc-beam (beam_size={args.beam_size}, token_prune={args.beam_token_prune})"
            )
        )
        print("Listening on stdin (16kHz, 16-bit, mono PCM)...", flush=True)

    buffer = torch.empty(0, dtype=torch.float32)
    total_samples = 0
    next_emit_at = min_audio_samples
    last_hyp = ""
    t0 = time.time()

    def decode_waveform(waveform: torch.Tensor) -> str:
        mel = mel_extractor(waveform)
        mel = (mel - mel.mean(dim=0, keepdim=True)) / (mel.std(dim=0, keepdim=True) + 1e-9)
        mel = mel.unsqueeze(0).to(device, non_blocking=True)
        mel_lengths = torch.tensor([mel.size(1)], dtype=torch.long, device=device)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
                log_probs, out_lengths = model(mel, mel_lengths)
            if lm_decoder is not None:
                return lm_beam_search_decode(log_probs, out_lengths, lm_decoder, args.lm_beam_width)[0]
            decoded = beam_search_decode(
                log_probs,
                out_lengths,
                beam_size=args.beam_size,
                blank_idx=BLANK_IDX,
                token_prune=token_prune,
            )
            return tokenizer.decode(decoded[0]) if decoded else ""

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

            wav_chunk = torch.tensor(pcm, dtype=torch.float32).div_(32768.0)
            total_samples += int(wav_chunk.numel())
            buffer = torch.cat((buffer, wav_chunk))
            if buffer.numel() > max_window_samples:
                buffer = buffer[-max_window_samples:]

            if total_samples < next_emit_at:
                continue

            hyp = decode_waveform(buffer)
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

    if buffer.numel() >= max(1, min_audio_samples):
        final_hyp = decode_waveform(buffer)
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
