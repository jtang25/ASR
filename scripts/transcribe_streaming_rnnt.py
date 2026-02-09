#!/usr/bin/env python3
"""Streaming-style RNN-T transcription with chunked encoder windows.

This script performs incremental chunked decoding by:
  1) Loading mel frames for the full utterance.
  2) Feeding the encoder a sliding window (left context + current chunk).
  3) Carrying predictor state/token history across chunks.

It is designed for low-latency validation and profiling (RTF, first-token latency),
while staying compatible with checkpoints produced by train.py.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import math
import time
from pathlib import Path
import sys
from typing import Optional

import torch

try:
    import torchaudio
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("torchaudio is required for streaming transcription.") from exc


# Allow running as: python3 scripts/transcribe_streaming_rnnt.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import ConformerTransducer
from preprocessing import LogMelSpectrogram
from tokenizer import BLANK_IDX, CharTokenizer, SentencePieceTokenizer


LOSSLESS_EXTENSIONS = {".wav", ".wave", ".flac"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Streaming-style RNN-T transcription.")
    p.add_argument("--checkpoint", required=True, help="Path to RNNT checkpoint.")
    p.add_argument("--audio", required=True, help="Path to input .wav/.wave/.flac.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument(
        "--amp-dtype",
        choices=["auto", "off", "fp16", "bf16"],
        default="off",
        help="Mixed precision for CUDA inference. 'auto' picks bf16 when supported.",
    )
    p.add_argument(
        "--compile",
        choices=["off", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        default="off",
        help="Optional torch.compile mode for encoder/joint.",
    )
    p.add_argument(
        "--chunk-size-enc",
        type=int,
        default=0,
        help=(
            "Chunk size in encoder steps. "
            "Default 0 uses checkpoint streaming_chunk_size (or fallback 8)."
        ),
    )
    p.add_argument(
        "--left-context-chunks",
        type=int,
        default=-2,
        help=(
            "Left-context chunks. "
            "Default -2 uses checkpoint config; -1 means unlimited; >=0 explicit."
        ),
    )
    p.add_argument(
        "--right-context",
        type=int,
        default=-1,
        help=(
            "Lookahead in encoder steps. "
            "Default -1 uses checkpoint config; >=0 explicit."
        ),
    )
    p.add_argument(
        "--max-symbols-per-step",
        type=int,
        default=5,
        help="Maximum emitted symbols per encoder frame during greedy decode (lower is faster).",
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate for loaded audio.",
    )
    return p.parse_args()


def resolve_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _enable_fast_cuda_backends(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def _resolve_amp_dtype(amp_arg: str, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda" or amp_arg == "off":
        return None
    if amp_arg == "bf16":
        return torch.bfloat16
    if amp_arg == "fp16":
        return torch.float16
    # auto
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _maybe_compile_for_infer(model: ConformerTransducer, compile_mode: str) -> bool:
    if compile_mode == "off":
        return False
    if not hasattr(torch, "compile"):
        print("warning: torch.compile unavailable; running eager mode.")
        return False
    try:
        model.encoder = torch.compile(model.encoder, mode=compile_mode, fullgraph=False, dynamic=True)
        model.joint = torch.compile(model.joint, mode=compile_mode, fullgraph=False, dynamic=True)
        return True
    except Exception as exc:  # pragma: no cover
        print(f"warning: torch.compile failed ({exc}); running eager mode.")
        return False


def _validate_audio_path(audio_path: str) -> None:
    p = Path(audio_path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if p.suffix.lower() not in LOSSLESS_EXTENSIONS:
        allowed = ", ".join(sorted(LOSSLESS_EXTENSIONS))
        raise ValueError(f"Unsupported audio extension {p.suffix}. Use one of: {allowed}")


def _read_librispeech_reference(flac_path: str) -> str:
    p = Path(flac_path)
    utt_id = p.stem
    parts = utt_id.split("-")
    if len(parts) < 2:
        return ""
    trans_file = p.parent / f"{parts[0]}-{parts[1]}.trans.txt"
    if not trans_file.exists():
        return ""
    with open(trans_file, "r", encoding="utf-8") as f:
        for line in f:
            row = line.strip().split(None, 1)
            if len(row) == 2 and row[0] == utt_id:
                return row[1].lower()
    return ""


def _load_audio(audio_path: str, sample_rate: int) -> torch.Tensor:
    _validate_audio_path(audio_path)
    waveform, sr = torchaudio.load(audio_path)
    if waveform.numel() == 0:
        raise ValueError(f"Audio is empty: {audio_path}")
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform


def _load_tokenizer(cfg: dict):
    tok_type = cfg.get("tokenizer", "sp")
    if tok_type == "sp":
        sp_model = cfg.get("sp_model")
        if not sp_model:
            raise ValueError("Checkpoint config tokenizer='sp' but sp_model missing.")
        return SentencePieceTokenizer(sp_model)
    return CharTokenizer()


def _build_model_from_config(cfg: dict, device: torch.device) -> ConformerTransducer:
    model = ConformerTransducer(
        n_mels=int(cfg.get("n_mels", 80)),
        encoder_dim=int(cfg.get("d_model", 256)),
        num_heads=int(cfg.get("num_heads", 4)),
        num_encoder_layers=int(cfg.get("num_layers", 12)),
        vocab_size=int(cfg.get("vocab_size", 1024)),
        conv_kernel_size=int(cfg.get("conv_kernel", 32)),
        max_len=int(cfg.get("max_len", 2048)),
        ffn_dropout=float(cfg.get("dropout", 0.1)),
        attn_dropout=float(cfg.get("dropout", 0.1)),
        conv_dropout=float(cfg.get("dropout", 0.1)),
        pred_embed_dim=int(cfg.get("pred_embed_dim", 256)),
        pred_hidden_dim=int(cfg.get("pred_hidden_dim", 640)),
        pred_num_layers=int(cfg.get("pred_num_layers", 1)),
        joint_dim=int(cfg.get("joint_dim", 640)),
        blank_idx=BLANK_IDX,
        streaming_chunk_size=int(cfg.get("streaming_chunk_size", 0)),
        streaming_left_context_chunks=int(cfg.get("streaming_left_context_chunks", -1)),
        streaming_right_context=int(cfg.get("streaming_right_context", 0)),
        streaming_causal_conv=bool(cfg.get("streaming_causal_conv", False)),
    ).to(device)
    return model


def _enc_len_for_mel(model: ConformerTransducer, mel_frames: int, device: torch.device) -> int:
    t = torch.tensor([int(mel_frames)], dtype=torch.long, device=device)
    out = model.subsampling.get_output_lengths(t)
    return int(out[0].item())


def _max_mel_frames_for_enc_steps(model: ConformerTransducer, enc_steps: int, device: torch.device) -> int:
    """Largest mel frame count whose subsampled length <= enc_steps."""
    if enc_steps < 1:
        raise ValueError("enc_steps must be >= 1")
    lo, hi = 1, max(8, enc_steps * 4)
    while _enc_len_for_mel(model, hi, device) <= enc_steps:
        hi *= 2
        if hi > 10_000_000:
            break
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _enc_len_for_mel(model, mid, device) <= enc_steps:
            lo = mid
        else:
            hi = mid - 1
    return lo


@torch.no_grad()
def streaming_rnnt_greedy_decode(
    model: ConformerTransducer,
    mel: torch.Tensor,
    chunk_size_enc: int,
    left_context_chunks: int,
    right_context: int,
    max_symbols_per_step: int,
    amp_dtype: torch.dtype | None = None,
) -> tuple[list[int], dict[str, float]]:
    """Chunked encoder windows + stateful predictor decode."""
    if chunk_size_enc <= 0:
        raise ValueError("chunk_size_enc must be > 0")
    if left_context_chunks < -1:
        raise ValueError("left_context_chunks must be >= -1")
    if right_context < 0:
        raise ValueError("right_context must be >= 0")
    if max_symbols_per_step < 1:
        raise ValueError("max_symbols_per_step must be >= 1")

    device = next(model.parameters()).device
    mel = mel.to(device)
    total_mel_frames = int(mel.size(0))

    # Convert encoder-step controls to mel-frame controls.
    chunk_mel = _max_mel_frames_for_enc_steps(model, chunk_size_enc, device)
    right_mel = _max_mel_frames_for_enc_steps(model, right_context, device) if right_context > 0 else 0
    if left_context_chunks == -1:
        left_mel = total_mel_frames
    elif left_context_chunks == 0:
        left_mel = 0
    else:
        left_enc = int(left_context_chunks * chunk_size_enc)
        left_mel = _max_mel_frames_for_enc_steps(model, left_enc, device)

    def _encoder_autocast_ctx():
        if device.type == "cuda" and amp_dtype is not None:
            return torch.autocast(device_type="cuda", dtype=amp_dtype)
        return nullcontext()

    # Predictor state carries across chunks.
    state: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    pred_input = torch.tensor([[model.blank_idx]], device=device, dtype=torch.long)
    pred_out, state = model.predictor(pred_input, state)
    pred_vec = pred_out[0, 0]
    pred_proj = model.joint.project_pred_step(pred_vec.unsqueeze(0))

    hyp: list[int] = []
    first_token_time_sec: Optional[float] = None
    processed_mel = 0
    chunk_count = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_start = time.perf_counter()

    while processed_mel < total_mel_frames:
        chunk_count += 1
        cur_start = processed_mel
        cur_end = min(total_mel_frames, cur_start + chunk_mel)
        ctx_start = max(0, cur_start - left_mel)
        window_end = min(total_mel_frames, cur_end + right_mel)

        chunk = mel[ctx_start:window_end].unsqueeze(0)
        chunk_len = torch.tensor([window_end - ctx_start], dtype=torch.long, device=device)

        with _encoder_autocast_ctx():
            enc_out, enc_len = model.encode(chunk, chunk_len)
        valid_len = int(enc_len[0].item())

        # Locate current chunk span in encoder steps.
        mel_before_cur = cur_start - ctx_start
        mel_until_cur_end = cur_end - ctx_start
        enc_before = _enc_len_for_mel(model, mel_before_cur, device)
        enc_until_end = _enc_len_for_mel(model, mel_until_cur_end, device)
        enc_before = min(enc_before, valid_len)
        enc_until_end = min(enc_until_end, valid_len)

        if enc_until_end > enc_before:
            cur_enc = enc_out[0, enc_before:enc_until_end]
            for i in range(cur_enc.size(0)):
                enc_vec = cur_enc[i]
                enc_proj = model.joint.project_enc_step(enc_vec.unsqueeze(0))
                for _ in range(max_symbols_per_step):
                    logits = model.joint.forward_projected_step(enc_proj, pred_proj)
                    token_id = int(logits.argmax(dim=-1).item())
                    if token_id == model.blank_idx:
                        break
                    hyp.append(token_id)
                    if first_token_time_sec is None:
                        # Approximate token time by consumed audio at current chunk edge.
                        first_token_time_sec = float(cur_end) * 0.010
                    pred_input = torch.tensor([[token_id]], device=device, dtype=torch.long)
                    pred_out, state = model.predictor(pred_input, state)
                    pred_vec = pred_out[0, 0]
                    pred_proj = model.joint.project_pred_step(pred_vec.unsqueeze(0))

        processed_mel = cur_end

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t_start
    audio_sec = float(total_mel_frames) * 0.010
    rtf = elapsed / max(audio_sec, 1e-9)
    metrics = {
        "audio_sec": audio_sec,
        "decode_sec": elapsed,
        "rtf": rtf,
        "first_token_latency_sec": first_token_time_sec if first_token_time_sec is not None else math.nan,
        "chunks": float(chunk_count),
        "chunk_mel_frames": float(chunk_mel),
    }
    return hyp, metrics


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    _enable_fast_cuda_backends(device)
    amp_dtype = _resolve_amp_dtype(args.amp_dtype, device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}) or ckpt.get("args", {})

    if cfg.get("loss_type", "rnnt") != "rnnt":
        raise ValueError("Checkpoint is not RNN-T. Use CTC transcribe script for CTC models.")

    tokenizer = _load_tokenizer(cfg)
    model = _build_model_from_config(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    compiled = _maybe_compile_for_infer(model, args.compile)

    waveform = _load_audio(args.audio, args.sample_rate)
    mel_extractor = LogMelSpectrogram(sample_rate=args.sample_rate, n_mels=int(cfg.get("n_mels", 80)))
    mel = mel_extractor(waveform)
    mel = (mel - mel.mean(dim=0, keepdim=True)) / (mel.std(dim=0, keepdim=True) + 1e-9)

    ckpt_chunk = int(cfg.get("streaming_chunk_size", 0))
    ckpt_left = int(cfg.get("streaming_left_context_chunks", -1))
    ckpt_right = int(cfg.get("streaming_right_context", 0))
    chunk_size_enc = int(args.chunk_size_enc) if args.chunk_size_enc > 0 else max(1, ckpt_chunk or 8)
    left_context_chunks = ckpt_left if args.left_context_chunks == -2 else int(args.left_context_chunks)
    right_context = ckpt_right if args.right_context == -1 else int(args.right_context)

    hyp_ids, metrics = streaming_rnnt_greedy_decode(
        model=model,
        mel=mel,
        chunk_size_enc=chunk_size_enc,
        left_context_chunks=left_context_chunks,
        right_context=right_context,
        max_symbols_per_step=args.max_symbols_per_step,
        amp_dtype=amp_dtype,
    )
    hyp_text = tokenizer.decode(hyp_ids)
    ref_text = _read_librispeech_reference(args.audio)

    print(f"checkpoint               : {ckpt_path}")
    print(f"device                   : {device}")
    print(f"amp_dtype                : {str(amp_dtype).replace('torch.', '') if amp_dtype else 'off'}")
    print(f"torch_compile            : {args.compile if compiled else 'off'}")
    print(f"audio                    : {args.audio}")
    print(f"chunk_size_enc           : {chunk_size_enc}")
    print(f"left_context_chunks      : {left_context_chunks}")
    print(f"right_context_enc        : {right_context}")
    print(f"chunks_processed         : {int(metrics['chunks'])}")
    print(f"chunk_mel_frames         : {int(metrics['chunk_mel_frames'])}")
    print(f"audio_sec                : {metrics['audio_sec']:.3f}")
    print(f"decode_sec               : {metrics['decode_sec']:.3f}")
    print(f"rtf                      : {metrics['rtf']:.4f}")
    if math.isfinite(metrics["first_token_latency_sec"]):
        print(f"first_token_latency_sec  : {metrics['first_token_latency_sec']:.3f}")
    else:
        print("first_token_latency_sec  : NaN (no non-blank token emitted)")
    if ref_text:
        print(f"ref                      : {ref_text}")
    print(f"hyp                      : {hyp_text}")


if __name__ == "__main__":
    main()
