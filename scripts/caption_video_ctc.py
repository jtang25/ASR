#!/usr/bin/env python3
"""Burn right-aligned, cropped live captions onto a video via streaming-like CTC."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch

try:
    import torchaudio
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("torchaudio is required for CTC video captioning.") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from decoding import beam_search_decode, build_lm_decoder, lm_beam_search_decode
from model import ConformerASR
from preprocessing import LogMelSpectrogram
from tokenizer import BLANK_IDX, CharTokenizer, SentencePieceTokenizer


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
            "Run streaming-like CTC ASR on a video and burn right-aligned captions "
            "that show only the trailing words (left-cropped)."
        )
    )
    p.add_argument("--checkpoint", required=True, help="Path to CTC checkpoint.")
    p.add_argument("--video", required=True, help="Input video path.")
    p.add_argument("--output", default="", help="Output captioned video path.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument(
        "--precision",
        choices=["auto", "fp32", "fp16", "bf16"],
        default="auto",
        help="Inference precision. 'auto' chooses bf16/fp16 on CUDA, fp32 on CPU.",
    )
    p.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate for ASR path.")
    p.add_argument("--chunk-ms", type=float, default=200.0, help="Streaming read chunk size in milliseconds.")
    p.add_argument("--window-sec", type=float, default=12.0, help="Sliding audio window size in seconds.")
    p.add_argument("--emit-interval-sec", type=float, default=0.5, help="Emit partials every N seconds.")
    p.add_argument("--min-audio-sec", type=float, default=0.5, help="Minimum audio before first decode.")
    p.add_argument("--beam-size", type=int, default=20, help="CTC prefix beam size.")
    p.add_argument("--beam-token-prune", type=int, default=0, help="Top-k token prune per frame (0 disables).")
    p.add_argument("--lm-path", default="", help="Optional KenLM .arpa/.bin path.")
    p.add_argument("--lm-alpha", type=float, default=0.5, help="LM fusion alpha.")
    p.add_argument("--lm-beta", type=float, default=1.0, help="LM fusion beta.")
    p.add_argument("--lm-beam-width", type=int, default=128, help="Beam width passed to LM decoder.")
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
    final_hyp: str,
    updates_raw: list[CaptionUpdate],
    audio_sec: float,
    context_words: int,
    display_words: int,
    min_segment_sec: float,
    tail_hold_sec: float,
) -> list[CaptionSegment]:
    updates = updates_raw[:]
    if not updates:
        final_text = _normalize_text(final_hyp)
        if final_text:
            display = _tail_words(_tail_words(final_text, context_words), display_words)
            if display:
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


def _load_audio(audio_path: Path, sample_rate: int) -> torch.Tensor:
    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.numel() == 0:
        raise ValueError(f"Audio is empty: {audio_path}")
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform


def _ctc_decoder_vocab(tokenizer) -> list[str]:
    return [str(tokenizer.id_to_piece(i)) for i in range(int(tokenizer.vocab_size))]


@torch.no_grad()
def _streaming_like_decode(
    *,
    model: ConformerASR,
    tokenizer,
    lm_decoder,
    waveform: torch.Tensor,
    sample_rate: int,
    mel_extractor: LogMelSpectrogram,
    beam_size: int,
    beam_token_prune: int,
    lm_beam_width: int,
    chunk_ms: float,
    window_sec: float,
    emit_interval_sec: float,
    min_audio_sec: float,
    context_words: int,
    display_words: int,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> tuple[str, list[CaptionUpdate], dict[str, float]]:
    amp_enabled = device.type == "cuda" and autocast_dtype is not None
    token_prune = beam_token_prune if beam_token_prune > 0 else None

    chunk_samples = max(1, int(sample_rate * chunk_ms / 1000.0))
    max_window_samples = max(1, int(window_sec * sample_rate))
    emit_every_samples = max(1, int(emit_interval_sec * sample_rate))
    min_audio_samples = max(0, int(min_audio_sec * sample_rate))

    buffer = torch.empty(0, dtype=torch.float32)
    total_samples = 0
    next_emit_at = min_audio_samples
    last_hyp = ""
    last_display = ""
    updates: list[CaptionUpdate] = []
    decode_calls = 0
    decode_sec = 0.0

    def decode_waveform(chunk_wave: torch.Tensor) -> str:
        nonlocal decode_calls
        nonlocal decode_sec
        t0 = time.perf_counter()
        mel = mel_extractor(chunk_wave)
        mel = (mel - mel.mean(dim=0, keepdim=True)) / (mel.std(dim=0, keepdim=True) + 1e-9)
        mel = mel.unsqueeze(0).to(device, non_blocking=True)
        mel_lengths = torch.tensor([mel.size(1)], dtype=torch.long, device=device)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
            log_probs, out_lengths = model(mel, mel_lengths)
        if lm_decoder is not None:
            hyp = lm_beam_search_decode(log_probs, out_lengths, lm_decoder, lm_beam_width)[0]
        else:
            decoded = beam_search_decode(
                log_probs,
                out_lengths,
                beam_size=beam_size,
                blank_idx=BLANK_IDX,
                token_prune=token_prune,
            )
            hyp = tokenizer.decode(decoded[0]) if decoded else ""
        decode_calls += 1
        decode_sec += time.perf_counter() - t0
        return _normalize_text(hyp)

    for start in range(0, int(waveform.numel()), chunk_samples):
        wav_chunk = waveform[start : start + chunk_samples]
        if wav_chunk.numel() <= 0:
            continue
        total_samples += int(wav_chunk.numel())
        buffer = torch.cat((buffer, wav_chunk))
        if buffer.numel() > max_window_samples:
            buffer = buffer[-max_window_samples:]
        if total_samples < next_emit_at:
            continue

        hyp = decode_waveform(buffer)
        now_sec = total_samples / sample_rate
        if hyp:
            context_tail = _tail_words(hyp, context_words)
            display_tail = _tail_words(context_tail, display_words)
            if display_tail and display_tail != last_display:
                updates.append(CaptionUpdate(time_sec=now_sec, text=display_tail))
                last_display = display_tail
        last_hyp = hyp
        next_emit_at += emit_every_samples

    if buffer.numel() >= max(1, min_audio_samples):
        final_hyp = decode_waveform(buffer)
        now_sec = total_samples / sample_rate
        if final_hyp:
            context_tail = _tail_words(final_hyp, context_words)
            display_tail = _tail_words(context_tail, display_words)
            if display_tail and display_tail != last_display:
                updates.append(CaptionUpdate(time_sec=now_sec, text=display_tail))
                last_display = display_tail
        last_hyp = final_hyp

    audio_sec = total_samples / sample_rate
    metrics = {
        "audio_sec": float(audio_sec),
        "decode_sec": float(decode_sec),
        "rtf": float(decode_sec / max(audio_sec, 1e-9)),
        "decode_calls": float(decode_calls),
    }
    return last_hyp, updates, metrics


def main() -> None:
    args = parse_args()
    if args.context_words < 1:
        raise ValueError("--context-words must be >= 1")
    if args.display_words < 1:
        raise ValueError("--display-words must be >= 1")
    if args.display_words > args.context_words:
        raise ValueError("--display-words must be <= --context-words")
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

    if not _tool_exists("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH.")
    if not _tool_exists("ffprobe"):
        raise RuntimeError("ffprobe not found on PATH.")

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else video_path.with_name(f"{video_path.stem}.captioned.ctc.mp4")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    autocast_dtype = _autocast_dtype(args.precision, device)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}) or ckpt.get("args", {}) or {}
    if cfg.get("loss_type", "ctc") != "ctc":
        raise ValueError("Checkpoint is not CTC. This script supports only CTC checkpoints.")

    tokenizer = _load_tokenizer(cfg)
    model = _build_model(cfg, int(cfg.get("vocab_size", tokenizer.vocab_size)), device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print(
            f"[WARN] Checkpoint/model mismatch "
            f"(missing={len(missing)}, unexpected={len(unexpected)})."
        )
    model.eval()
    mel_extractor = LogMelSpectrogram(n_mels=int(cfg.get("n_mels", 80)))

    lm_decoder = None
    decode_mode = "ctc-beam"
    if args.lm_path:
        lm_decoder = build_lm_decoder(
            vocab=_ctc_decoder_vocab(tokenizer),
            lm_path=args.lm_path,
            blank_idx=BLANK_IDX,
            alpha=float(args.lm_alpha),
            beta=float(args.lm_beta),
        )
        decode_mode = "ctc+lm"

    with tempfile.TemporaryDirectory(prefix="asr_caption_ctc_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        wav_path = tmp_dir / "audio.wav"
        ass_tmp_path = tmp_dir / "captions.ass"

        _extract_audio_ffmpeg(video_path=video_path, wav_out=wav_path, sample_rate=args.sample_rate)
        waveform = _load_audio(wav_path, args.sample_rate)

        final_hyp, updates, metrics = _streaming_like_decode(
            model=model,
            tokenizer=tokenizer,
            lm_decoder=lm_decoder,
            waveform=waveform,
            sample_rate=int(args.sample_rate),
            mel_extractor=mel_extractor,
            beam_size=int(args.beam_size),
            beam_token_prune=int(args.beam_token_prune),
            lm_beam_width=int(args.lm_beam_width),
            chunk_ms=float(args.chunk_ms),
            window_sec=float(args.window_sec),
            emit_interval_sec=float(args.emit_interval_sec),
            min_audio_sec=float(args.min_audio_sec),
            context_words=int(args.context_words),
            display_words=int(args.display_words),
            device=device,
            autocast_dtype=autocast_dtype,
        )

        width, height, video_sec = _probe_video(video_path)
        total_sec = max(float(metrics["audio_sec"]), video_sec, 0.0)
        segments = _build_caption_segments(
            final_hyp=final_hyp,
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
    print(f"precision                : {args.precision}")
    print(f"decode_mode              : {decode_mode}")
    if args.lm_path:
        print(f"lm_path                  : {Path(args.lm_path).expanduser().resolve()}")
        print(f"lm_alpha                 : {args.lm_alpha}")
        print(f"lm_beta                  : {args.lm_beta}")
        print(f"lm_beam_width            : {args.lm_beam_width}")
    print(f"beam_size                : {args.beam_size}")
    print(f"beam_token_prune         : {args.beam_token_prune}")
    print(f"chunk_ms                 : {args.chunk_ms}")
    print(f"window_sec               : {args.window_sec}")
    print(f"emit_interval_sec        : {args.emit_interval_sec}")
    print(f"min_audio_sec            : {args.min_audio_sec}")
    print(f"context_words            : {args.context_words}")
    print(f"display_words            : {args.display_words}")
    print(f"caption_segments         : {len(segments)}")
    print(f"decode_calls             : {int(metrics['decode_calls'])}")
    print(f"audio_sec                : {metrics['audio_sec']:.3f}")
    print(f"decode_sec               : {metrics['decode_sec']:.3f}")
    print(f"rtf                      : {metrics['rtf']:.4f}")
    print(f"hyp                      : {final_hyp}")


if __name__ == "__main__":
    main()
