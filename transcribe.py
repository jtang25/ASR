import argparse
from pathlib import Path

import torch
import torchaudio

from decoding import beam_search_decode
from model import ConformerASR
from preprocessing import BLANK_IDX, VOCAB_SIZE, LogMelSpectrogram, tokens_to_text


LOSSLESS_EXTENSIONS = {".wav", ".wave", ".flac"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CTC beam-search transcription with a trained Conformer model."
    )
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/best.pt",
        help="Path to model checkpoint (.pt).",
    )
    parser.add_argument(
        "--audio",
        default="./audio/example.wav",
        help="Path to input audio file (.wav/.wave/.flac).",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=10,
        help="Beam size for CTC prefix beam search.",
    )
    parser.add_argument(
        "--beam-token-prune",
        type=int,
        default=0,
        help="Per-frame top-k token pruning (0 disables).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=0,
        help="Max decoder length after subsampling (0 uses checkpoint/default).",
    )
    parser.add_argument(
        "--chunk-overlap-frames",
        type=int,
        default=40,
        help="Overlap between inference chunks in input mel frames.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate for loaded audio.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Compute device to use.",
    )
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _validate_args(args: argparse.Namespace) -> None:
    if args.beam_size < 1:
        raise ValueError("--beam-size must be >= 1")
    if args.beam_token_prune < 0:
        raise ValueError("--beam-token-prune must be >= 0")
    if args.max_seq_len < 0:
        raise ValueError("--max-seq-len must be >= 0")
    if args.chunk_overlap_frames < 0:
        raise ValueError("--chunk-overlap-frames must be >= 0")
    if args.sample_rate < 1:
        raise ValueError("--sample-rate must be >= 1")


def _output_len_after_subsampling(model: ConformerASR, input_frames: int) -> int:
    lengths = torch.tensor([input_frames], dtype=torch.long)
    return int(model.subsampling.get_output_lengths(lengths)[0].item())


def _max_input_frames_for_output_limit(model: ConformerASR, max_output_len: int) -> int:
    if max_output_len < 1:
        raise ValueError("max_output_len must be >= 1")

    lo, hi = 1, max_output_len * 4
    while _output_len_after_subsampling(model, hi) <= max_output_len:
        hi *= 2
        if hi > 10_000_000:
            break

    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _output_len_after_subsampling(model, mid) <= max_output_len:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _load_lossless_audio(audio_path_: str, sample_rate_: int = 16000) -> torch.Tensor:
    path = Path(audio_path_)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path_}")

    if path.suffix.lower() not in LOSSLESS_EXTENSIONS:
        allowed = ", ".join(sorted(LOSSLESS_EXTENSIONS))
        raise ValueError(f"Unsupported format: {path.suffix}. Use one of: {allowed}")

    waveform, sr = torchaudio.load(str(path))
    if waveform.numel() == 0:
        raise ValueError(f"Audio file is empty: {audio_path_}")

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)

    if sr != sample_rate_:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate_)
    return waveform


@torch.no_grad()
def _infer_with_chunking(
    model: ConformerASR,
    mel: torch.Tensor,
    device: torch.device,
    max_output_len: int,
    overlap_frames: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    total_frames = mel.size(0)
    chunk_frames = _max_input_frames_for_output_limit(model, max_output_len)

    if total_frames <= chunk_frames:
        mel_batch = mel.unsqueeze(0).to(device)
        mel_lengths = torch.tensor([total_frames], dtype=torch.long, device=device)
        log_probs, output_lengths = model(mel_batch, mel_lengths)
        return log_probs.detach().cpu(), output_lengths.detach().cpu(), chunk_frames

    if overlap_frames >= chunk_frames:
        raise ValueError(
            f"overlap_frames ({overlap_frames}) must be smaller than chunk size ({chunk_frames})"
        )

    stride = chunk_frames - overlap_frames
    overlap_out = (
        _output_len_after_subsampling(model, overlap_frames) if overlap_frames > 0 else 0
    )

    merged_chunks: list[torch.Tensor] = []
    start = 0
    while start < total_frames:
        end = min(start + chunk_frames, total_frames)
        chunk = mel[start:end].unsqueeze(0).to(device)
        chunk_len = torch.tensor([end - start], dtype=torch.long, device=device)
        chunk_log_probs, chunk_out_lens = model(chunk, chunk_len)

        valid = int(chunk_out_lens[0].item())
        chunk_seq = chunk_log_probs[0, :valid].detach().cpu()

        if merged_chunks and overlap_out > 0:
            if overlap_out < chunk_seq.size(0):
                chunk_seq = chunk_seq[overlap_out:]
            else:
                chunk_seq = chunk_seq[:0]

        if chunk_seq.numel() > 0:
            merged_chunks.append(chunk_seq)

        if end >= total_frames:
            break
        start += stride

    if not merged_chunks:
        vocab_size = model.ctc_head.out_features
        empty = torch.empty(1, 0, vocab_size)
        return empty, torch.tensor([0], dtype=torch.long), chunk_frames

    merged = torch.cat(merged_chunks, dim=0).unsqueeze(0)
    lengths = torch.tensor([merged.size(1)], dtype=torch.long)
    return merged, lengths, chunk_frames


def _load_model(checkpoint: str, device: torch.device) -> tuple[ConformerASR, dict]:
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {}) or ckpt.get("config", {}) or {}

    model = ConformerASR(
        n_mels=ckpt_args.get("n_mels", 80),
        d_model=ckpt_args.get("d_model", 256),
        num_heads=ckpt_args.get("num_heads", 4),
        num_layers=ckpt_args.get("num_layers", 12),
        vocab_size=VOCAB_SIZE,
        conv_kernel_size=ckpt_args.get("conv_kernel", 31),
        max_len=ckpt_args.get("max_len", 2048),
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt_args


def main() -> None:
    args = parse_args()
    _validate_args(args)

    device = _resolve_device(args.device)
    model, ckpt_args = _load_model(args.checkpoint, device)

    waveform = _load_lossless_audio(args.audio, sample_rate_=args.sample_rate)

    mel_extractor = LogMelSpectrogram(
        sample_rate=args.sample_rate,
        n_mels=ckpt_args.get("n_mels", 80),
    )
    mel = mel_extractor(waveform)
    mel = (mel - mel.mean()) / (mel.std() + 1e-9)

    max_len_after_subsampling = (
        args.max_seq_len if args.max_seq_len > 0 else ckpt_args.get("max_len", 2048)
    )

    log_probs, output_lengths, chunk_frames = _infer_with_chunking(
        model=model,
        mel=mel,
        device=device,
        max_output_len=max_len_after_subsampling,
        overlap_frames=args.chunk_overlap_frames,
    )

    token_prune = args.beam_token_prune if args.beam_token_prune > 0 else None
    decoded = beam_search_decode(
        log_probs=log_probs,
        lengths=output_lengths,
        beam_size=args.beam_size,
        blank_idx=BLANK_IDX,
        token_prune=token_prune,
    )
    text = tokens_to_text(decoded[0])

    print(f"Device: {device}")
    print(f"Audio: {args.audio}")
    print(f"Mel frames: {mel.size(0)}")
    print(f"Chunk frames (max): {chunk_frames}")
    print(f"Output frames: {int(output_lengths[0].item())}")
    print(f"Transcript: {text}")


if __name__ == "__main__":
    main()
