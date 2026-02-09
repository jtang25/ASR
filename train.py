"""Train a Conformer ASR model (CTC or RNN-T).

Paper defaults:
  - Loss: rnnt
  - Encoder: 17 layers, d=512, 8 heads, kernel=32
  - Decoder: 1-layer LSTM, dim=640, joint_dim=640
  - Tokenizer: 1k SentencePiece unigram (vocab=1024 with blank)
  - Optimizer: Adam (beta1=0.9, beta2=0.98, eps=1e-9)
  - LR: warmup 10k steps, peak=0.05/d, inverse-sqrt decay
  - Regularization: dropout=0.1, weight_decay=1e-6, variational_noise=0.075
  - SpecAugment: F=27, 2 freq masks, 10 time masks with pS=0.05
  - Data: LibriSpeech 960h (train-clean-100 + train-clean-360 + train-other-500)
"""

from __future__ import annotations

import argparse
import csv
import datetime
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

try:
    import torchaudio
    _RNNT_LOSS_AVAILABLE = hasattr(torchaudio.functional, "rnnt_loss")
except ImportError:
    _RNNT_LOSS_AVAILABLE = False

from decoding import (
    beam_search_decode,
    build_lm_decoder,
    lm_beam_search_decode,
    rnnt_greedy_decode,
    rnnt_beam_search,
)
from model import ConformerASR, ConformerTransducer
from preprocessing import LogMelSpectrogram, get_dataloader
from tokenizer import BLANK_IDX, CharTokenizer, SentencePieceTokenizer


# ===================================================================
# CLI
# ===================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Conformer CTC / RNN-T ASR.")

    # Data
    p.add_argument("--data-root", default="./data")
    p.add_argument("--train-splits", nargs="+",
                   default=["train-clean-100", "train-clean-360", "train-other-500"])
    p.add_argument("--val-split", default="dev-other")

    # Tokenizer
    p.add_argument("--tokenizer", default="sp", choices=["char", "sp"],
                   help="char = 29-token character vocab, sp = SentencePiece.")
    p.add_argument("--sp-model", default="./tokenizer/sp_1k.model",
                   help="Path to trained SentencePiece .model file.")

    # Model
    p.add_argument("--loss-type", default="rnnt", choices=["ctc", "rnnt"])
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=17)
    p.add_argument("--n-mels", type=int, default=80)
    p.add_argument("--conv-kernel", type=int, default=32)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument(
        "--streaming-mode",
        action="store_true",
        help=(
            "Enable streaming-friendly encoder masking. "
            "Uses chunked left-context attention and optional right context."
        ),
    )
    p.add_argument(
        "--streaming-chunk-size",
        type=int,
        default=0,
        help="Chunk size in encoder time steps after subsampling (0 disables streaming mask).",
    )
    p.add_argument(
        "--streaming-left-context-chunks",
        type=int,
        default=-1,
        help="How many previous chunks each chunk can attend to (-1 = unlimited).",
    )
    p.add_argument(
        "--streaming-right-context",
        type=int,
        default=0,
        help="Per-frame lookahead in encoder steps for streaming attention.",
    )
    p.add_argument(
        "--streaming-causal-conv",
        action="store_true",
        help="Use causal depthwise convolution in Conformer conv modules.",
    )
    p.add_argument(
        "--no-streaming-causal-conv",
        action="store_false",
        dest="streaming_causal_conv",
    )
    p.set_defaults(streaming_causal_conv=False)

    # RNN-T decoder
    p.add_argument("--pred-embed-dim", type=int, default=256)
    p.add_argument("--pred-hidden-dim", type=int, default=640)
    p.add_argument("--pred-num-layers", type=int, default=1)
    p.add_argument("--joint-dim", type=int, default=640)

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32,
                   help="Per-GPU batch size. Paper uses effective ~256 across TPU pods.")
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--optimizer", default="adam", choices=["adam", "adamw"])
    p.add_argument("--fused-optimizer", action="store_true", default=True)
    p.add_argument("--no-fused-optimizer", action="store_false", dest="fused_optimizer")
    p.add_argument("--precision", default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--tf32", action="store_true", default=True,
                   help="Enable TensorFloat-32 matmul/convolution on CUDA for throughput.")
    p.add_argument("--no-tf32", action="store_false", dest="tf32")
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument("--compile-mode", default="max-autotune-no-cudagraphs",
                   choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])

    # LR schedule
    p.add_argument("--lr-schedule", default="paper", choices=["paper", "cosine"])
    p.add_argument("--lr", type=float, default=3e-4, help="Peak LR (cosine mode only).")
    p.add_argument("--paper-peak-factor", type=float, default=0.05,
                   help="Paper mode: peak_lr = factor / d_model.")
    p.add_argument("--warmup-steps", type=int, default=10_000)
    p.add_argument("--min-lr", type=float, default=1e-6, help="Cosine mode floor.")

    # Regularization
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--variational-noise", type=float, default=0.075,
                   help="Std of Gaussian noise added to weights each step (0 disables).")

    # Decoding / eval
    p.add_argument("--beam-size", type=int, default=20)
    p.add_argument("--beam-token-prune", type=int, default=0)
    p.add_argument("--eval-lm-path", default=None)
    p.add_argument("--eval-lm-alpha", type=float, default=0.5)
    p.add_argument("--eval-lm-beta", type=float, default=1.0)
    p.add_argument("--eval-lm-beam-width", type=int, default=128)
    p.add_argument(
        "--rnnt-eval-decoder",
        default="greedy",
        choices=["greedy", "beam", "both"],
        help=(
            "RNN-T validation WER decoder. "
            "'greedy' is fastest, 'beam' is higher quality, 'both' logs both and uses beam WER as primary."
        ),
    )
    p.add_argument("--rnnt-eval-beam-size", type=int, default=8)
    p.add_argument("--rnnt-eval-beam-topk", type=int, default=10)
    p.add_argument("--rnnt-eval-max-symbols-per-step", type=int, default=10)

    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--pin-memory", action="store_true", default=True)
    p.add_argument("--no-pin-memory", action="store_false", dest="pin_memory")
    p.add_argument("--persistent-workers", action="store_true", default=True)
    p.add_argument("--no-persistent-workers", action="store_false", dest="persistent_workers")
    p.add_argument("--eval-sample-decode", action="store_true", default=True)
    p.add_argument("--no-eval-sample-decode", action="store_false", dest="eval_sample_decode")
    p.add_argument("--eval-every", type=int, default=1,
                   help="Run full validation every N epochs (1 = every epoch).")
    p.add_argument(
        "--eval-sample-path",
        default=None,
        help="Optional path to a .flac file for epoch-end sample decoding.",
    )
    p.add_argument(
        "--rnnt-fused-log-softmax",
        action="store_true",
        dest="rnnt_fused_log_softmax",
        help="Use fused log_softmax path inside torchaudio rnnt_loss (can be unstable on some CUDA builds).",
    )
    p.add_argument(
        "--no-rnnt-fused-log-softmax",
        action="store_false",
        dest="rnnt_fused_log_softmax",
        help="Disable fused log_softmax and apply log_softmax explicitly before rnnt_loss.",
    )
    p.add_argument(
        "--rnnt-loss-device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Where to run torchaudio rnnt_loss. Use cpu as a stability fallback for CUDA kernel crashes.",
    )
    p.add_argument(
        "--rnnt-max-batch-tu",
        type=int,
        default=1_500_000,
        help=(
            "Split RNNT loss computation into smaller chunks so (chunk_batch * max_T * max_U) "
            "stays under this limit. Set 0 to disable chunking."
        ),
    )
    p.set_defaults(rnnt_fused_log_softmax=False)

    # Checkpointing
    p.add_argument("--ckpt-dir", default="./checkpoints")
    p.add_argument("--log-csv", default=None)
    p.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save last.pt every N epochs (always on final epoch).",
    )
    p.add_argument("--resume-path", default=None)
    p.add_argument(
        "--init-encoder-from",
        default=None,
        help=(
            "Path to checkpoint used to initialize only encoder weights. "
            "Useful for CTC->RNN-T transfer. Ignored when full resume is used."
        ),
    )
    p.add_argument("--reset-scheduler-on-resume", action="store_true")
    p.add_argument("--override-lr-on-resume", action="store_true")
    p.add_argument("--auto-resume", action="store_true", default=True)
    p.add_argument("--no-auto-resume", action="store_false", dest="auto_resume")

    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--download", action="store_true", default=True)
    p.add_argument("--no-download", action="store_false", dest="download")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--sync-bn", action="store_true", default=False,
                   help="Enable SyncBatchNorm in distributed training (slower).")
    p.add_argument("--ddp-bucket-cap-mb", type=int, default=100)
    p.add_argument("--ddp-grad-as-bucket-view", action="store_true", default=True)
    p.add_argument("--no-ddp-grad-as-bucket-view", action="store_false", dest="ddp_grad_as_bucket_view")
    p.add_argument(
        "--ddp-timeout-sec",
        type=int,
        default=300,
        help=(
            "Distributed collective timeout in seconds. Lower values fail faster on hangs "
            "to reduce wasted GPU time."
        ),
    )
    p.add_argument(
        "--ddp-broadcast-buffers",
        action="store_true",
        default=False,
        help=(
            "Broadcast module buffers (e.g., BatchNorm running stats) from rank 0 before "
            "forward. Disabled by default to avoid syncing large static buffers every step."
        ),
    )
    p.add_argument(
        "--no-ddp-broadcast-buffers",
        action="store_false",
        dest="ddp_broadcast_buffers",
    )
    p.add_argument("--paper-strict", action="store_true",
                   help="Fail fast if key hyperparameters deviate from Conformer Transducer Large paper defaults.")

    return p.parse_args()


# ===================================================================
# Utilities
# ===================================================================

def _resolve_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _init_distributed(arg: str, timeout_sec: int = 300):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_dist = world_size > 1
    if not is_dist:
        return False, rank, local_rank, world_size, _resolve_device(arg)
    use_cuda = arg == "cuda" or (arg == "auto" and torch.cuda.is_available())
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if not dist.is_initialized():
        timeout = datetime.timedelta(seconds=max(1, int(timeout_sec)))
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timeout,
        )
    return True, rank, local_rank, world_size, device


def _cleanup(is_dist: bool):
    if is_dist and dist.is_initialized():
        dist.destroy_process_group()


def _dist_barrier(device: torch.device):
    if not dist.is_initialized():
        return
    if device.type == "cuda":
        dev_idx = device.index if device.index is not None else torch.cuda.current_device()
        dist.barrier(device_ids=[dev_idx])
    else:
        dist.barrier()


def _extract_encoder_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract encoder.* parameters from a checkpoint state_dict."""
    for prefix in ("encoder.", "module.encoder."):
        enc_state = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if enc_state:
            return enc_state
    return {}


def _init_encoder_from_checkpoint(
    model_raw: nn.Module,
    ckpt_path: str,
    device: torch.device,
    is_main: bool,
) -> None:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Encoder init checkpoint not found: {ckpt_path}")
    ckpt_obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt_obj, dict) and "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        src_state = ckpt_obj["model"]
    elif isinstance(ckpt_obj, dict):
        src_state = ckpt_obj
    else:
        raise ValueError(f"Unsupported checkpoint format for encoder init: {ckpt_path}")

    enc_state = _extract_encoder_state_dict(src_state)
    if not enc_state:
        raise ValueError(
            f"No encoder.* weights found in checkpoint: {ckpt_path}. "
            "Expected a CTC/RNN-T model checkpoint with encoder parameters."
        )

    missing, unexpected = model_raw.encoder.load_state_dict(enc_state, strict=False)
    if is_main:
        loaded = len(enc_state)
        print(f"Initialized encoder from: {ckpt_path} | loaded_keys={loaded}")
        if missing:
            print(f"  [encoder-init] missing keys: {len(missing)}")
        if unexpected:
            print(f"  [encoder-init] unexpected keys: {len(unexpected)}")


def _reduce_sum(val: float, device: torch.device, is_dist: bool) -> float:
    if not is_dist:
        return val
    t = torch.tensor(val, device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def _set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast_dtype(precision: str) -> torch.dtype | None:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return None


def _configure_runtime(args: argparse.Namespace, device: torch.device, is_main: bool):
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    if args.precision in ("fp16", "bf16") or args.tf32:
        torch.set_float32_matmul_precision("high")
    if is_main:
        print(f"Runtime: precision={args.precision} tf32={args.tf32} cudnn_benchmark={torch.backends.cudnn.benchmark}")


def _paper_fidelity_mismatches(args: argparse.Namespace, vocab_size: int | None = None) -> list[str]:
    mismatches: list[str] = []
    streaming_enabled = bool(getattr(args, "streaming_mode", False) or int(getattr(args, "streaming_chunk_size", 0)) > 0)
    if streaming_enabled:
        mismatches.append("streaming_mode=True (expected False for paper baseline)")

    expected_str = {
        "loss_type": "rnnt",
        "tokenizer": "sp",
        "optimizer": "adam",
        "lr_schedule": "paper",
        "precision": "bf16",
    }
    for key, expected in expected_str.items():
        actual = getattr(args, key)
        if actual != expected:
            mismatches.append(f"{key}={actual!r} (expected {expected!r})")

    expected_int = {
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 17,
        "conv_kernel": 32,
        "n_mels": 80,
        "pred_hidden_dim": 640,
        "pred_num_layers": 1,
        "joint_dim": 640,
        "warmup_steps": 10_000,
    }
    for key, expected in expected_int.items():
        actual = int(getattr(args, key))
        if actual != expected:
            mismatches.append(f"{key}={actual} (expected {expected})")

    expected_float = {
        "dropout": 0.1,
        "variational_noise": 0.075,
        "weight_decay": 1e-6,
        "paper_peak_factor": 0.05,
    }
    for key, expected in expected_float.items():
        actual = float(getattr(args, key))
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            mismatches.append(f"{key}={actual} (expected {expected})")

    expected_splits = ["train-clean-100", "train-clean-360", "train-other-500"]
    if list(args.train_splits) != expected_splits:
        mismatches.append(f"train_splits={args.train_splits!r} (expected {expected_splits!r})")

    if args.rnnt_loss_device != "cuda":
        mismatches.append(f"rnnt_loss_device={args.rnnt_loss_device!r} (expected 'cuda')")

    if vocab_size is not None and int(vocab_size) != 1024:
        mismatches.append(f"vocab_size={vocab_size} (expected 1024 incl blank)")

    return mismatches


# ===================================================================
# Variational noise (Graves 2012)
# ===================================================================

def apply_variational_noise(model: nn.Module, std: float) -> dict[str, torch.Tensor]:
    """Add Gaussian noise to all trainable parameters. Returns noise dict for removal."""
    noise_dict: dict[str, torch.Tensor] = {}
    if std <= 0:
        return noise_dict
    for name, param in model.named_parameters():
        if param.requires_grad:
            noise = torch.randn_like(param.data) * std
            noise_dict[name] = noise
            param.data.add_(noise)
    return noise_dict


def remove_variational_noise(model: nn.Module, noise_dict: dict[str, torch.Tensor]):
    """Remove previously applied noise."""
    for name, param in model.named_parameters():
        if name in noise_dict:
            param.data.sub_(noise_dict[name])


# ===================================================================
# WER computation
# ===================================================================

def edit_distance(ref: list[str], hyp: list[str]) -> int:
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j - 1], prev[j], dp[j - 1])
    return dp[m]


def _compute_wer_ctc(
    log_probs, output_lengths, targets, target_lengths,
    tokenizer, beam_size, token_prune, lm_decoder, lm_beam_width,
) -> tuple[int, int]:
    if lm_decoder is not None:
        hyp_texts = lm_beam_search_decode(log_probs, output_lengths, lm_decoder, lm_beam_width)
    else:
        decoded = beam_search_decode(
            log_probs, output_lengths,
            beam_size=beam_size, blank_idx=BLANK_IDX, token_prune=token_prune,
        )
        hyp_texts = [tokenizer.decode(tokens) for tokens in decoded]

    total_errors = 0
    total_words = 0
    for i, hyp_text in enumerate(hyp_texts):
        ref_len = int(target_lengths[i].item())
        ref_ids = targets[i, :ref_len].tolist()
        ref_text = tokenizer.decode(ref_ids)
        ref_words = ref_text.split()
        hyp_words = hyp_text.split()
        total_words += max(len(ref_words), 1)
        total_errors += edit_distance(ref_words, hyp_words)
    return total_errors, total_words


def _compute_wer_rnnt(
    model, enc_out, enc_lengths, targets, target_lengths, tokenizer,
    decoder: str = "greedy",
    beam_size: int = 8,
    beam_topk: int = 10,
    max_symbols_per_step: int = 10,
) -> tuple[int, int]:
    if decoder == "beam":
        decoded = rnnt_beam_search(
            model,
            enc_out,
            enc_lengths,
            blank_idx=model.blank_idx,
            beam_size=max(1, int(beam_size)),
            max_symbols_per_step=max(1, int(max_symbols_per_step)),
            top_k_tokens=max(1, int(beam_topk)),
        )
    else:
        decoded = rnnt_greedy_decode(
            model, enc_out, enc_lengths, blank_idx=model.blank_idx, max_symbols_per_step=max(1, int(max_symbols_per_step)),
        )
    total_errors = 0
    total_words = 0
    for i, hyp_ids in enumerate(decoded):
        ref_len = int(target_lengths[i].item())
        ref_ids = targets[i, :ref_len].tolist()
        ref_text = tokenizer.decode(ref_ids)
        hyp_text = tokenizer.decode(hyp_ids)
        ref_words = ref_text.split()
        hyp_words = hyp_text.split()
        total_words += max(len(ref_words), 1)
        total_errors += edit_distance(ref_words, hyp_words)
    return total_errors, total_words


def _find_eval_sample_flac(data_root: str, split: str) -> str | None:
    split_root = Path(data_root) / "LibriSpeech" / split
    if not split_root.exists():
        return None
    flacs = sorted(split_root.rglob("*.flac"))
    return str(flacs[0]) if flacs else None


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


@torch.no_grad()
def decode_eval_sample(
    model,
    device: torch.device,
    loss_type: str,
    tokenizer,
    sample_path: str,
    mel_extractor: LogMelSpectrogram,
    beam_size: int = 10,
    token_prune: int | None = None,
    rnnt_decoder: str = "greedy",
    rnnt_beam_size: int = 8,
    rnnt_beam_topk: int = 10,
    rnnt_max_symbols_per_step: int = 10,
) -> tuple[str, str]:
    waveform, sample_rate = torchaudio.load(sample_path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

    mel = mel_extractor(waveform)
    mel = (mel - mel.mean(dim=0, keepdim=True)) / (mel.std(dim=0, keepdim=True) + 1e-9)

    mel = mel.unsqueeze(0).to(device, non_blocking=True)
    mel_lengths = torch.tensor([mel.size(1)], dtype=torch.long, device=device)

    if loss_type == "ctc":
        log_probs, out_lengths = model(mel, mel_lengths)
        decoded = beam_search_decode(
            log_probs,
            out_lengths,
            beam_size=beam_size,
            blank_idx=BLANK_IDX,
            token_prune=token_prune,
        )
        hyp = tokenizer.decode(decoded[0]) if decoded else ""
    else:
        enc_out, enc_lengths = model.encode(mel.float(), mel_lengths)
        if rnnt_decoder == "beam":
            decoded = rnnt_beam_search(
                model,
                enc_out,
                enc_lengths,
                blank_idx=model.blank_idx,
                beam_size=max(1, int(rnnt_beam_size)),
                max_symbols_per_step=max(1, int(rnnt_max_symbols_per_step)),
                top_k_tokens=max(1, int(rnnt_beam_topk)),
            )
        else:
            decoded = rnnt_greedy_decode(
                model, enc_out, enc_lengths, blank_idx=model.blank_idx, max_symbols_per_step=max(1, int(rnnt_max_symbols_per_step)),
            )
        hyp = tokenizer.decode(decoded[0]) if decoded else ""

    ref = _read_librispeech_reference(sample_path)
    return ref, hyp


# ===================================================================
# LR Schedulers
# ===================================================================

class PaperTransformerSchedule:
    """Warmup then inverse-sqrt. peak_lr = peak_factor / d_model."""

    def __init__(self, optimizer, d_model: int, warmup_steps: int, peak_factor: float = 0.05):
        self.optimizer = optimizer
        self.d_model = int(d_model)
        self.warmup_steps = max(1, int(warmup_steps))
        self.peak_factor = float(peak_factor)
        self.lr_peak = self.peak_factor / max(1, self.d_model)
        self.step_count = 0
        # Set initial LR to step-1 value so first optimizer step is not wasted
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.lr_peak / self.warmup_steps

    def step(self) -> float:
        self.step_count += 1
        lr = self._compute_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def get_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _compute_lr(self) -> float:
        s = max(1, self.step_count)
        w = self.warmup_steps
        if s <= w:
            return self.lr_peak * (s / w)
        return self.lr_peak * math.sqrt(w / s)

    def state_dict(self) -> dict:
        return {
            "name": "paper", "step_count": self.step_count,
            "d_model": self.d_model, "warmup_steps": self.warmup_steps,
            "peak_factor": self.peak_factor, "lr_peak": self.lr_peak,
        }

    def load_state_dict(self, state: dict):
        self.step_count = int(state.get("step_count", 0))
        self.d_model = int(state.get("d_model", self.d_model))
        self.warmup_steps = max(1, int(state.get("warmup_steps", self.warmup_steps)))
        self.peak_factor = float(state.get("peak_factor", self.peak_factor))
        self.lr_peak = float(state.get("lr_peak", self.peak_factor / max(1, self.d_model)))
        for pg in self.optimizer.param_groups:
            pg["lr"] = self._compute_lr()


class WarmupCosineScheduler:
    """Linear warmup then cosine decay to min_lr."""

    def __init__(self, optimizer, warmup_steps, total_steps, peak_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps = int(total_steps)
        self.peak_lr = float(peak_lr)
        self.min_lr = float(min_lr)
        self.step_count = 0

    def step(self) -> float:
        self.step_count += 1
        lr = self._compute_lr(self.step_count)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def get_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _compute_lr(self, s: int) -> float:
        if s <= self.warmup_steps:
            return self.peak_lr * s / self.warmup_steps
        progress = (s - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def state_dict(self) -> dict:
        return {
            "name": "cosine", "step_count": self.step_count,
            "warmup_steps": self.warmup_steps, "total_steps": self.total_steps,
            "peak_lr": self.peak_lr, "min_lr": self.min_lr,
        }

    def load_state_dict(self, state: dict):
        self.step_count = int(state.get("step_count", 0))
        self.warmup_steps = max(1, int(state.get("warmup_steps", self.warmup_steps)))
        self.total_steps = int(state.get("total_steps", self.total_steps))
        self.peak_lr = float(state.get("peak_lr", self.peak_lr))
        self.min_lr = float(state.get("min_lr", self.min_lr))
        for pg in self.optimizer.param_groups:
            pg["lr"] = self._compute_lr(self.step_count)


# ===================================================================
# Combined loader (multiple splits)
# ===================================================================

def combined_loader(loaders, shuffle=True):
    iters = [iter(l) for l in loaders]
    order: list[int] = []
    for i, l in enumerate(loaders):
        order += [i] * len(l)
    if shuffle:
        random.shuffle(order)
    for i in order:
        yield next(iters[i])


def combined_len(loaders) -> int:
    return sum(len(l) for l in loaders)


# ===================================================================
# CSV logging
# ===================================================================

CSV_FIELDS = [
    "record_type", "epoch", "step", "global_step", "optimizer_step",
    "train_step_loss", "train_running_loss", "train_epoch_loss",
    "val_loss", "val_wer", "lr", "elapsed_sec", "best_wer",
]


def append_csv_rows(csv_path: str, rows: list[dict]):
    if not rows:
        return
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists or os.path.getsize(csv_path) == 0:
            w.writeheader()
        w.writerows(rows)


# ===================================================================
# RNN-T loss helper
# ===================================================================

def _build_rnnt_chunks(
    logit_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    max_batch_tu: int,
) -> list[tuple[int, int, int, int]]:
    """Greedy chunking along batch to bound chunk_batch * max_T * max_U."""
    batch_size = int(logit_lengths.size(0))
    t_list = logit_lengths.detach().to("cpu", dtype=torch.int64).tolist()
    # U in logits includes the prepended blank step.
    u_list = [u + 1 for u in target_lengths.detach().to("cpu", dtype=torch.int64).tolist()]

    if batch_size == 0:
        return []

    if max_batch_tu <= 0:
        return [(0, batch_size, max(t_list), max(u_list))]

    chunks: list[tuple[int, int, int, int]] = []
    start = 0
    while start < batch_size:
        end = start + 1
        t_max = int(t_list[start])
        u_max = int(u_list[start])
        while end < batch_size:
            next_t = max(t_max, int(t_list[end]))
            next_u = max(u_max, int(u_list[end]))
            estimate = (end - start + 1) * next_t * next_u
            if estimate > max_batch_tu:
                break
            t_max, u_max = next_t, next_u
            end += 1
        chunks.append((start, end, t_max, u_max))
        start = end
    return chunks


def _rnnt_loss_chunked(
    logits: torch.Tensor,
    targets: torch.Tensor,
    logit_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_idx: int,
    fused_log_softmax: bool,
    loss_device: str,
    max_batch_tu: int,
) -> tuple[torch.Tensor, int]:
    """Compute RNNT loss, chunking long/large batches to avoid CUDA kernel failures."""
    chunks = _build_rnnt_chunks(logit_lengths, target_lengths, max_batch_tu)
    if not chunks:
        raise RuntimeError("Empty batch passed to RNNT loss.")

    total = None
    total_weight = 0
    for s, e, t_max, u_max in chunks:
        logits_chunk = logits[s:e, :t_max, :u_max, :].float()
        targets_chunk = targets[s:e, : max(0, u_max - 1)].int()
        logit_len_chunk = logit_lengths[s:e].clamp(min=1, max=t_max).int()
        target_len_chunk = target_lengths[s:e].clamp(min=0, max=max(0, u_max - 1)).int()
        chunk_bs = int(e - s)

        if not fused_log_softmax:
            logits_chunk = torch.log_softmax(logits_chunk, dim=-1)

        if loss_device == "cpu":
            logits_chunk = logits_chunk.cpu()
            targets_chunk = targets_chunk.cpu()
            logit_len_chunk = logit_len_chunk.cpu()
            target_len_chunk = target_len_chunk.cpu()

        chunk_mean = torchaudio.functional.rnnt_loss(
            logits=logits_chunk,
            targets=targets_chunk,
            logit_lengths=logit_len_chunk,
            target_lengths=target_len_chunk,
            blank=blank_idx,
            reduction="mean",
            fused_log_softmax=fused_log_softmax,
        )
        weighted = chunk_mean * max(1, chunk_bs)
        total = weighted if total is None else total + weighted
        total_weight += chunk_bs

    # Preserve chunking invariance: weighted average over chunk means.
    loss = total / max(1, total_weight)
    return loss, len(chunks)


# ===================================================================
# Training loop
# ===================================================================

def train_one_epoch(
    model,
    loader,
    loader_len: int,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    grad_clip: float,
    accum_steps: int,
    epoch_num: int,
    global_step_offset: int,
    loss_type: str,
    var_noise_std: float = 0.0,
    is_distributed: bool = False,
    is_main: bool = True,
    autocast_dtype: torch.dtype | None = None,
    rnnt_fused_log_softmax: bool = False,
    rnnt_loss_device: str = "cuda",
    rnnt_max_batch_tu: int = 1_500_000,
):
    model.train()

    if loss_type == "ctc":
        loss_fn = nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)
    else:
        if not _RNNT_LOSS_AVAILABLE:
            raise RuntimeError(
                "torchaudio.functional.rnnt_loss not available. "
                "Upgrade torchaudio or install a compatible version."
            )

    total_loss = 0.0
    num_batches = 0
    step_rows: list[dict] = []
    optimizer.zero_grad(set_to_none=True)

    # Get the raw model for variational noise (not wrapped DDP)
    raw_model = model.module if isinstance(model, DDP) else model
    rnnt_chunk_count = 0

    for step, (mel, tokens, mel_lengths, token_lengths) in enumerate(loader):
        mel = mel.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)
        mel_lengths = mel_lengths.to(device, non_blocking=True)
        token_lengths = token_lengths.to(device, non_blocking=True)

        is_last = (step + 1) == loader_len
        should_step = ((step + 1) % accum_steps == 0) or is_last

        # Apply variational noise before forward pass
        noise_dict = apply_variational_noise(raw_model, var_noise_std)

        amp_enabled = device.type == "cuda" and autocast_dtype is not None
        use_scaler = (
            scaler.is_enabled()
            and not (loss_type == "rnnt" and rnnt_loss_device == "cpu")
        )
        raw_loss = 0.0

        if loss_type == "ctc":
            sync_ctx = nullcontext()
            if is_distributed and isinstance(model, DDP) and not should_step:
                sync_ctx = model.no_sync()

            with sync_ctx:
                with autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
                    log_probs, output_lengths = model(mel, mel_lengths)
                    log_probs_t = log_probs.transpose(0, 1)  # (T, B, V)
                    output_lengths = output_lengths.clamp(min=1)
                    loss = loss_fn(log_probs_t.float(), tokens, output_lengths, token_lengths)
                    raw_loss = float(loss.item())
                    loss = loss / accum_steps
                    if use_scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
        else:
            # Keep rnnt_loss in FP32 but allow autocast in model forward for throughput.
            rnnt_amp = amp_enabled and rnnt_loss_device == "cuda"
            est_enc_lengths = raw_model.encoder.get_output_lengths(mel_lengths).clamp(min=1).int()
            batch_chunks = _build_rnnt_chunks(est_enc_lengths, token_lengths, rnnt_max_batch_tu)
            if not batch_chunks:
                raise RuntimeError("RNNT chunking produced no chunks for a non-empty batch.")
            batch_size = max(1, int(tokens.size(0)))
            n_batch_chunks = len(batch_chunks)

            # Important for DDP: even if chunk counts differ by rank, only one chunk per outer
            # batch performs synchronized backward when `should_step` is True.
            for chunk_idx, (s, e, _t_max, _u_max) in enumerate(batch_chunks):
                do_sync_chunk = should_step and (chunk_idx == (n_batch_chunks - 1))
                chunk_sync_ctx = nullcontext()
                if is_distributed and isinstance(model, DDP) and not do_sync_chunk:
                    chunk_sync_ctx = model.no_sync()

                with chunk_sync_ctx:
                    with autocast(device_type=device.type, dtype=autocast_dtype, enabled=rnnt_amp):
                        chunk_bs = e - s
                        logits, enc_lengths = model(
                            mel[s:e],
                            mel_lengths[s:e],
                            tokens[s:e],
                            token_lengths[s:e],
                        )
                        enc_lengths = enc_lengths.clamp(min=1).int()
                        loss_chunk, n_chunks = _rnnt_loss_chunked(
                            logits=logits,
                            targets=tokens[s:e],
                            logit_lengths=enc_lengths,
                            target_lengths=token_lengths[s:e],
                            blank_idx=BLANK_IDX,
                            fused_log_softmax=rnnt_fused_log_softmax,
                            loss_device=rnnt_loss_device,
                            max_batch_tu=rnnt_max_batch_tu,
                        )

                    rnnt_chunk_count += n_chunks
                    weighted_loss = loss_chunk * (float(chunk_bs) / float(batch_size))
                    raw_loss += float(weighted_loss.detach().item())
                    loss_for_backward = weighted_loss / accum_steps
                    if use_scaler:
                        scaler.scale(loss_for_backward).backward()
                    else:
                        loss_for_backward.backward()

        # Remove variational noise after backward
        remove_variational_noise(raw_model, noise_dict)

        if should_step:
            if use_scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += raw_loss
        num_batches += 1
        running_loss = total_loss / max(1, num_batches)
        lr_now = scheduler.get_lr()
        gs = global_step_offset + step + 1

        if is_main:
            step_rows.append({
                "record_type": "train_step", "epoch": epoch_num,
                "step": step + 1, "global_step": gs,
                "optimizer_step": scheduler.step_count,
                "train_step_loss": raw_loss, "train_running_loss": running_loss,
                "train_epoch_loss": "", "val_loss": "", "val_wer": "",
                "lr": lr_now, "elapsed_sec": "", "best_wer": "",
            })
            if (step + 1) % 100 == 0:
                if loss_type == "rnnt":
                    avg_chunks = rnnt_chunk_count / max(1, num_batches)
                    print(f"  step {step+1:5d} | loss {running_loss:.4f} | lr {lr_now:.2e} | rnnt_chunks {avg_chunks:.2f}")
                else:
                    print(f"  step {step+1:5d} | loss {running_loss:.4f} | lr {lr_now:.2e}")

    total_loss = _reduce_sum(total_loss, device, is_distributed)
    nb_global = _reduce_sum(float(num_batches), device, is_distributed)
    return total_loss / max(nb_global, 1.0), step_rows, num_batches


@torch.no_grad()
def evaluate(
    model,
    loader,
    device: torch.device,
    loss_type: str,
    tokenizer,
    beam_size: int = 10,
    token_prune: int | None = None,
    lm_decoder=None,
    lm_beam_width: int = 100,
    autocast_dtype: torch.dtype | None = None,
    rnnt_fused_log_softmax: bool = False,
    rnnt_loss_device: str = "cuda",
    rnnt_max_batch_tu: int = 1_500_000,
    rnnt_eval_decoder: str = "greedy",
    rnnt_eval_beam_size: int = 8,
    rnnt_eval_beam_topk: int = 10,
    rnnt_eval_max_symbols_per_step: int = 10,
    is_distributed: bool = False,
) -> tuple[float, float, float, float]:
    model.eval()

    if loss_type == "ctc":
        loss_fn = nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)

    total_loss = 0.0
    total_errors = 0
    total_words = 0
    total_errors_greedy = 0
    total_words_greedy = 0
    total_errors_beam = 0
    total_words_beam = 0
    num_batches = 0
    amp_enabled = device.type == "cuda" and autocast_dtype is not None
    want_greedy = loss_type == "rnnt" and rnnt_eval_decoder in ("greedy", "both")
    want_beam = loss_type == "rnnt" and rnnt_eval_decoder in ("beam", "both")

    for mel, tokens, mel_lengths, token_lengths in loader:
        mel = mel.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)
        mel_lengths = mel_lengths.to(device, non_blocking=True)
        token_lengths = token_lengths.to(device, non_blocking=True)

        if loss_type == "ctc":
            with autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
                log_probs, output_lengths = model(mel, mel_lengths)
            log_probs_t = log_probs.transpose(0, 1)
            output_lengths = output_lengths.clamp(min=1)
            loss = loss_fn(log_probs_t.float(), tokens, output_lengths, token_lengths)
            errors, words = _compute_wer_ctc(
                log_probs, output_lengths, tokens, token_lengths,
                tokenizer, beam_size, token_prune, lm_decoder, lm_beam_width,
            )
        else:
            # RNN-T
            rnnt_amp = amp_enabled and rnnt_loss_device == "cuda"
            with autocast(device_type=device.type, dtype=autocast_dtype, enabled=rnnt_amp):
                est_enc_lengths = model.encoder.get_output_lengths(mel_lengths).clamp(min=1).int()
                batch_chunks = _build_rnnt_chunks(est_enc_lengths, token_lengths, rnnt_max_batch_tu)
                batch_size = max(1, int(tokens.size(0)))
                loss_val = 0.0
                for s, e, _t_max, _u_max in batch_chunks:
                    chunk_bs = e - s
                    logits, enc_lengths = model(
                        mel[s:e],
                        mel_lengths[s:e],
                        tokens[s:e],
                        token_lengths[s:e],
                    )
                    enc_lengths = enc_lengths.clamp(min=1).int()
                    loss_chunk, _ = _rnnt_loss_chunked(
                        logits=logits,
                        targets=tokens[s:e],
                        logit_lengths=enc_lengths,
                        target_lengths=token_lengths[s:e],
                        blank_idx=BLANK_IDX,
                        fused_log_softmax=rnnt_fused_log_softmax,
                        loss_device=rnnt_loss_device,
                        max_batch_tu=rnnt_max_batch_tu,
                    )
                    loss_val += float(loss_chunk.item()) * (float(chunk_bs) / float(batch_size))
            enc_out, enc_lens = model.encode(mel, mel_lengths)
            enc_lens = enc_lens.clamp(min=1)
            errors_g = words_g = errors_b = words_b = 0
            if want_greedy:
                errors_g, words_g = _compute_wer_rnnt(
                    model,
                    enc_out,
                    enc_lens,
                    tokens,
                    token_lengths,
                    tokenizer,
                    decoder="greedy",
                    max_symbols_per_step=rnnt_eval_max_symbols_per_step,
                )
                total_errors_greedy += errors_g
                total_words_greedy += words_g
            if want_beam:
                errors_b, words_b = _compute_wer_rnnt(
                    model,
                    enc_out,
                    enc_lens,
                    tokens,
                    token_lengths,
                    tokenizer,
                    decoder="beam",
                    beam_size=rnnt_eval_beam_size,
                    beam_topk=rnnt_eval_beam_topk,
                    max_symbols_per_step=rnnt_eval_max_symbols_per_step,
                )
                total_errors_beam += errors_b
                total_words_beam += words_b
            if rnnt_eval_decoder == "greedy":
                errors, words = errors_g, words_g
            else:
                # beam or both: use beam WER as primary validation metric
                errors, words = errors_b, words_b
            loss = torch.tensor(loss_val, device=device, dtype=torch.float32)

        total_loss += float(loss.item())
        total_errors += errors
        total_words += words
        num_batches += 1

    if is_distributed and dist.is_initialized():
        metrics = torch.tensor(
            [
                total_loss,
                float(num_batches),
                float(total_errors),
                float(total_words),
                float(total_errors_greedy),
                float(total_words_greedy),
                float(total_errors_beam),
                float(total_words_beam),
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss = float(metrics[0].item())
        num_batches = int(metrics[1].item())
        total_errors = int(metrics[2].item())
        total_words = int(metrics[3].item())
        total_errors_greedy = int(metrics[4].item())
        total_words_greedy = int(metrics[5].item())
        total_errors_beam = int(metrics[6].item())
        total_words_beam = int(metrics[7].item())

    avg_loss = total_loss / max(num_batches, 1)
    wer = total_errors / max(total_words, 1)
    wer_greedy = (
        total_errors_greedy / max(total_words_greedy, 1)
        if want_greedy and total_words_greedy > 0
        else float("nan")
    )
    wer_beam = (
        total_errors_beam / max(total_words_beam, 1)
        if want_beam and total_words_beam > 0
        else float("nan")
    )
    return avg_loss, wer, wer_greedy, wer_beam


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    args = parse_args()
    if args.streaming_mode and args.streaming_chunk_size <= 0:
        args.streaming_chunk_size = 8
    if args.streaming_chunk_size < 0:
        raise ValueError("--streaming-chunk-size must be >= 0")
    if args.streaming_left_context_chunks < -1:
        raise ValueError("--streaming-left-context-chunks must be >= -1")
    if args.streaming_right_context < 0:
        raise ValueError("--streaming-right-context must be >= 0")
    if args.rnnt_eval_beam_size < 1:
        raise ValueError("--rnnt-eval-beam-size must be >= 1")
    if args.rnnt_eval_beam_topk < 1:
        raise ValueError("--rnnt-eval-beam-topk must be >= 1")
    if args.rnnt_eval_max_symbols_per_step < 1:
        raise ValueError("--rnnt-eval-max-symbols-per-step must be >= 1")
    streaming_enabled = bool(args.streaming_mode or args.streaming_chunk_size > 0)
    if not streaming_enabled:
        args.streaming_chunk_size = 0
        args.streaming_left_context_chunks = -1
        args.streaming_right_context = 0
        args.streaming_causal_conv = False

    is_distributed, rank, local_rank, world_size, device = _init_distributed(
        args.device, timeout_sec=args.ddp_timeout_sec
    )
    is_main = rank == 0
    _set_seed(args.seed + rank)
    _configure_runtime(args, device, is_main)

    if args.loss_type == "rnnt" and not _RNNT_LOSS_AVAILABLE:
        raise RuntimeError(
            "RNN-T loss requires torchaudio with rnnt_loss support. "
            "Install a recent version of torchaudio."
        )

    try:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        log_csv = args.log_csv or os.path.join(args.ckpt_dir, "training_log.csv")

        # ---- Tokenizer ----
        if args.tokenizer == "sp":
            tokenizer = SentencePieceTokenizer(args.sp_model)
            if is_main:
                print(f"SentencePiece tokenizer: {args.sp_model} (vocab={tokenizer.vocab_size})")
        else:
            tokenizer = CharTokenizer()
            if is_main:
                print(f"Character tokenizer (vocab={tokenizer.vocab_size})")

        vocab_size = tokenizer.vocab_size

        paper_mismatches = _paper_fidelity_mismatches(args, vocab_size=vocab_size)
        if args.paper_strict and paper_mismatches:
            msg = "Paper-strict mode mismatch:\n  - " + "\n  - ".join(paper_mismatches)
            raise ValueError(msg)
        if is_main:
            if paper_mismatches:
                print("Paper fidelity: NOT exact")
                for item in paper_mismatches:
                    print(f"  - {item}")
            else:
                print("Paper fidelity: key hyperparameters match Conformer Transducer Large defaults.")

        if is_main:
            print(f"Device: {device}")
            print(f"Distributed: {'enabled' if is_distributed else 'disabled'}" +
                  (f" | world_size={world_size}" if is_distributed else ""))
            print(f"Loss type: {args.loss_type}")
            print(f"Loading train splits: {args.train_splits} | val: {args.val_split}")

        # ---- Data loaders ----
        def _build_train(dl_flag):
            pairs = [
                get_dataloader(
                    args.data_root, split, args.batch_size,
                    n_mels=args.n_mels, augment=True,
                    num_workers=args.num_workers,
                    prefetch_factor=args.prefetch_factor,
                    pin_memory=args.pin_memory,
                    persistent_workers=args.persistent_workers,
                    download=dl_flag,
                    distributed=is_distributed, rank=rank, world_size=world_size,
                    return_sampler=True, tokenizer=tokenizer,
                )
                for split in args.train_splits
            ]
            loaders = [x[0] for x in pairs]
            samplers = [x[1] for x in pairs if isinstance(x[1], DistributedSampler)]
            return loaders, samplers

        def _build_val(dl_flag):
            return get_dataloader(
                args.data_root, args.val_split, args.batch_size,
                n_mels=args.n_mels, augment=False,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                pin_memory=args.pin_memory,
                persistent_workers=args.persistent_workers,
                download=dl_flag,
                distributed=is_distributed, rank=rank, world_size=world_size,
                tokenizer=tokenizer,
            )

        if is_distributed and args.download:
            if is_main:
                train_loaders, train_samplers = _build_train(True)
                val_loader = _build_val(True)
            _dist_barrier(device)
            if not is_main:
                train_loaders, train_samplers = _build_train(False)
                val_loader = _build_val(False)
        else:
            train_loaders, train_samplers = _build_train(args.download)
            val_loader = _build_val(args.download)

        # LM decoder (CTC + LM only)
        eval_lm_decoder = None
        if args.eval_lm_path and args.loss_type == "ctc":
            from tokenizer import CHAR_VOCAB
            eval_lm_decoder = build_lm_decoder(
                vocab=CHAR_VOCAB, lm_path=args.eval_lm_path,
                blank_idx=BLANK_IDX, alpha=args.eval_lm_alpha, beta=args.eval_lm_beta,
            )
            if is_main:
                print(f"CTC LM decoding: {args.eval_lm_path} "
                      f"alpha={args.eval_lm_alpha} beta={args.eval_lm_beta}")

        train_len = combined_len(train_loaders)
        mel_extractor = LogMelSpectrogram(n_mels=args.n_mels)

        sample_decode_path = None
        if args.eval_sample_decode:
            sample_decode_path = args.eval_sample_path or _find_eval_sample_flac(args.data_root, args.val_split)
            if is_main:
                if sample_decode_path is not None and os.path.exists(sample_decode_path):
                    print(f"Eval sample decode file: {sample_decode_path}")
                else:
                    print("[WARN] No eval .flac found for sample decode; disabling sample decode.")
                    sample_decode_path = None

        # ---- Model ----
        if args.loss_type == "ctc":
            model = ConformerASR(
                n_mels=args.n_mels, d_model=args.d_model,
                num_heads=args.num_heads, num_layers=args.num_layers,
                vocab_size=vocab_size,
                conv_kernel_size=args.conv_kernel, max_len=args.max_len,
                ffn_dropout=args.dropout, attn_dropout=args.dropout,
                conv_dropout=args.dropout,
                streaming_chunk_size=args.streaming_chunk_size,
                streaming_left_context_chunks=args.streaming_left_context_chunks,
                streaming_right_context=args.streaming_right_context,
                streaming_causal_conv=args.streaming_causal_conv,
            ).to(device)
        else:
            model = ConformerTransducer(
                n_mels=args.n_mels, encoder_dim=args.d_model,
                num_heads=args.num_heads, num_encoder_layers=args.num_layers,
                vocab_size=vocab_size,
                conv_kernel_size=args.conv_kernel, max_len=args.max_len,
                ffn_dropout=args.dropout, attn_dropout=args.dropout,
                conv_dropout=args.dropout,
                pred_embed_dim=args.pred_embed_dim,
                pred_hidden_dim=args.pred_hidden_dim,
                pred_num_layers=args.pred_num_layers,
                joint_dim=args.joint_dim,
                blank_idx=BLANK_IDX,
                streaming_chunk_size=args.streaming_chunk_size,
                streaming_left_context_chunks=args.streaming_left_context_chunks,
                streaming_right_context=args.streaming_right_context,
                streaming_causal_conv=args.streaming_causal_conv,
            ).to(device)

        if is_distributed and device.type == "cuda" and args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.compile:
            model = torch.compile(model, mode=args.compile_mode)
        if is_distributed:
            ddp_kw = {
                "bucket_cap_mb": int(args.ddp_bucket_cap_mb),
                "gradient_as_bucket_view": bool(args.ddp_grad_as_bucket_view),
                "broadcast_buffers": bool(args.ddp_broadcast_buffers),
            }
            if device.type == "cuda":
                ddp_kw.update({"device_ids": [local_rank], "output_device": local_rank})
            model = DDP(model, **ddp_kw)

        model_raw = model.module if isinstance(model, DDP) else model

        if is_main:
            n_params = sum(p.numel() for p in model_raw.parameters() if p.requires_grad)
            print(f"Model parameters: {n_params / 1e6:.1f}M")

        resume_path = args.resume_path
        if resume_path is None and args.auto_resume:
            auto = os.path.join(args.ckpt_dir, "last.pt")
            if os.path.exists(auto):
                resume_path = auto
                if is_main:
                    print(f"Auto-resume: {resume_path}")

        if args.init_encoder_from:
            if resume_path:
                if is_main:
                    print(
                        "Ignoring --init-encoder-from because full resume is active "
                        f"(resume_path={resume_path})."
                    )
            else:
                _init_encoder_from_checkpoint(
                    model_raw=model_raw,
                    ckpt_path=args.init_encoder_from,
                    device=device,
                    is_main=is_main,
                )

        # ---- Optimizer ----
        opt_cls = torch.optim.Adam if args.optimizer == "adam" else torch.optim.AdamW
        base_opt_kwargs = {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "betas": (0.9, 0.98),
            "eps": 1e-9,
        }

        optimizer = None
        opt_candidates: list[dict] = [dict(base_opt_kwargs)]
        if device.type == "cuda":
            if args.fused_optimizer:
                # fused and foreach cannot both be True in many torch builds.
                opt_candidates = [
                    {**base_opt_kwargs, "fused": True},
                    {**base_opt_kwargs, "foreach": True},
                    dict(base_opt_kwargs),
                ]
            else:
                opt_candidates = [
                    {**base_opt_kwargs, "foreach": True},
                    dict(base_opt_kwargs),
                ]

        last_err: Exception | None = None
        for candidate in opt_candidates:
            try:
                optimizer = opt_cls(model.parameters(), **candidate)
                break
            except (TypeError, RuntimeError, ValueError) as err:
                last_err = err
                continue

        if optimizer is None:
            raise RuntimeError(
                f"Failed to initialize optimizer {args.optimizer} with supported kwargs. "
                f"Last error: {last_err}"
            )

        steps_per_epoch = math.ceil(train_len / max(1, args.accum_steps))
        total_steps = steps_per_epoch * args.epochs

        if args.lr_schedule == "paper":
            scheduler = PaperTransformerSchedule(
                optimizer, d_model=args.d_model,
                warmup_steps=args.warmup_steps, peak_factor=args.paper_peak_factor,
            )
        else:
            scheduler = WarmupCosineScheduler(
                optimizer, warmup_steps=args.warmup_steps,
                total_steps=total_steps, peak_lr=args.lr, min_lr=args.min_lr,
            )

        autocast_dtype = _autocast_dtype(args.precision)
        scaler = GradScaler(device=device.type, enabled=(device.type == "cuda" and args.precision == "fp16"))

        # ---- Resume ----
        start_epoch = 0
        best_wer = float("inf")

        if resume_path:
            if not os.path.exists(resume_path):
                raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            ckpt_cfg = ckpt.get("config", {}) or {}
            model_raw.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if args.override_lr_on_resume:
                for pg in optimizer.param_groups:
                    pg["lr"] = args.lr
            if not args.reset_scheduler_on_resume and "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            elif args.reset_scheduler_on_resume:
                scheduler.step_count = 0
            if "scaler" in ckpt and ckpt["scaler"]:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_wer = float(ckpt.get("best_wer", float("inf")))
            if is_main:
                resume_warnings: list[str] = []
                if not args.reset_scheduler_on_resume:
                    old_warmup = ckpt_cfg.get("warmup_steps", None)
                    if old_warmup is not None and int(old_warmup) != int(args.warmup_steps):
                        resume_warnings.append(
                            f"warmup_steps differs (ckpt={int(old_warmup)} vs cli={int(args.warmup_steps)})"
                        )
                    old_peak_factor = ckpt_cfg.get("paper_peak_factor", None)
                    if old_peak_factor is not None and not math.isclose(
                        float(old_peak_factor), float(args.paper_peak_factor), rel_tol=0.0, abs_tol=1e-12
                    ):
                        resume_warnings.append(
                            f"paper_peak_factor differs (ckpt={float(old_peak_factor)} vs cli={float(args.paper_peak_factor)})"
                        )
                    old_sched = ckpt_cfg.get("lr_schedule", None)
                    if old_sched is not None and str(old_sched) != str(args.lr_schedule):
                        resume_warnings.append(
                            f"lr_schedule differs (ckpt={old_sched!r} vs cli={args.lr_schedule!r})"
                        )
                if not args.override_lr_on_resume and args.lr_schedule == "cosine":
                    opt_state = ckpt.get("optimizer", {})
                    try:
                        old_lr = float(opt_state.get("param_groups", [{}])[0].get("lr"))
                    except Exception:
                        old_lr = None
                    if old_lr is not None and not math.isclose(old_lr, float(args.lr), rel_tol=0.0, abs_tol=1e-12):
                        resume_warnings.append(
                            f"optimizer lr differs (ckpt={old_lr:.3e} vs cli={float(args.lr):.3e})"
                        )
                if resume_warnings:
                    print("[WARN] Resume checkpoint hyperparameters differ from CLI; current run will keep checkpoint state.")
                    for item in resume_warnings:
                        print(f"  - {item}")
                    print(
                        "  Use --reset-scheduler-on-resume to apply new LR schedule settings, "
                        "and --override-lr-on-resume to force optimizer lr."
                    )
                print(f"Resumed: epoch {start_epoch+1}/{args.epochs} | "
                      f"best WER {best_wer:.2%} | step={scheduler.step_count}")
            if start_epoch >= args.epochs:
                if is_main:
                    print("Already trained to target epochs.")
                return

        # ---- Training ----
        if is_main:
            print(f"\nScheduler: {args.lr_schedule} | "
                  f"peak_lr={scheduler.lr_peak if hasattr(scheduler, 'lr_peak') else args.lr:.2e} | "
                  f"warmup={args.warmup_steps}")
            print(f"Optimizer: {args.optimizer} | fused={args.fused_optimizer} | precision={args.precision}")
            print(f"DataLoader: workers={args.num_workers} prefetch={args.prefetch_factor} "
                  f"pin_memory={args.pin_memory} persistent_workers={args.persistent_workers}")
            if streaming_enabled:
                print(
                    "Streaming encoder: "
                    f"chunk={args.streaming_chunk_size} "
                    f"left_chunks={args.streaming_left_context_chunks} "
                    f"right={args.streaming_right_context} "
                    f"causal_conv={args.streaming_causal_conv}"
                )
            else:
                print("Streaming encoder: disabled")
            print(f"Eval/checkpoint cadence: eval_every={args.eval_every} save_every={args.save_every}")
            if is_distributed:
                print(f"DDP: bucket_cap_mb={args.ddp_bucket_cap_mb} "
                      f"timeout_sec={args.ddp_timeout_sec} "
                      f"grad_as_bucket_view={args.ddp_grad_as_bucket_view} "
                      f"broadcast_buffers={args.ddp_broadcast_buffers} sync_bn={args.sync_bn}")
            print(f"torch.compile: {args.compile} ({args.compile_mode})")
            print(f"Variational noise: {args.variational_noise}")
            if args.loss_type == "rnnt":
                print(f"RNN-T fused log_softmax: {args.rnnt_fused_log_softmax}")
                print(f"RNN-T loss device: {args.rnnt_loss_device}")
                print(f"RNN-T max batch*T*U: {args.rnnt_max_batch_tu}")
                print(
                    "RNN-T eval decoder: "
                    f"{args.rnnt_eval_decoder} "
                    f"(beam_size={args.rnnt_eval_beam_size}, topk={args.rnnt_eval_beam_topk}, "
                    f"max_symbols={args.rnnt_eval_max_symbols_per_step})"
                )
            print(f"Effective batch: {args.batch_size} x {args.accum_steps} x {world_size} = "
                  f"{args.batch_size * args.accum_steps * world_size}")
            print()

        for epoch_idx in range(start_epoch, args.epochs):
            t0 = time.time()
            if is_main:
                print("=" * 60)
                print(f"Epoch {epoch_idx + 1}/{args.epochs}")
                print("=" * 60)

            for sampler in train_samplers:
                sampler.set_epoch(epoch_idx)

            train_loader = combined_loader(train_loaders, shuffle=True)
            gs_offset = epoch_idx * train_len

            train_loss, step_rows, num_batches = train_one_epoch(
                model=model, loader=train_loader, loader_len=train_len,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                device=device, grad_clip=args.grad_clip,
                accum_steps=args.accum_steps, epoch_num=epoch_idx + 1,
                global_step_offset=gs_offset, loss_type=args.loss_type,
                var_noise_std=args.variational_noise,
                is_distributed=is_distributed, is_main=is_main,
                autocast_dtype=autocast_dtype,
                rnnt_fused_log_softmax=args.rnnt_fused_log_softmax,
                rnnt_loss_device=args.rnnt_loss_device,
                rnnt_max_batch_tu=args.rnnt_max_batch_tu,
            )

            run_eval = ((epoch_idx + 1) % max(1, args.eval_every) == 0) or ((epoch_idx + 1) == args.epochs)
            if run_eval:
                tp = args.beam_token_prune if args.beam_token_prune > 0 else None
                val_loss, val_wer, val_wer_greedy, val_wer_beam = evaluate(
                    model_raw, val_loader, device,
                    loss_type=args.loss_type, tokenizer=tokenizer,
                    beam_size=args.beam_size, token_prune=tp,
                    lm_decoder=eval_lm_decoder, lm_beam_width=args.eval_lm_beam_width,
                    autocast_dtype=autocast_dtype,
                    rnnt_fused_log_softmax=args.rnnt_fused_log_softmax,
                    rnnt_loss_device=args.rnnt_loss_device,
                    rnnt_max_batch_tu=args.rnnt_max_batch_tu,
                    rnnt_eval_decoder=args.rnnt_eval_decoder,
                    rnnt_eval_beam_size=args.rnnt_eval_beam_size,
                    rnnt_eval_beam_topk=args.rnnt_eval_beam_topk,
                    rnnt_eval_max_symbols_per_step=args.rnnt_eval_max_symbols_per_step,
                    is_distributed=is_distributed,
                )
            else:
                val_loss, val_wer = float("nan"), float("nan")
                val_wer_greedy, val_wer_beam = float("nan"), float("nan")

            elapsed = time.time() - t0

            if is_main:
                if run_eval and math.isfinite(val_wer):
                    print(f"\nEpoch {epoch_idx+1}: train_loss={train_loss:.4f} "
                          f"val_loss={val_loss:.4f} val_wer={val_wer:.2%} time={elapsed:.0f}s")
                    if args.loss_type == "rnnt":
                        if math.isfinite(val_wer_greedy):
                            print(f"  val_wer_greedy={val_wer_greedy:.2%}")
                        if math.isfinite(val_wer_beam):
                            print(f"  val_wer_beam={val_wer_beam:.2%}")
                else:
                    print(f"\nEpoch {epoch_idx+1}: train_loss={train_loss:.4f} "
                          f"eval=skipped time={elapsed:.0f}s")

                if run_eval and sample_decode_path is not None:
                    try:
                        tp = args.beam_token_prune if args.beam_token_prune > 0 else None
                        ref_text, hyp_text = decode_eval_sample(
                            model=model_raw,
                            device=device,
                            loss_type=args.loss_type,
                            tokenizer=tokenizer,
                            sample_path=sample_decode_path,
                            mel_extractor=mel_extractor,
                            beam_size=args.beam_size,
                            token_prune=tp,
                            rnnt_decoder=("beam" if args.rnnt_eval_decoder in ("beam", "both") else "greedy"),
                            rnnt_beam_size=args.rnnt_eval_beam_size,
                            rnnt_beam_topk=args.rnnt_eval_beam_topk,
                            rnnt_max_symbols_per_step=args.rnnt_eval_max_symbols_per_step,
                        )
                        print("Sample decode:")
                        print(f"  file: {sample_decode_path}")
                        if ref_text:
                            print(f"  ref : {ref_text}")
                        print(f"  hyp : {hyp_text}")
                    except Exception as e:
                        print(f"[WARN] Sample decode failed: {e}")

                best_for_log = min(best_wer, val_wer) if (run_eval and math.isfinite(val_wer)) else best_wer
                epoch_rows = step_rows + [{
                    "record_type": "epoch_summary", "epoch": epoch_idx + 1,
                    "step": num_batches, "global_step": gs_offset + num_batches,
                    "optimizer_step": scheduler.step_count,
                    "train_step_loss": "", "train_running_loss": "",
                    "train_epoch_loss": train_loss,
                    "val_loss": val_loss, "val_wer": val_wer,
                    "lr": scheduler.get_lr(), "elapsed_sec": elapsed,
                    "best_wer": best_for_log,
                }]
                append_csv_rows(log_csv, epoch_rows)

                ckpt_data = {
                    "epoch": epoch_idx,
                    "model": model_raw.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": scheduler.step_count,
                    "val_loss": val_loss, "val_wer": val_wer,
                    "best_wer": best_for_log,
                    "config": {
                        "loss_type": args.loss_type,
                        "tokenizer": args.tokenizer,
                        "sp_model": args.sp_model if args.tokenizer == "sp" else None,
                        "d_model": args.d_model,
                        "num_heads": args.num_heads,
                        "num_layers": args.num_layers,
                        "n_mels": args.n_mels,
                        "conv_kernel": args.conv_kernel,
                        "max_len": args.max_len,
                        "streaming_mode": streaming_enabled,
                        "streaming_chunk_size": args.streaming_chunk_size,
                        "streaming_left_context_chunks": args.streaming_left_context_chunks,
                        "streaming_right_context": args.streaming_right_context,
                        "streaming_causal_conv": args.streaming_causal_conv,
                        "pred_embed_dim": args.pred_embed_dim,
                        "pred_hidden_dim": args.pred_hidden_dim,
                        "pred_num_layers": args.pred_num_layers,
                        "joint_dim": args.joint_dim,
                        "vocab_size": vocab_size,
                        "dropout": args.dropout,
                        "variational_noise": args.variational_noise,
                        "optimizer": args.optimizer,
                        "precision": args.precision,
                        "tf32": args.tf32,
                        "fused_optimizer": args.fused_optimizer,
                        "lr_schedule": args.lr_schedule,
                        "paper_peak_factor": args.paper_peak_factor,
                        "warmup_steps": args.warmup_steps,
                        "lr": args.lr,
                        "min_lr": args.min_lr,
                        "weight_decay": args.weight_decay,
                        "batch_size": args.batch_size,
                        "accum_steps": args.accum_steps,
                        "grad_clip": args.grad_clip,
                        "world_size": world_size,
                        "rnnt_fused_log_softmax": args.rnnt_fused_log_softmax,
                        "rnnt_loss_device": args.rnnt_loss_device,
                        "rnnt_max_batch_tu": args.rnnt_max_batch_tu,
                        "rnnt_eval_decoder": args.rnnt_eval_decoder,
                        "rnnt_eval_beam_size": args.rnnt_eval_beam_size,
                        "rnnt_eval_beam_topk": args.rnnt_eval_beam_topk,
                        "rnnt_eval_max_symbols_per_step": args.rnnt_eval_max_symbols_per_step,
                        "init_encoder_from": args.init_encoder_from,
                    },
                }
                should_save_last = (
                    (epoch_idx + 1) == args.epochs
                    or ((epoch_idx + 1) % max(1, args.save_every) == 0)
                )
                if should_save_last:
                    torch.save(ckpt_data, os.path.join(args.ckpt_dir, "last.pt"))
                if run_eval and math.isfinite(val_wer) and val_wer < best_wer:
                    best_wer = val_wer
                    torch.save(ckpt_data, os.path.join(args.ckpt_dir, "best.pt"))
                    print(f"  ** New best WER: {best_wer:.2%} **")

        if is_main:
            print(f"\nTraining complete. Best WER: {best_wer:.2%}")

    finally:
        _cleanup(is_distributed)


if __name__ == "__main__":
    main()
