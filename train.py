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
    p.add_argument("--tokenizer", default="char", choices=["char", "sp"],
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

    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--eval-sample-decode", action="store_true", default=True)
    p.add_argument("--no-eval-sample-decode", action="store_false", dest="eval_sample_decode")
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
    p.add_argument("--resume-path", default=None)
    p.add_argument("--reset-scheduler-on-resume", action="store_true")
    p.add_argument("--override-lr-on-resume", action="store_true")
    p.add_argument("--auto-resume", action="store_true", default=True)
    p.add_argument("--no-auto-resume", action="store_false", dest="auto_resume")

    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--download", action="store_true", default=True)
    p.add_argument("--no-download", action="store_false", dest="download")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

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


def _init_distributed(arg: str):
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
        dist.init_process_group(backend=backend, init_method="env://")
    return True, rank, local_rank, world_size, device


def _cleanup(is_dist: bool):
    if is_dist and dist.is_initialized():
        dist.destroy_process_group()


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
) -> tuple[int, int]:
    decoded = rnnt_greedy_decode(
        model, enc_out, enc_lengths, blank_idx=model.blank_idx, max_symbols_per_step=10,
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
        decoded = rnnt_greedy_decode(
            model, enc_out, enc_lengths, blank_idx=model.blank_idx, max_symbols_per_step=10,
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
    for s, e, t_max, u_max in chunks:
        logits_chunk = logits[s:e, :t_max, :u_max, :].float()
        targets_chunk = targets[s:e, : max(0, u_max - 1)].int()
        logit_len_chunk = logit_lengths[s:e].clamp(min=1, max=t_max).int()
        target_len_chunk = target_lengths[s:e].clamp(min=0, max=max(0, u_max - 1)).int()

        if not fused_log_softmax:
            logits_chunk = torch.log_softmax(logits_chunk, dim=-1)

        if loss_device == "cpu":
            logits_chunk = logits_chunk.cpu()
            targets_chunk = targets_chunk.cpu()
            logit_len_chunk = logit_len_chunk.cpu()
            target_len_chunk = target_len_chunk.cpu()

        chunk_sum = torchaudio.functional.rnnt_loss(
            logits=logits_chunk,
            targets=targets_chunk,
            logit_lengths=logit_len_chunk,
            target_lengths=target_len_chunk,
            blank=blank_idx,
            reduction="sum",
            fused_log_softmax=fused_log_softmax,
        )
        total = chunk_sum if total is None else total + chunk_sum

    # torchaudio reduction='mean' is equivalent to sum / batch_size
    loss = total / max(1, int(targets.size(0)))
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

        sync_ctx = nullcontext()
        if is_distributed and isinstance(model, DDP) and not should_step:
            sync_ctx = model.no_sync()

        # Apply variational noise before forward pass
        noise_dict = apply_variational_noise(raw_model, var_noise_std)

        use_scaler = (
            device.type == "cuda"
            and not (loss_type == "rnnt" and rnnt_loss_device == "cpu")
        )
        raw_loss = 0.0

        with sync_ctx:
            if loss_type == "ctc":
                with autocast(device_type=device.type, enabled=(device.type == "cuda")):
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
                # Keep RNNT forward/loss in FP32 to avoid CUDA kernel instability on some setups.
                with autocast(device_type=device.type, enabled=False):
                    est_enc_lengths = raw_model.encoder.get_output_lengths(mel_lengths).clamp(min=1).int()
                    batch_chunks = _build_rnnt_chunks(est_enc_lengths, token_lengths, rnnt_max_batch_tu)
                    batch_size = max(1, int(tokens.size(0)))

                    # Forward/backward per chunk avoids allocating a huge (B,T,U,V) logits tensor.
                    for s, e, _t_max, _u_max in batch_chunks:
                        chunk_bs = e - s
                        logits, enc_lengths = model(
                            mel[s:e].float(),
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
    rnnt_fused_log_softmax: bool = False,
    rnnt_loss_device: str = "cuda",
    rnnt_max_batch_tu: int = 1_500_000,
) -> tuple[float, float]:
    model.eval()

    if loss_type == "ctc":
        loss_fn = nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)

    total_loss = 0.0
    total_errors = 0
    total_words = 0
    num_batches = 0

    for mel, tokens, mel_lengths, token_lengths in loader:
        mel = mel.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)
        mel_lengths = mel_lengths.to(device, non_blocking=True)
        token_lengths = token_lengths.to(device, non_blocking=True)

        if loss_type == "ctc":
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
            est_enc_lengths = model.encoder.get_output_lengths(mel_lengths).clamp(min=1).int()
            batch_chunks = _build_rnnt_chunks(est_enc_lengths, token_lengths, rnnt_max_batch_tu)
            batch_size = max(1, int(tokens.size(0)))
            loss_val = 0.0
            for s, e, _t_max, _u_max in batch_chunks:
                chunk_bs = e - s
                logits, enc_lengths = model(
                    mel[s:e].float(),
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
            # WER via greedy decode (fast)
            enc_out, enc_lens = model.encode(mel, mel_lengths)
            enc_lens = enc_lens.clamp(min=1)
            errors, words = _compute_wer_rnnt(
                model, enc_out, enc_lens, tokens, token_lengths, tokenizer,
            )
            loss = torch.tensor(loss_val, device=device, dtype=torch.float32)

        total_loss += float(loss.item())
        total_errors += errors
        total_words += words
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    wer = total_errors / max(total_words, 1)
    return avg_loss, wer


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    args = parse_args()
    is_distributed, rank, local_rank, world_size, device = _init_distributed(args.device)
    is_main = rank == 0
    _set_seed(args.seed + rank)

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
                    num_workers=args.num_workers, download=dl_flag,
                    distributed=is_distributed, rank=rank, world_size=world_size,
                    return_sampler=True, tokenizer=tokenizer,
                )
                for split in args.train_splits
            ]
            loaders = [x[0] for x in pairs]
            samplers = [x[1] for x in pairs if isinstance(x[1], DistributedSampler)]
            return loaders, samplers

        def _build_val(dl_flag):
            if not is_main:
                return None
            return get_dataloader(
                args.data_root, args.val_split, args.batch_size,
                n_mels=args.n_mels, augment=False,
                num_workers=args.num_workers, download=dl_flag,
                distributed=False, tokenizer=tokenizer,
            )

        if is_distributed and args.download:
            if is_main:
                train_loaders, train_samplers = _build_train(True)
                val_loader = _build_val(True)
            dist.barrier()
            if not is_main:
                train_loaders, train_samplers = _build_train(False)
                val_loader = _build_val(False)
        else:
            train_loaders, train_samplers = _build_train(args.download)
            val_loader = _build_val(args.download)

        # LM decoder (CTC + LM only)
        eval_lm_decoder = None
        if is_main and args.eval_lm_path and args.loss_type == "ctc":
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
            ).to(device)

        if is_distributed and device.type == "cuda":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if is_distributed:
            ddp_kw = ({"device_ids": [local_rank], "output_device": local_rank}
                      if device.type == "cuda" else {})
            model = DDP(model, **ddp_kw)

        model_raw = model.module if isinstance(model, DDP) else model

        if is_main:
            n_params = sum(p.numel() for p in model_raw.parameters() if p.requires_grad)
            print(f"Model parameters: {n_params / 1e6:.1f}M")

        # ---- Optimizer ----
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=1e-9,
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

        scaler = GradScaler(device=device.type, enabled=(device.type == "cuda"))

        # ---- Resume ----
        start_epoch = 0
        best_wer = float("inf")

        resume_path = args.resume_path
        if resume_path is None and args.auto_resume:
            auto = os.path.join(args.ckpt_dir, "last.pt")
            if os.path.exists(auto):
                resume_path = auto
                if is_main:
                    print(f"Auto-resume: {resume_path}")

        if resume_path:
            if not os.path.exists(resume_path):
                raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
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
            print(f"Variational noise: {args.variational_noise}")
            if args.loss_type == "rnnt":
                print(f"RNN-T fused log_softmax: {args.rnnt_fused_log_softmax}")
                print(f"RNN-T loss device: {args.rnnt_loss_device}")
                print(f"RNN-T max batch*T*U: {args.rnnt_max_batch_tu}")
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
                rnnt_fused_log_softmax=args.rnnt_fused_log_softmax,
                rnnt_loss_device=args.rnnt_loss_device,
                rnnt_max_batch_tu=args.rnnt_max_batch_tu,
            )

            if is_distributed:
                dist.barrier()

            if is_main:
                tp = args.beam_token_prune if args.beam_token_prune > 0 else None
                val_loss, val_wer = evaluate(
                    model_raw, val_loader, device,
                    loss_type=args.loss_type, tokenizer=tokenizer,
                    beam_size=args.beam_size, token_prune=tp,
                    lm_decoder=eval_lm_decoder, lm_beam_width=args.eval_lm_beam_width,
                    rnnt_fused_log_softmax=args.rnnt_fused_log_softmax,
                    rnnt_loss_device=args.rnnt_loss_device,
                    rnnt_max_batch_tu=args.rnnt_max_batch_tu,
                )
            else:
                val_loss, val_wer = 0.0, 0.0

            if is_distributed:
                vm = torch.tensor([val_loss, val_wer], device=device, dtype=torch.float64)
                dist.broadcast(vm, src=0)
                val_loss, val_wer = float(vm[0]), float(vm[1])

            elapsed = time.time() - t0

            if is_main:
                print(f"\nEpoch {epoch_idx+1}: train_loss={train_loss:.4f} "
                      f"val_loss={val_loss:.4f} val_wer={val_wer:.2%} time={elapsed:.0f}s")
                if sample_decode_path is not None:
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
                        )
                        print("Sample decode:")
                        print(f"  file: {sample_decode_path}")
                        if ref_text:
                            print(f"  ref : {ref_text}")
                        print(f"  hyp : {hyp_text}")
                    except Exception as e:
                        print(f"[WARN] Sample decode failed: {e}")

                epoch_rows = step_rows + [{
                    "record_type": "epoch_summary", "epoch": epoch_idx + 1,
                    "step": num_batches, "global_step": gs_offset + num_batches,
                    "optimizer_step": scheduler.step_count,
                    "train_step_loss": "", "train_running_loss": "",
                    "train_epoch_loss": train_loss,
                    "val_loss": val_loss, "val_wer": val_wer,
                    "lr": scheduler.get_lr(), "elapsed_sec": elapsed,
                    "best_wer": min(best_wer, val_wer),
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
                    "best_wer": min(best_wer, val_wer),
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
                        "pred_embed_dim": args.pred_embed_dim,
                        "pred_hidden_dim": args.pred_hidden_dim,
                        "pred_num_layers": args.pred_num_layers,
                        "joint_dim": args.joint_dim,
                        "vocab_size": vocab_size,
                        "dropout": args.dropout,
                        "variational_noise": args.variational_noise,
                        "lr_schedule": args.lr_schedule,
                        "paper_peak_factor": args.paper_peak_factor,
                        "warmup_steps": args.warmup_steps,
                        "weight_decay": args.weight_decay,
                        "world_size": world_size,
                    },
                }

                torch.save(ckpt_data, os.path.join(args.ckpt_dir, "last.pt"))
                if val_wer < best_wer:
                    best_wer = val_wer
                    torch.save(ckpt_data, os.path.join(args.ckpt_dir, "best.pt"))
                    print(f"  ** New best WER: {best_wer:.2%} **")

        if is_main:
            print(f"\nTraining complete. Best WER: {best_wer:.2%}")

    finally:
        _cleanup(is_distributed)


if __name__ == "__main__":
    main()
