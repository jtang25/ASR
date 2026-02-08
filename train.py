import argparse
import csv
from contextlib import nullcontext
import math
import os
import random
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from decoding import beam_search_decode, build_lm_decoder, lm_beam_search_decode
from model import ConformerASR
from preprocessing import BLANK_IDX, VOCAB, VOCAB_SIZE, get_dataloader, tokens_to_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Conformer CTC ASR model.")

    parser.add_argument("--data-root", default="./data", help="Dataset root directory.")
    parser.add_argument(
        "--train-splits",
        nargs="+",
        default=["train-clean-100", "train-clean-360"],
        help="LibriSpeech train splits to mix.",
    )
    parser.add_argument("--val-split", default="dev-clean", help="LibriSpeech validation split.")

    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=17)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--conv-kernel", type=int, default=31)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=400)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument(
        "--beam-token-prune",
        type=int,
        default=0,
        help="Per-frame top-k pruning for beam search (0 disables).",
    )
    parser.add_argument(
        "--eval-lm-path",
        default=None,
        help="Optional KenLM model (.arpa/.bin) path for validation-time LM decoding.",
    )
    parser.add_argument(
        "--eval-lm-alpha",
        type=float,
        default=0.5,
        help="LM weight alpha used when --eval-lm-path is set.",
    )
    parser.add_argument(
        "--eval-lm-beta",
        type=float,
        default=1.0,
        help="Word insertion bonus beta used when --eval-lm-path is set.",
    )
    parser.add_argument(
        "--eval-lm-beam-width",
        type=int,
        default=128,
        help="Beam width for LM decoder when --eval-lm-path is set.",
    )

    parser.add_argument("--ckpt-dir", default="./checkpoints")
    parser.add_argument(
        "--log-csv",
        default=None,
        help="CSV path for training logs (default: <ckpt-dir>/training_log.csv).",
    )
    parser.add_argument(
        "--resume-path",
        default=None,
        help="Optional checkpoint path to resume from.",
    )
    parser.add_argument(
        "--reset-scheduler-on-resume",
        action="store_true",
        help="Do not load scheduler state from checkpoint when resuming.",
    )
    parser.add_argument(
        "--override-lr-on-resume",
        action="store_true",
        help="Force optimizer LR to --lr after loading checkpoint optimizer state.",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        default=True,
        help="If --resume-path is not set, auto-resume from <ckpt-dir>/last.pt when available.",
    )
    parser.add_argument(
        "--no-auto-resume",
        action="store_false",
        dest="auto_resume",
        help="Disable automatic resume-from-last checkpoint behavior.",
    )

    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--download", action="store_true", default=True)
    parser.add_argument("--no-download", action="store_false", dest="download")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _init_distributed(device_arg: str) -> tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    is_distributed = world_size > 1
    if not is_distributed:
        return False, rank, local_rank, world_size, _resolve_device(device_arg)

    use_cuda = device_arg == "cuda" or (device_arg == "auto" and torch.cuda.is_available())
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("Distributed CUDA training requested but CUDA is not available.")

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


def _cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def _reduce_sum(value: float, device: torch.device, is_distributed: bool) -> float:
    if not is_distributed:
        return value
    t = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def _validate_args(args: argparse.Namespace) -> None:
    if not args.train_splits:
        raise ValueError("--train-splits must not be empty")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.accum_steps < 1:
        raise ValueError("--accum-steps must be >= 1")
    if args.lr <= 0:
        raise ValueError("--lr must be > 0")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be >= 0")
    if args.grad_clip <= 0:
        raise ValueError("--grad-clip must be > 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.beam_size < 1:
        raise ValueError("--beam-size must be >= 1")
    if args.beam_token_prune < 0:
        raise ValueError("--beam-token-prune must be >= 0")
    if args.eval_lm_beam_width < 1:
        raise ValueError("--eval-lm-beam-width must be >= 1")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Word Error Rate
# ---------------------------------------------------------------------------


def edit_distance(ref: list, hyp: list) -> int:
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


def compute_wer(
    log_probs: torch.Tensor,
    output_lengths: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    beam_size: int = 10,
    token_prune: int | None = None,
    lm_decoder=None,
    lm_beam_width: int = 100,
) -> float:
    if lm_decoder is not None:
        hyp_texts = lm_beam_search_decode(
            log_probs,
            output_lengths,
            decoder=lm_decoder,
            beam_width=lm_beam_width,
        )
    else:
        decoded = beam_search_decode(
            log_probs,
            output_lengths,
            beam_size=beam_size,
            blank_idx=BLANK_IDX,
            token_prune=token_prune,
        )
        hyp_texts = [tokens_to_text(tokens) for tokens in decoded]
    total_words = 0
    total_errors = 0

    for i, hyp_text in enumerate(hyp_texts):
        ref_len = int(target_lengths[i].item())
        ref_tokens = targets[i, :ref_len].tolist()
        ref_text = tokens_to_text(ref_tokens)

        ref_words = ref_text.split()
        hyp_words = hyp_text.split()

        total_words += len(ref_words)
        total_errors += edit_distance(ref_words, hyp_words)

    return total_errors / max(total_words, 1)


# ---------------------------------------------------------------------------
# Learning Rate Scheduler: Linear warmup + Cosine decay
# ---------------------------------------------------------------------------


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, peak_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.step_count = 0

    def step(self):
        self.step_count += 1
        lr = self._compute_lr(self.step_count)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def _compute_lr(self, step_count: int) -> float:
        if step_count <= self.warmup_steps:
            return self.peak_lr * step_count / max(1, self.warmup_steps)
        progress = (step_count - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        return self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
            1 + math.cos(math.pi * progress)
        )

    def state_dict(self) -> dict:
        return {
            "step_count": self.step_count,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "peak_lr": self.peak_lr,
            "min_lr": self.min_lr,
        }

    def load_state_dict(self, state: dict) -> None:
        self.step_count = int(state.get("step_count", 0))
        for pg in self.optimizer.param_groups:
            pg["lr"] = self._compute_lr(self.step_count)


# ---------------------------------------------------------------------------
# Simple combined loader: mix batches from multiple dataloaders
# ---------------------------------------------------------------------------


def combined_loader(loaders, shuffle=True):
    iters = [iter(l) for l in loaders]
    order = []
    for i, l in enumerate(loaders):
        order += [i] * len(l)
    if shuffle:
        random.shuffle(order)
    for i in order:
        try:
            yield next(iters[i])
        except StopIteration:
            continue


def combined_len(loaders):
    return sum(len(l) for l in loaders)


CSV_FIELDS = [
    "record_type",
    "epoch",
    "step",
    "global_step",
    "optimizer_step",
    "train_step_loss",
    "train_running_loss",
    "train_epoch_loss",
    "val_loss",
    "val_wer",
    "lr",
    "elapsed_sec",
    "best_wer",
]


def append_csv_rows(csv_path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if (not file_exists) or os.path.getsize(csv_path) == 0:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_epoch(
    model,
    loader,
    loader_len,
    optimizer,
    scheduler,
    scaler,
    device,
    grad_clip,
    accum_steps,
    epoch_num: int,
    global_step_offset: int,
    is_distributed: bool = False,
    is_main_process: bool = True,
):
    model.train()
    ctc_loss_fn = nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)

    total_loss = 0.0
    num_batches = 0
    step_rows: list[dict] = []
    optimizer.zero_grad(set_to_none=True)

    for step, (mel, tokens, mel_lengths, token_lengths) in enumerate(loader):
        mel = mel.to(device)
        tokens = tokens.to(device)
        mel_lengths = mel_lengths.to(device)
        token_lengths = token_lengths.to(device)
        is_last_batch = (step + 1) == loader_len
        should_step = ((step + 1) % accum_steps == 0) or is_last_batch

        sync_ctx = nullcontext()
        if is_distributed and isinstance(model, DDP) and not should_step:
            sync_ctx = model.no_sync()
        with sync_ctx:
            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                log_probs, output_lengths = model(mel, mel_lengths)
                log_probs_ctc = log_probs.transpose(0, 1)
                output_lengths = output_lengths.clamp(min=1)
                loss = ctc_loss_fn(log_probs_ctc, tokens, output_lengths, token_lengths)
                loss = loss / accum_steps
            scaler.scale(loss).backward()

        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        raw_loss = loss.item() * accum_steps
        total_loss += raw_loss
        num_batches += 1
        running_loss = total_loss / num_batches
        lr_now = scheduler.get_lr()
        global_step = global_step_offset + step + 1
        if is_main_process:
            step_rows.append(
                {
                    "record_type": "train_step",
                    "epoch": epoch_num,
                    "step": step + 1,
                    "global_step": global_step,
                    "optimizer_step": scheduler.step_count,
                    "train_step_loss": raw_loss,
                    "train_running_loss": running_loss,
                    "train_epoch_loss": "",
                    "val_loss": "",
                    "val_wer": "",
                    "lr": lr_now,
                    "elapsed_sec": "",
                    "best_wer": "",
                }
            )

        if is_main_process and (step + 1) % 100 == 0:
            print(f"  step {step+1:5d} | loss {running_loss:.4f} | lr {lr_now:.2e}")

    total_loss = _reduce_sum(total_loss, device=device, is_distributed=is_distributed)
    num_batches_global = _reduce_sum(float(num_batches), device=device, is_distributed=is_distributed)
    return total_loss / max(num_batches_global, 1.0), step_rows, num_batches


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    beam_size: int = 10,
    token_prune: int | None = None,
    lm_decoder=None,
    lm_beam_width: int = 100,
):
    model.eval()
    ctc_loss_fn = nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)

    total_loss = 0.0
    total_wer = 0.0
    num_batches = 0

    for mel, tokens, mel_lengths, token_lengths in loader:
        mel = mel.to(device)
        tokens = tokens.to(device)
        mel_lengths = mel_lengths.to(device)
        token_lengths = token_lengths.to(device)

        log_probs, output_lengths = model(mel, mel_lengths)
        log_probs_ctc = log_probs.transpose(0, 1)
        output_lengths = output_lengths.clamp(min=1)

        loss = ctc_loss_fn(log_probs_ctc, tokens, output_lengths, token_lengths)
        wer = compute_wer(
            log_probs,
            output_lengths,
            tokens,
            token_lengths,
            beam_size=beam_size,
            token_prune=token_prune,
            lm_decoder=lm_decoder,
            lm_beam_width=lm_beam_width,
        )

        total_loss += loss.item()
        total_wer += wer
        num_batches += 1

    return total_loss / max(num_batches, 1), total_wer / max(num_batches, 1)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    is_distributed, rank, local_rank, world_size, device = _init_distributed(args.device)
    is_main_process = rank == 0
    _set_seed(args.seed + rank)

    try:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        log_csv_path = args.log_csv or os.path.join(args.ckpt_dir, "training_log.csv")

        if is_main_process:
            print(f"Device: {device}")
            if is_distributed:
                print(
                    f"Distributed: enabled | rank={rank} local_rank={local_rank} "
                    f"world_size={world_size}"
                )
            else:
                print("Distributed: disabled")
            print(f"CSV logging: {log_csv_path}")
            print(f"Loading train splits: {args.train_splits} | val: {args.val_split} ...")

        def _build_train_loaders(download_flag: bool):
            train_loader_and_samplers = [
                get_dataloader(
                    args.data_root,
                    split,
                    args.batch_size,
                    n_mels=args.n_mels,
                    augment=True,
                    num_workers=args.num_workers,
                    download=download_flag,
                    distributed=is_distributed,
                    rank=rank,
                    world_size=world_size,
                    return_sampler=True,
                )
                for split in args.train_splits
            ]
            local_train_loaders = [x[0] for x in train_loader_and_samplers]
            local_train_samplers = [
                x[1] for x in train_loader_and_samplers if isinstance(x[1], DistributedSampler)
            ]
            return local_train_loaders, local_train_samplers

        def _build_val_loader(download_flag: bool):
            if not is_main_process:
                return None
            return get_dataloader(
                args.data_root,
                args.val_split,
                args.batch_size,
                n_mels=args.n_mels,
                augment=False,
                num_workers=args.num_workers,
                download=download_flag,
                distributed=False,
            )

        if is_distributed and args.download:
            if is_main_process:
                train_loaders, train_samplers = _build_train_loaders(download_flag=True)
                val_loader = _build_val_loader(download_flag=True)
            dist.barrier()
            if not is_main_process:
                train_loaders, train_samplers = _build_train_loaders(download_flag=False)
                val_loader = _build_val_loader(download_flag=False)
        else:
            train_loaders, train_samplers = _build_train_loaders(download_flag=args.download)
            val_loader = _build_val_loader(download_flag=args.download)

        eval_lm_decoder = None
        if is_main_process and args.eval_lm_path:
            eval_lm_decoder = build_lm_decoder(
                vocab=VOCAB,
                lm_path=args.eval_lm_path,
                blank_idx=BLANK_IDX,
                alpha=args.eval_lm_alpha,
                beta=args.eval_lm_beta,
            )
            print(
                "Validation LM decoding enabled: "
                f"path={args.eval_lm_path} alpha={args.eval_lm_alpha} "
                f"beta={args.eval_lm_beta} beam_width={args.eval_lm_beam_width}"
            )

        train_len = combined_len(train_loaders)

        model = ConformerASR(
            n_mels=args.n_mels,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            vocab_size=VOCAB_SIZE,
            conv_kernel_size=args.conv_kernel,
        ).to(device)
        if is_distributed:
            ddp_kwargs = (
                {"device_ids": [local_rank], "output_device": local_rank}
                if device.type == "cuda"
                else {}
            )
            model = DDP(model, **ddp_kwargs)
        model_for_ckpt = model.module if isinstance(model, DDP) else model

        if is_main_process:
            num_params = sum(p.numel() for p in model_for_ckpt.parameters() if p.requires_grad)
            print(f"Model parameters: {num_params / 1e6:.1f}M")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        steps_per_epoch = math.ceil(train_len / max(1, args.accum_steps))
        total_steps = steps_per_epoch * args.epochs
        scheduler = WarmupCosineScheduler(
            optimizer,
            args.warmup_steps,
            total_steps,
            peak_lr=args.lr,
        )

        scaler = GradScaler(device=device.type, enabled=(device.type == "cuda"))

        start_epoch = 0
        best_wer = float("inf")

        resume_path = args.resume_path
        if resume_path is None and args.auto_resume:
            auto_path = os.path.join(args.ckpt_dir, "last.pt")
            if os.path.exists(auto_path):
                resume_path = auto_path
                if is_main_process:
                    print(f"Auto-resume checkpoint found: {resume_path}")

        if resume_path:
            if not os.path.exists(resume_path):
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model_for_ckpt.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if args.override_lr_on_resume:
                for pg in optimizer.param_groups:
                    pg["lr"] = args.lr
            if not args.reset_scheduler_on_resume:
                if "scheduler" in ckpt:
                    scheduler.load_state_dict(ckpt["scheduler"])
                else:
                    scheduler.load_state_dict({"step_count": ckpt.get("step", 0)})
            else:
                scheduler.step_count = 0
                for pg in optimizer.param_groups:
                    pg["lr"] = args.lr
            if "scaler" in ckpt and ckpt["scaler"] is not None:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_wer = ckpt.get("best_wer", float("inf"))
            if is_main_process:
                resume_mode = (
                    "reset_scheduler"
                    if args.reset_scheduler_on_resume
                    else f"restore_scheduler(step={scheduler.step_count})"
                )
                print(
                    f"Resumed from {resume_path} | next epoch: {start_epoch + 1} / {args.epochs} | "
                    f"best WER: {best_wer:.2%} | {resume_mode} | "
                    f"lr_now: {optimizer.param_groups[0]['lr']:.2e}"
                )
            if start_epoch >= args.epochs:
                if is_main_process:
                    print(
                        f"Checkpoint already reached epoch {start_epoch}. "
                        f"Increase --epochs above {start_epoch} to continue training."
                    )
                return

        for epoch_idx in range(start_epoch, args.epochs):
            t0 = time.time()
            if is_main_process:
                print("\n" + "=" * 60)
                print(f"Epoch {epoch_idx + 1}/{args.epochs}")
                print("=" * 60)

            for sampler in train_samplers:
                sampler.set_epoch(epoch_idx)

            train_loader = combined_loader(train_loaders, shuffle=True)
            global_step_offset = epoch_idx * train_len

            train_loss, step_rows, num_batches = train_one_epoch(
                model,
                train_loader,
                train_len,
                optimizer,
                scheduler,
                scaler,
                device,
                args.grad_clip,
                args.accum_steps,
                epoch_num=epoch_idx + 1,
                global_step_offset=global_step_offset,
                is_distributed=is_distributed,
                is_main_process=is_main_process,
            )

            if is_distributed:
                dist.barrier()

            if is_main_process:
                token_prune = args.beam_token_prune if args.beam_token_prune > 0 else None
                val_loss, val_wer = evaluate(
                    model_for_ckpt,
                    val_loader,
                    device,
                    beam_size=args.beam_size,
                    token_prune=token_prune,
                    lm_decoder=eval_lm_decoder,
                    lm_beam_width=args.eval_lm_beam_width,
                )
            else:
                val_loss, val_wer = 0.0, 0.0

            if is_distributed:
                val_metrics = torch.tensor([val_loss, val_wer], device=device, dtype=torch.float64)
                dist.broadcast(val_metrics, src=0)
                val_loss = float(val_metrics[0].item())
                val_wer = float(val_metrics[1].item())

            elapsed = time.time() - t0

            if is_main_process:
                print(f"\nEpoch {epoch_idx + 1} summary:")
                print(f"  Train loss: {train_loss:.4f}")
                print(f"  Val loss:   {val_loss:.4f}")
                print(f"  Val WER:    {val_wer:.2%}")
                print(f"  Time:       {elapsed:.0f}s")

                epoch_rows = step_rows + [
                    {
                        "record_type": "epoch_summary",
                        "epoch": epoch_idx + 1,
                        "step": num_batches,
                        "global_step": global_step_offset + num_batches,
                        "optimizer_step": scheduler.step_count,
                        "train_step_loss": "",
                        "train_running_loss": "",
                        "train_epoch_loss": train_loss,
                        "val_loss": val_loss,
                        "val_wer": val_wer,
                        "lr": scheduler.get_lr(),
                        "elapsed_sec": elapsed,
                        "best_wer": min(best_wer, val_wer),
                    }
                ]
                append_csv_rows(log_csv_path, epoch_rows)

                ckpt = {
                    "epoch": epoch_idx,
                    "model": model_for_ckpt.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": scheduler.step_count,
                    "val_loss": val_loss,
                    "val_wer": val_wer,
                    "best_wer": min(best_wer, val_wer),
                    "config": {
                        "data_root": args.data_root,
                        "train_splits": args.train_splits,
                        "val_split": args.val_split,
                        "d_model": args.d_model,
                        "num_heads": args.num_heads,
                        "num_layers": args.num_layers,
                        "n_mels": args.n_mels,
                        "conv_kernel": args.conv_kernel,
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "accum_steps": args.accum_steps,
                        "lr": args.lr,
                        "warmup_steps": args.warmup_steps,
                        "weight_decay": args.weight_decay,
                        "grad_clip": args.grad_clip,
                        "num_workers": args.num_workers,
                        "beam_size": args.beam_size,
                        "beam_token_prune": args.beam_token_prune,
                        "eval_lm_path": args.eval_lm_path,
                        "eval_lm_alpha": args.eval_lm_alpha,
                        "eval_lm_beta": args.eval_lm_beta,
                        "eval_lm_beam_width": args.eval_lm_beam_width,
                        "world_size": world_size,
                    },
                }
                torch.save(ckpt, os.path.join(args.ckpt_dir, "last.pt"))

                if val_wer < best_wer:
                    best_wer = val_wer
                    torch.save(ckpt, os.path.join(args.ckpt_dir, "best.pt"))
                    print(f"  ** New best WER: {best_wer:.2%} **")

        if is_main_process:
            print(f"\nTraining complete. Best WER: {best_wer:.2%}")
    finally:
        _cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
