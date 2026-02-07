import argparse
import math
import os
import random
import time

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from decoding import beam_search_decode
from model import ConformerASR
from preprocessing import BLANK_IDX, VOCAB_SIZE, get_dataloader, tokens_to_text


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

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument(
        "--beam-token-prune",
        type=int,
        default=0,
        help="Per-frame top-k pruning for beam search (0 disables).",
    )

    parser.add_argument("--ckpt-dir", default="./checkpoints")
    parser.add_argument(
        "--resume-path",
        default=None,
        help="Optional checkpoint path to resume from.",
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
) -> float:
    decoded = beam_search_decode(
        log_probs,
        output_lengths,
        beam_size=beam_size,
        blank_idx=BLANK_IDX,
        token_prune=token_prune,
    )
    total_words = 0
    total_errors = 0

    for i in range(len(decoded)):
        hyp_text = tokens_to_text(decoded[i])
        ref_tokens = targets[i, : target_lengths[i]].tolist()
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
        if self.step_count <= self.warmup_steps:
            lr = self.peak_lr * self.step_count / max(1, self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_epoch(model, loader, loader_len, optimizer, scheduler, scaler, device, grad_clip, accum_steps):
    model.train()
    ctc_loss_fn = nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)

    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad(set_to_none=True)

    for step, (mel, tokens, mel_lengths, token_lengths) in enumerate(loader):
        mel = mel.to(device)
        tokens = tokens.to(device)
        mel_lengths = mel_lengths.to(device)
        token_lengths = token_lengths.to(device)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            log_probs, output_lengths = model(mel, mel_lengths)
            log_probs_ctc = log_probs.transpose(0, 1)
            output_lengths = output_lengths.clamp(min=1)
            loss = ctc_loss_fn(log_probs_ctc, tokens, output_lengths, token_lengths)
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        should_step = (step + 1) % accum_steps == 0
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.item() * accum_steps
        num_batches += 1

        if (step + 1) % 100 == 0:
            avg = total_loss / num_batches
            lr_now = scheduler.get_lr()
            print(f"  step {step+1:5d} | loss {avg:.4f} | lr {lr_now:.2e}")

    if (loader_len % accum_steps) != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, beam_size: int = 10, token_prune: int | None = None):
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
        )

        total_loss += loss.item()
        total_wer += wer
        num_batches += 1

    return total_loss / max(num_batches, 1), total_wer / max(num_batches, 1)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    _set_seed(args.seed)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = _resolve_device(args.device)
    print(f"Device: {device}")

    print(f"Loading train splits: {args.train_splits} | val: {args.val_split} ...")
    train_loaders = [
        get_dataloader(
            args.data_root,
            split,
            args.batch_size,
            n_mels=args.n_mels,
            augment=True,
            num_workers=args.num_workers,
            download=args.download,
        )
        for split in args.train_splits
    ]
    train_len = combined_len(train_loaders)

    val_loader = get_dataloader(
        args.data_root,
        args.val_split,
        args.batch_size,
        n_mels=args.n_mels,
        augment=False,
        num_workers=args.num_workers,
        download=args.download,
    )

    model = ConformerASR(
        n_mels=args.n_mels,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        vocab_size=VOCAB_SIZE,
        conv_kernel_size=args.conv_kernel,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
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

    if args.resume_path:
        if not os.path.exists(args.resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_path}")
        ckpt = torch.load(args.resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.step_count = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_wer = ckpt.get("best_wer", float("inf"))
        print(f"Resumed from epoch {start_epoch}, best WER: {best_wer:.2%}")

    for epoch_idx in range(start_epoch, args.epochs):
        t0 = time.time()
        print("\n" + "=" * 60)
        print(f"Epoch {epoch_idx + 1}/{args.epochs}")
        print("=" * 60)

        train_loader = combined_loader(train_loaders, shuffle=True)

        train_loss = train_one_epoch(
            model,
            train_loader,
            train_len,
            optimizer,
            scheduler,
            scaler,
            device,
            args.grad_clip,
            args.accum_steps,
        )

        token_prune = args.beam_token_prune if args.beam_token_prune > 0 else None
        val_loss, val_wer = evaluate(
            model,
            val_loader,
            device,
            beam_size=args.beam_size,
            token_prune=token_prune,
        )
        elapsed = time.time() - t0

        print(f"\nEpoch {epoch_idx + 1} summary:")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss:   {val_loss:.4f}")
        print(f"  Val WER:    {val_wer:.2%}")
        print(f"  Time:       {elapsed:.0f}s")

        ckpt = {
            "epoch": epoch_idx,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
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
            },
        }
        torch.save(ckpt, os.path.join(args.ckpt_dir, "last.pt"))

        if val_wer < best_wer:
            best_wer = val_wer
            torch.save(ckpt, os.path.join(args.ckpt_dir, "best.pt"))
            print(f"  ** New best WER: {best_wer:.2%} **")

    print(f"\nTraining complete. Best WER: {best_wer:.2%}")


if __name__ == "__main__":
    main()
