import argparse
import os
import time
import math

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from preprocessing import get_dataloader, tokens_to_text, VOCAB_SIZE, BLANK_IDX
from model import ConformerASR

# ---------------------------------------------------------------------------
# Greedy CTC Decoding
# ---------------------------------------------------------------------------

def greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor):
    """
    Greedy CTC decoding: argmax -> collapse repeats -> remove blanks.

    Args:
        log_probs: (B, T, V) log probabilities
        lengths: (B,) valid lengths
    Returns:
        List of decoded token lists
    """
    predictions = log_probs.argmax(dim=-1)  # (B, T)
    decoded = []
    for i in range(predictions.size(0)):
        seq = predictions[i, : lengths[i]].tolist()
        # Collapse repeated tokens
        collapsed = []
        prev = None
        for token in seq:
            if token != prev:
                collapsed.append(token)
            prev = token
        # Remove blanks
        collapsed = [t for t in collapsed if t != BLANK_IDX]
        decoded.append(collapsed)
    return decoded


# ---------------------------------------------------------------------------
# Word Error Rate
# ---------------------------------------------------------------------------

def edit_distance(ref: list, hyp: list) -> int:
    """Compute Levenshtein edit distance between two sequences."""
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
) -> float:
    """Compute Word Error Rate for a batch."""
    decoded = greedy_decode(log_probs, output_lengths)
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
            lr = self.peak_lr * self.step_count / self.warmup_steps
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
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, grad_clip, accum_steps):
    model.train()
    ctc_loss_fn = nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)

    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    for step, (mel, tokens, mel_lengths, token_lengths) in enumerate(loader):
        mel = mel.to(device)
        tokens = tokens.to(device)
        mel_lengths = mel_lengths.to(device)
        token_lengths = token_lengths.to(device)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            log_probs, output_lengths = model(mel, mel_lengths)

            # CTC loss expects (T, B, C) format
            log_probs_ctc = log_probs.transpose(0, 1)  # (T, B, V)

            # Clamp output lengths to be >= target lengths (safety check)
            output_lengths = output_lengths.clamp(min=1)

            loss = ctc_loss_fn(log_probs_ctc, tokens, output_lengths, token_lengths)
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * accum_steps
        num_batches += 1

        if (step + 1) % 100 == 0:
            avg = total_loss / num_batches
            lr = scheduler.get_lr()
            print(f"  step {step+1:5d} | loss {avg:.4f} | lr {lr:.2e}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
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
        wer = compute_wer(log_probs, output_lengths, tokens, token_lengths)

        total_loss += loss.item()
        total_wer += wer
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_wer = total_wer / max(num_batches, 1)
    return avg_loss, avg_wer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Conformer ASR Training")
    # Data
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--train_split", type=str, default="train-clean-100")
    parser.add_argument("--val_split", type=str, default="dev-clean")
    # Model
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--conv_kernel", type=int, default=31)
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accum_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--num_workers", type=int, default=4)
    # Checkpoints
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Data ----
    print(f"Loading {args.train_split} and {args.val_split}...")
    train_loader = get_dataloader(
        args.data_root, args.train_split, args.batch_size,
        n_mels=args.n_mels, augment=True, num_workers=args.num_workers,
    )
    val_loader = get_dataloader(
        args.data_root, args.val_split, args.batch_size,
        n_mels=args.n_mels, augment=False, num_workers=args.num_workers,
    )

    # ---- Model ----
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

    # ---- Optimizer & Scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.98), eps=1e-9,
    )

    steps_per_epoch = math.ceil(len(train_loader) / args.accum_steps)
    total_steps = steps_per_epoch * args.epochs
    scheduler = WarmupCosineScheduler(
        optimizer, args.warmup_steps, total_steps, peak_lr=args.lr,
    )

    scaler = GradScaler(device=device.type, enabled=(device.type == "cuda"))

    start_epoch = 0
    best_wer = float("inf")

    # ---- Resume ----
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.step_count = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_wer = ckpt.get("best_wer", float("inf"))
        print(f"Resumed from epoch {start_epoch}, best WER: {best_wer:.2%}")

    # ---- Training Loop ----
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, args.grad_clip, args.accum_steps,
        )

        val_loss, val_wer = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        print(f"\nEpoch {epoch+1} summary:")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss:   {val_loss:.4f}")
        print(f"  Val WER:    {val_wer:.2%}")
        print(f"  Time:       {elapsed:.0f}s")

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": scheduler.step_count,
            "val_loss": val_loss,
            "val_wer": val_wer,
            "best_wer": min(best_wer, val_wer),
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.ckpt_dir, "last.pt"))

        if val_wer < best_wer:
            best_wer = val_wer
            torch.save(ckpt, os.path.join(args.ckpt_dir, "best.pt"))
            print(f"  ** New best WER: {best_wer:.2%} **")

    print(f"\nTraining complete. Best WER: {best_wer:.2%}")


if __name__ == "__main__":
    main()