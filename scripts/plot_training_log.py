#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path


def to_float(value: str) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def moving_average(values: list[float], window: int) -> list[float]:
    out: list[float] = []
    running_sum = 0.0
    buf: list[float] = []

    for val in values:
        if math.isnan(val):
            out.append(math.nan)
            continue
        buf.append(val)
        running_sum += val
        if len(buf) > window:
            running_sum -= buf.pop(0)
        out.append(running_sum / len(buf))
    return out


def parse_log(log_path: Path):
    train_rows: list[dict[str, float]] = []
    summary_rows: list[dict[str, float]] = []

    with log_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record_type = row.get("record_type", "")
            if record_type == "train_step":
                train_rows.append(
                    {
                        "global_step": to_float(row.get("global_step")),
                        "train_step_loss": to_float(row.get("train_step_loss")),
                        "train_running_loss": to_float(row.get("train_running_loss")),
                        "lr": to_float(row.get("lr")),
                    }
                )
            elif record_type == "epoch_summary":
                summary_rows.append(
                    {
                        "epoch": to_float(row.get("epoch")),
                        "global_step": to_float(row.get("global_step")),
                        "train_epoch_loss": to_float(row.get("train_epoch_loss")),
                        "val_loss": to_float(row.get("val_loss")),
                        "val_wer": to_float(row.get("val_wer")),
                    }
                )

    train_rows.sort(key=lambda r: (math.inf if math.isnan(r["global_step"]) else r["global_step"]))
    summary_rows.sort(key=lambda r: (math.inf if math.isnan(r["global_step"]) else r["global_step"]))
    return train_rows, summary_rows


def make_plot(log_path: Path, out_path: Path, smooth_window: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required. Install it with: pip install matplotlib "
            "or pip install -r requirements.txt"
        ) from exc

    train_rows, summary_rows = parse_log(log_path)

    if not train_rows and not summary_rows:
        raise ValueError(f"No usable rows found in {log_path}")

    train_global_step = [r["global_step"] for r in train_rows]
    train_step_loss = [r["train_step_loss"] for r in train_rows]
    train_running_loss = [r["train_running_loss"] for r in train_rows]
    train_lr = [r["lr"] for r in train_rows]
    train_step_loss_smooth = moving_average(train_step_loss, smooth_window)

    epoch_idx = [r["epoch"] for r in summary_rows]
    train_epoch_loss = [r["train_epoch_loss"] for r in summary_rows]
    val_loss = [r["val_loss"] for r in summary_rows]
    val_wer = [r["val_wer"] for r in summary_rows]

    fig, ax = plt.subplots(2, 2, figsize=(14, 8))

    if train_rows:
        ax[0, 0].plot(train_global_step, train_step_loss, alpha=0.18, label="train_step_loss")
        ax[0, 0].plot(
            train_global_step,
            train_step_loss_smooth,
            linewidth=2,
            label=f"train_step_loss_ma{smooth_window}",
        )
        ax[0, 0].plot(train_global_step, train_running_loss, linewidth=1.5, label="train_running_loss")
        ax[1, 1].plot(train_global_step, train_lr, label="lr")
    else:
        ax[0, 0].text(0.5, 0.5, "No train_step rows", ha="center", va="center")
        ax[1, 1].text(0.5, 0.5, "No train_step rows", ha="center", va="center")

    if summary_rows:
        ax[0, 1].plot(epoch_idx, train_epoch_loss, marker="o", label="train_epoch_loss")
        ax[0, 1].plot(epoch_idx, val_loss, marker="o", label="val_loss")
        ax[1, 0].plot(epoch_idx, val_wer, marker="o", label="val_wer")
    else:
        ax[0, 1].text(0.5, 0.5, "No epoch_summary rows", ha="center", va="center")
        ax[1, 0].text(0.5, 0.5, "No epoch_summary rows", ha="center", va="center")

    ax[0, 0].set_title("Training Loss vs Global Step")
    ax[0, 0].set_xlabel("global_step")
    ax[0, 0].set_ylabel("loss")
    ax[0, 0].grid(alpha=0.3)
    ax[0, 0].legend()

    ax[0, 1].set_title("Epoch Summary Loss")
    ax[0, 1].set_xlabel("epoch")
    ax[0, 1].set_ylabel("loss")
    ax[0, 1].grid(alpha=0.3)
    ax[0, 1].legend()

    ax[1, 0].set_title("Validation WER")
    ax[1, 0].set_xlabel("epoch")
    ax[1, 0].set_ylabel("WER")
    ax[1, 0].grid(alpha=0.3)
    ax[1, 0].legend()

    ax[1, 1].set_title("Learning Rate vs Global Step")
    ax[1, 1].set_xlabel("global_step")
    ax[1, 1].set_ylabel("lr")
    ax[1, 1].grid(alpha=0.3)
    ax[1, 1].legend()

    fig.suptitle(str(log_path), fontsize=10)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    print(f"Saved: {out_path}")
    plt.show()


def default_log_path() -> Path:
    preferred = Path("checkpoints_dev_other/training_log.csv")
    fallback = Path("checkpoints/training_log.csv")
    if preferred.exists():
        return preferred
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ASR training log CSV.")
    parser.add_argument(
        "log_path",
        nargs="?",
        type=Path,
        default=default_log_path(),
        help="Path to training_log.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Path to output image (default: <log_dir>/training_log_plot.png)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=100,
        help="Moving-average window for train_step_loss.",
    )
    args = parser.parse_args()

    if args.smooth_window <= 0:
        raise ValueError("--smooth-window must be > 0")

    log_path = args.log_path
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    out_path = args.out if args.out is not None else log_path.with_name("training_log_plot.png")
    make_plot(log_path=log_path, out_path=out_path, smooth_window=args.smooth_window)


if __name__ == "__main__":
    main()
