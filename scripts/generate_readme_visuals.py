#!/usr/bin/env python3
"""Generate story-driven README visuals from local ASR artifacts."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib import ticker


TIMESTAMP_RE = re.compile(
    r"^\[(?P<h>\d+):(?P<m>\d+):(?P<s>\d+(?:\.\d+)?)\]\s*(?P<text>.*)$"
)
WORD_RE = re.compile(r"[A-Za-z']+")

BG = "#060f1d"
PANEL_BG = "#0f172a"
GRID = "#1f314f"
TXT = "#dbe9ff"
MUTED = "#8aa4c8"
ACCENT = "#59c3ff"
ACCENT_2 = "#66e0a3"
ACCENT_3 = "#fbbf24"
ACCENT_4 = "#ff7b72"


@dataclass
class TrainStep:
    global_step: float
    train_step_loss: float
    train_running_loss: float


@dataclass
class EpochSummary:
    epoch: float
    global_step: float
    train_epoch_loss: float
    val_loss: float
    val_wer: float


@dataclass
class TimestampLine:
    start_sec: float
    text: str
    words: int


@dataclass
class ChapterStats:
    chapter: str
    asr_segments: int
    asr_words: int
    asr_minutes: float
    final_segments: int
    final_words: int


def _to_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def parse_training_log(path: Path) -> tuple[list[TrainStep], list[EpochSummary]]:
    train_steps: list[TrainStep] = []
    epoch_summaries: list[EpochSummary] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record_type = row.get("record_type", "")
            if record_type == "train_step":
                train_steps.append(
                    TrainStep(
                        global_step=_to_float(row.get("global_step")),
                        train_step_loss=_to_float(row.get("train_step_loss")),
                        train_running_loss=_to_float(row.get("train_running_loss")),
                    )
                )
            elif record_type == "epoch_summary":
                epoch_summaries.append(
                    EpochSummary(
                        epoch=_to_float(row.get("epoch")),
                        global_step=_to_float(row.get("global_step")),
                        train_epoch_loss=_to_float(row.get("train_epoch_loss")),
                        val_loss=_to_float(row.get("val_loss")),
                        val_wer=_to_float(row.get("val_wer")),
                    )
                )
    train_steps.sort(key=lambda x: x.global_step)
    epoch_summaries.sort(key=lambda x: x.epoch)
    return train_steps, epoch_summaries


def parse_timestamped_file(path: Path) -> list[TimestampLine]:
    lines: list[TimestampLine] = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            match = TIMESTAMP_RE.match(raw.strip())
            if not match:
                continue
            h = int(match.group("h"))
            m = int(match.group("m"))
            s = float(match.group("s"))
            text = match.group("text").strip()
            total_sec = (h * 3600) + (m * 60) + s
            words = len(WORD_RE.findall(text))
            lines.append(TimestampLine(start_sec=total_sec, text=text, words=words))
    return lines


def estimate_segment_durations(lines: list[TimestampLine]) -> list[float]:
    if not lines:
        return []
    starts = [line.start_sec for line in lines]
    gaps = [max(0.15, starts[i + 1] - starts[i]) for i in range(len(starts) - 1)]
    fallback = median(gaps) if gaps else 3.0
    durations = gaps + [fallback]
    return [min(max(d, 0.15), 20.0) for d in durations]


def parse_final_transcript(path: Path) -> tuple[int, int]:
    segments = 0
    words = 0
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            text = parts[1] if len(parts) > 1 else ""
            segments += 1
            words += len(WORD_RE.findall(text))
    return segments, words


def collect_chapter_stats(asr_root: Path, final_root: Path) -> tuple[list[ChapterStats], dict[str, list[TimestampLine]]]:
    chapter_lines: dict[str, list[TimestampLine]] = {}
    for asr_path in sorted(asr_root.glob("*/*.asr_timestamped.txt")):
        chapter = asr_path.parent.name
        chapter_lines[chapter] = parse_timestamped_file(asr_path)

    chapter_stats: list[ChapterStats] = []
    chapters = sorted(chapter_lines.keys())
    for chapter in chapters:
        lines = chapter_lines[chapter]
        durations = estimate_segment_durations(lines)
        asr_segments = len(lines)
        asr_words = sum(line.words for line in lines)
        asr_minutes = sum(durations) / 60.0

        trans_path_candidates = sorted((final_root / chapter).glob("*.trans.txt"))
        final_segments = 0
        final_words = 0
        if trans_path_candidates:
            final_segments, final_words = parse_final_transcript(trans_path_candidates[0])

        chapter_stats.append(
            ChapterStats(
                chapter=chapter,
                asr_segments=asr_segments,
                asr_words=asr_words,
                asr_minutes=asr_minutes,
                final_segments=final_segments,
                final_words=final_words,
            )
        )
    return chapter_stats, chapter_lines


def moving_average(values: list[float], window: int) -> np.ndarray:
    if not values:
        return np.array([])
    arr = np.asarray(values, dtype=float)
    if window <= 1:
        return arr
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(arr, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def setup_matplotlib_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL_BG,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TXT,
            "axes.titlecolor": TXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "text.color": TXT,
            "savefig.facecolor": BG,
            "savefig.edgecolor": BG,
            "font.size": 10,
            "font.family": "DejaVu Sans",
        }
    )


def _style_axis(ax) -> None:
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(alpha=0.35, linewidth=0.7)
    ax.tick_params(colors=MUTED)


def plot_pipeline_story(
    out_path: Path, chapter_stats: list[ChapterStats], epoch_summaries: list[EpochSummary]
) -> None:
    total_asr_segments = sum(s.asr_segments for s in chapter_stats)
    total_final_segments = sum(s.final_segments for s in chapter_stats)
    total_minutes = sum(s.asr_minutes for s in chapter_stats)
    total_asr_words = sum(s.asr_words for s in chapter_stats)
    total_final_words = sum(s.final_words for s in chapter_stats)

    valid_wer = [e for e in epoch_summaries if not math.isnan(e.val_wer)]
    best = min(valid_wer, key=lambda e: e.val_wer) if valid_wer else None
    final_epoch = epoch_summaries[-1] if epoch_summaries else None
    compression = (
        (total_final_segments / total_asr_segments) if total_asr_segments else math.nan
    )

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.03,
        0.95,
        "ASR PROJECT STORYBOARD",
        fontsize=22,
        fontweight="bold",
        color=TXT,
        va="top",
    )
    ax.text(
        0.03,
        0.90,
        "From timestamped long-form speech to trainable utterances, then into Conformer training and decode paths.",
        fontsize=11,
        color=MUTED,
        va="top",
    )

    def draw_box(x: float, y: float, w: float, h: float, title: str, body: str, edge: str) -> None:
        patch = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            linewidth=1.8,
            edgecolor=edge,
            facecolor=PANEL_BG,
            alpha=1.0,
        )
        ax.add_patch(patch)
        ax.text(x + 0.02, y + h - 0.06, title, fontsize=13, fontweight="bold", color=TXT, va="top")
        ax.text(x + 0.02, y + h - 0.11, body, fontsize=10, color=MUTED, va="top", linespacing=1.45)

    src_body = (
        "Source corpus\n"
        f"  - {len(chapter_stats)} chapter transcripts\n"
        f"  - {total_asr_segments:,} ASR timestamp lines\n"
        f"  - ~{total_minutes:,.1f} minutes\n"
        f"  - {total_asr_words:,} detected words"
    )
    prep_body = (
        "Data shaping (`scripts/convert_to_librispeech.py`)\n"
        f"  - ASR lines -> utterance units\n"
        f"  - {total_asr_segments:,} -> {total_final_segments:,} segments\n"
        f"  - compression ratio: {compression:.2f}\n"
        f"  - normalized words: {total_final_words:,}"
    )

    if best is not None and final_epoch is not None:
        train_body = (
            "Model optimization (`train.py`)\n"
            f"  - epoch summaries: {len(epoch_summaries)}\n"
            f"  - best val WER: {best.val_wer:.3f} (epoch {int(best.epoch)})\n"
            f"  - final val loss: {final_epoch.val_loss:.3f}\n"
            f"  - global step: {int(final_epoch.global_step):,}"
        )
    else:
        train_body = "Model optimization (`train.py`)\n  - epoch summaries unavailable"

    decode_body = (
        "Decode and deployment paths\n"
        "  - CTC beam / optional KenLM fusion\n"
        "  - RNN-T greedy/beam streaming decode\n"
        "  - checkpoints + CSV logs + plotting\n"
        "  - custom transcript conversion pipeline"
    )

    x_left = 0.06
    x_right = 0.54
    y_top = 0.56
    y_bottom = 0.16
    box_w = 0.40
    box_h = 0.28

    draw_box(x_left, y_top, box_w, box_h, "1. Ingest", src_body, ACCENT)
    draw_box(x_right, y_top, box_w, box_h, "2. Prepare", prep_body, ACCENT_2)
    draw_box(x_right, y_bottom, box_w, box_h, "3. Train", train_body, ACCENT_3)
    draw_box(x_left, y_bottom, box_w, box_h, "4. Decode", decode_body, ACCENT_4)

    def arrow(x1: float, y1: float, x2: float, y2: float, color: str = "#94a3b8") -> None:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", lw=2.2, color=color, shrinkA=4, shrinkB=4),
        )

    # Clockwise square flow: ingest -> prepare -> train -> decode.
    arrow(x_left + box_w, y_top + box_h * 0.50, x_right, y_top + box_h * 0.50, ACCENT)
    arrow(x_right + box_w * 0.50, y_top, x_right + box_w * 0.50, y_bottom + box_h, ACCENT_2)
    arrow(x_right, y_bottom + box_h * 0.50, x_left + box_w, y_bottom + box_h * 0.50, ACCENT_3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_training_dynamics(
    out_path: Path, train_steps: list[TrainStep], epoch_summaries: list[EpochSummary]
) -> None:
    step_x = [s.global_step for s in train_steps if not math.isnan(s.global_step)]
    step_loss = [s.train_step_loss for s in train_steps if not math.isnan(s.train_step_loss)]
    running_loss = [s.train_running_loss for s in train_steps if not math.isnan(s.train_running_loss)]

    smooth_window = max(150, len(step_loss) // 120) if step_loss else 150
    smooth_loss = moving_average(step_loss, smooth_window)

    epochs = [e.epoch for e in epoch_summaries]
    train_epoch_loss = [e.train_epoch_loss for e in epoch_summaries]
    val_loss = [e.val_loss for e in epoch_summaries]
    val_wer = [e.val_wer for e in epoch_summaries]

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, hspace=0.25)
    ax_top = fig.add_subplot(gs[0])
    ax_bottom = fig.add_subplot(gs[1])

    if step_x and step_loss:
        step_loss_logsafe = [v if v > 0 else np.nan for v in step_loss]
        smooth_loss_logsafe = [v if v > 0 else np.nan for v in smooth_loss]
        ax_top.plot(step_x, step_loss_logsafe, color=ACCENT, alpha=0.08, linewidth=1.0, label="train_step_loss")
        ax_top.plot(step_x, smooth_loss_logsafe, color=ACCENT_2, linewidth=2.0, label=f"moving_avg ({smooth_window})")
    if step_x and running_loss:
        running_loss_logsafe = [v if v > 0 else np.nan for v in running_loss]
        ax_top.plot(step_x, running_loss_logsafe, color=ACCENT_3, alpha=0.95, linewidth=1.3, label="train_running_loss")
    ax_top.set_title("Training Dynamics: Step-Level Loss")
    ax_top.set_xlabel("global step")
    ax_top.set_ylabel("loss (log scale)")
    ax_top.set_yscale("log", nonpositive="clip")
    ax_top.legend(loc="upper right", frameon=False)
    _style_axis(ax_top)

    train_epoch_loss_logsafe = [v if v > 0 else np.nan for v in train_epoch_loss]
    val_loss_logsafe = [v if v > 0 else np.nan for v in val_loss]
    ax_bottom.plot(epochs, train_epoch_loss_logsafe, marker="o", color=ACCENT_2, linewidth=1.8, label="train_epoch_loss")
    ax_bottom.plot(epochs, val_loss_logsafe, marker="o", color=ACCENT_3, linewidth=1.8, label="val_loss")
    ax_bottom.set_xlabel("epoch")
    ax_bottom.set_ylabel("loss (log scale)")
    ax_bottom.set_yscale("log", nonpositive="clip")
    ax_bottom.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    _style_axis(ax_bottom)

    ax_wer = ax_bottom.twinx()
    ax_wer.plot(epochs, val_wer, marker="D", color=ACCENT_4, linewidth=1.7, label="val_wer")
    ax_wer.set_ylabel("WER", color=ACCENT_4)
    ax_wer.tick_params(axis="y", colors=ACCENT_4)
    for spine in ax_wer.spines.values():
        spine.set_color(GRID)

    valid_wer = [e for e in epoch_summaries if not math.isnan(e.val_wer)]
    if valid_wer:
        best = min(valid_wer, key=lambda e: e.val_wer)
        ax_wer.scatter([best.epoch], [best.val_wer], s=70, color=ACCENT_4, zorder=5)
        ax_wer.annotate(
            f"best WER {best.val_wer:.3f}\n@ epoch {int(best.epoch)}",
            xy=(best.epoch, best.val_wer),
            xytext=(best.epoch + 1.0, best.val_wer + 0.06),
            color=TXT,
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2),
        )

    lines1, labels1 = ax_bottom.get_legend_handles_labels()
    lines2, labels2 = ax_wer.get_legend_handles_labels()
    ax_bottom.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=False)
    ax_bottom.set_title("Epoch-Level Validation Behavior")

    fig.suptitle("Conformer CTC Stage: Optimization Story", fontsize=16, fontweight="bold", y=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_dataset_footprint(out_path: Path, chapter_stats: list[ChapterStats]) -> None:
    chapters = [s.chapter for s in chapter_stats]
    asr_segments = np.array([s.asr_segments for s in chapter_stats], dtype=float)
    final_segments = np.array([s.final_segments for s in chapter_stats], dtype=float)
    asr_minutes = np.array([s.asr_minutes for s in chapter_stats], dtype=float)
    asr_words = np.array([s.asr_words for s in chapter_stats], dtype=float)
    final_words = np.array([s.final_words for s in chapter_stats], dtype=float)
    compression = np.divide(final_segments, asr_segments, out=np.zeros_like(final_segments), where=asr_segments > 0)
    asr_wpm = np.divide(asr_words, asr_minutes, out=np.zeros_like(asr_words), where=asr_minutes > 0)

    x = np.arange(len(chapters))
    width = 0.36

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

    ax_left.bar(x - width / 2, asr_segments, width=width, color=ACCENT, alpha=0.85, label="ASR timestamp segments")
    ax_left.bar(x + width / 2, final_segments, width=width, color=ACCENT_2, alpha=0.9, label="Final training utterances")
    ax_left.set_title("Chapter Density: Raw ASR vs Final Utterances")
    ax_left.set_xticks(x, chapters)
    ax_left.set_ylabel("segment count")
    ax_left.legend(frameon=False, loc="upper right")
    _style_axis(ax_left)

    ax_right.plot(chapters, asr_wpm, marker="o", linewidth=2.0, color=ACCENT_3, label="ASR words/min")
    ax_right.plot(chapters, compression * 100.0, marker="D", linewidth=1.8, color=ACCENT_4, label="utterance compression (%)")
    ax_right.set_title("Chapter Rhythm and Compression")
    ax_right.set_ylabel("rate")
    ax_right.set_xlabel("chapter")
    _style_axis(ax_right)
    ax_right.legend(frameon=False, loc="upper right")

    total_asr = int(np.sum(asr_segments))
    total_final = int(np.sum(final_segments))
    total_minutes = np.sum(asr_minutes)
    total_final_words = int(np.sum(final_words))
    summary = (
        f"Totals: {total_asr:,} timestamp segments -> {total_final:,} utterances\n"
        f"Approx timeline: {total_minutes:,.1f} minutes | Final words: {total_final_words:,}"
    )
    fig.text(0.02, 0.02, summary, color=MUTED, fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_chapter_timeline(out_path: Path, chapter: str, lines: list[TimestampLine]) -> None:
    if not lines:
        return
    starts_sec = np.array([line.start_sec for line in lines], dtype=float)
    durations_sec = np.array(estimate_segment_durations(lines), dtype=float)
    words = np.array([line.words for line in lines], dtype=float)

    starts_min = starts_sec / 60.0
    widths_min = durations_sec / 60.0
    wpm = np.divide(words * 60.0, durations_sec, out=np.zeros_like(words), where=durations_sec > 0)
    smooth_wpm = moving_average(wpm.tolist(), max(7, len(wpm) // 45))

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(14, 7),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.2, 1.0], "hspace": 0.18},
    )

    ax_top.plot(starts_min, wpm, color=ACCENT, alpha=0.25, linewidth=1.0, label="segment WPM")
    ax_top.plot(starts_min, smooth_wpm, color=ACCENT_2, linewidth=2.2, label="smoothed WPM")
    ax_top.fill_between(starts_min, smooth_wpm, color=ACCENT_2, alpha=0.12)
    ax_top.set_title(f"Chapter {chapter}: Temporal Speech Dynamics")
    ax_top.set_ylabel("words per minute")
    ax_top.legend(loc="upper right", frameon=False)
    _style_axis(ax_top)

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(words.min(), words.max() if words.max() > words.min() else words.min() + 1.0)
    colors = [cmap(norm(v)) for v in words]
    ax_bottom.bar(
        starts_min,
        np.ones_like(starts_min),
        width=widths_min,
        align="edge",
        color=colors,
        edgecolor="none",
    )
    ax_bottom.set_ylim(0, 1)
    ax_bottom.set_yticks([])
    ax_bottom.set_xlabel("timeline minute")
    ax_bottom.set_title("Timestamp Segment Strip (color = words per segment)")
    _style_axis(ax_bottom)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax_bottom,
        orientation="horizontal",
        pad=0.19,
        fraction=0.08,
    )
    cbar.set_label("words in segment", color=MUTED)
    cbar.ax.tick_params(colors=MUTED)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate eye-catching README visuals from project data.")
    parser.add_argument(
        "--training-log",
        type=Path,
        default=Path("checkpoints_conformer_m_ctc_then_rnnt/ctc_stage/training_log.csv"),
        help="Path to training_log.csv",
    )
    parser.add_argument(
        "--asr-root",
        type=Path,
        default=Path("dataset/JeromePowell_asr/9999"),
        help="Root directory for ASR timestamp transcript chapters.",
    )
    parser.add_argument(
        "--final-root",
        type=Path,
        default=Path("dataset/JeromePowell/9999"),
        help="Root directory for final LibriSpeech-style chapter outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/readme"),
        help="Directory to store generated images.",
    )
    parser.add_argument(
        "--focus-chapter",
        type=str,
        default="008",
        help="Chapter id used for timeline visualization.",
    )
    args = parser.parse_args()

    if not args.training_log.exists():
        raise FileNotFoundError(f"Training log missing: {args.training_log}")
    if not args.asr_root.exists():
        raise FileNotFoundError(f"ASR root missing: {args.asr_root}")
    if not args.final_root.exists():
        raise FileNotFoundError(f"Final root missing: {args.final_root}")

    setup_matplotlib_theme()

    train_steps, epoch_summaries = parse_training_log(args.training_log)
    chapter_stats, chapter_lines = collect_chapter_stats(args.asr_root, args.final_root)

    if not chapter_stats:
        raise RuntimeError("No chapter stats parsed from ASR transcripts.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = args.output_dir / "01_pipeline_story.png"
    training_path = args.output_dir / "02_training_dynamics.png"
    footprint_path = args.output_dir / "03_dataset_footprint.png"
    timeline_path = args.output_dir / f"04_chapter_{args.focus_chapter}_timeline.png"

    plot_pipeline_story(pipeline_path, chapter_stats, epoch_summaries)
    plot_training_dynamics(training_path, train_steps, epoch_summaries)
    plot_dataset_footprint(footprint_path, chapter_stats)
    plot_chapter_timeline(
        timeline_path,
        args.focus_chapter,
        chapter_lines.get(args.focus_chapter, []),
    )

    print("Generated README visual assets:")
    print(f"- {pipeline_path}")
    print(f"- {training_path}")
    print(f"- {footprint_path}")
    if timeline_path.exists():
        print(f"- {timeline_path}")
    else:
        print(f"- skipped timeline for chapter {args.focus_chapter} (no data)")


if __name__ == "__main__":
    main()
