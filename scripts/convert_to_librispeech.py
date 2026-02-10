"""
Convert timestamped transcript to LibriSpeech format.

Usage:
    python convert_to_librispeech.py --transcript transcript.txt --audio source.flac --speaker-id 1234 --output-dir dataset/

This will:
1. Parse timestamps and merge lines into sentence-level utterances
2. Normalize text (lowercase, remove punctuation, expand numbers)
3. Create LibriSpeech directory structure
4. Generate ffmpeg commands to split audio into utterance-level .flac files
5. Write .trans.txt files
"""

import re
import os
import argparse
import subprocess
from pathlib import Path


def parse_timestamp(ts_str: str) -> float:
    """Convert [HH:MM:SS] or HH:MM:SS to seconds."""
    match = re.match(r"\[?(\d+):(\d+):(\d+)\]?", ts_str)
    if not match:
        raise ValueError(f"Invalid timestamp: {ts_str}")
    h, m, s = int(match.group(1)), int(match.group(2)), int(match.group(3))
    return h * 3600 + m * 60 + s


def is_metadata_line(text: str) -> bool:
    """Check if a line is a metadata header or speaker label to be skipped."""
    stripped = text.strip().rstrip(".")

    # Speaker labels: "CHAIR POWELL", "NICK TIMIRAOS", etc.
    # Pattern: 1-5 words, all uppercase (possibly with punctuation)
    if re.match(r'^[A-Z][A-Z\s\'.,-]+$', stripped) and len(stripped.split()) <= 5:
        return True

    # Lines that are just a year + speaker label: "2025 CHAIR POWELL"
    if re.match(r'^\d{4}\s+[A-Z][A-Z\s\'.,-]+$', stripped):
        return True

    # Transcript headers: "Transcript of Chair Powell's Press Conference..."
    if re.match(r'(?i)^transcript\s+of\b', stripped):
        return True

    return False


def strip_trailing_speaker_label(text: str) -> str:
    """Remove trailing ALL-CAPS speaker labels appended to regular text.

    Handles patterns like:
        "that you're particularly- CHAIR POWELL."  -> "that you're particularly"
        "Daniel. DANIEL AVIS."                     -> "Daniel."
    """
    # Remove trailing ALL-CAPS names (2+ uppercase words at end of line)
    cleaned = re.sub(r'[,\-\s]*\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\.?\s*$', '', text)
    # Clean up trailing dashes, commas, spaces
    cleaned = re.sub(r'[\-,\s]+$', '', cleaned)
    return cleaned.strip()


def parse_transcript(filepath: str, time_offset_seconds: float = 0.0) -> list[tuple[float, str]]:
    """Parse transcript file into list of (timestamp_seconds, text) tuples.

    `time_offset_seconds` is added to each parsed timestamp and then clamped to >= 0.
    Use negative values when transcript timestamps are late relative to the audio.
    """
    entries = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(r"(\[?\d+:\d+:\d+\]?)\s*(.*)", line)
            if match:
                ts = parse_timestamp(match.group(1))
                ts = max(0.0, ts + time_offset_seconds)
                text = match.group(2).strip()
                if text and not is_metadata_line(text):
                    text = strip_trailing_speaker_label(text)
                    if text:
                        entries.append((ts, text))
    return entries


# ---- Text normalization ----

# Common number expansions
ONES = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_to_words(n: int) -> str:
    """Convert integer to English words. Handles up to millions."""
    if n == 0:
        return "zero"
    if n < 0:
        return "negative " + number_to_words(-n)

    parts = []

    if n >= 1_000_000:
        parts.append(number_to_words(n // 1_000_000) + " million")
        n %= 1_000_000

    if n >= 1000:
        parts.append(number_to_words(n // 1000) + " thousand")
        n %= 1000

    if n >= 100:
        parts.append(ONES[n // 100] + " hundred")
        n %= 100

    if n >= 20:
        parts.append(TENS[n // 10])
        n %= 10

    if 0 < n < 20:
        parts.append(ONES[n])

    return " ".join(p for p in parts if p)


def expand_year(year: int) -> str:
    """Expand a year like 2026 -> 'twenty twenty six'."""
    if 2000 <= year <= 2009:
        return "two thousand" + (" " + ONES[year - 2000] if year > 2000 else "")
    if 2010 <= year <= 2099:
        first_half = "twenty"
        second_half = year % 100
        if second_half < 20:
            return first_half + " " + ONES[second_half]
        else:
            t = TENS[second_half // 10]
            o = ONES[second_half % 10]
            return first_half + " " + (t + " " + o if o else t)
    if 1900 <= year <= 1999:
        first = number_to_words(year // 100)
        rest = year % 100
        if rest == 0:
            return first + " hundred"
        return first + " " + number_to_words(rest)
    return number_to_words(year)


def expand_percentage(text: str) -> str:
    """Expand percentage patterns like '3.0%' -> 'three point zero percent'."""
    def _repl(m):
        num = m.group(1)
        if "." in num:
            integer, decimal = num.split(".", 1)
            int_words = number_to_words(int(integer)) if integer else "zero"
            dec_words = " ".join(ONES[int(d)] if int(d) > 0 else "zero" for d in decimal)
            return f"{int_words} point {dec_words} percent"
        else:
            return number_to_words(int(num)) + " percent"
    return re.sub(r"(\d+\.?\d*)%", _repl, text)


def expand_fractions(text: str) -> str:
    """Expand simple fractions like '3/4' -> 'three quarters'."""
    fraction_map = {
        "1/2": "one half",
        "1/3": "one third",
        "2/3": "two thirds",
        "1/4": "one quarter",
        "3/4": "three quarters",
    }
    for frac, words in fraction_map.items():
        text = text.replace(frac, words)
    return text


def normalize_text(text: str) -> str:
    """
    Normalize transcript text to LibriSpeech convention:
    - lowercase
    - remove punctuation
    - expand numbers, percentages, ordinals
    - normalize whitespace
    """
    # Remove speaker markers like ">>"
    text = re.sub(r">+", "", text)

    # Remove bracketed annotations like [Cough], [Laughter], [Music], etc.
    text = re.sub(r"\[[\w\s]+\]", "", text)

    # Expand common abbreviations
    text = text.replace("U6", "u six")
    text = text.replace("SCP", "s c p")
    text = text.replace("GDP", "g d p")
    text = text.replace("PCE", "p c e")
    text = text.replace("AI", "a i")
    text = text.replace("FOMC", "f o m c")
    text = text.replace("BIS", "b i s")
    text = text.replace("AFP", "a f p")
    text = text.replace("CBS", "c b s")
    text = text.replace("NBC", "n b c")
    text = text.replace("ABC", "a b c")
    text = text.replace("CNN", "c n n")
    text = text.replace("CNBC", "c n b c")
    text = text.replace("US", "u s")

    text = expand_percentage(text)
    text = expand_fractions(text)

    # Expand years (4-digit numbers that look like years)
    def _expand_year_match(m):
        y = int(m.group(0))
        if 1900 <= y <= 2099:
            return expand_year(y)
        return number_to_words(y)

    # Handle decimal numbers first (e.g., "3.65")
    def _expand_decimal(m):
        num = m.group(0)
        integer, decimal = num.split(".", 1)
        int_words = number_to_words(int(integer))
        dec_words = " ".join(ONES[int(d)] if int(d) > 0 else "zero" for d in decimal)
        return f"{int_words} point {dec_words}"

    text = re.sub(r"\d+\.\d+", _expand_decimal, text)

    # Expand comma-separated numbers (e.g., "22,000")
    def _expand_comma_num(m):
        num_str = m.group(0).replace(",", "")
        return number_to_words(int(num_str))

    text = re.sub(r"\d{1,3}(?:,\d{3})+", _expand_comma_num, text)

    # Expand remaining standalone numbers
    def _expand_num(m):
        n = int(m.group(0))
        if 1900 <= n <= 2099:
            return expand_year(n)
        return number_to_words(n)

    text = re.sub(r"\b\d+\b", _expand_num, text)

    # Lowercase
    text = text.lower()

    # Remove everything except lowercase letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---- Sentence segmentation ----

# Sentence-ending punctuation pattern
SENT_END = re.compile(r'[.?!](?:\s|$)')


def merge_into_sentences(entries: list[tuple[float, str]],
                         min_duration: float = 2.0,
                         max_duration: float = 25.0) -> list[dict]:
    """
    Merge timestamped text fragments into sentence-level utterances.

    Handles cases where a sentence boundary occurs mid-fragment, e.g.:
        [00:00:24] people. The US economy expanded at a

    Returns list of dicts:
        {"start": float, "end": float, "text": str, "normalized": str}
    """
    # First, split entries at sentence boundaries within fragments.
    # If a fragment contains "X. Y", split into two sub-entries both with same timestamp.
    split_entries = []
    for ts, text in entries:
        # Split on sentence-ending punctuation followed by a space and uppercase letter
        # This catches "people. The" but not "U.S." or "2%."
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        for j, part in enumerate(parts):
            split_entries.append((ts, part.strip()))

    sentences = []
    current_text = ""
    current_start = split_entries[0][0] if split_entries else 0.0

    for i, (ts, text) in enumerate(split_entries):
        if not current_text:
            current_start = ts
        current_text += " " + text if current_text else text

        # Check if this fragment ends a sentence
        has_sent_end = bool(SENT_END.search(text))

        # Get the next timestamp for the end time
        if i + 1 < len(split_entries):
            next_ts = split_entries[i + 1][0]
        else:
            # Last entry: estimate end as start + 3 seconds
            next_ts = ts + 3.0

        duration = next_ts - current_start

        # Commit sentence if we hit punctuation and meet minimum duration,
        # or if we exceed max duration
        if (has_sent_end and duration >= min_duration) or duration >= max_duration:
            sentences.append({
                "start": current_start,
                "end": next_ts,
                "text": current_text.strip(),
                "normalized": normalize_text(current_text.strip()),
            })
            current_text = ""

    # Flush remaining text
    if current_text.strip():
        end_ts = split_entries[-1][0] + 3.0 if split_entries else 0.0
        sentences.append({
            "start": current_start,
            "end": end_ts,
            "text": current_text.strip(),
            "normalized": normalize_text(current_text.strip()),
        })

    return sentences


def filter_sentences(sentences: list[dict],
                     min_duration: float = 1.5,
                     max_duration: float = 30.0,
                     min_words: int = 2) -> list[dict]:
    """Filter out segments that are too short, too long, or empty."""
    filtered = []
    for s in sentences:
        duration = s["end"] - s["start"]
        word_count = len(s["normalized"].split())
        if duration >= min_duration and duration <= max_duration and word_count >= min_words:
            filtered.append(s)
    return filtered


# ---- LibriSpeech output ----

def create_librispeech_structure(sentences: list[dict],
                                 speaker_id: str,
                                 chapter_id: str,
                                 audio_path: str,
                                 output_dir: str,
                                 buffer_ms: int = 150):
    """
    Create LibriSpeech directory structure and generate ffmpeg split commands.

    Structure:
        output_dir/speaker_id/chapter_id/speaker_id-chapter_id-NNNN.flac
        output_dir/speaker_id/chapter_id/speaker_id-chapter_id.trans.txt
    """
    chapter_dir = Path(output_dir) / speaker_id / chapter_id
    chapter_dir.mkdir(parents=True, exist_ok=True)

    trans_lines = []
    ffmpeg_commands = []
    buffer_s = buffer_ms / 1000.0

    for i, sent in enumerate(sentences):
        utt_id = f"{speaker_id}-{chapter_id}-{i:04d}"
        flac_path = chapter_dir / f"{utt_id}.flac"

        # Add buffer around segment
        start = max(0, sent["start"] - buffer_s)
        duration = (sent["end"] - sent["start"]) + 2 * buffer_s

        # ffmpeg command to extract segment
        cmd = (
            f'ffmpeg -y -i "{audio_path}" '
            f"-ss {start:.3f} -t {duration:.3f} "
            f'-ar 16000 -ac 1 "{flac_path}"'
        )
        ffmpeg_commands.append(cmd)

        # Transcript line (LibriSpeech uses UPPERCASE by convention in .trans.txt,
        # but many pipelines immediately lowercase it. We store uppercase here.)
        trans_lines.append(f"{utt_id} {sent['normalized'].upper()}")

    # Write trans.txt
    trans_path = chapter_dir / f"{speaker_id}-{chapter_id}.trans.txt"
    with open(trans_path, "w") as f:
        f.write("\n".join(trans_lines) + "\n")

    # Write ffmpeg script
    script_path = chapter_dir / "split_audio.sh"
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("set -e\n\n")
        for cmd in ffmpeg_commands:
            f.write(cmd + "\n")

    return trans_path, script_path, len(sentences)


def print_preview(sentences: list[dict], n: int = 10):
    """Print a preview of the first n sentences."""
    print(f"\n{'='*80}")
    print(f"PREVIEW (first {min(n, len(sentences))} of {len(sentences)} utterances)")
    print(f"{'='*80}\n")
    for i, s in enumerate(sentences[:n]):
        dur = s["end"] - s["start"]
        print(f"  [{i:04d}] {s['start']:.1f}s - {s['end']:.1f}s  ({dur:.1f}s)")
        print(f"         RAW:  {s['text'][:100]}{'...' if len(s['text']) > 100 else ''}")
        print(f"         NORM: {s['normalized'][:100]}{'...' if len(s['normalized']) > 100 else ''}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Convert timestamped transcript to LibriSpeech format")
    parser.add_argument("--transcript", required=True, help="Path to timestamped transcript (.txt)")
    parser.add_argument("--audio", required=True, help="Path to source audio file")
    parser.add_argument("--speaker-id", default="9999", help="Speaker ID (default: 9999)")
    parser.add_argument("--chapter-id", default="001", help="Chapter ID (default: 001)")
    parser.add_argument("--output-dir", default="./dataset", help="Output directory")
    parser.add_argument(
        "--time-offset-seconds",
        type=float,
        default=0.0,
        help="Global timestamp shift applied before segmentation (negative shifts earlier)",
    )
    parser.add_argument("--min-duration", type=float, default=2.0, help="Min utterance duration in seconds")
    parser.add_argument("--max-duration", type=float, default=25.0, help="Max utterance duration in seconds")
    parser.add_argument("--buffer-ms", type=int, default=150, help="Audio buffer in ms added before/after each segment")
    parser.add_argument("--preview-only", action="store_true", help="Just preview segments, don't write files")
    parser.add_argument("--run-split", action="store_true", help="Also run ffmpeg to split audio")
    args = parser.parse_args()

    print(f"Parsing transcript: {args.transcript}")
    print(f"  Timestamp offset: {args.time_offset_seconds:+.3f}s")
    entries = parse_transcript(args.transcript, time_offset_seconds=args.time_offset_seconds)
    print(f"  Found {len(entries)} timestamped lines")

    print("Merging into sentences...")
    sentences = merge_into_sentences(entries, min_duration=args.min_duration, max_duration=args.max_duration)
    print(f"  Merged into {len(sentences)} raw sentences")

    sentences = filter_sentences(sentences, min_duration=args.min_duration, max_duration=args.max_duration)
    print(f"  After filtering: {len(sentences)} utterances")

    durations = [s["end"] - s["start"] for s in sentences]
    total = sum(durations)
    print(f"  Total audio: {total:.0f}s ({total/60:.1f} min)")
    print(f"  Avg duration: {total/len(sentences):.1f}s")
    print(f"  Min/Max: {min(durations):.1f}s / {max(durations):.1f}s")

    print_preview(sentences)

    if args.preview_only:
        return

    print("Creating LibriSpeech structure...")
    trans_path, script_path, n = create_librispeech_structure(
        sentences, args.speaker_id, args.chapter_id,
        args.audio, args.output_dir, args.buffer_ms
    )
    print(f"  Wrote {n} entries to {trans_path}")
    print(f"  Wrote split script to {script_path}")

    if args.run_split:
        print("\nRunning ffmpeg to split audio...")
        subprocess.run(["bash", str(script_path)], check=True)
        print("  Done!")
    else:
        print(f"\nTo split the audio, run:")
        print(f"  bash {script_path}")


if __name__ == "__main__":
    main()
