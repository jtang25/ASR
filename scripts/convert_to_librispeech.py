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
    """Convert [HH:MM:SS] or [HH:MM:SS.mmm] or bare variants to seconds."""
    match = re.match(r"\[?(\d+):(\d+):(\d+(?:\.\d+)?)\]?", ts_str)
    if not match:
        raise ValueError(f"Invalid timestamp: {ts_str}")
    h, m = int(match.group(1)), int(match.group(2))
    s = float(match.group(3))
    return h * 3600 + m * 60 + s


def get_audio_duration_seconds(audio_path: str) -> float | None:
    """Return audio duration via ffprobe, or None if unavailable."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            text=True,
        ).strip()
        return float(out)
    except Exception:
        return None


def is_metadata_line(text: str) -> bool:
    """Check if a line is a metadata header or speaker label to be skipped."""
    stripped = text.strip().rstrip(".")

    # Common all-caps phrases that are actual speech, not metadata
    SPEECH_PHRASES = {
        "THANK YOU", "GOOD AFTERNOON", "GOOD MORNING", "GOOD EVENING",
        "YES", "NO", "OKAY", "OK", "SURE", "RIGHT", "EXACTLY",
        "THANK YOU VERY MUCH", "THANKS", "PLEASE",
    }
    if stripped.upper() in SPEECH_PHRASES:
        return False

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

    Only strips when the label follows sentence-ending punctuation or a dash,
    and requires at least 2 uppercase words to avoid stripping legitimate
    acronyms like FOMC, GDP, etc.

    Handles patterns like:
        "that you're particularly- CHAIR POWELL."  -> "that you're particularly"
        "Daniel. DANIEL AVIS."                     -> "Daniel."
    """
    # Require punctuation or dash before the all-caps name (2+ uppercase words)
    cleaned = re.sub(r'(?<=[.!?\-])\s+[A-Z]{2,}(?:\s+[A-Z]{2,})+\.?\s*$', '', text)
    # Also handle dash-separated: "particularly- CHAIR POWELL"
    cleaned = re.sub(r'\s*-\s*[A-Z]{2,}(?:\s+[A-Z]{2,})+\.?\s*$', '', cleaned)
    # Clean up trailing dashes, commas, spaces
    cleaned = re.sub(r'[\-,\s]+$', '', cleaned)
    return cleaned.strip()


def parse_transcript(filepath: str,
                     time_offset_seconds: float = 0.0,
                     drift_start_seconds: float | None = None,
                     drift_end_seconds: float | None = None) -> list[tuple[float, str]]:
    """Parse transcript file into list of (timestamp_seconds, text) tuples.

    Applies optional timestamp correction:
    - `time_offset_seconds`: constant shift applied to all timestamps.
    - `drift_start_seconds` / `drift_end_seconds`: linear drift shift from the
      first to last timestamp (e.g. start=-14.5, end=-16.0).
    Final timestamps are clamped to >= 0.
    """
    raw_entries = []
    # utf-8-sig strips BOM when present; some raw transcripts include it on line 1.
    with open(filepath, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.lstrip("\ufeff").strip()
            if not line:
                continue
            match = re.match(r"(\[?\d+:\d+:\d+(?:\.\d+)?\]?)\s*(.*)", line)
            if match:
                ts = parse_timestamp(match.group(1))
                text = match.group(2).strip()
                if text and not is_metadata_line(text):
                    text = strip_trailing_speaker_label(text)
                    if text:
                        raw_entries.append((ts, text))

    if not raw_entries:
        return []

    # Optional linear drift over the transcript timeline.
    if drift_start_seconds is None and drift_end_seconds is None:
        start_shift = 0.0
        end_shift = 0.0
    else:
        start_shift = drift_start_seconds if drift_start_seconds is not None else 0.0
        end_shift = drift_end_seconds if drift_end_seconds is not None else start_shift

    first_ts = raw_entries[0][0]
    last_ts = raw_entries[-1][0]
    span = max(1e-6, last_ts - first_ts)

    entries = []
    for ts, text in raw_entries:
        progress = (ts - first_ts) / span if span > 0 else 0.0
        drift_shift = start_shift + progress * (end_shift - start_shift)
        adj_ts = max(0.0, ts + time_offset_seconds + drift_shift)
        entries.append((adj_ts, text))

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
    - expand numbers, percentages, ordinals, dollar amounts
    - normalize whitespace
    """
    # Remove speaker markers like ">>"
    text = re.sub(r">+", "", text)

    # Remove bracketed annotations like [Cough], [Laughter], [Music], etc.
    text = re.sub(r"\[[\w\s]+\]", "", text)

    # Expand common titles/abbreviations (before lowercasing)
    text = re.sub(r'\bMr\.\s*', 'Mister ', text)
    text = re.sub(r'\bMrs\.\s*', 'Misses ', text)
    text = re.sub(r'\bDr\.\s*', 'Doctor ', text)
    text = re.sub(r'\bSt\.\s*', 'Saint ', text)
    text = re.sub(r'\bGov\.\s*', 'Governor ', text)
    text = re.sub(r'\bSen\.\s*', 'Senator ', text)
    text = re.sub(r'\bRep\.\s*', 'Representative ', text)

    # Expand acronyms with word boundaries to avoid matching inside words
    ABBREVIATIONS = {
        "U6": "u six",
        "SCP": "s c p",
        "GDP": "g d p",
        "PCE": "p c e",
        "AI": "a i",
        "FOMC": "f o m c",
        "BIS": "b i s",
        "AFP": "a f p",
        "CBS": "c b s",
        "NBC": "n b c",
        "ABC": "a b c",
        "CNN": "c n n",
        "CNBC": "c n b c",
        "CPI": "c p i",
        "QE": "q e",
        "QT": "q t",
        "IMF": "i m f",
        "ECB": "e c b",
        "Fed": "fed",
        "US": "u s",
    }
    for abbr, expansion in ABBREVIATIONS.items():
        text = re.sub(rf'\b{re.escape(abbr)}\b', expansion, text)

    # Expand dollar amounts: "$3.5 billion", "$500 million", "$100"
    def _expand_dollar(m):
        num_str = m.group(1)
        scale = m.group(2) or ""
        if "." in num_str:
            integer, decimal = num_str.split(".", 1)
            int_words = number_to_words(int(integer)) if integer else "zero"
            dec_words = " ".join(ONES[int(d)] if int(d) > 0 else "zero" for d in decimal)
            num_words = f"{int_words} point {dec_words}"
        else:
            num_words = number_to_words(int(num_str))
        scale = scale.strip().lower()
        if scale:
            return f"{num_words} {scale} dollars"
        return f"{num_words} dollars"

    text = re.sub(r'\$(\d+\.?\d*)\s*(billion|million|trillion|thousand)?', _expand_dollar, text, flags=re.IGNORECASE)

    text = expand_percentage(text)
    text = expand_fractions(text)

    # Expand ordinals: 1st, 2nd, 3rd, 4th, 21st, etc.
    ORDINAL_MAP = {
        1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
        6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
        11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
        15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth",
        19: "nineteenth", 20: "twentieth", 30: "thirtieth", 40: "fortieth",
        50: "fiftieth", 60: "sixtieth", 70: "seventieth", 80: "eightieth",
        90: "ninetieth",
    }

    def _expand_ordinal(m):
        n = int(m.group(1))
        if n in ORDINAL_MAP:
            return ORDINAL_MAP[n]
        if n < 100:
            tens = (n // 10) * 10
            ones = n % 10
            if ones == 0:
                return ORDINAL_MAP.get(tens, number_to_words(n) + "th")
            tens_word = TENS[n // 10]
            ones_ordinal = ORDINAL_MAP.get(ones, number_to_words(ones) + "th")
            return f"{tens_word} {ones_ordinal}"
        # For larger ordinals, just append "th" to the cardinal
        return number_to_words(n) + "th"

    text = re.sub(r'\b(\d+)(?:st|nd|rd|th)\b', _expand_ordinal, text)

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
                         max_duration: float = 25.0,
                         final_end_seconds: float | None = None) -> list[dict]:
    """
    Merge timestamped text fragments into sentence-level utterances.

    Handles cases where a sentence boundary occurs mid-fragment, e.g.:
        [00:00:24] people. The US economy expanded at a

    Returns list of dicts:
        {"start": float, "end": float, "text": str, "normalized": str}
    """
    # First, split entries at sentence boundaries within fragments.
    # When a fragment is split, estimate sub-timestamps proportionally by
    # character count within the interval to the next original timestamp.
    split_entries = []
    for idx, (ts, text) in enumerate(entries):
        # Get the next original timestamp to compute interval length
        if idx + 1 < len(entries):
            next_original_ts = entries[idx + 1][0]
        else:
            next_original_ts = final_end_seconds if final_end_seconds is not None else ts + 3.0

        interval = next_original_ts - ts

        # Split on sentence-ending punctuation followed by a space and uppercase letter
        # This catches "people. The" but not "U.S." or "2%."
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

        if len(parts) == 1:
            split_entries.append((ts, parts[0].strip()))
        else:
            # Distribute the interval proportionally by character length
            total_chars = sum(len(p) for p in parts)
            running_offset = 0.0
            for j, part in enumerate(parts):
                sub_ts = ts + running_offset
                char_fraction = len(part) / total_chars if total_chars > 0 else 0
                running_offset += interval * char_fraction
                split_entries.append((sub_ts, part.strip()))

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
            # Last entry: use audio end if known, else estimate +3 seconds.
            next_ts = final_end_seconds if final_end_seconds is not None else ts + 3.0

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
        if split_entries:
            end_ts = final_end_seconds if final_end_seconds is not None else split_entries[-1][0] + 3.0
        else:
            end_ts = 0.0
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


def segment_from_timestamps(entries: list[tuple[float, str]],
                            final_end_seconds: float | None = None) -> list[dict]:
    """Create one segment per timestamp interval [ts_i, ts_{i+1}).

    This mode uses only explicit transcript timestamps for boundaries:
    - start = current timestamp
    - end   = next timestamp
    The final timestamp uses audio end if provided; otherwise it is dropped.
    """
    if len(entries) < 2:
        return []

    segments = []
    for i in range(len(entries)):
        start, text = entries[i]
        if i + 1 < len(entries):
            end = entries[i + 1][0]
        elif final_end_seconds is not None:
            end = final_end_seconds
        else:
            continue
        if end <= start:
            continue
        norm = normalize_text(text)
        if not norm:
            continue
        segments.append({
            "start": start,
            "end": end,
            "text": text,
            "normalized": norm,
        })
    return segments


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
        flac_path_posix = flac_path.as_posix()

        # Add buffer around segment
        start = max(0, sent["start"] - buffer_s)
        duration = (sent["end"] - sent["start"]) + 2 * buffer_s

        # ffmpeg command to extract segment
        cmd = (
            f'ffmpeg -y -loglevel error -i "{audio_path}" '
            f"-ss {start:.3f} -t {duration:.3f} "
            f'-ar 16000 -ac 1 "{flac_path_posix}"'
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
    parser.add_argument(
        "--drift-start-seconds",
        type=float,
        default=None,
        help="Linear drift shift at transcript start (seconds)",
    )
    parser.add_argument(
        "--drift-end-seconds",
        type=float,
        default=None,
        help="Linear drift shift at transcript end (seconds)",
    )
    parser.add_argument("--min-duration", type=float, default=2.0, help="Min utterance duration in seconds")
    parser.add_argument("--max-duration", type=float, default=25.0, help="Max utterance duration in seconds")
    parser.add_argument("--buffer-ms", type=int, default=150, help="Audio buffer in ms added before/after each segment")
    parser.add_argument("--preview-only", action="store_true", help="Just preview segments, don't write files")
    parser.add_argument("--run-split", action="store_true", help="Also run ffmpeg to split audio")
    parser.add_argument(
        "--segment-mode",
        choices=["sentence", "timestamp"],
        default="sentence",
        help=(
            "sentence: merge into sentence-level chunks (default); "
            "timestamp: one segment per consecutive timestamp interval"
        ),
    )
    args = parser.parse_args()

    if (args.drift_start_seconds is None) ^ (args.drift_end_seconds is None):
        parser.error("Use both --drift-start-seconds and --drift-end-seconds together.")

    print(f"Parsing transcript: {args.transcript}")
    print(f"  Timestamp offset: {args.time_offset_seconds:+.3f}s")
    if args.drift_start_seconds is not None and args.drift_end_seconds is not None:
        print(f"  Drift: {args.drift_start_seconds:+.3f}s -> {args.drift_end_seconds:+.3f}s")
    entries = parse_transcript(
        args.transcript,
        time_offset_seconds=args.time_offset_seconds,
        drift_start_seconds=args.drift_start_seconds,
        drift_end_seconds=args.drift_end_seconds,
    )
    print(f"  Found {len(entries)} timestamped lines")
    audio_end_seconds = get_audio_duration_seconds(args.audio)
    if audio_end_seconds is not None:
        print(f"  Audio duration: {audio_end_seconds:.3f}s")
    else:
        print("  Audio duration: unavailable (ffprobe failed); using +3.0s fallback for final segment.")

    if args.segment_mode == "timestamp":
        print("Segmenting strictly by consecutive timestamps...")
        sentences = segment_from_timestamps(entries, final_end_seconds=audio_end_seconds)
        print(f"  Timestamp-derived utterances: {len(sentences)}")
        if len(entries) >= 1 and audio_end_seconds is None:
            print("  Note: final timestamp entry is dropped (no following end timestamp).")
    else:
        print("Merging into sentences...")
        sentences = merge_into_sentences(
            entries,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            final_end_seconds=audio_end_seconds,
        )
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
