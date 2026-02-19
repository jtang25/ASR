#!/usr/bin/env python3
"""Find utterance IDs that contain requested keywords/phrases.

Scans LibriSpeech-style chapter transcripts under:
  dataset/JeromePowell/9999/<chapter>/9999-<chapter>.trans.txt

Writes a JSON map:
  {
    "term one": ["9999-001-0001", ...],
    "term two": [...]
  }
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_TERMS = [
    "our dual mandate goals",
    "the us",
    "american",
    "policy rate",
    "federal",
    "growth",
    "low",
    "payroll(s)",
    "reflects",
    "months",
    "inflation",
    "stabilize",
    "tariff",
    "maximum employment",
    "goals",
    "dual",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--transcript-root",
        default="dataset/JeromePowell/9999",
        help="Root with chapter dirs (001..008) and chapter .trans.txt files.",
    )
    p.add_argument(
        "--chapters",
        nargs="*",
        default=None,
        help="Optional chapter IDs (e.g. 001 002). Defaults to all numeric directories.",
    )
    p.add_argument(
        "--terms",
        default=None,
        help="Comma-separated terms/phrases. Defaults to the built-in list.",
    )
    p.add_argument(
        "--terms-file",
        default=None,
        help="Optional file with one term/phrase per line.",
    )
    p.add_argument(
        "--output-json",
        default="dataset/JeromePowell_asr/keyword_occurrences_001_008.json",
        help="Output JSON path.",
    )
    return p.parse_args()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_terms(args: argparse.Namespace) -> list[str]:
    if args.terms_file:
        lines = Path(args.terms_file).read_text(encoding="utf-8").splitlines()
        raw = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    elif args.terms:
        raw = [t.strip() for t in args.terms.split(",") if t.strip()]
    else:
        raw = list(DEFAULT_TERMS)

    deduped: list[str] = []
    seen: set[str] = set()
    for term in raw:
        key = term.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(term.strip())
    return deduped


def expand_optional_plural(term: str) -> list[str]:
    if "(s)" in term:
        return [term.replace("(s)", ""), term.replace("(s)", "s")]
    return [term]


def expand_us_variants(tokens: list[str]) -> list[list[str]]:
    seqs: list[list[str]] = [[]]
    for token in tokens:
        if token == "us":
            next_seqs: list[list[str]] = []
            for seq in seqs:
                next_seqs.append(seq + ["us"])
                next_seqs.append(seq + ["u", "s"])
            seqs = next_seqs
        else:
            seqs = [seq + [token] for seq in seqs]
    return seqs


def build_term_sequences(term: str) -> list[list[str]]:
    variants: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for variant in expand_optional_plural(term):
        tokens = normalize_text(variant).split()
        if not tokens:
            continue
        for seq in expand_us_variants(tokens):
            key = tuple(seq)
            if key in seen:
                continue
            seen.add(key)
            variants.append(seq)
    return variants


def has_sequence(haystack: list[str], needle: list[str]) -> bool:
    n = len(needle)
    if n == 0:
        return False
    if n == 1:
        return needle[0] in set(haystack)
    limit = len(haystack) - n + 1
    for i in range(limit):
        if haystack[i : i + n] == needle:
            return True
    return False


def iter_chapter_transcripts(root: Path, selected: list[str] | None) -> list[Path]:
    if selected:
        chapter_dirs = [root / ch.zfill(3) for ch in selected]
    else:
        chapter_dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit())

    trans_paths: list[Path] = []
    for chapter_dir in chapter_dirs:
        chapter = chapter_dir.name
        trans_path = chapter_dir / f"9999-{chapter}.trans.txt"
        if trans_path.exists():
            trans_paths.append(trans_path)
    return trans_paths


def main() -> None:
    args = parse_args()
    transcript_root = Path(args.transcript_root)
    if not transcript_root.exists():
        raise FileNotFoundError(f"Transcript root not found: {transcript_root}")

    terms = parse_terms(args)
    if not terms:
        raise RuntimeError("No terms to search.")

    term_to_sequences = {term: build_term_sequences(term) for term in terms}
    term_to_ids: dict[str, list[str]] = {term: [] for term in terms}
    term_seen_ids: dict[str, set[str]] = {term: set() for term in terms}

    trans_paths = iter_chapter_transcripts(transcript_root, args.chapters)
    if not trans_paths:
        raise RuntimeError(f"No chapter transcript files found under: {transcript_root}")

    total_lines = 0
    for trans_path in trans_paths:
        for line in trans_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) != 2:
                continue
            utt_id, text = parts
            tokens = normalize_text(text).split()
            total_lines += 1

            for term, sequences in term_to_sequences.items():
                if any(has_sequence(tokens, seq) for seq in sequences):
                    if utt_id not in term_seen_ids[term]:
                        term_seen_ids[term].add(utt_id)
                        term_to_ids[term].append(utt_id)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(term_to_ids, indent=2), encoding="utf-8")

    print(f"Wrote: {out_path}")
    print(f"Scanned utterances: {total_lines}")
    for term in terms:
        print(f"{term}: {len(term_to_ids[term])}")


if __name__ == "__main__":
    main()
