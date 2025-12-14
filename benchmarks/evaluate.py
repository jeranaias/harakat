#!/usr/bin/env python3
"""
Evaluation script for Harakat Arabic diacritization.

Calculates Diacritic Error Rate (DER) and Word Error Rate (WER)
on the Tashkeela test set using the paper's methodology.

DER is calculated per-character-position (not per-diacritic-mark):
  - Each diacritizable character position is counted once
  - Errors are mismatches in the diacritics at each position

Metrics reported:
  - DER (with case): All diacritic positions including word-final
  - DER (no case): Excluding word-final position (case vowels)
  - WER (with case): Words with any diacritic error
  - WER (no case): Words with non-final diacritic errors

Usage:
    python evaluate.py                    # Evaluate on default test set
    python evaluate.py --file custom.txt  # Evaluate on custom file
    python evaluate.py --sample 1000      # Evaluate on random sample
"""

import sys
import argparse
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, '..')
from harakat import diacritize

# Arabic diacritical marks
HARAKAT = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670')


def strip_diacritics(text: str) -> str:
    """Remove all diacritical marks from text."""
    return ''.join(c for c in text if c not in HARAKAT)


def extract_diacritic_sequence(word: str) -> Tuple[List[str], List[str]]:
    """
    Extract base characters and their associated diacritics.

    Returns:
        (base_chars, diacritics): Lists where diacritics[i] contains
        the diacritics following base_chars[i]
    """
    base_chars = []
    diacritics = []
    current_diacritics = []

    for c in word:
        if c in HARAKAT:
            current_diacritics.append(c)
        else:
            if base_chars:
                # Normalize by sorting (shadda + vowel vs vowel + shadda)
                diacritics.append(''.join(sorted(current_diacritics)))
            base_chars.append(c)
            current_diacritics = []

    # Don't forget the final character's diacritics
    if base_chars:
        diacritics.append(''.join(sorted(current_diacritics)))

    return base_chars, diacritics


def calculate_word_metrics(pred_word: str, gold_word: str,
                           exclude_final: bool = False) -> Dict:
    """
    Calculate per-position diacritic errors for a single word.

    Args:
        pred_word: Predicted diacritized word
        gold_word: Gold standard diacritized word
        exclude_final: If True, exclude final position (for "no case" metric)

    Returns:
        Dictionary with 'char_errors', 'char_total', 'word_error'
        or {'skip': True} if words don't align
    """
    pred_base = strip_diacritics(pred_word)
    gold_base = strip_diacritics(gold_word)

    # Must have same base form
    if pred_base != gold_base:
        return {'skip': True}

    pred_bases, pred_diacs = extract_diacritic_sequence(pred_word)
    gold_bases, gold_diacs = extract_diacritic_sequence(gold_word)

    if len(pred_diacs) != len(gold_diacs):
        return {'skip': True}

    positions = len(pred_diacs)
    if positions == 0:
        return {'skip': True}

    # Optionally exclude final position (case vowels)
    end_pos = positions - 1 if exclude_final else positions

    errors = 0
    total = 0

    for i in range(end_pos):
        total += 1
        if pred_diacs[i] != gold_diacs[i]:
            errors += 1

    return {
        'skip': False,
        'char_errors': errors,
        'char_total': total,
        'word_error': 1 if errors > 0 else 0
    }


def calculate_line_metrics(pred_line: str, gold_line: str) -> Dict:
    """
    Calculate metrics for an entire line.

    Uses word alignment with skipping for mismatched words.
    """
    pred_words = pred_line.split()
    gold_words = gold_line.split()

    results = {
        'char_errors_with_case': 0,
        'char_total_with_case': 0,
        'char_errors_no_case': 0,
        'char_total_no_case': 0,
        'word_errors_with_case': 0,
        'word_errors_no_case': 0,
        'word_total': 0,
        'skipped': 0
    }

    # Simple alignment: try to match words by their base forms
    pi, gi = 0, 0
    while pi < len(pred_words) and gi < len(gold_words):
        pred_base = strip_diacritics(pred_words[pi])
        gold_base = strip_diacritics(gold_words[gi])

        if pred_base == gold_base:
            # Words align - calculate metrics
            m_with = calculate_word_metrics(pred_words[pi], gold_words[gi], False)
            if not m_with.get('skip'):
                results['char_errors_with_case'] += m_with['char_errors']
                results['char_total_with_case'] += m_with['char_total']
                results['word_errors_with_case'] += m_with['word_error']
                results['word_total'] += 1

            m_no = calculate_word_metrics(pred_words[pi], gold_words[gi], True)
            if not m_no.get('skip'):
                results['char_errors_no_case'] += m_no['char_errors']
                results['char_total_no_case'] += m_no['char_total']
                results['word_errors_no_case'] += m_no['word_error']

            pi += 1
            gi += 1
        elif pi < len(pred_words) - 1 and strip_diacritics(pred_words[pi+1]) == gold_base:
            # Skip predicted word (insertion)
            pi += 1
        elif gi < len(gold_words) - 1 and pred_base == strip_diacritics(gold_words[gi+1]):
            # Skip gold word (deletion)
            gi += 1
        else:
            # Can't align - skip both
            pi += 1
            gi += 1
            results['skipped'] += 1

    return results


def evaluate_file(filepath: str, sample_size: int = None, verbose: bool = False) -> dict:
    """
    Evaluate Harakat on a file of gold-standard diacritized text.

    Args:
        filepath: Path to file with diacritized Arabic text
        sample_size: If set, evaluate on random sample of lines
        verbose: Print progress

    Returns:
        Dictionary with evaluation metrics
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    if sample_size and sample_size < len(lines):
        lines = random.sample(lines, sample_size)

    totals = {
        'char_errors_with_case': 0,
        'char_total_with_case': 0,
        'char_errors_no_case': 0,
        'char_total_no_case': 0,
        'word_errors_with_case': 0,
        'word_errors_no_case': 0,
        'word_total': 0,
        'skipped': 0,
        'processing_errors': 0
    }

    start_time = time.time()

    for i, gold_text in enumerate(lines):
        if verbose and (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            if elapsed > 0:
                rate = (i + 1) / elapsed
                if totals['char_total_with_case'] > 0:
                    current_der = 100 * totals['char_errors_with_case'] / totals['char_total_with_case']
                    print(f"Processing line {i + 1}/{len(lines)}... DER: {current_der:.2f}%")
                else:
                    print(f"Processing line {i + 1}/{len(lines)}...")

        # Strip diacritics and re-diacritize
        undiacritized = strip_diacritics(gold_text)

        try:
            predicted = diacritize(undiacritized)
        except Exception as e:
            totals['processing_errors'] += 1
            continue

        # Calculate metrics
        metrics = calculate_line_metrics(predicted, gold_text)

        for key in metrics:
            totals[key] += metrics[key]

    elapsed = time.time() - start_time

    # Calculate percentages
    der_with = (totals['char_errors_with_case'] / totals['char_total_with_case'] * 100) if totals['char_total_with_case'] > 0 else 0
    der_no = (totals['char_errors_no_case'] / totals['char_total_no_case'] * 100) if totals['char_total_no_case'] > 0 else 0
    wer_with = (totals['word_errors_with_case'] / totals['word_total'] * 100) if totals['word_total'] > 0 else 0
    wer_no = (totals['word_errors_no_case'] / totals['word_total'] * 100) if totals['word_total'] > 0 else 0

    results = {
        "lines_processed": len(lines),
        "lines_skipped": totals['skipped'],
        "processing_errors": totals['processing_errors'],

        # With case (including final vowels)
        "der_with_case": der_with,
        "der_with_case_errors": totals['char_errors_with_case'],
        "der_with_case_total": totals['char_total_with_case'],

        # Without case (excluding final vowels)
        "der_no_case": der_no,
        "der_no_case_errors": totals['char_errors_no_case'],
        "der_no_case_total": totals['char_total_no_case'],

        # WER with case
        "wer_with_case": wer_with,
        "wer_with_case_errors": totals['word_errors_with_case'],

        # WER without case
        "wer_no_case": wer_no,
        "wer_no_case_errors": totals['word_errors_no_case'],

        "word_total": totals['word_total'],
        "time_seconds": elapsed,
        "lines_per_second": len(lines) / elapsed if elapsed > 0 else 0,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Harakat diacritization')
    parser.add_argument('--file', '-f', default='tashkeela_test.txt',
                        help='Path to gold-standard test file')
    parser.add_argument('--sample', '-s', type=int, default=None,
                        help='Evaluate on random sample of N lines')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print progress')
    args = parser.parse_args()

    filepath = Path(args.file)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    print(f"Evaluating Harakat on: {filepath}")
    if args.sample:
        print(f"Using random sample of {args.sample} lines")
    print()

    results = evaluate_file(str(filepath), args.sample, args.verbose)

    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Lines processed: {results['lines_processed']:,}")
    print(f"Words evaluated: {results['word_total']:,}")
    print()

    print("DER (Diacritic Error Rate):")
    print(f"  With case endings:    {results['der_with_case']:.2f}%")
    print(f"    Errors: {results['der_with_case_errors']:,} / {results['der_with_case_total']:,} positions")
    print(f"  Without case endings: {results['der_no_case']:.2f}%")
    print(f"    Errors: {results['der_no_case_errors']:,} / {results['der_no_case_total']:,} positions")
    print()

    print("WER (Word Error Rate):")
    print(f"  With case endings:    {results['wer_with_case']:.2f}%")
    print(f"    Errors: {results['wer_with_case_errors']:,} / {results['word_total']:,} words")
    print(f"  Without case endings: {results['wer_no_case']:.2f}%")
    print(f"    Errors: {results['wer_no_case_errors']:,} / {results['word_total']:,} words")
    print()

    print(f"Processing time: {results['time_seconds']:.1f} seconds")
    print(f"Speed: {results['lines_per_second']:.1f} lines/second")
    print("=" * 60)


if __name__ == "__main__":
    main()
