#!/usr/bin/env python3
"""
Full evaluation of Harakat: DER and WER with and without case endings.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'c:\\tashkeel\\harakat')

import importlib
import harakat
importlib.reload(harakat)

from harakat import diacritize, __version__

print("=" * 70)
print(f"Harakat Full Evaluation")
print(f"Version: {__version__}")
print("=" * 70)

# Diacritics
HARAKAT = '\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0670'
HARAKAT_SET = set(HARAKAT)

# Case endings (final position diacritics)
CASE_MARKS = set('\u064B\u064C\u064D\u064E\u064F\u0650')  # tanween + short vowels

def strip_harakat(text):
    """Remove all diacritical marks."""
    return ''.join(c for c in text if c not in HARAKAT_SET)

def extract_diacritics(word):
    """Extract diacritic sequence from a word (one per base char)."""
    base_chars = []
    diacritics = []
    current = []

    for c in word:
        if c in HARAKAT_SET:
            current.append(c)
        else:
            if base_chars:
                diacritics.append(''.join(sorted(current)))
            base_chars.append(c)
            current = []

    if base_chars:
        diacritics.append(''.join(sorted(current)))

    return base_chars, diacritics

def remove_final_case(diacritics):
    """Remove case endings from the final position."""
    if not diacritics:
        return diacritics

    # Remove case marks from the last position
    final = diacritics[-1]
    final_no_case = ''.join(c for c in final if c not in CASE_MARKS)
    return diacritics[:-1] + [final_no_case]

def evaluate():
    test_file = r'c:\tashkeel\tashkeela_v2\tashkeela_test.txt'

    with open(test_file, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    print(f"\nEvaluating {len(lines):,} lines...")

    # Metrics
    total_der_positions = 0
    total_der_errors = 0
    total_der_nocase_positions = 0
    total_der_nocase_errors = 0

    total_words = 0
    total_wer_errors = 0
    total_wer_nocase_errors = 0

    for i, gold_line in enumerate(lines):
        if (i + 1) % 500 == 0:
            der = 100 * total_der_errors / total_der_positions if total_der_positions > 0 else 0
            print(f"  {i+1:,}/{len(lines):,} - DER: {der:.2f}%")

        # Get prediction
        undiac = strip_harakat(gold_line)
        pred_line = diacritize(undiac)

        # Split into words
        pred_words = pred_line.split()
        gold_words = gold_line.split()

        # Align words by base form
        pi, gi = 0, 0
        while pi < len(pred_words) and gi < len(gold_words):
            pred_base = strip_harakat(pred_words[pi])
            gold_base = strip_harakat(gold_words[gi])

            if pred_base == gold_base:
                pred_bases, pred_diacs = extract_diacritics(pred_words[pi])
                gold_bases, gold_diacs = extract_diacritics(gold_words[gi])

                if len(pred_diacs) == len(gold_diacs):
                    total_words += 1

                    # DER with case
                    word_errors = 0
                    for j in range(len(pred_diacs)):
                        total_der_positions += 1
                        if pred_diacs[j] != gold_diacs[j]:
                            total_der_errors += 1
                            word_errors += 1

                    # WER with case
                    if word_errors > 0:
                        total_wer_errors += 1

                    # DER without case
                    pred_nocase = remove_final_case(pred_diacs)
                    gold_nocase = remove_final_case(gold_diacs)
                    word_errors_nocase = 0
                    for j in range(len(pred_nocase)):
                        total_der_nocase_positions += 1
                        if pred_nocase[j] != gold_nocase[j]:
                            total_der_nocase_errors += 1
                            word_errors_nocase += 1

                    # WER without case
                    if word_errors_nocase > 0:
                        total_wer_nocase_errors += 1

                pi += 1
                gi += 1
            else:
                # Base mismatch - skip
                pi += 1
                gi += 1

    # Calculate final metrics
    der = 100 * total_der_errors / total_der_positions if total_der_positions > 0 else 0
    der_nocase = 100 * total_der_nocase_errors / total_der_nocase_positions if total_der_nocase_positions > 0 else 0
    wer = 100 * total_wer_errors / total_words if total_words > 0 else 0
    wer_nocase = 100 * total_wer_nocase_errors / total_words if total_words > 0 else 0

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nTotal words evaluated: {total_words:,}")
    print(f"Total diacritic positions: {total_der_positions:,}")
    print()
    print(f"{'Metric':<20} {'With Case':>15} {'Without Case':>15}")
    print("-" * 50)
    print(f"{'DER':<20} {der:>14.2f}% {der_nocase:>14.2f}%")
    print(f"{'WER':<20} {wer:>14.2f}% {wer_nocase:>14.2f}%")
    print()
    print(f"DER errors: {total_der_errors:,} / {total_der_positions:,}")
    print(f"DER (no case) errors: {total_der_nocase_errors:,} / {total_der_nocase_positions:,}")
    print(f"WER errors: {total_wer_errors:,} / {total_words:,}")
    print(f"WER (no case) errors: {total_wer_nocase_errors:,} / {total_words:,}")

    return {
        'der': der,
        'der_nocase': der_nocase,
        'wer': wer,
        'wer_nocase': wer_nocase,
    }

if __name__ == '__main__':
    evaluate()
