#!/usr/bin/env python3
"""
Dual Gold Standard Evaluation for Harakat

Evaluates the diacritization model against BOTH:
1. Original Tashkeela gold (for comparability with published benchmarks)
2. Tashkeela V2 (normalized gold, for true quality assessment)

This reveals the impact of gold standard inconsistencies on DER.

Usage:
    python eval_dual_gold.py                    # Evaluate V2 model
    python eval_dual_gold.py --model v3         # Evaluate V3 model
"""

import sys
import os
import argparse
from collections import defaultdict

if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')

harakat_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, harakat_dir)

# Constants
HARAKAT = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670')


def strip_harakat(text):
    return ''.join(c for c in text if c not in HARAKAT)


def extract_diacritic_sequence(word):
    """Canonical diacritic extraction."""
    base_chars = []
    diacritics = []
    current_diacritics = []

    for c in word:
        if c in HARAKAT:
            current_diacritics.append(c)
        else:
            if base_chars:
                diacritics.append(''.join(sorted(current_diacritics)))
            base_chars.append(c)
            current_diacritics = []

    if base_chars:
        diacritics.append(''.join(sorted(current_diacritics)))

    return base_chars, diacritics


def calculate_metrics(pred_word, gold_word, exclude_final=False):
    """Calculate DER metrics for a word pair."""
    pred_base = strip_harakat(pred_word)
    gold_base = strip_harakat(gold_word)

    if pred_base != gold_base:
        return None

    pred_bases, pred_diacs = extract_diacritic_sequence(pred_word)
    gold_bases, gold_diacs = extract_diacritic_sequence(gold_word)

    if len(pred_diacs) != len(gold_diacs):
        return None

    positions = len(pred_diacs)
    if positions == 0:
        return None

    end_pos = positions - 1 if exclude_final else positions

    errors = 0
    total = 0

    for i in range(end_pos):
        total += 1
        if pred_diacs[i] != gold_diacs[i]:
            errors += 1

    return {
        'errors': errors,
        'total': total,
        'word_error': 1 if errors > 0 else 0
    }


def evaluate_file(diacritize_fn, gold_path, name):
    """Evaluate against a single gold file."""
    with open(gold_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    totals = defaultdict(int)

    for i, gold_line in enumerate(lines):
        undiac = strip_harakat(gold_line)
        pred_line = diacritize_fn(undiac)

        pred_words = pred_line.split()
        gold_words = gold_line.split()

        pi, gi = 0, 0
        while pi < len(pred_words) and gi < len(gold_words):
            pred_base = strip_harakat(pred_words[pi])
            gold_base = strip_harakat(gold_words[gi])

            if pred_base == gold_base:
                # With case
                m = calculate_metrics(pred_words[pi], gold_words[gi], False)
                if m:
                    totals['errors_with_case'] += m['errors']
                    totals['total_with_case'] += m['total']
                    totals['word_errors_with_case'] += m['word_error']
                    totals['words'] += 1

                # Without case
                m = calculate_metrics(pred_words[pi], gold_words[gi], True)
                if m:
                    totals['errors_no_case'] += m['errors']
                    totals['total_no_case'] += m['total']
                    totals['word_errors_no_case'] += m['word_error']

                pi += 1
                gi += 1
            else:
                if pi < len(pred_words) - 1 and strip_harakat(pred_words[pi+1]) == gold_base:
                    pi += 1
                elif gi < len(gold_words) - 1 and pred_base == strip_harakat(gold_words[gi+1]):
                    gi += 1
                else:
                    pi += 1
                    gi += 1

    return {
        'lines': len(lines),
        'der_with_case': 100 * totals['errors_with_case'] / totals['total_with_case'],
        'der_no_case': 100 * totals['errors_no_case'] / totals['total_no_case'],
        'wer': 100 * totals['word_errors_with_case'] / totals['words'],
        'errors': totals['errors_with_case'],
        'total': totals['total_with_case'],
    }


def main():
    parser = argparse.ArgumentParser(description='Dual gold standard evaluation')
    parser.add_argument('--model', default='v2_sweep', choices=['v2', 'v2_sweep', 'v3'])
    args = parser.parse_args()

    print("=" * 70)
    print("DUAL GOLD STANDARD EVALUATION")
    print("=" * 70)
    print(f"\nModel: {args.model}")

    # Load model
    if args.model == 'v2':
        from harakat_v2 import diacritize
        diacritize_fn = diacritize
    elif args.model == 'v2_sweep':
        from harakat_v2 import diacritize_with_sweep
        diacritize_fn = diacritize_with_sweep
    else:
        from harakat_v3 import diacritize
        diacritize_fn = diacritize

    # Find gold files
    tashkeel_dir = os.path.dirname(harakat_dir)
    original_dir = os.path.join(tashkeel_dir, 'paper_eval')
    v2_dir = os.path.join(tashkeel_dir, 'tashkeela_v2')

    datasets = [
        ('test', 'tashkeela_test.txt', 'Test (2,500 lines)'),
        ('val', 'tashkeela_val.txt', 'Validation (5,000 lines)'),
    ]

    results = {}

    for key, filename, desc in datasets:
        original_path = os.path.join(original_dir, filename)
        v2_path = os.path.join(v2_dir, filename)

        if not os.path.exists(original_path):
            print(f"\nSkipping {desc}: file not found")
            continue

        print(f"\n{'='*70}")
        print(f"EVALUATING: {desc}")
        print(f"{'='*70}")

        # Evaluate against original gold
        print(f"\nAgainst ORIGINAL Tashkeela...")
        orig_results = evaluate_file(diacritize_fn, original_path, f"Original-{key}")

        # Evaluate against V2 gold
        if os.path.exists(v2_path):
            print(f"Against NORMALIZED Tashkeela V2...")
            v2_results = evaluate_file(diacritize_fn, v2_path, f"V2-{key}")
        else:
            v2_results = None

        results[key] = {
            'original': orig_results,
            'v2': v2_results
        }

        # Print comparison
        print(f"\n{'-'*70}")
        print(f"{'Gold Standard':<25} {'DER':<12} {'DER (no case)':<15} {'WER':<10}")
        print(f"{'-'*70}")

        print(f"{'Original Tashkeela':<25} {orig_results['der_with_case']:.2f}%      "
              f"{orig_results['der_no_case']:.2f}%          {orig_results['wer']:.2f}%")

        if v2_results:
            print(f"{'Tashkeela V2 (normalized)':<25} {v2_results['der_with_case']:.2f}%      "
                  f"{v2_results['der_no_case']:.2f}%          {v2_results['wer']:.2f}%")

            # Improvement from normalization
            der_diff = orig_results['der_with_case'] - v2_results['der_with_case']
            print(f"\n{'Improvement from V2 gold:':<25} {der_diff:+.2f}%")
            print(f"{'Errors revealed as false:':<25} {orig_results['errors'] - v2_results['errors']:,}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("""
Key findings:
- Original gold has inconsistencies that artificially inflate DER
- V2 gold fixes particle kasra and sun letter assimilation
- True model quality is better than original DER suggests

For publications:
- Report BOTH DERs for transparency
- Original DER for comparability with prior work
- V2 DER for true quality assessment
""")

    # Table for documentation
    print("\n" + "=" * 70)
    print("COPY FOR DOCUMENTATION")
    print("=" * 70)
    print("""
| Dataset | Gold | DER (with case) | DER (no case) | WER | Errors |
|---------|------|-----------------|---------------|-----|--------|""")

    for key in ['test', 'val']:
        if key not in results:
            continue
        r = results[key]
        name = 'Test' if key == 'test' else 'Validation'

        print(f"| {name} | Original | {r['original']['der_with_case']:.2f}% | "
              f"{r['original']['der_no_case']:.2f}% | {r['original']['wer']:.2f}% | "
              f"{r['original']['errors']} |")

        if r['v2']:
            print(f"| {name} | V2 (norm) | {r['v2']['der_with_case']:.2f}% | "
                  f"{r['v2']['der_no_case']:.2f}% | {r['v2']['wer']:.2f}% | "
                  f"{r['v2']['errors']} |")


if __name__ == '__main__':
    main()
