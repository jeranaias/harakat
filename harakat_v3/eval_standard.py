#!/usr/bin/env python3
"""
Standard Evaluation Script for Harakat V3

This script implements the canonical evaluation protocol from METHODOLOGY.md.
It tests on BOTH validation and test sets to check for overfitting.

Usage:
    python eval_standard.py                    # Evaluate current V2 + sweep
    python eval_standard.py --model v2         # Evaluate V2 without sweep
    python eval_standard.py --model v3         # Evaluate V3 (when ready)

Output includes:
    - DER with case endings
    - DER without case endings
    - WER
    - Comparison between validation and test performance
    - Overfitting warning if test >> validation improvement
"""

import sys
import os
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict

if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')

# Add paths - use absolute path for reliability
harakat_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, harakat_dir)

# Canonical diacritic set (from METHODOLOGY.md)
HARAKAT = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670')


def strip_harakat(text: str) -> str:
    """Remove all diacritics from text."""
    return ''.join(c for c in text if c not in HARAKAT)


def extract_diacritic_sequence(word: str) -> Tuple[List[str], List[str]]:
    """
    Extract base chars and their associated diacritics.
    This is the CANONICAL implementation from METHODOLOGY.md.
    """
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


def calculate_word_metrics(pred_word: str, gold_word: str,
                           exclude_final: bool = False) -> Dict:
    """Calculate metrics for a single word."""
    pred_base = strip_harakat(pred_word)
    gold_base = strip_harakat(gold_word)

    if pred_base != gold_base:
        return {'skip': True}

    pred_bases, pred_diacs = extract_diacritic_sequence(pred_word)
    gold_bases, gold_diacs = extract_diacritic_sequence(gold_word)

    if len(pred_diacs) != len(gold_diacs):
        return {'skip': True}

    positions = len(pred_diacs)
    if positions == 0:
        return {'skip': True}

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
    """Calculate metrics for a full line."""
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

    pi, gi = 0, 0
    while pi < len(pred_words) and gi < len(gold_words):
        pred_base = strip_harakat(pred_words[pi])
        gold_base = strip_harakat(gold_words[gi])

        if pred_base == gold_base:
            # With case endings
            m_with = calculate_word_metrics(pred_words[pi], gold_words[gi], False)
            if not m_with.get('skip'):
                results['char_errors_with_case'] += m_with['char_errors']
                results['char_total_with_case'] += m_with['char_total']
                results['word_errors_with_case'] += m_with['word_error']
                results['word_total'] += 1

            # Without case endings
            m_no = calculate_word_metrics(pred_words[pi], gold_words[gi], True)
            if not m_no.get('skip'):
                results['char_errors_no_case'] += m_no['char_errors']
                results['char_total_no_case'] += m_no['char_total']
                results['word_errors_no_case'] += m_no['word_error']

            pi += 1
            gi += 1
        elif pi < len(pred_words) - 1 and strip_harakat(pred_words[pi+1]) == gold_base:
            pi += 1
        elif gi < len(gold_words) - 1 and pred_base == strip_harakat(gold_words[gi+1]):
            gi += 1
        else:
            pi += 1
            gi += 1
            results['skipped'] += 1

    return results


def evaluate_dataset(diacritize_fn, dataset_path: str, name: str) -> Dict:
    """Evaluate on a single dataset."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    totals = defaultdict(int)

    print(f"\nEvaluating on {name} ({len(lines)} lines)...")

    for i, gold_line in enumerate(lines):
        undiac = strip_harakat(gold_line)
        pred_line = diacritize_fn(undiac)

        metrics = calculate_line_metrics(pred_line, gold_line)
        for k, v in metrics.items():
            totals[k] += v

        if (i + 1) % 500 == 0:
            der = 100 * totals['char_errors_with_case'] / totals['char_total_with_case']
            print(f"  {i+1}/{len(lines)} - DER: {der:.3f}%")

    # Calculate final metrics
    results = {
        'name': name,
        'lines': len(lines),
        'der_with_case': 100 * totals['char_errors_with_case'] / totals['char_total_with_case'],
        'der_no_case': 100 * totals['char_errors_no_case'] / totals['char_total_no_case'],
        'wer': 100 * totals['word_errors_with_case'] / totals['word_total'],
        'errors': totals['char_errors_with_case'],
        'total': totals['char_total_with_case'],
    }

    return results


def run_standard_evaluation(model_name: str = 'v2_sweep'):
    """
    Run standard evaluation on both validation and test sets.

    This is the canonical evaluation procedure from METHODOLOGY.md.
    """
    print("=" * 70)
    print("HARAKAT STANDARD EVALUATION")
    print("=" * 70)
    print(f"\nModel: {model_name}")
    print("Following protocol from METHODOLOGY.md")

    # Load appropriate model
    if model_name == 'v2':
        from harakat_v2 import diacritize
        diacritize_fn = diacritize
        print("Using: V2 neural pipeline (no sweep)")
    elif model_name == 'v2_sweep':
        from harakat_v2 import diacritize_with_sweep
        diacritize_fn = diacritize_with_sweep
        print("Using: V2 + Final Sweep")
    elif model_name == 'v3':
        from harakat_v3 import diacritize
        diacritize_fn = diacritize
        print("Using: V3 (V2 + particle rules)")
    else:
        print(f"ERROR: Unknown model '{model_name}'")
        return

    # Find data files
    base_paths = ['../..', '../../..', '../../../paper_eval']
    val_path = None
    test_path = None

    for base in base_paths:
        if os.path.exists(f'{base}/tashkeela_val.txt'):
            val_path = f'{base}/tashkeela_val.txt'
        if os.path.exists(f'{base}/tashkeela_test.txt'):
            test_path = f'{base}/tashkeela_test.txt'

    if not val_path or not test_path:
        print("ERROR: Could not find data files")
        print("Expected: tashkeela_val.txt and tashkeela_test.txt")
        return

    print(f"\nValidation set: {val_path}")
    print(f"Test set: {test_path}")

    # Evaluate on both sets
    val_results = evaluate_dataset(diacritize_fn, val_path, "Validation")
    test_results = evaluate_dataset(diacritize_fn, test_path, "Test")

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Dataset':<15} {'Lines':<8} {'DER':<10} {'DER (no case)':<15} {'WER':<10}")
    print("-" * 60)

    for r in [val_results, test_results]:
        print(f"{r['name']:<15} {r['lines']:<8} {r['der_with_case']:.2f}%     "
              f"{r['der_no_case']:.2f}%          {r['wer']:.2f}%")

    # Check for overfitting
    print("\n" + "=" * 70)
    print("GENERALIZATION CHECK")
    print("=" * 70)

    val_der = val_results['der_with_case']
    test_der = test_results['der_with_case']
    gap = test_der - val_der

    print(f"\nValidation DER: {val_der:.2f}%")
    print(f"Test DER:       {test_der:.2f}%")
    print(f"Gap:            {gap:+.2f}%")

    if gap < -0.3:
        print("\n⚠️  WARNING: Test significantly better than validation")
        print("   This may indicate overfitting to test set patterns.")
        print("   Review METHODOLOGY.md generalization guidelines.")
    elif gap > 0.3:
        print("\n⚠️  WARNING: Test significantly worse than validation")
        print("   Model may not generalize well to unseen data.")
    else:
        print("\n✓ Results are consistent between validation and test sets.")

    # Summary for documentation
    print("\n" + "=" * 70)
    print("COPY FOR PROGRESS.md")
    print("=" * 70)
    print(f"""
| Dataset | DER (with case) | DER (no case) | WER | Errors |
|---------|-----------------|---------------|-----|--------|
| Validation | {val_results['der_with_case']:.2f}% | {val_results['der_no_case']:.2f}% | {val_results['wer']:.2f}% | {val_results['errors']} |
| Test | {test_results['der_with_case']:.2f}% | {test_results['der_no_case']:.2f}% | {test_results['wer']:.2f}% | {test_results['errors']} |
""")

    return val_results, test_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Standard Harakat Evaluation')
    parser.add_argument('--model', default='v2_sweep',
                        choices=['v2', 'v2_sweep', 'v3'],
                        help='Model to evaluate')
    args = parser.parse_args()

    run_standard_evaluation(args.model)
