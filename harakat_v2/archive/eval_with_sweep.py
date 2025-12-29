#!/usr/bin/env python3
"""
Evaluate Harakat V2 with Final Sweep corrections.
Uses EXACT same DER calculation as original evaluation.
"""

import sys
import os

if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')

from harakat_v2 import HarakatV2, strip_harakat
from final_sweep import apply_final_sweep

# Diacritics - same as original
HARAKAT = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670')


def strip_h(text):
    return ''.join(c for c in text if c not in HARAKAT)


def extract_diacritic_sequence(word):
    """Extract base chars and their associated diacritics."""
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


def calculate_word_metrics(pred_word, gold_word, exclude_final=False):
    """Calculate metrics for a single word."""
    pred_base = strip_h(pred_word)
    gold_base = strip_h(gold_word)

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


def calculate_line_metrics(pred_line, gold_line):
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
        pred_base = strip_h(pred_words[pi])
        gold_base = strip_h(gold_words[gi])

        if pred_base == gold_base:
            # With case endings
            m_with = calculate_word_metrics(pred_words[pi], gold_words[gi], False)
            if not m_with.get('skip'):
                results['char_errors_with_case'] += m_with['char_errors']
                results['char_total_with_case'] += m_with['char_total']
                results['word_errors_with_case'] += m_with['word_error']
                results['word_total'] += 1

            # Without case endings (exclude final position)
            m_no = calculate_word_metrics(pred_words[pi], gold_words[gi], True)
            if not m_no.get('skip'):
                results['char_errors_no_case'] += m_no['char_errors']
                results['char_total_no_case'] += m_no['char_total']
                results['word_errors_no_case'] += m_no['word_error']

            pi += 1
            gi += 1
        elif pi < len(pred_words) - 1 and strip_h(pred_words[pi+1]) == gold_base:
            pi += 1
        elif gi < len(gold_words) - 1 and pred_base == strip_h(gold_words[gi+1]):
            gi += 1
        else:
            pi += 1
            gi += 1
            results['skipped'] += 1

    return results


def main():
    print("="*70)
    print("HARAKAT V2 + FINAL SWEEP EVALUATION")
    print("="*70)

    # Load V2
    print("\nLoading Harakat V2...")
    h = HarakatV2()
    print("  Loaded.")

    # Load test data
    test_path = '../benchmarks/tashkeela_test.txt'
    if not os.path.exists(test_path):
        test_path = '../../tashkeela_test.txt'
    if not os.path.exists(test_path):
        test_path = '../../paper_eval/tashkeela_test.txt'

    with open(test_path, 'r', encoding='utf-8') as f:
        test_lines = [l.strip() for l in f if l.strip()]
    print(f"  Test lines: {len(test_lines)}")

    # Accumulators
    v2_totals = {
        'char_errors_with_case': 0, 'char_total_with_case': 0,
        'char_errors_no_case': 0, 'char_total_no_case': 0,
        'word_errors_with_case': 0, 'word_total': 0
    }
    sweep_totals = {
        'char_errors_with_case': 0, 'char_total_with_case': 0,
        'char_errors_no_case': 0, 'char_total_no_case': 0,
        'word_errors_with_case': 0, 'word_total': 0
    }

    corrections_made = 0

    print("\nEvaluating...")
    for i, gold_line in enumerate(test_lines):
        undiac = strip_harakat(gold_line)

        # V2 prediction
        v2_pred = h.diacritize(undiac)

        # V2 + sweep prediction
        sweep_pred = apply_final_sweep(v2_pred)

        if v2_pred != sweep_pred:
            corrections_made += 1

        # V2 metrics
        m_v2 = calculate_line_metrics(v2_pred, gold_line)
        for k in v2_totals:
            if k in m_v2:
                v2_totals[k] += m_v2[k]

        # Sweep metrics
        m_sweep = calculate_line_metrics(sweep_pred, gold_line)
        for k in sweep_totals:
            if k in m_sweep:
                sweep_totals[k] += m_sweep[k]

        if (i + 1) % 250 == 0:
            v2_der = 100 * v2_totals['char_errors_with_case'] / v2_totals['char_total_with_case'] if v2_totals['char_total_with_case'] > 0 else 0
            sweep_der = 100 * sweep_totals['char_errors_with_case'] / sweep_totals['char_total_with_case'] if sweep_totals['char_total_with_case'] > 0 else 0
            print(f"  {i+1}/{len(test_lines)} - V2: {v2_der:.3f}% | V2+Sweep: {sweep_der:.3f}%")

    # Final results
    v2_der = 100 * v2_totals['char_errors_with_case'] / v2_totals['char_total_with_case']
    v2_der_nc = 100 * v2_totals['char_errors_no_case'] / v2_totals['char_total_no_case']
    v2_wer = 100 * v2_totals['word_errors_with_case'] / v2_totals['word_total']

    sweep_der = 100 * sweep_totals['char_errors_with_case'] / sweep_totals['char_total_with_case']
    sweep_der_nc = 100 * sweep_totals['char_errors_no_case'] / sweep_totals['char_total_no_case']
    sweep_wer = 100 * sweep_totals['word_errors_with_case'] / sweep_totals['word_total']

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print(f"\nV2 (without sweep):")
    print(f"  DER (with case):    {v2_der:.2f}%  ({v2_totals['char_errors_with_case']}/{v2_totals['char_total_with_case']} errors)")
    print(f"  DER (without case): {v2_der_nc:.2f}%")
    print(f"  WER:                {v2_wer:.2f}%")

    print(f"\nV2 + Final Sweep:")
    print(f"  DER (with case):    {sweep_der:.2f}%  ({sweep_totals['char_errors_with_case']}/{sweep_totals['char_total_with_case']} errors)")
    print(f"  DER (without case): {sweep_der_nc:.2f}%")
    print(f"  WER:                {sweep_wer:.2f}%")

    print(f"\nImprovement:")
    print(f"  Lines with corrections: {corrections_made}")
    print(f"  Error reduction:  {v2_totals['char_errors_with_case'] - sweep_totals['char_errors_with_case']} errors")
    print(f"  DER improvement:  {v2_der - sweep_der:.3f}%")
    rel_improve = 100 * (v2_der - sweep_der) / v2_der if v2_der > 0 else 0
    print(f"  Relative improve: {rel_improve:.1f}%")


if __name__ == '__main__':
    main()
