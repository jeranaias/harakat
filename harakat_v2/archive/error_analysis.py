#!/usr/bin/env python3
"""
Deep Error Analysis for Harakat V2

Analyzes remaining errors after V1 + Case V3 + Hybrid LSTM pipeline.
Identifies patterns and categories of errors for potential fixes.
"""

import sys
import os

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')

from collections import Counter, defaultdict
import json

# Import V2
from harakat_v2 import HarakatV2, strip_harakat

# Constants
HARAKAT = 'ًٌٍَُِّْ'
HARAKAT_SET = set(HARAKAT)

DIAC_NAMES = {
    '': 'none',
    '\u064e': 'fatha',
    '\u064f': 'damma',
    '\u0650': 'kasra',
    '\u064b': 'tanwin_fath',
    '\u064c': 'tanwin_damm',
    '\u064d': 'tanwin_kasr',
    '\u0652': 'sukun',
    '\u0651': 'shadda',
}


def get_diac_name(d):
    if '\u0651' in d:
        other = d.replace('\u0651', '')
        if other:
            return f"shadda+{DIAC_NAMES.get(other, other)}"
        return 'shadda'
    return DIAC_NAMES.get(d, d if d else 'none')


def extract_diacritics(word):
    """Extract base characters and their diacritics."""
    base, diacs, curr = [], [], []
    for c in word:
        if c in HARAKAT_SET:
            curr.append(c)
        else:
            if base:
                diacs.append(''.join(sorted(curr)))
            base.append(c)
            curr = []
    if base:
        diacs.append(''.join(sorted(curr)))
    return base, diacs


def analyze_errors(pred_words, gold_words, errors_list, is_final_position=None):
    """Collect detailed error information."""
    pi, gi = 0, 0
    while pi < len(pred_words) and gi < len(gold_words):
        pb = strip_harakat(pred_words[pi])
        gb = strip_harakat(gold_words[gi])

        if pb == gb:
            base_p, diacs_p = extract_diacritics(pred_words[pi])
            base_g, diacs_g = extract_diacritics(gold_words[gi])

            if len(diacs_p) == len(diacs_g):
                for j, (p, g) in enumerate(zip(diacs_p, diacs_g)):
                    is_final = (j == len(diacs_p) - 1)

                    if is_final_position is not None and is_final != is_final_position:
                        continue

                    if p != g:
                        char_at_pos = base_p[j] if j < len(base_p) else '?'
                        errors_list.append({
                            'word_base': pb,
                            'word_gold': gold_words[gi],
                            'word_pred': pred_words[pi],
                            'position': j,
                            'is_final': is_final,
                            'char': char_at_pos,
                            'pred_diac': p,
                            'gold_diac': g,
                            'pred_name': get_diac_name(p),
                            'gold_name': get_diac_name(g),
                        })
            pi += 1
            gi += 1
        elif pi < len(pred_words) - 1 and strip_harakat(pred_words[pi+1]) == gb:
            pi += 1
        elif gi < len(gold_words) - 1 and pb == strip_harakat(gold_words[gi+1]):
            gi += 1
        else:
            pi += 1
            gi += 1


def main():
    print("="*70, flush=True)
    print("DEEP ERROR ANALYSIS - HARAKAT V2", flush=True)
    print("="*70, flush=True)

    # Load V2
    print("\nLoading Harakat V2...", flush=True)
    h = HarakatV2()
    print("  V2 loaded successfully", flush=True)

    # Load test data
    test_path = '../benchmarks/tashkeela_test.txt'
    if not os.path.exists(test_path):
        test_path = '../../tashkeela_test.txt'
    if not os.path.exists(test_path):
        test_path = '../../paper_eval/tashkeela_test.txt'

    with open(test_path, 'r', encoding='utf-8') as f:
        test_lines = [l.strip() for l in f if l.strip()]
    print(f"  Test lines: {len(test_lines)}", flush=True)

    # Process and collect errors
    print("\nProcessing through V2 pipeline...", flush=True)

    all_errors = []
    case_errors = []
    internal_errors = []

    for i, gold_line in enumerate(test_lines):
        undiac = strip_harakat(gold_line)
        pred_line = h.diacritize(undiac)

        pred_words = pred_line.split()
        gold_words = gold_line.split()

        analyze_errors(pred_words, gold_words, all_errors)
        analyze_errors(pred_words, gold_words, case_errors, is_final_position=True)
        analyze_errors(pred_words, gold_words, internal_errors, is_final_position=False)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(test_lines)} - {len(all_errors)} errors so far", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"Total errors collected: {len(all_errors)}", flush=True)
    print(f"  Case ending errors: {len(case_errors)}", flush=True)
    print(f"  Internal vowel errors: {len(internal_errors)}", flush=True)

    # ===== CONFUSION ANALYSIS =====
    print("\n" + "="*70, flush=True)
    print("ERROR TYPE BREAKDOWN", flush=True)
    print("="*70, flush=True)

    confusion = Counter()
    for e in all_errors:
        confusion[(e['pred_name'], e['gold_name'])] += 1

    print("\nTop 25 Confusion Pairs (pred -> gold):", flush=True)
    print("-"*55, flush=True)
    for (pred, gold), count in confusion.most_common(25):
        pct = 100 * count / len(all_errors) if all_errors else 0
        print(f"  {pred:20} -> {gold:20} : {count:5} ({pct:5.2f}%)", flush=True)

    # ===== CASE ENDING ERRORS =====
    print("\n" + "="*70, flush=True)
    print("CASE ENDING ERRORS (Final Position)", flush=True)
    print("="*70, flush=True)

    case_confusion = Counter()
    for e in case_errors:
        case_confusion[(e['pred_name'], e['gold_name'])] += 1

    print(f"\nTotal case errors: {len(case_errors)}", flush=True)
    print("\nTop 15 Case Confusions:", flush=True)
    print("-"*55, flush=True)
    for (pred, gold), count in case_confusion.most_common(15):
        pct = 100 * count / len(case_errors) if case_errors else 0
        print(f"  {pred:20} -> {gold:20} : {count:5} ({pct:5.2f}%)", flush=True)

    # ===== INTERNAL VOWEL ERRORS =====
    print("\n" + "="*70, flush=True)
    print("INTERNAL VOWEL ERRORS (Non-Final Positions)", flush=True)
    print("="*70, flush=True)

    internal_confusion = Counter()
    for e in internal_errors:
        internal_confusion[(e['pred_name'], e['gold_name'])] += 1

    print(f"\nTotal internal errors: {len(internal_errors)}", flush=True)
    print("\nTop 15 Internal Confusions:", flush=True)
    print("-"*55, flush=True)
    for (pred, gold), count in internal_confusion.most_common(15):
        pct = 100 * count / len(internal_errors) if internal_errors else 0
        print(f"  {pred:20} -> {gold:20} : {count:5} ({pct:5.2f}%)", flush=True)

    # ===== SHADDA-RELATED ERRORS =====
    print("\n" + "="*70, flush=True)
    print("SHADDA-RELATED ERRORS", flush=True)
    print("="*70, flush=True)

    shadda_errors = [e for e in all_errors if 'shadda' in e['pred_name'] or 'shadda' in e['gold_name']]
    print(f"\nTotal shadda-related errors: {len(shadda_errors)}", flush=True)

    shadda_confusion = Counter()
    for e in shadda_errors:
        shadda_confusion[(e['pred_name'], e['gold_name'])] += 1

    print("\nShadda Confusions:", flush=True)
    print("-"*55, flush=True)
    for (pred, gold), count in shadda_confusion.most_common(15):
        pct = 100 * count / len(shadda_errors) if shadda_errors else 0
        print(f"  {pred:20} -> {gold:20} : {count:5} ({pct:5.2f}%)", flush=True)

    # ===== MOST PROBLEMATIC WORDS =====
    print("\n" + "="*70, flush=True)
    print("MOST PROBLEMATIC WORDS", flush=True)
    print("="*70, flush=True)

    word_errors = Counter()
    for e in all_errors:
        word_errors[e['word_base']] += 1

    print("\nTop 40 Words with Most Errors:", flush=True)
    print("-"*65, flush=True)
    for word, count in word_errors.most_common(40):
        examples = [e for e in all_errors if e['word_base'] == word][:3]
        ex_str = "; ".join([f"{e['pred_name']}->{e['gold_name']}" for e in examples])
        print(f"  {word:15} : {count:4} errors  (e.g., {ex_str})", flush=True)

    # ===== SPECIFIC PATTERN ANALYSIS =====
    print("\n" + "="*70, flush=True)
    print("SPECIFIC ERROR PATTERNS", flush=True)
    print("="*70, flush=True)

    # Missing shadda (predicted no shadda, gold has shadda)
    missing_shadda = [e for e in all_errors if 'shadda' not in e['pred_name'] and 'shadda' in e['gold_name']]
    print(f"\nMissing shadda errors: {len(missing_shadda)}", flush=True)

    # Extra shadda (predicted shadda, gold has no shadda)
    extra_shadda = [e for e in all_errors if 'shadda' in e['pred_name'] and 'shadda' not in e['gold_name']]
    print(f"Extra shadda errors: {len(extra_shadda)}", flush=True)

    # Sukun vs vowel
    sukun_to_vowel = [e for e in all_errors if e['pred_name'] == 'sukun' and e['gold_name'] in ['fatha', 'damma', 'kasra']]
    vowel_to_sukun = [e for e in all_errors if e['gold_name'] == 'sukun' and e['pred_name'] in ['fatha', 'damma', 'kasra']]
    print(f"Sukun -> vowel errors: {len(sukun_to_vowel)}", flush=True)
    print(f"Vowel -> sukun errors: {len(vowel_to_sukun)}", flush=True)

    # None vs something
    none_to_diac = [e for e in all_errors if e['pred_name'] == 'none' and e['gold_name'] != 'none']
    diac_to_none = [e for e in all_errors if e['gold_name'] == 'none' and e['pred_name'] != 'none']
    print(f"Missing diacritic (none->X): {len(none_to_diac)}", flush=True)
    print(f"Extra diacritic (X->none): {len(diac_to_none)}", flush=True)

    # ===== SAMPLE ERRORS =====
    print("\n" + "="*70, flush=True)
    print("SAMPLE ERROR EXAMPLES", flush=True)
    print("="*70, flush=True)

    error_types = defaultdict(list)
    for e in all_errors:
        key = (e['pred_name'], e['gold_name'], e['is_final'])
        error_types[key].append(e)

    print("\nExamples of top error types:", flush=True)
    for (pred, gold, is_final), examples in sorted(error_types.items(), key=lambda x: -len(x[1]))[:12]:
        pos_type = "CASE" if is_final else "INTERNAL"
        print(f"\n[{pos_type}] {pred} -> {gold} ({len(examples)} errors):", flush=True)
        for ex in examples[:5]:
            print(f"    {ex['word_base']:12} | pred: {ex['word_pred']:18} | gold: {ex['word_gold']}", flush=True)

    # ===== SAVE DATA =====
    print("\n" + "="*70, flush=True)
    print("SAVING DETAILED ERROR DATA", flush=True)
    print("="*70, flush=True)

    error_data = {
        'summary': {
            'total_errors': len(all_errors),
            'case_errors': len(case_errors),
            'internal_errors': len(internal_errors),
            'shadda_related': len(shadda_errors),
            'missing_shadda': len(missing_shadda),
            'extra_shadda': len(extra_shadda),
        },
        'confusion_matrix': {f"{p}->{g}": c for (p, g), c in confusion.most_common(50)},
        'case_confusion': {f"{p}->{g}": c for (p, g), c in case_confusion.most_common(30)},
        'internal_confusion': {f"{p}->{g}": c for (p, g), c in internal_confusion.most_common(30)},
        'top_problem_words': dict(word_errors.most_common(100)),
        'sample_errors': all_errors[:500],
    }

    with open('error_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(error_data, f, ensure_ascii=False, indent=2)

    print("  Saved to error_analysis.json", flush=True)

    print("\n" + "="*70, flush=True)
    print("ANALYSIS COMPLETE", flush=True)
    print("="*70, flush=True)


if __name__ == '__main__':
    main()
