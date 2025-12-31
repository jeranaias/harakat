#!/usr/bin/env python3
"""
Verify the 1.71% DER claim by testing harakat_v3 module on both test sets.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'c:\\tashkeel\\harakat')

# Import harakat_v3 directly (this is what the README claims achieves 1.71%)
from harakat_v3 import diacritize as v3_diacritize, __version__ as v3_version

print(f"Verifying 1.71% DER Claim")
print(f"Using: harakat_v3 module (V{v3_version})")
print("=" * 70)

HARAKAT = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670')

def strip_h(text):
    return ''.join(c for c in text if c not in HARAKAT)

def extract_diacritic_sequence(word):
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
    pred_base = strip_h(pred_word)
    gold_base = strip_h(gold_word)

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

    return {'errors': errors, 'total': total, 'word_error': 1 if errors > 0 else 0}

def evaluate_file(diacritize_fn, filepath, name):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    totals = {
        'errors_with': 0, 'total_with': 0,
        'errors_no': 0, 'total_no': 0,
        'word_errors': 0, 'words': 0
    }

    for i, gold_line in enumerate(lines):
        if (i + 1) % 500 == 0:
            der = 100 * totals['errors_with'] / totals['total_with'] if totals['total_with'] > 0 else 0
            print(f"  {name}: {i+1}/{len(lines)} - DER: {der:.2f}%")

        undiac = strip_h(gold_line)
        pred_line = diacritize_fn(undiac)

        pred_words = pred_line.split()
        gold_words = gold_line.split()

        pi, gi = 0, 0
        while pi < len(pred_words) and gi < len(gold_words):
            pred_base = strip_h(pred_words[pi])
            gold_base = strip_h(gold_words[gi])

            if pred_base == gold_base:
                m = calculate_metrics(pred_words[pi], gold_words[gi], False)
                if m:
                    totals['errors_with'] += m['errors']
                    totals['total_with'] += m['total']
                    totals['word_errors'] += m['word_error']
                    totals['words'] += 1

                m_no = calculate_metrics(pred_words[pi], gold_words[gi], True)
                if m_no:
                    totals['errors_no'] += m_no['errors']
                    totals['total_no'] += m_no['total']

                pi += 1
                gi += 1
            else:
                pi += 1
                gi += 1

    der_with = 100 * totals['errors_with'] / totals['total_with'] if totals['total_with'] > 0 else 0
    der_no = 100 * totals['errors_no'] / totals['total_no'] if totals['total_no'] > 0 else 0
    wer = 100 * totals['word_errors'] / totals['words'] if totals['words'] > 0 else 0

    return {
        'name': name,
        'lines': len(lines),
        'der_with': der_with,
        'der_no': der_no,
        'wer': wer,
        'errors': totals['errors_with'],
        'total': totals['total_with']
    }

# Test files
test_files = [
    ('c:\\tashkeel\\tashkeela_v2\\tashkeela_test.txt', 'V2 Gold (Test)'),
    ('c:\\tashkeel\\tashkeela_v2\\tashkeela_val.txt', 'V2 Gold (Validation)'),
    ('c:\\tashkeel\\paper_eval\\tashkeela_test.txt', 'Original Gold (Test)'),
]

import time
results = []

for filepath, name in test_files:
    try:
        print(f"\n{'='*70}")
        print(f"Evaluating: {name}")
        print(f"File: {filepath}")
        print(f"{'='*70}")

        start = time.time()
        result = evaluate_file(v3_diacritize, filepath, name)
        elapsed = time.time() - start
        result['time'] = elapsed
        results.append(result)

        print(f"\nResults for {name}:")
        print(f"  DER (with case): {result['der_with']:.2f}%")
        print(f"  DER (no case):   {result['der_no']:.2f}%")
        print(f"  WER:             {result['wer']:.2f}%")
        print(f"  Time: {elapsed:.1f}s")
    except FileNotFoundError:
        print(f"  File not found: {filepath}")

print("\n" + "=" * 70)
print("SUMMARY - Verifying 1.71% DER Claim")
print("=" * 70)
print(f"\n{'Dataset':<25} {'DER':<10} {'DER (no case)':<15} {'WER':<10}")
print("-" * 60)
for r in results:
    print(f"{r['name']:<25} {r['der_with']:.2f}%     {r['der_no']:.2f}%          {r['wer']:.2f}%")

print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)
for r in results:
    if 'Test' in r['name']:
        if r['der_with'] < 1.75:
            print(f"✓ {r['name']}: {r['der_with']:.2f}% - MATCHES 1.71% claim!")
        elif r['der_with'] < 2.0:
            print(f"~ {r['name']}: {r['der_with']:.2f}% - Close to 1.71% claim")
        else:
            print(f"✗ {r['name']}: {r['der_with']:.2f}% - Does NOT match 1.71% claim")
