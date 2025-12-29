#!/usr/bin/env python3
"""
Analyze remaining من errors to improve ML classifier.
"""
import sys
import os
import io

script_dir = os.path.dirname(os.path.abspath(__file__))
harakat_dir = os.path.dirname(script_dir)
sys.path.insert(0, harakat_dir)

from harakat_v3 import diacritize
from harakat_v3.rules.anna_fix import strip_harakat

def main():
    if sys.platform == 'win32':
        os.system('chcp 65001 > nul 2>&1')
    if not isinstance(sys.stdout, io.TextIOWrapper):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        except:
            pass

    tashkeel_dir = os.path.dirname(harakat_dir)
    v2_gold = os.path.join(tashkeel_dir, 'tashkeela_v2', 'tashkeela_test.txt')

    with open(v2_gold, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()][:500]

    print("=" * 70)
    print("من ERROR ANALYSIS")
    print("=" * 70)

    # من variants
    min_variants = {
        'مِنْ': 'preposition',
        'مِنَ': 'preposition',
        'مَنْ': 'pronoun/relative',
        'مَنَ': 'pronoun/relative',
    }

    errors = []

    for line_num, gold_line in enumerate(lines):
        undiac = strip_harakat(gold_line)
        pred_line = diacritize(undiac)

        gold_words = gold_line.split()
        pred_words = pred_line.split()

        for i, (gw, pw) in enumerate(zip(gold_words, pred_words)):
            gb = strip_harakat(gw)
            if gb == 'من':
                if gw != pw:
                    # Get context
                    left_ctx = gold_words[max(0, i-3):i]
                    right_ctx = gold_words[i+1:i+4]

                    gold_type = min_variants.get(gw, 'unknown')
                    pred_type = min_variants.get(pw, 'unknown')

                    errors.append({
                        'line': line_num,
                        'gold': gw,
                        'pred': pw,
                        'gold_type': gold_type,
                        'pred_type': pred_type,
                        'left': ' '.join(left_ctx),
                        'right': ' '.join(right_ctx),
                    })

    print(f"\nTotal من errors: {len(errors)}")

    # Group by error type
    from collections import Counter
    error_types = Counter()
    for e in errors:
        key = f"{e['pred_type']} → {e['gold_type']}"
        error_types[key] += 1

    print(f"\nError type distribution:")
    for key, count in error_types.most_common():
        print(f"  {key}: {count}")

    print(f"\n{'='*70}")
    print("DETAILED ERRORS")
    print("=" * 70)

    for e in errors[:25]:
        print(f"\nLine {e['line']}:")
        print(f"  Gold: {e['gold']} ({e['gold_type']})")
        print(f"  Pred: {e['pred']} ({e['pred_type']})")
        print(f"  Context: {e['left']} [{e['pred']}] {e['right']}")

    # Analyze patterns in right context for errors
    print(f"\n{'='*70}")
    print("CONTEXT PATTERNS FOR ERRORS")
    print("=" * 70)

    right_words = Counter()
    for e in errors:
        right = e['right'].split()
        if right:
            right_words[strip_harakat(right[0])] += 1

    print(f"\nWords following erroneous من predictions:")
    for word, count in right_words.most_common(15):
        print(f"  {word}: {count}")

if __name__ == '__main__':
    main()
