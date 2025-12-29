#!/usr/bin/env python3
"""
Analyze سنة errors - year vs tradition (sunna).
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
    print("سنة ERROR ANALYSIS (year vs sunna/tradition)")
    print("=" * 70)

    # سنة variants
    sana_variants = {
        'سَنَة': 'year',
        'سَنَةً': 'year',
        'سَنَةٍ': 'year',
        'سَنَةِ': 'year',
        'سُنَّة': 'tradition',
        'سُنَّةً': 'tradition',
        'سُنَّةٍ': 'tradition',
        'سُنَّةِ': 'tradition',
        'السَّنَة': 'year',
        'السُّنَّة': 'tradition',
        'السُّنَّةِ': 'tradition',
    }

    errors = []

    for line_num, gold_line in enumerate(lines):
        undiac = strip_harakat(gold_line)
        pred_line = diacritize(undiac)

        gold_words = gold_line.split()
        pred_words = pred_line.split()

        for i, (gw, pw) in enumerate(zip(gold_words, pred_words)):
            gb = strip_harakat(gw)
            if 'سنة' in gb or gb == 'السنة':
                if gw != pw:
                    # Get context
                    left_ctx = gold_words[max(0, i-3):i]
                    right_ctx = gold_words[i+1:i+4]

                    gold_type = 'tradition' if 'ُنّ' in gw or 'ُّ' in gw else 'year'
                    pred_type = 'tradition' if 'ُنّ' in pw or 'ُّ' in pw else 'year'

                    errors.append({
                        'line': line_num,
                        'gold': gw,
                        'pred': pw,
                        'gold_type': gold_type,
                        'pred_type': pred_type,
                        'left': ' '.join(left_ctx),
                        'right': ' '.join(right_ctx),
                    })

    print(f"\nTotal سنة errors: {len(errors)}")

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

    for e in errors[:20]:
        print(f"\nLine {e['line']}:")
        print(f"  Gold: {e['gold']} ({e['gold_type']})")
        print(f"  Pred: {e['pred']} ({e['pred_type']})")
        print(f"  Context: {e['left']} [{e['pred']}] {e['right']}")

    # Analyze patterns
    print(f"\n{'='*70}")
    print("CONTEXT PATTERNS")
    print("=" * 70)

    tradition_context = []
    year_context = []
    for e in errors:
        ctx = f"{e['left']} {e['right']}"
        if e['gold_type'] == 'tradition':
            tradition_context.append(ctx)
        else:
            year_context.append(ctx)

    print(f"\nContexts where سُنَّة (tradition) is correct:")
    for ctx in tradition_context[:10]:
        print(f"  {ctx}")

    print(f"\nContexts where سَنَة (year) is correct:")
    for ctx in year_context[:10]:
        print(f"  {ctx}")

if __name__ == '__main__':
    main()
