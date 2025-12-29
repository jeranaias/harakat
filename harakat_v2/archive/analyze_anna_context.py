#!/usr/bin/env python3
"""
Analyze contexts where أَنَّ/أَنْ errors occur to find patterns.
"""

import sys
import os

if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')

from harakat_v2 import HarakatV2, strip_harakat

HARAKAT = 'ًٌٍَُِّْ'
HARAKAT_SET = set(HARAKAT)
SHADDA = 'ّ'


def has_shadda_on_char(word, target_char='ن'):
    """Check if word has shadda on target character."""
    for i, c in enumerate(word):
        if c == target_char:
            for j in range(i+1, len(word)):
                if word[j] == SHADDA:
                    return True
                if word[j] not in HARAKAT_SET:
                    break
    return False


def main():
    print("="*70)
    print("ANALYZING أَنَّ vs أَنْ ERROR CONTEXTS")
    print("="*70)

    # Load V2
    h = HarakatV2()

    # Load test data
    test_path = '../benchmarks/tashkeela_test.txt'
    if not os.path.exists(test_path):
        test_path = '../../tashkeela_test.txt'
    if not os.path.exists(test_path):
        test_path = '../../paper_eval/tashkeela_test.txt'

    with open(test_path, 'r', encoding='utf-8') as f:
        test_lines = [l.strip() for l in f if l.strip()]

    print(f"Test lines: {len(test_lines)}")

    # Target bases
    target_bases = {'أن', 'إن', 'بأن', 'فإن', 'لأن', 'كأن', 'وأن', 'وإن', 'بأنه', 'فأن', 'لأنه'}

    # Collect errors
    missing_shadda = []  # Predicted no shadda, gold has shadda
    extra_shadda = []    # Predicted shadda, gold has no shadda

    print("\nProcessing...")

    for i, gold_line in enumerate(test_lines):
        undiac = strip_harakat(gold_line)
        pred_line = h.diacritize(undiac)

        gold_words = gold_line.split()
        pred_words = pred_line.split()

        # Align words
        pi, gi = 0, 0
        while pi < len(pred_words) and gi < len(gold_words):
            pb = strip_harakat(pred_words[pi])
            gb = strip_harakat(gold_words[gi])

            if pb == gb:
                # Check if target word
                if gb in target_bases or (gb.startswith('ا') and 'ن' in gb):
                    pred_has_shadda = has_shadda_on_char(pred_words[pi], 'ن')
                    gold_has_shadda = has_shadda_on_char(gold_words[gi], 'ن')

                    if pred_has_shadda != gold_has_shadda:
                        # Get context
                        start = max(0, gi - 5)
                        end = min(len(gold_words), gi + 6)
                        context_gold = ' '.join(gold_words[start:end])
                        context_undiac = strip_harakat(context_gold)

                        if gold_has_shadda and not pred_has_shadda:
                            missing_shadda.append({
                                'word_base': gb,
                                'pred': pred_words[pi],
                                'gold': gold_words[gi],
                                'context_gold': context_gold,
                                'context_undiac': context_undiac,
                            })
                        else:
                            extra_shadda.append({
                                'word_base': gb,
                                'pred': pred_words[pi],
                                'gold': gold_words[gi],
                                'context_gold': context_gold,
                                'context_undiac': context_undiac,
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

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(test_lines)}")

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"\nMissing shadda (أَنَ → أَنَّ): {len(missing_shadda)} errors")
    print(f"Extra shadda (أَنَّ → أَنَ): {len(extra_shadda)} errors")

    # Analyze missing shadda patterns
    print(f"\n{'='*70}")
    print("MISSING SHADDA EXAMPLES (should be أَنَّ)")
    print(f"{'='*70}")

    from collections import Counter
    word_counts = Counter(e['word_base'] for e in missing_shadda)
    print(f"\nBy word: {dict(word_counts.most_common(10))}")

    print("\nSample contexts:")
    for ex in missing_shadda[:30]:
        print(f"\n  Base: {ex['word_base']}")
        print(f"  Pred: {ex['pred']} -> Gold: {ex['gold']}")
        print(f"  Context: {ex['context_gold']}")

    # Look for patterns
    print(f"\n{'='*70}")
    print("PATTERN ANALYSIS")
    print(f"{'='*70}")

    # Check what comes after
    after_word = Counter()
    for ex in missing_shadda:
        ctx_words = ex['context_gold'].split()
        try:
            idx = next(i for i, w in enumerate(ctx_words) if strip_harakat(w) == ex['word_base'])
            if idx + 1 < len(ctx_words):
                next_word_base = strip_harakat(ctx_words[idx + 1])
                after_word[next_word_base] += 1
        except StopIteration:
            pass

    print("\nWords following missing shadda (top 20):")
    for word, count in after_word.most_common(20):
        print(f"  {word}: {count}")

    # Check what comes before
    before_word = Counter()
    for ex in missing_shadda:
        ctx_words = ex['context_gold'].split()
        try:
            idx = next(i for i, w in enumerate(ctx_words) if strip_harakat(w) == ex['word_base'])
            if idx > 0:
                prev_word_base = strip_harakat(ctx_words[idx - 1])
                before_word[prev_word_base] += 1
        except StopIteration:
            pass

    print("\nWords before missing shadda (top 20):")
    for word, count in before_word.most_common(20):
        print(f"  {word}: {count}")

    print(f"\n{'='*70}")
    print("EXTRA SHADDA EXAMPLES (should be أَنْ)")
    print(f"{'='*70}")

    word_counts2 = Counter(e['word_base'] for e in extra_shadda)
    print(f"\nBy word: {dict(word_counts2.most_common(10))}")

    print("\nSample contexts:")
    for ex in extra_shadda[:20]:
        print(f"\n  Base: {ex['word_base']}")
        print(f"  Pred: {ex['pred']} -> Gold: {ex['gold']}")
        print(f"  Context: {ex['context_gold']}")


if __name__ == '__main__':
    main()
