#!/usr/bin/env python3
"""
Analyze remaining أن/أنّ errors specifically.
"""

import sys
import os

if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')

script_dir = os.path.dirname(os.path.abspath(__file__))
harakat_dir = os.path.dirname(script_dir)
sys.path.insert(0, harakat_dir)

from harakat_v3 import diacritize

HARAKAT = set('ًٌٍَُِّْ')
SHADDA = 'ّ'


def strip_harakat(text):
    return ''.join(c for c in text if c not in HARAKAT)


def has_shadda_on_nun(word):
    """Check if word has shadda on ن."""
    for i, c in enumerate(word):
        if c == 'ن':
            for j in range(i+1, len(word)):
                if word[j] == SHADDA:
                    return True
                if word[j] not in HARAKAT:
                    break
    return False


def main():
    import io
    if not isinstance(sys.stdout, io.TextIOWrapper):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        except:
            pass

    print("=" * 70)
    print("أَنَّ/أَنْ ERROR ANALYSIS")
    print("=" * 70)

    tashkeel_dir = os.path.dirname(harakat_dir)
    v2_gold = os.path.join(tashkeel_dir, 'tashkeela_v2', 'tashkeela_test.txt')

    with open(v2_gold, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()][:500]

    target_bases = {'أن', 'إن', 'بأن', 'فإن', 'لأن', 'كأن', 'وأن', 'وإن'}

    errors = []
    correct = []

    for line_num, gold_line in enumerate(lines):
        undiac = strip_harakat(gold_line)
        pred_line = diacritize(undiac)

        gold_words = gold_line.split()
        pred_words = pred_line.split()

        gi, pi = 0, 0
        while gi < len(gold_words) and pi < len(pred_words):
            gw, pw = gold_words[gi], pred_words[pi]
            gb, pb = strip_harakat(gw), strip_harakat(pw)

            if gb == pb and gb in target_bases:
                gold_shadda = has_shadda_on_nun(gw)
                pred_shadda = has_shadda_on_nun(pw)

                # Get context
                left_context = gold_words[max(0, gi-2):gi]
                right_context = gold_words[gi+1:gi+3]

                if gold_shadda != pred_shadda:
                    errors.append({
                        'line': line_num,
                        'gold': gw,
                        'pred': pw,
                        'gold_shadda': gold_shadda,
                        'pred_shadda': pred_shadda,
                        'left': left_context,
                        'right': right_context,
                        'base': gb,
                    })
                else:
                    correct.append({
                        'gold': gw,
                        'base': gb,
                    })

            gi += 1
            pi += 1

    print(f"\nTotal أن/إن etc. instances: {len(errors) + len(correct)}")
    print(f"Correct: {len(correct)}")
    print(f"Errors: {len(errors)}")

    print(f"\n{'='*70}")
    print("ERRORS (missing or extra shadda)")
    print("="*70)

    # Group by error type
    missing_shadda = [e for e in errors if e['gold_shadda'] and not e['pred_shadda']]
    extra_shadda = [e for e in errors if not e['gold_shadda'] and e['pred_shadda']]

    print(f"\nMissing shadda (should be أَنَّ but predicted أَنْ): {len(missing_shadda)}")
    for e in missing_shadda[:15]:
        context = ' '.join(e['left']) + f" [{e['pred']} → {e['gold']}] " + ' '.join(e['right'])
        next_word = e['right'][0] if e['right'] else 'END'
        next_base = strip_harakat(next_word)
        print(f"  Line {e['line']}: {context}")
        print(f"    Next word: {next_word} (base: {next_base})")

    print(f"\nExtra shadda (should be أَنْ but predicted أَنَّ): {len(extra_shadda)}")
    for e in extra_shadda[:15]:
        context = ' '.join(e['left']) + f" [{e['pred']} → {e['gold']}] " + ' '.join(e['right'])
        next_word = e['right'][0] if e['right'] else 'END'
        next_base = strip_harakat(next_word)
        print(f"  Line {e['line']}: {context}")
        print(f"    Next word: {next_word} (base: {next_base})")

    # Analyze patterns
    print(f"\n{'='*70}")
    print("PATTERN ANALYSIS")
    print("="*70)

    from collections import Counter
    next_words = Counter()
    for e in missing_shadda:
        if e['right']:
            next_words[strip_harakat(e['right'][0])] += 1

    print(f"\nWords following errors (missing shadda):")
    for word, count in next_words.most_common(10):
        print(f"  {word}: {count}")


if __name__ == '__main__':
    main()
