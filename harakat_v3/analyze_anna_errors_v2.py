#!/usr/bin/env python3
"""
Analyze anna fix errors - compare before/after anna_fix.
"""
import sys
import os
import io

script_dir = os.path.dirname(os.path.abspath(__file__))
harakat_dir = os.path.dirname(script_dir)
sys.path.insert(0, harakat_dir)

from harakat_v3 import diacritize
from harakat_v3.rules.anna_fix import strip_harakat, has_shadda_on_nun

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

    target_bases = {'أن', 'إن', 'بأن', 'فإن', 'لأن', 'كأن', 'وأن', 'وإن'}

    print("=" * 70)
    print("ANNA FIX ERROR ANALYSIS")
    print("=" * 70)

    false_positives = []  # Added shadda but gold doesn't have it
    true_positives = []   # Added shadda correctly
    false_negatives = []  # Gold has shadda but we don't
    true_negatives = []   # Correctly didn't add shadda

    for line_num, gold_line in enumerate(lines):
        undiac = strip_harakat(gold_line)
        pred_no_anna = diacritize(undiac, apply_anna=False)
        pred_with_anna = diacritize(undiac, apply_anna=True)

        gold_words = gold_line.split()
        pred_no_words = pred_no_anna.split()
        pred_with_words = pred_with_anna.split()

        for i, (gw, pn, pw) in enumerate(zip(gold_words, pred_no_words, pred_with_words)):
            gb = strip_harakat(gw)
            if gb in target_bases:
                gold_shadda = has_shadda_on_nun(gw)
                pred_no_shadda = has_shadda_on_nun(pn)
                pred_with_shadda = has_shadda_on_nun(pw)

                next_word = gold_words[i+1] if i+1 < len(gold_words) else ''
                next_base = strip_harakat(next_word)

                # Was shadda added by anna_fix?
                shadda_added = pred_with_shadda and not pred_no_shadda

                if shadda_added:
                    if gold_shadda:
                        true_positives.append({
                            'line': line_num, 'word': gb, 'gold': gw,
                            'next': next_word, 'next_base': next_base
                        })
                    else:
                        false_positives.append({
                            'line': line_num, 'word': gb, 'gold': gw,
                            'next': next_word, 'next_base': next_base
                        })
                elif gold_shadda and not pred_with_shadda:
                    false_negatives.append({
                        'line': line_num, 'word': gb, 'gold': gw,
                        'next': next_word, 'next_base': next_base
                    })
                elif not gold_shadda and not pred_with_shadda:
                    true_negatives.append({
                        'line': line_num, 'word': gb, 'gold': gw,
                        'next': next_word, 'next_base': next_base
                    })

    print(f"\nTotal instances of أن/إن etc: {len(true_positives) + len(false_positives) + len(false_negatives) + len(true_negatives)}")
    print(f"\nTrue Positives (correctly added shadda): {len(true_positives)}")
    print(f"False Positives (wrongly added shadda): {len(false_positives)}")
    print(f"False Negatives (missed shadda): {len(false_negatives)}")
    print(f"True Negatives (correctly no shadda): {len(true_negatives)}")

    print(f"\n{'='*70}")
    print(f"FALSE POSITIVES - Added shadda when gold doesn't have it")
    print("=" * 70)
    for fp in false_positives[:20]:
        print(f"\n  Line {fp['line']}: {fp['word']}")
        print(f"    Gold: {fp['gold']}")
        print(f"    Next word: {fp['next']} (base: {fp['next_base']})")

    print(f"\n{'='*70}")
    print(f"TRUE POSITIVES - Correctly added shadda")
    print("=" * 70)
    for tp in true_positives[:10]:
        print(f"\n  Line {tp['line']}: {tp['word']}")
        print(f"    Gold: {tp['gold']}")
        print(f"    Next word: {tp['next']} (base: {tp['next_base']})")

    # Pattern analysis
    print(f"\n{'='*70}")
    print("PATTERN ANALYSIS - What follows false positives?")
    print("=" * 70)
    from collections import Counter
    fp_patterns = Counter()
    for fp in false_positives:
        fp_patterns[fp['next_base']] += 1

    for pattern, count in fp_patterns.most_common(15):
        print(f"  {pattern}: {count}")

if __name__ == '__main__':
    main()
