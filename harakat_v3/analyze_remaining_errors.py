#!/usr/bin/env python3
"""
Analyze remaining errors after V3 grammar fixes.

With V3 at 1.67% DER (test) vs V2 gold, we need to find what's left to fix.
Goal: Sub-1% DER

Categories to analyze:
1. Case ending errors (final vowels)
2. Internal vowel errors (mid-word)
3. Shadda errors (gemination)
4. Sukun errors
5. Specific problematic words
"""

import sys
import os
from collections import defaultdict, Counter

if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')

script_dir = os.path.dirname(os.path.abspath(__file__))
harakat_dir = os.path.dirname(script_dir)
sys.path.insert(0, harakat_dir)

# Import V3 diacritize
from harakat_v3 import diacritize

HARAKAT = 'ًٌٍَُِّْ'
HARAKAT_SET = set(HARAKAT)
FATHA = 'َ'
KASRA = 'ِ'
DAMMA = 'ُ'
SUKUN = 'ْ'
SHADDA = 'ّ'
TANWEEN_FATH = 'ً'
TANWEEN_KASR = 'ٍ'
TANWEEN_DAMM = 'ٌ'


def strip_harakat(text):
    return ''.join(c for c in text if c not in HARAKAT_SET)


def extract_diacritics(word):
    """Extract (base_char, diacritics) pairs."""
    result = []
    current_diacs = ''
    for c in word:
        if c in HARAKAT_SET:
            current_diacs += c
        else:
            if result:
                result[-1] = (result[-1][0], current_diacs)
            result.append((c, ''))
            current_diacs = ''
    if result:
        result[-1] = (result[-1][0], current_diacs)
    return result


def categorize_error(pred_diac, gold_diac, is_final):
    """Categorize the type of diacritization error."""
    p = set(pred_diac) if pred_diac else set()
    g = set(gold_diac) if gold_diac else set()

    has_shadda_pred = SHADDA in p
    has_shadda_gold = SHADDA in g

    # Remove shadda for vowel comparison
    p_vowels = p - {SHADDA}
    g_vowels = g - {SHADDA}

    # Categorize
    if has_shadda_gold and not has_shadda_pred:
        return 'missing_shadda'
    elif has_shadda_pred and not has_shadda_gold:
        return 'extra_shadda'
    elif SUKUN in g_vowels and SUKUN not in p_vowels:
        return 'missing_sukun'
    elif SUKUN in p_vowels and SUKUN not in g_vowels:
        return 'extra_sukun'
    elif is_final and (TANWEEN_FATH in g or TANWEEN_KASR in g or TANWEEN_DAMM in g):
        return 'case_tanween'
    elif is_final:
        return 'case_vowel'
    else:
        return 'internal_vowel'


def analyze_file(gold_path, name, limit=None):
    """Analyze errors in a file."""
    tashkeel_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    v2_gold_path = os.path.join(tashkeel_dir, 'tashkeela_v2', os.path.basename(gold_path))

    print(f"\nAnalyzing {name}...")
    print(f"Using V2 normalized gold: {v2_gold_path}")

    with open(v2_gold_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    if limit:
        lines = lines[:limit]

    error_categories = Counter()
    error_words = defaultdict(lambda: {'count': 0, 'examples': []})
    confusion_matrix = defaultdict(Counter)

    total_positions = 0
    total_errors = 0

    for line in lines:
        undiac = strip_harakat(line)
        pred = diacritize(undiac)

        pred_words = pred.split()
        gold_words = line.split()

        pi, gi = 0, 0
        while pi < len(pred_words) and gi < len(gold_words):
            pw, gw = pred_words[pi], gold_words[gi]
            pb, gb = strip_harakat(pw), strip_harakat(gw)

            if pb == gb:
                pred_diacs = extract_diacritics(pw)
                gold_diacs = extract_diacritics(gw)

                if len(pred_diacs) == len(gold_diacs):
                    for i, ((pc, pd), (gc, gd)) in enumerate(zip(pred_diacs, gold_diacs)):
                        total_positions += 1
                        is_final = (i == len(pred_diacs) - 1)

                        if pd != gd:
                            total_errors += 1
                            category = categorize_error(pd, gd, is_final)
                            error_categories[category] += 1

                            # Track problematic words
                            error_words[pb]['count'] += 1
                            if len(error_words[pb]['examples']) < 3:
                                error_words[pb]['examples'].append({
                                    'pred': pw, 'gold': gw, 'category': category
                                })

                            # Confusion matrix
                            confusion_matrix[gd][pd] += 1

                pi += 1
                gi += 1
            else:
                pi += 1
                gi += 1

    # Print results
    der = 100 * total_errors / total_positions if total_positions > 0 else 0
    print(f"\n{name} Results:")
    print("=" * 70)
    print(f"Total positions: {total_positions:,}")
    print(f"Total errors: {total_errors:,}")
    print(f"DER: {der:.2f}%")

    print(f"\nError breakdown:")
    print("-" * 50)
    for cat, count in error_categories.most_common():
        pct = 100 * count / total_errors if total_errors > 0 else 0
        der_impact = 100 * count / total_positions
        print(f"  {cat:<20} {count:>6} ({pct:.1f}% of errors, {der_impact:.3f}% DER)")

    # Top problematic words
    print(f"\nTop 20 problematic words:")
    print("-" * 50)
    top_words = sorted(error_words.items(), key=lambda x: -x[1]['count'])[:20]
    for word, data in top_words:
        print(f"  {word}: {data['count']} errors")
        for ex in data['examples'][:2]:
            print(f"    pred: {ex['pred']} | gold: {ex['gold']} ({ex['category']})")

    # Confusion patterns
    print(f"\nTop confusion patterns (gold → pred):")
    print("-" * 50)
    confusions = []
    for gold_d, preds in confusion_matrix.items():
        for pred_d, count in preds.items():
            if count >= 10:
                confusions.append((gold_d or 'none', pred_d or 'none', count))
    for g, p, c in sorted(confusions, key=lambda x: -x[2])[:15]:
        print(f"  '{g}' → '{p}': {c}")

    return {
        'der': der,
        'errors': total_errors,
        'categories': dict(error_categories),
        'top_words': [(w, d['count']) for w, d in top_words]
    }


def main():
    print("=" * 70)
    print("V3 REMAINING ERROR ANALYSIS")
    print("=" * 70)
    print("\nGoal: Identify what to fix to reach sub-1% DER")

    harakat_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tashkeel_dir = os.path.dirname(harakat_dir)

    test_path = os.path.join(tashkeel_dir, 'paper_eval', 'tashkeela_test.txt')

    if os.path.exists(test_path):
        results = analyze_file(test_path, "Test Set", limit=500)

        print("\n" + "=" * 70)
        print("RECOMMENDATIONS FOR SUB-1% DER")
        print("=" * 70)

        if results['categories']:
            top_cat = max(results['categories'].items(), key=lambda x: x[1])
            print(f"\n1. Focus on: {top_cat[0]} ({top_cat[1]} errors)")

        if results['top_words']:
            print(f"\n2. High-impact words to target:")
            for word, count in results['top_words'][:5]:
                print(f"   - {word}: {count} errors")


if __name__ == '__main__':
    main()
