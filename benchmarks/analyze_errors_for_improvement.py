#!/usr/bin/env python3
"""
Analyze the 2.16% DER errors to find where improvements can come from.
Goal: Get from 2.16% to <1.5% DER
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'c:\\tashkeel\\harakat')

import importlib
import harakat
importlib.reload(harakat)

from harakat import diacritize, __version__
from collections import Counter, defaultdict

print(f"Error Analysis for DER Improvement")
print(f"Current Version: V{__version__}")
print("=" * 70)
print("Goal: 2.16% → <1.5% DER (need to fix ~2,900+ errors)")
print("=" * 70)

HARAKAT = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670')
FATHA = '\u064E'
KASRA = '\u0650'
DAMMA = '\u064F'
SUKUN = '\u0652'
SHADDA = '\u0651'
TANWEEN_FATH = '\u064B'
TANWEEN_KASR = '\u064D'
TANWEEN_DAMM = '\u064C'

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

def categorize_error(pred_diac, gold_diac, is_final, base_char):
    """Categorize the error type for targeting fixes."""
    p = set(pred_diac) if pred_diac else set()
    g = set(gold_diac) if gold_diac else set()

    # Check shadda
    if SHADDA in g and SHADDA not in p:
        return 'missing_shadda'
    if SHADDA in p and SHADDA not in g:
        return 'extra_shadda'

    # Check sukun
    if SUKUN in g and SUKUN not in p:
        return 'missing_sukun'
    if SUKUN in p and SUKUN not in g:
        return 'extra_sukun'

    # Case endings (final position)
    if is_final:
        if any(t in g for t in [TANWEEN_FATH, TANWEEN_KASR, TANWEEN_DAMM]):
            return 'case_tanween'
        return 'case_vowel'

    # Internal vowel
    return 'internal_vowel'

def analyze_errors():
    v2_gold_file = r'c:\tashkeel\tashkeela_v2\tashkeela_test.txt'

    with open(v2_gold_file, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    print(f"\nAnalyzing {len(lines)} lines from V2 Gold...")

    error_categories = Counter()
    error_words = defaultdict(lambda: {'count': 0, 'examples': [], 'categories': Counter()})
    confusion_pairs = Counter()  # (gold_diac, pred_diac) -> count
    position_errors = {'final': 0, 'internal': 0}

    total_errors = 0
    total_positions = 0

    for i, gold_line in enumerate(lines):
        if (i + 1) % 500 == 0:
            print(f"  Processing {i+1}/{len(lines)}...")

        undiac = strip_h(gold_line)
        pred_line = diacritize(undiac)

        pred_words = pred_line.split()
        gold_words = gold_line.split()

        pi, gi = 0, 0
        while pi < len(pred_words) and gi < len(gold_words):
            pred_base = strip_h(pred_words[pi])
            gold_base = strip_h(gold_words[gi])

            if pred_base == gold_base:
                pred_bases, pred_diacs = extract_diacritic_sequence(pred_words[pi])
                gold_bases, gold_diacs = extract_diacritic_sequence(gold_words[gi])

                if len(pred_diacs) == len(gold_diacs):
                    for j in range(len(pred_diacs)):
                        total_positions += 1
                        is_final = (j == len(pred_diacs) - 1)

                        if pred_diacs[j] != gold_diacs[j]:
                            total_errors += 1

                            # Categorize
                            base_char = pred_bases[j] if j < len(pred_bases) else ''
                            category = categorize_error(pred_diacs[j], gold_diacs[j], is_final, base_char)
                            error_categories[category] += 1

                            # Track position
                            if is_final:
                                position_errors['final'] += 1
                            else:
                                position_errors['internal'] += 1

                            # Track word
                            error_words[pred_base]['count'] += 1
                            error_words[pred_base]['categories'][category] += 1
                            if len(error_words[pred_base]['examples']) < 5:
                                error_words[pred_base]['examples'].append({
                                    'pred': pred_words[pi],
                                    'gold': gold_words[gi],
                                    'pos': j,
                                    'is_final': is_final,
                                    'pred_diac': pred_diacs[j],
                                    'gold_diac': gold_diacs[j]
                                })

                            # Confusion pair
                            confusion_pairs[(gold_diacs[j], pred_diacs[j])] += 1

                pi += 1
                gi += 1
            else:
                pi += 1
                gi += 1

    der = 100 * total_errors / total_positions
    target_errors = int(total_positions * 0.015)  # 1.5% target
    errors_to_fix = total_errors - target_errors

    print(f"\n{'='*70}")
    print(f"CURRENT STATE")
    print(f"{'='*70}")
    print(f"Total positions: {total_positions:,}")
    print(f"Total errors: {total_errors:,}")
    print(f"Current DER: {der:.2f}%")
    print(f"\nTarget: 1.5% DER = {target_errors:,} errors")
    print(f"Errors to fix: {errors_to_fix:,} ({100*errors_to_fix/total_errors:.1f}% of current errors)")

    print(f"\n{'='*70}")
    print(f"ERROR BREAKDOWN BY CATEGORY")
    print(f"{'='*70}")
    print(f"{'Category':<20} {'Count':>8} {'% of Errors':>12} {'DER Impact':>12}")
    print("-" * 55)
    for cat, count in error_categories.most_common():
        pct = 100 * count / total_errors
        der_impact = 100 * count / total_positions
        print(f"{cat:<20} {count:>8,} {pct:>11.1f}% {der_impact:>11.3f}%")

    print(f"\n{'='*70}")
    print(f"ERROR BREAKDOWN BY POSITION")
    print(f"{'='*70}")
    print(f"Final (case endings): {position_errors['final']:,} ({100*position_errors['final']/total_errors:.1f}%)")
    print(f"Internal (mid-word):  {position_errors['internal']:,} ({100*position_errors['internal']/total_errors:.1f}%)")

    print(f"\n{'='*70}")
    print(f"TOP 30 ERROR WORDS (targeting these = maximum impact)")
    print(f"{'='*70}")
    top_words = sorted(error_words.items(), key=lambda x: -x[1]['count'])[:30]

    print(f"{'Word':<15} {'Errors':>8} {'DER %':>8} {'Top Category':<20}")
    print("-" * 55)
    cumulative_fixed = 0
    for word, data in top_words:
        top_cat = data['categories'].most_common(1)[0][0] if data['categories'] else 'unknown'
        der_pct = 100 * data['count'] / total_positions
        cumulative_fixed += data['count']
        cumulative_der = 100 * (total_errors - cumulative_fixed) / total_positions
        print(f"{word:<15} {data['count']:>8,} {der_pct:>7.3f}% {top_cat:<20}")

    print(f"\nIf we fix top 30 words: {cumulative_der:.2f}% DER remaining")

    print(f"\n{'='*70}")
    print(f"TOP CONFUSION PAIRS (gold → pred)")
    print(f"{'='*70}")
    print(f"{'Gold':<10} {'Pred':<10} {'Count':>8} {'DER %':>8}")
    print("-" * 40)
    for (gold, pred), count in confusion_pairs.most_common(20):
        gold_disp = gold if gold else '(none)'
        pred_disp = pred if pred else '(none)'
        der_pct = 100 * count / total_positions
        print(f"{gold_disp:<10} {pred_disp:<10} {count:>8,} {der_pct:>7.3f}%")

    print(f"\n{'='*70}")
    print(f"EXAMPLES FROM TOP ERROR WORDS")
    print(f"{'='*70}")
    for word, data in top_words[:10]:
        print(f"\n{word} ({data['count']} errors):")
        for ex in data['examples'][:3]:
            pos_type = "FINAL" if ex['is_final'] else "internal"
            print(f"  pred: {ex['pred']}")
            print(f"  gold: {ex['gold']}")
            print(f"  pos {ex['pos']} ({pos_type}): '{ex['pred_diac']}' → '{ex['gold_diac']}'")

    print(f"\n{'='*70}")
    print(f"ACTIONABLE RECOMMENDATIONS")
    print(f"{'='*70}")

    # Calculate what to focus on
    case_errors = error_categories.get('case_vowel', 0) + error_categories.get('case_tanween', 0)
    internal_errors = error_categories.get('internal_vowel', 0)
    shadda_errors = error_categories.get('missing_shadda', 0) + error_categories.get('extra_shadda', 0)
    sukun_errors = error_categories.get('missing_sukun', 0) + error_categories.get('extra_sukun', 0)

    print(f"""
1. CASE ENDINGS ({case_errors:,} errors = {100*case_errors/total_positions:.2f}% DER)
   - These are final position vowel/tanween errors
   - Hardest to fix (requires syntactic understanding)
   - But if we exclude case: DER drops significantly

2. INTERNAL VOWELS ({internal_errors:,} errors = {100*internal_errors/total_positions:.2f}% DER)
   - Mid-word vowel patterns (فَعَلَ vs فَعِلَ)
   - Can be fixed with word-specific overrides
   - High-frequency words give most bang for buck

3. SHADDA ERRORS ({shadda_errors:,} errors = {100*shadda_errors/total_positions:.2f}% DER)
   - Missing or extra gemination marks
   - Often systematic (sun letters, specific words)

4. SUKUN ERRORS ({sukun_errors:,} errors = {100*sukun_errors/total_positions:.2f}% DER)
   - Often pattern-based (verb forms)

STRATEGY TO REACH <1.5% DER:
- Need to fix {errors_to_fix:,} errors
- Top 30 words alone could fix ~{cumulative_fixed:,} errors
- Focus on high-frequency words with consistent error patterns
""")

    return {
        'total_errors': total_errors,
        'total_positions': total_positions,
        'der': der,
        'categories': dict(error_categories),
        'top_words': [(w, d['count'], d['examples']) for w, d in top_words],
        'confusion_pairs': dict(confusion_pairs.most_common(50))
    }

if __name__ == '__main__':
    results = analyze_errors()
