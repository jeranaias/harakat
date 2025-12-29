#!/usr/bin/env python3
"""
Detailed Error Categorization for V3 Planning

Categorizes V2 errors into actionable groups for targeted fixes.
"""

import sys
import os

if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')

sys.path.insert(0, '../..')
sys.path.insert(0, '../../harakat_v2')

from collections import Counter, defaultdict
import json

# Import V2
from harakat_v2 import HarakatV2, strip_harakat, diacritize_with_sweep

# Constants
HARAKAT = 'ًٌٍَُِّْ'
HARAKAT_SET = set(HARAKAT)
SHADDA = 'ّ'

# Particles that should have fixed diacritization
FIXED_PARTICLES = {
    'إلى': 'إِلَى',
    'إلا': 'إِلَّا',
    'إذا': 'إِذَا',
    'على': 'عَلَى',
    'حتى': 'حَتَّى',
    'إذ': 'إِذْ',
    'إن': None,  # Context-dependent, handled by sweep
    'أن': None,  # Context-dependent, handled by sweep
}

# High-frequency homographs
HOMOGRAPHS = {
    'من': ['مِنْ', 'مَنْ'],  # from / who
    'أم': ['أَمْ', 'أُمّ', 'أَمَّ'],  # or / mother / led
    'غير': ['غَيْر', 'غَيَّر'],  # other / changed
    'ملك': ['مَلِك', 'مُلْك', 'مَلَك', 'مَلَّك'],  # king/property/angel/owned
    'علم': ['عِلْم', 'عَلِمَ', 'عَلَم', 'عُلِمَ'],  # knowledge/knew/flag/was known
    'ذكر': ['ذِكْر', 'ذَكَرَ', 'ذُكِرَ', 'ذَكَر'],  # mention/mentioned/was mentioned/male
    'قتل': ['قَتْل', 'قَتَلَ', 'قُتِلَ'],  # killing/killed/was killed
    'أخذ': ['أَخْذ', 'أَخَذَ', 'أُخِذَ'],  # taking/took/was taken
    'وجد': ['وَجَدَ', 'وُجِدَ', 'وَجْد'],  # found/was found/ecstasy
    'ثم': ['ثُمَّ', 'ثَمَّ'],  # then/there
}

# Verb form patterns (for active/passive detection)
ACTIVE_PASSIVE_PATTERNS = {
    # Form I
    'فَعَلَ': 'فُعِلَ',
    'يَفْعَلُ': 'يُفْعَلُ',
    'يَفْعِلُ': 'يُفْعَلُ',
    'فَعِلَ': 'فُعِلَ',
}


def get_diac_name(d):
    """Get human-readable name for diacritic."""
    names = {
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
    if SHADDA in d:
        other = d.replace(SHADDA, '')
        if other:
            return f"shadda+{names.get(other, other)}"
        return 'shadda'
    return names.get(d, d if d else 'none')


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


def categorize_error(error):
    """Categorize a single error into actionable groups."""
    categories = []

    word_base = error['word_base']
    pred_name = error['pred_name']
    gold_name = error['gold_name']
    is_final = error['is_final']

    # Category 1: Fixed particle errors
    if word_base in FIXED_PARTICLES:
        if FIXED_PARTICLES[word_base] is not None:
            categories.append('fixable_particle')

    # Category 2: Homograph errors
    if word_base in HOMOGRAPHS:
        categories.append('homograph')

    # Category 3: Shadda-related
    if 'shadda' in pred_name or 'shadda' in gold_name:
        if 'shadda' in gold_name and 'shadda' not in pred_name:
            categories.append('missing_shadda')
        elif 'shadda' in pred_name and 'shadda' not in gold_name:
            categories.append('extra_shadda')
        else:
            categories.append('shadda_vowel_wrong')

    # Category 4: Active/Passive confusion
    if pred_name in ['fatha', 'damma'] and gold_name in ['fatha', 'damma']:
        if not is_final:  # Internal vowel
            categories.append('possible_voice_error')

    # Category 5: Case ending errors
    if is_final:
        if 'tanwin' in pred_name or 'tanwin' in gold_name:
            categories.append('tanween_case')
        else:
            categories.append('case_ending')

    # Category 6: Sukun confusion
    if pred_name == 'sukun' or gold_name == 'sukun':
        categories.append('sukun_confusion')

    # Default
    if not categories:
        categories.append('other_vowel')

    return categories


def analyze_errors_detailed(test_path, output_path):
    """Perform detailed error analysis and categorization."""

    print("="*70)
    print("DETAILED ERROR CATEGORIZATION FOR V3")
    print("="*70)

    # Load V2
    print("\nLoading Harakat V2 + Sweep...")
    h = HarakatV2()

    # Load test data
    with open(test_path, 'r', encoding='utf-8') as f:
        test_lines = [l.strip() for l in f if l.strip()]
    print(f"Test lines: {len(test_lines)}")

    # Collect all errors with categories
    all_errors = []
    category_counts = Counter()
    category_errors = defaultdict(list)

    print("\nProcessing...")
    for i, gold_line in enumerate(test_lines):
        undiac = strip_harakat(gold_line)
        pred_line = diacritize_with_sweep(undiac)

        gold_words = gold_line.split()
        pred_words = pred_line.split()

        # Align and compare
        pi, gi = 0, 0
        while pi < len(pred_words) and gi < len(gold_words):
            pb = strip_harakat(pred_words[pi])
            gb = strip_harakat(gold_words[gi])

            if pb == gb:
                base_p, diacs_p = extract_diacritics(pred_words[pi])
                base_g, diacs_g = extract_diacritics(gold_words[gi])

                if len(diacs_p) == len(diacs_g):
                    for j, (pd, gd) in enumerate(zip(diacs_p, diacs_g)):
                        if pd != gd:
                            error = {
                                'word_base': gb,
                                'word_gold': gold_words[gi],
                                'word_pred': pred_words[pi],
                                'position': j,
                                'is_final': (j == len(diacs_g) - 1),
                                'char': base_g[j] if j < len(base_g) else '?',
                                'pred_diac': pd,
                                'gold_diac': gd,
                                'pred_name': get_diac_name(pd),
                                'gold_name': get_diac_name(gd),
                            }

                            # Categorize
                            cats = categorize_error(error)
                            error['categories'] = cats

                            all_errors.append(error)

                            for cat in cats:
                                category_counts[cat] += 1
                                if len(category_errors[cat]) < 100:
                                    category_errors[cat].append(error)

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
            print(f"  {i+1}/{len(test_lines)} - {len(all_errors)} errors")

    # Results
    print(f"\n{'='*70}")
    print("CATEGORY BREAKDOWN")
    print(f"{'='*70}")
    print(f"\nTotal errors: {len(all_errors)}")

    print("\nBy category:")
    for cat, count in category_counts.most_common():
        pct = 100 * count / len(all_errors)
        print(f"  {cat:25} : {count:5} ({pct:5.2f}%)")

    # Actionable categories
    print(f"\n{'='*70}")
    print("ACTIONABLE CATEGORIES (V3 TARGETS)")
    print(f"{'='*70}")

    actionable = ['fixable_particle', 'homograph', 'missing_shadda',
                  'possible_voice_error', 'case_ending']

    for cat in actionable:
        count = category_counts.get(cat, 0)
        if count > 0:
            print(f"\n## {cat.upper()} ({count} errors)")
            examples = category_errors.get(cat, [])[:10]
            for ex in examples:
                print(f"  {ex['word_base']:12} | {ex['pred_name']:15} -> {ex['gold_name']:15}")

    # Save detailed data
    output_data = {
        'total_errors': len(all_errors),
        'category_counts': dict(category_counts),
        'category_examples': {k: v[:50] for k, v in category_errors.items()},
        'all_errors': all_errors[:1000],  # First 1000 for inspection
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved detailed analysis to {output_path}")

    return category_counts, category_errors


if __name__ == '__main__':
    test_path = '../../benchmarks/tashkeela_test.txt'
    if not os.path.exists(test_path):
        test_path = '../../../tashkeela_test.txt'
    if not os.path.exists(test_path):
        test_path = '../../../paper_eval/tashkeela_test.txt'

    output_path = 'categorized_errors.json'

    analyze_errors_detailed(test_path, output_path)
