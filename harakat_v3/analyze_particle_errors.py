#!/usr/bin/env python3
"""
Analyze particle-related errors in the test and validation sets.
Count how many errors could be fixed by particle rules.
"""

import sys
import os
from collections import defaultdict

if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')

harakat_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, harakat_dir)

from harakat_v2 import diacritize_with_sweep

# Constants
HARAKAT = 'ًٌٍَُِّْ'
HARAKAT_SET = set(HARAKAT)


def strip_harakat(text):
    return ''.join(c for c in text if c not in HARAKAT_SET)


# Particles with fixed internal patterns
FIXED_PARTICLES = {
    # word_base: (correct_pattern, description)
    'إلى': ('إِلَى', 'kasra on hamza'),
    'إلا': ('إِلَّا', 'kasra on hamza + shadda'),
    'إذا': ('إِذَا', 'kasra on hamza'),
    'إذ': ('إِذْ', 'kasra on hamza'),
    'حتى': ('حَتَّى', 'shadda on ta'),
}


def get_first_vowel(word):
    """Get the first vowel after the first base letter."""
    for i, c in enumerate(word):
        if c in HARAKAT_SET:
            return c
    return None


def words_match_base(pred_word, gold_word):
    """Check if two words have the same base."""
    return strip_harakat(pred_word) == strip_harakat(gold_word)


def compare_word_diacritics(pred, gold):
    """Compare diacritics between predicted and gold words."""
    pred_base = strip_harakat(pred)
    gold_base = strip_harakat(gold)

    if pred_base != gold_base:
        return None, None, None

    # Extract diacritics per position
    pred_diacs = []
    gold_diacs = []

    pred_diac = ''
    gold_diac = ''
    pred_i = 0
    gold_i = 0

    # Simple diacritic extraction
    for c in pred:
        if c in HARAKAT_SET:
            pred_diac += c
        else:
            pred_diacs.append(pred_diac)
            pred_diac = ''
    pred_diacs.append(pred_diac)

    for c in gold:
        if c in HARAKAT_SET:
            gold_diac += c
        else:
            gold_diacs.append(gold_diac)
            gold_diac = ''
    gold_diacs.append(gold_diac)

    if len(pred_diacs) != len(gold_diacs):
        return None, None, None

    errors = 0
    for p, g in zip(pred_diacs, gold_diacs):
        if p != g:
            errors += 1

    return errors, pred_diacs, gold_diacs


def analyze_file(filepath, name, limit=None):
    """Analyze particle errors in a file."""
    print(f"\nAnalyzing {name}...")

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    if limit:
        lines = lines[:limit]

    particle_errors = defaultdict(lambda: {'count': 0, 'examples': []})
    total_particle_words = defaultdict(int)

    for line_num, gold_line in enumerate(lines):
        undiac = strip_harakat(gold_line)
        pred_line = diacritize_with_sweep(undiac)

        pred_words = pred_line.split()
        gold_words = gold_line.split()

        # Align words
        pi, gi = 0, 0
        while pi < len(pred_words) and gi < len(gold_words):
            pred_base = strip_harakat(pred_words[pi])
            gold_base = strip_harakat(gold_words[gi])

            if pred_base == gold_base:
                # Check if this is a target particle
                if pred_base in FIXED_PARTICLES:
                    total_particle_words[pred_base] += 1

                    # Compare diacritics
                    if pred_words[pi] != gold_words[gi]:
                        particle_errors[pred_base]['count'] += 1
                        if len(particle_errors[pred_base]['examples']) < 3:
                            particle_errors[pred_base]['examples'].append({
                                'pred': pred_words[pi],
                                'gold': gold_words[gi],
                                'context': ' '.join(gold_words[max(0,gi-2):gi+3])
                            })

                pi += 1
                gi += 1
            else:
                # Skip misaligned words
                if pi < len(pred_words) - 1 and strip_harakat(pred_words[pi+1]) == gold_base:
                    pi += 1
                elif gi < len(gold_words) - 1 and pred_base == strip_harakat(gold_words[gi+1]):
                    gi += 1
                else:
                    pi += 1
                    gi += 1

        if (line_num + 1) % 500 == 0:
            print(f"  Processed {line_num + 1}/{len(lines)} lines")

    # Print results
    print(f"\n{name} Results:")
    print("=" * 60)

    total_errors = 0
    for particle, data in sorted(particle_errors.items(), key=lambda x: -x[1]['count']):
        total = total_particle_words[particle]
        errors = data['count']
        total_errors += errors
        accuracy = 100 * (total - errors) / total if total > 0 else 0

        correct_pattern, desc = FIXED_PARTICLES[particle]
        print(f"\n{particle} ({desc})")
        print(f"  Total occurrences: {total}")
        print(f"  Errors: {errors} ({100*errors/total:.1f}% error rate)")
        print(f"  Accuracy: {accuracy:.1f}%")
        print(f"  Expected: {correct_pattern}")

        for ex in data['examples']:
            print(f"    Pred: {ex['pred']} | Gold: {ex['gold']}")
            print(f"    Context: {ex['context']}")

    print(f"\nTotal fixable particle errors: {total_errors}")
    return total_errors, particle_errors


def main():
    print("=" * 70)
    print("PARTICLE ERROR ANALYSIS")
    print("=" * 70)

    # Find data files
    base_paths = [
        os.path.join(harakat_dir, '..'),
        os.path.join(harakat_dir, '..', 'paper_eval'),
        harakat_dir,
    ]

    val_path = None
    test_path = None

    for base in base_paths:
        if os.path.exists(os.path.join(base, 'tashkeela_val.txt')):
            val_path = os.path.join(base, 'tashkeela_val.txt')
        if os.path.exists(os.path.join(base, 'tashkeela_test.txt')):
            test_path = os.path.join(base, 'tashkeela_test.txt')

    if not test_path:
        print("Could not find test data files")
        return

    print(f"Test set: {test_path}")
    if val_path:
        print(f"Validation set: {val_path}")

    # Analyze test set
    test_errors, test_data = analyze_file(test_path, "Test Set (2,500 lines)")

    # Analyze validation set (sample for speed)
    if val_path:
        val_errors, val_data = analyze_file(val_path, "Validation Set (5,000 lines)")

    print("\n" + "=" * 70)
    print("POTENTIAL DER IMPACT")
    print("=" * 70)

    # Estimate DER impact (443,641 total positions in test set)
    test_positions = 443641
    der_impact = 100 * test_errors / test_positions
    print(f"\nTest set:")
    print(f"  Fixable errors: {test_errors}")
    print(f"  Estimated DER reduction: {der_impact:.3f}%")
    print(f"  New DER would be: {2.23 - der_impact:.2f}%")


if __name__ == '__main__':
    main()
