#!/usr/bin/env python3
"""
Test script for particle rules.

Analyzes current V2 output for particles that have fixed patterns.
"""

import sys
import os

if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harakat_v2 import diacritize_with_sweep

# Constants
HARAKAT = 'ًٌٍَُِّْ'
HARAKAT_SET = set(HARAKAT)


def strip_harakat(text):
    """Remove all diacritics."""
    return ''.join(c for c in text if c not in HARAKAT_SET)


# Test cases: (input, expected_pattern_description)
TEST_PARTICLES = [
    # Prepositions with kasra on initial hamza
    ("إلى المدرسة", "إِلَى", "kasra on hamza-below"),
    ("إلا الحق", "إِلَّا", "kasra on hamza-below + shadda"),
    ("إذا جاء", "إِذَا", "kasra on hamza-below"),
    ("إذ قال", "إِذْ", "kasra on hamza-below"),

    # Sun letter assimilation
    ("الشمس", "الشَّ", "shadda on sun letter after ال"),
    ("الرحمن", "الرَّ", "shadda on sun letter after ال"),
    ("النور", "النُّ", "shadda on sun letter after ال"),
    ("السماء", "السَّ", "shadda on sun letter after ال"),
    ("الدار", "الدَّ", "shadda on sun letter after ال"),
    ("التاجر", "التَّ", "shadda on sun letter after ال"),

    # Non-sun letters (should NOT have shadda)
    ("الكتاب", "الْكِ", "sukun on lam, no shadda"),
    ("المدرسة", "الْمَ", "sukun on lam, no shadda"),
    ("البيت", "الْبَ", "sukun on lam, no shadda"),

    # Allah (special pattern)
    ("الله أكبر", "اللَّهُ", "shadda on lam"),

    # Question particles
    ("ماذا فعلت", "مَاذَا", "fatha-alif-fatha"),
    ("كيف حالك", "كَيْفَ", "fatha-sukun-fatha"),
    ("أين الباب", "أَيْنَ", "fatha-sukun-fatha"),
    ("متى جاء", "مَتَى", "fatha-fatha"),

    # Relative pronouns
    ("الذي جاء", "الَّذِي", "fatha+shadda on lam"),
    ("التي جاءت", "الَّتِي", "fatha+shadda on lam"),
]


def has_shadda_after_al(word):
    """Check if word has shadda on third letter (after ال)."""
    base = strip_harakat(word)
    if not base.startswith('ال') or len(base) < 3:
        return False

    # Find the third base letter and check for shadda
    base_count = 0
    for i, c in enumerate(word):
        if c not in HARAKAT_SET:
            base_count += 1
            if base_count == 3:  # Third base letter
                # Check following chars for shadda
                for j in range(i+1, min(i+4, len(word))):
                    if word[j] == 'ّ':
                        return True
                    if word[j] not in HARAKAT_SET:
                        break
                return False
    return False


def get_hamza_vowel(word):
    """Get vowel on initial hamza (إ or أ)."""
    if not word:
        return None

    for i, c in enumerate(word):
        if c in 'إأ':
            # Check following char for vowel
            for j in range(i+1, min(i+3, len(word))):
                if word[j] in 'ِ':  # kasra
                    return 'kasra'
                elif word[j] in 'َ':  # fatha
                    return 'fatha'
                elif word[j] in 'ُ':  # damma
                    return 'damma'
                elif word[j] not in HARAKAT_SET:
                    break
            return 'none'
    return None


def main():
    print("=" * 70)
    print("PARTICLE RULES ANALYSIS")
    print("=" * 70)
    print()

    errors = []

    for test_input, expected_pattern, description in TEST_PARTICLES:
        result = diacritize_with_sweep(test_input)
        first_word = result.split()[0]

        # Analyze
        base = strip_harakat(first_word)

        print(f"Input:    {test_input}")
        print(f"Output:   {result}")
        print(f"Expected: {expected_pattern} ({description})")

        # Check specific patterns
        if base.startswith('إل'):
            vowel = get_hamza_vowel(first_word)
            if vowel != 'kasra':
                print(f"  ERROR: Initial hamza should have kasra, got {vowel}")
                errors.append((test_input, expected_pattern, first_word, 'wrong hamza vowel'))
            else:
                print(f"  OK: Has kasra on hamza")

        if base.startswith('ال') and len(base) >= 3:
            sun_letters = set('تثدذرزسشصضطظلن')
            third = base[2]
            has_shadda = has_shadda_after_al(first_word)

            if third in sun_letters:
                if not has_shadda:
                    print(f"  ERROR: Sun letter '{third}' missing shadda")
                    errors.append((test_input, expected_pattern, first_word, 'missing sun shadda'))
                else:
                    print(f"  OK: Sun letter has shadda")
            else:
                if has_shadda:
                    print(f"  WARNING: Non-sun letter '{third}' has shadda")
                else:
                    print(f"  OK: Non-sun letter no shadda")

        print()

    print("=" * 70)
    print(f"SUMMARY: {len(errors)} errors found")
    print("=" * 70)

    for inp, exp, got, issue in errors:
        print(f"  {strip_harakat(inp.split()[0])}: {issue}")
        print(f"    Expected: {exp}")
        print(f"    Got:      {got}")


if __name__ == '__main__':
    main()
