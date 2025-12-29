#!/usr/bin/env python3
"""
Particle Rules for V3

Fixed diacritization patterns for common Arabic particles.
These have consistent, predictable diacritization regardless of context.

Key fix: Adding kasra (ِ) to hamza-below (إ) which the V2 model often misses.
"""

import sys
import os

if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')

# Constants
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
    """Remove all diacritics."""
    return ''.join(c for c in text if c not in HARAKAT_SET)


# ============================================================
# PARTICLE PATTERNS
# ============================================================
# These particles have fixed internal voweling.
# The key issue is missing kasra on hamza-below (إ).
#
# Pattern format: base -> list of (correct_internal, preserve_case_ending)
# If preserve_case_ending is True, we only fix internal vowels.

# Particles with hamza-below that need kasra
HAMZA_BELOW_PARTICLES = {
    # word_base: internal_pattern (excluding final case diacritics)
    'إلى': 'إِلَ',      # إِلَى - kasra on hamza, fatha on lam
    'إلا': 'إِلَّ',     # إِلَّا - kasra on hamza, fatha+shadda on lam
    'إذا': 'إِذَ',      # إِذَا - kasra on hamza, fatha on dhal
    'إذ': 'إِذْ',       # إِذْ - kasra on hamza, sukun on dhal
    'إن': 'إِنْ',       # إِنْ or إِنَّ - context dependent (handled by final_sweep)
}


def fix_hamza_below_kasra(word, word_base):
    """
    Fix missing kasra on hamza-below (إ) for known particles.

    The V2 model often outputs إلَى instead of إِلَى.
    This function adds the missing kasra.

    Args:
        word: Diacritized word
        word_base: Undiacritized base form

    Returns:
        Fixed word, or original if no fix needed
    """
    if word_base not in HAMZA_BELOW_PARTICLES:
        return word

    # Find hamza-below (إ) in the word
    hamza_pos = None
    for i, c in enumerate(word):
        if c == 'إ':
            hamza_pos = i
            break

    if hamza_pos is None:
        return word

    # Check if kasra already follows the hamza
    has_kasra = False
    for j in range(hamza_pos + 1, min(hamza_pos + 3, len(word))):
        if word[j] == KASRA:
            has_kasra = True
            break
        if word[j] not in HARAKAT_SET:
            break

    if has_kasra:
        return word  # Already correct

    # Insert kasra after hamza
    # First, remove any existing vowel after hamza
    new_word = []
    skip_vowel = False
    for i, c in enumerate(word):
        if i == hamza_pos:
            new_word.append(c)
            new_word.append(KASRA)  # Add kasra
            skip_vowel = True
        elif skip_vowel and c in {FATHA, DAMMA, KASRA}:
            skip_vowel = False
            # Skip this vowel (we already added kasra)
            continue
        else:
            skip_vowel = False
            new_word.append(c)

    return ''.join(new_word)


def apply_particle_rules(text):
    """
    Apply particle rules to diacritized text.

    Currently handles:
    1. Missing kasra on hamza-below (إ) particles

    Args:
        text: Diacritized text

    Returns:
        Text with particle corrections applied
    """
    words = text.split()
    result = []
    corrections = 0

    for word in words:
        word_base = strip_harakat(word)

        # Fix hamza-below kasra
        if word_base in HAMZA_BELOW_PARTICLES:
            fixed = fix_hamza_below_kasra(word, word_base)
            if fixed != word:
                corrections += 1
            result.append(fixed)
        else:
            result.append(word)

    return ' '.join(result)


# ============================================================
# TESTING
# ============================================================

def test_particle_rules():
    """Test particle rules with specific examples."""
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 60)
    print("PARTICLE RULES TEST")
    print("=" * 60)

    test_cases = [
        # (input, expected, description)
        ("إلَى الْمَدْرَسَةِ", "إِلَى الْمَدْرَسَةِ", "إلى: add kasra"),
        ("إلَّا الْحَقَّ", "إِلَّا الْحَقَّ", "إلا: add kasra"),
        ("إذَا جَاءَ", "إِذَا جَاءَ", "إذا: add kasra"),
        ("إذْ قَالَ", "إِذْ قَالَ", "إذ: add kasra"),
        ("إِلَى الْبَيْتِ", "إِلَى الْبَيْتِ", "إلى: already correct"),
        ("كَتَبَ الْوَلَدُ", "كَتَبَ الْوَلَدُ", "non-particle: unchanged"),
    ]

    all_passed = True
    for inp, expected, desc in test_cases:
        result = apply_particle_rules(inp)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_passed = False

        print(f"\n{desc}")
        print(f"  Input:    {inp}")
        print(f"  Expected: {expected}")
        print(f"  Got:      {result}")
        print(f"  Status:   {status}")

    print("\n" + "=" * 60)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    test_particle_rules()
