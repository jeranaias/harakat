#!/usr/bin/env python3
"""
Case Ending Rules for V3.5

Strategy 2: Grammar-based case ending corrections.

Arabic case endings (i'rab):
- Nominative (مرفوع): damma ُ or dammatan ٌ
- Accusative (منصوب): fatha َ or fathatan ً
- Genitive (مجرور): kasra ِ or kasratan ٍ

Key rules:
1. After prepositions → genitive (kasra)
2. Subject position → nominative (damma)
3. Object position → accusative (fatha)

Top case errors from analysis:
- غير: 14 errors (غَيْرَ vs غَيْرِ)
- بن: 8 errors (بْنُ vs بْنِ)
- علم: case errors
"""

import sys
import os
import re

if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')

# Constants
HARAKAT = 'ًٌٍَُِّْ'
HARAKAT_SET = set(HARAKAT)
FATHA = 'َ'
KASRA = 'ِ'
DAMMA = 'ُ'
SUKUN = 'ْ'
TANWEEN_FATH = 'ً'
TANWEEN_KASR = 'ٍ'
TANWEEN_DAMM = 'ٌ'


def strip_harakat(text):
    """Remove diacritics."""
    return ''.join(c for c in text if c not in HARAKAT_SET)


def extract_case_ending(word):
    """
    Extract the case ending (final vowel) from a word.

    Returns: (case_vowel, is_tanween)
    """
    if not word:
        return None, False

    # Look for the final vowel (after last consonant)
    for i in range(len(word) - 1, -1, -1):
        c = word[i]
        if c in {FATHA, KASRA, DAMMA}:
            return c, False
        if c in {TANWEEN_FATH, TANWEEN_KASR, TANWEEN_DAMM}:
            return c, True
        if c not in HARAKAT_SET:
            # Hit a consonant before finding vowel
            break

    return None, False


def replace_case_ending(word, new_case, use_tanween=False):
    """
    Replace the case ending of a word.

    Args:
        word: Diacritized word
        new_case: New case vowel (FATHA, KASRA, or DAMMA)
        use_tanween: If True, use tanween version

    Returns:
        Word with new case ending
    """
    if not word:
        return word

    # Map vowels to tanween
    tanween_map = {
        FATHA: TANWEEN_FATH,
        KASRA: TANWEEN_KASR,
        DAMMA: TANWEEN_DAMM,
    }

    target = tanween_map[new_case] if use_tanween else new_case

    # First, find the position of the last consonant
    last_consonant_idx = -1
    for i in range(len(word) - 1, -1, -1):
        if word[i] not in HARAKAT_SET:
            last_consonant_idx = i
            break

    if last_consonant_idx == -1:
        return word

    # Now look for case vowels AFTER the last consonant
    result = list(word)
    for i in range(last_consonant_idx + 1, len(result)):
        c = result[i]
        if c in {FATHA, KASRA, DAMMA, TANWEEN_FATH, TANWEEN_KASR, TANWEEN_DAMM}:
            # This is the case ending - replace it
            result[i] = target
            return ''.join(result)

    # If no case ending found after last consonant, append it
    result.insert(last_consonant_idx + 1, target)
    return ''.join(result)


# Prepositions that trigger genitive case on the following word
PREPOSITIONS = {
    'من', 'إلى', 'على', 'في', 'عن', 'ب', 'ل', 'ك',
    'منذ', 'مذ', 'رب', 'حتى', 'خلا', 'عدا', 'حاشا',
    'بين', 'فوق', 'تحت', 'أمام', 'خلف', 'قبل', 'بعد',
    'عند', 'لدى', 'مع', 'نحو', 'دون', 'غير', 'سوى',
}

# Prepositions with attached particles
ATTACHED_PREPS = {
    'بال', 'لل', 'كال', 'فال', 'وال',  # ب/ل/ك/ف/و + ال
    'بما', 'لما', 'مما', 'عما', 'فيما', 'إلى', 'على',
    'منها', 'منه', 'منهم', 'عليه', 'عليها', 'عليهم',
    'فيه', 'فيها', 'فيهم', 'بها', 'به', 'بهم',
    'لها', 'له', 'لهم', 'إليه', 'إليها', 'إليهم',
}

# Words that commonly follow prepositions and need genitive
GENITIVE_TARGETS = {
    'غير': True,   # غَيْرِ after preposition
    'بن': True,    # بْنِ (son of)
    'ابن': True,   # ابْنِ
    'أهل': True,   # أَهْلِ
    'ذي': True,    # ذِي (possessor)
    'كل': True,    # كُلِّ
    'بعض': True,   # بَعْضِ
    'جميع': True,  # جَمِيعِ
    'ذلك': True,   # ذَلِكَ (unchanged but common)
}


def is_preposition(word):
    """Check if word is a preposition (jarr)."""
    word_base = strip_harakat(word)

    # Direct match
    if word_base in PREPOSITIONS:
        return True

    # Attached preposition check
    if word_base in ATTACHED_PREPS:
        return True

    # Check for ب/ل/ك prefix on definite nouns
    if len(word_base) > 3:
        if word_base.startswith('بال') or word_base.startswith('لل') or word_base.startswith('كال'):
            return False  # These are complete words, not prepositions

    # Single letter prepositions attached to words
    if len(word_base) > 2:
        first = word_base[0]
        if first in 'بلك' and word_base[1:].startswith('ال'):
            # ب/ل/ك + الـ pattern - the whole thing is governed
            return False

    return False


def needs_genitive(word, prev_word):
    """
    Check if word should be in genitive case based on preceding word.

    Returns: True if word should have kasra, False otherwise
    """
    if not prev_word:
        return False

    word_base = strip_harakat(word)
    prev_base = strip_harakat(prev_word)

    # Check if previous word is a preposition
    if prev_base in PREPOSITIONS:
        return True

    # Check for attached preposition patterns
    # If previous word ends with preposition attached to something
    # e.g., "مِنْ" before غير
    if prev_base in {'من', 'عن', 'في', 'على', 'إلى'}:
        return True

    return False


def apply_case_rules(text):
    """
    Apply case ending rules to diacritized text.

    Currently focused on genitive after prepositions.
    """
    words = text.split()
    if len(words) < 2:
        return text

    result = []
    corrections = 0

    for i, word in enumerate(words):
        word_base = strip_harakat(word)

        # Check if this word should be genitive
        if i > 0:
            prev_word = words[i - 1]
            prev_base = strip_harakat(prev_word)

            # Rule 1: After specific prepositions, targets get kasra
            if prev_base in PREPOSITIONS and word_base in GENITIVE_TARGETS:
                current_case, is_tanween = extract_case_ending(word)

                if current_case and current_case not in {KASRA, TANWEEN_KASR}:
                    # Change to genitive
                    new_word = replace_case_ending(word, KASRA, is_tanween)
                    if new_word != word:
                        result.append(new_word)
                        corrections += 1
                        continue

            # Rule 2: بْنِ pattern (son of) - always genitive after name
            if word_base == 'بن' or word_base == 'ابن':
                current_case, _ = extract_case_ending(word)
                if current_case == DAMMA:
                    # Change to kasra
                    new_word = replace_case_ending(word, KASRA, False)
                    if new_word != word:
                        result.append(new_word)
                        corrections += 1
                        continue

        result.append(word)

    return ' '.join(result)


def test_case_rules():
    """Test case ending rules."""
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except:
        pass

    print("=" * 60)
    print("CASE ENDING RULES TEST")
    print("=" * 60)

    test_cases = [
        # غير after preposition
        ("مِنْ غَيْرَ أَنْ", "من غير → غَيْرِ"),
        ("عَلَى غَيْرَ ذَلِكَ", "على غير → غَيْرِ"),
        ("فِي غَيْرَ مَكَانِهِ", "في غير → غَيْرِ"),

        # بن pattern
        ("مُحَمَّدُ بْنُ عَلِيٍّ", "بن → بْنِ"),
        ("عَبْدُ اللَّهِ بْنُ عُمَرَ", "بن → بْنِ"),

        # Should not change
        ("هَذَا غَيْرُ ذَلِكَ", "غير as subject - keep damma"),
    ]

    for text, description in test_cases:
        result = apply_case_rules(text)
        changed = text != result
        print(f"\n{description}")
        print(f"  Input:  {text}")
        print(f"  Output: {result}")
        print(f"  Changed: {'Yes' if changed else 'No'}")


if __name__ == '__main__':
    test_case_rules()
