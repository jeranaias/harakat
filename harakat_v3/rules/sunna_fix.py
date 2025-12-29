#!/usr/bin/env python3
"""
سُنَّة/سَنَة Disambiguation

سُنَّة (sunna) = tradition/practice - used in religious contexts
سَنَة (sana) = year - used in temporal contexts

Key indicators for سُنَّة (tradition):
- Followed by: مُؤَكَّدَة، رَاتِبَة
- Context: religious practice, sunnah of prophet

Key indicators for سَنَة (year):
- Followed by: numbers, الحريق (year of fire), etc.
- Context: temporal references
"""

import sys
import os
import io

if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')

# Constants
HARAKAT = 'ًٌٍَُِّْ'
HARAKAT_SET = set(HARAKAT)
SHADDA = 'ّ'
FATHA = 'َ'
DAMMA = 'ُ'


def strip_harakat(text):
    """Remove diacritics."""
    return ''.join(c for c in text if c not in HARAKAT_SET)


def is_year_form(word):
    """Check if word has year form (سَنَة)."""
    # سَنَة has fatha on س and no shadda
    base = strip_harakat(word)
    if base not in {'سنة', 'السنة', 'بالسنة', 'وسنة', 'فسنة', 'لسنة'}:
        return False
    # Check for shadda - if present, it's sunna
    return SHADDA not in word


def is_sunna_form(word):
    """Check if word has sunna form (سُنَّة)."""
    base = strip_harakat(word)
    if base not in {'سنة', 'السنة', 'بالسنة', 'وسنة', 'فسنة', 'لسنة'}:
        return False
    # Check for shadda and damma
    return SHADDA in word


def convert_to_sunna(word):
    """Convert سَنَة to سُنَّة form."""
    base = strip_harakat(word)

    # Handle different prefixes
    if base.startswith('ال'):
        core = 'سنة'
        prefix = word[:word.find('س')]
    elif base.startswith('بال'):
        core = 'سنة'
        prefix = word[:word.find('س')]
    elif base.startswith('ب') or base.startswith('و') or base.startswith('ف') or base.startswith('ل'):
        core = 'سنة'
        prefix = word[0]
    else:
        core = 'سنة'
        prefix = ''

    # Get the suffix/case ending
    suffix_start = word.rfind('ة')
    if suffix_start == -1:
        return word  # No ة found

    suffix = word[suffix_start:]  # ة with any following harakat

    # Build سُنَّة form
    return prefix + 'سُنَّ' + suffix


def should_be_sunna(word_base, next_word, prev_word=None):
    """
    Determine if سنة should be سُنَّة (tradition).

    Returns True if should be sunna, False if year, None if uncertain.
    """
    if next_word:
        next_base = strip_harakat(next_word)

        # Strong indicators for سُنَّة (sunna/tradition)
        sunna_indicators = {
            'مؤكدة', 'مؤكدا', 'راتبة', 'راتبا', 'النبي',
            'الرسول', 'الأكل', 'الشرب', 'الصلاة',
        }

        if next_base in sunna_indicators:
            return True

        # Followed by adjective describing sunna
        if next_base.startswith('مؤكد') or next_base.startswith('راتب'):
            return True

    if prev_word:
        prev_base = strip_harakat(prev_word)

        # Preceded by words indicating religious context
        sunna_prev = {'فوت', 'خشي', 'خشية'}
        if prev_base in sunna_prev:
            return True

    return None


def apply_sunna_fix(text):
    """
    Apply سُنَّة/سَنَة disambiguation to diacritized text.
    """
    words = text.split()
    if len(words) < 1:
        return text

    result = []

    for i, word in enumerate(words):
        word_base = strip_harakat(word)

        # Check if this is a سنة word in year form
        if word_base in {'سنة', 'السنة', 'بالسنة', 'وسنة', 'فسنة', 'لسنة'}:
            if is_year_form(word):
                next_word = words[i+1] if i + 1 < len(words) else None
                prev_word = words[i-1] if i > 0 else None

                should_sunna = should_be_sunna(word_base, next_word, prev_word)

                if should_sunna is True:
                    word = convert_to_sunna(word)

        result.append(word)

    return ' '.join(result)


def test_sunna_fix():
    """Test sunna fix."""
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except:
        pass

    print("=" * 60)
    print("SUNNA FIX TEST")
    print("=" * 60)

    test_cases = [
        ("يَكُونُ سَنَةً مُؤَكَّدَةً", "سنة مؤكدة → sunna"),
        ("خَشِيَ فَوْتَ سَنَةٍ رَاتِبَةٍ", "فوت سنة راتبة → sunna"),
        ("وَالْخُشُوعُ سَنَةٌ", "الخشوع سنة → keep (uncertain)"),
        ("سَنَةَ الْحَرِيقِ", "سنة الحريق → year (keep)"),
    ]

    for text, description in test_cases:
        result = apply_sunna_fix(text)
        changed = text != result
        print(f"\n{description}:")
        print(f"  Input:  {text}")
        print(f"  Output: {result}")
        print(f"  Changed: {changed}")


if __name__ == '__main__':
    test_sunna_fix()
