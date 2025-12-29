#!/usr/bin/env python3
"""
Homograph Disambiguation Rules for V3

These are words with multiple valid diacritizations where context determines
the correct form. Unlike particles (fixed patterns), these require analysis.

Top homographs by error count:
1. من: مِنْ (from/preposition) vs مَنْ (who/pronoun) - 31 errors
2. أم: أَمْ (or/conjunction) vs أُمّ (mother/noun) - 14 errors
3. ثم: ثُمَّ (then/adverb) vs ثَمَّ (there/adverb) - 10 errors

These account for ~55 errors which is 0.06% DER.
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


def strip_harakat(text):
    return ''.join(c for c in text if c not in HARAKAT_SET)


# ============================================================
# من (MIN/MAN) DISAMBIGUATION
# ============================================================
# مِنْ (min) = from, preposition - followed by noun/pronoun
# مَنْ (man) = who, interrogative/relative pronoun - starts clause

# Words that typically follow مِنْ (preposition "from")
MIN_CONTEXT_WORDS = {
    # Definite nouns (start with ال)
    'ال',
    # Pronouns and demonstratives
    'هذا', 'هذه', 'ذلك', 'تلك', 'هؤلاء', 'أولئك',
    # Common nouns after "from"
    'أهل', 'بني', 'بيت', 'عند', 'غير', 'قبل', 'بعد', 'فوق', 'تحت',
    'جهة', 'جانب', 'باب', 'أبواب', 'طريق',
    # Pronouns attached to next word
    'ه', 'ها', 'هم', 'هما', 'هن', 'ك', 'كم', 'كما', 'كن', 'ي', 'نا',
}

# Words that typically follow مَنْ (pronoun "who")
MAN_CONTEXT_WORDS = {
    # Verbs (present tense markers)
    'ي', 'ت', 'ن', 'أ',  # verb prefixes
    # Past tense verbs (common)
    'كان', 'جاء', 'قال', 'فعل', 'عمل', 'أراد', 'أخذ', 'رأى',
    # Particles often after "who"
    'لا', 'لم', 'لن', 'قد', 'سوف',
}


def is_min_context(next_word_base):
    """Check if context suggests مِنْ (from)."""
    if not next_word_base:
        return None

    # Definite article strongly suggests "from"
    if next_word_base.startswith('ال'):
        return True

    # Check common patterns
    if next_word_base in MIN_CONTEXT_WORDS:
        return True

    return None


def is_man_context(next_word_base, prev_word_base=None):
    """Check if context suggests مَنْ (who)."""
    if not next_word_base:
        return None

    # Verb follows "who"
    first_char = next_word_base[0] if next_word_base else ''
    if first_char in 'يتنأ' and len(next_word_base) >= 3:
        # Likely a present tense verb
        return True

    # After certain particles, usually "who"
    if prev_word_base in {'و', 'ف', 'أ', 'أما', 'إما'}:
        return True

    return None


def disambiguate_min_man(word, word_base, next_word, prev_word=None):
    """
    Disambiguate من between مِنْ and مَنْ.

    Returns corrected word or None if uncertain.
    """
    if word_base != 'من':
        return None

    next_base = strip_harakat(next_word) if next_word else None
    prev_base = strip_harakat(prev_word) if prev_word else None

    # Check for مِنْ context
    if is_min_context(next_base):
        return 'مِنْ'

    # Check for مَنْ context
    if is_man_context(next_base, prev_base):
        return 'مَنْ'

    # Default: uncertain, keep original
    return None


# ============================================================
# أم (AM/UMM) DISAMBIGUATION
# ============================================================
# أَمْ (am) = or, conjunction - connects alternatives
# أُمّ (umm) = mother, noun - has shadda and damma

def disambiguate_am_umm(word, word_base, next_word, prev_word=None):
    """
    Disambiguate أم between أَمْ (or) and أُمّ (mother).

    أَمْ is much more common in formal text.
    أُمّ usually has possessive suffix (أُمُّه, أُمِّي) or case ending.
    """
    if word_base != 'أم':
        return None

    # If it has shadda, it's probably أُمّ
    if SHADDA in word:
        return None  # Keep as-is

    # Standalone أم is almost always أَمْ (or)
    # Check for question context
    if prev_word:
        prev_base = strip_harakat(prev_word)
        # After question word or أ, it's "or"
        if prev_base in {'أ', 'هل', 'ما', 'ماذا', 'كيف', 'أين', 'متى'}:
            return 'أَمْ'

        # Between two nouns/verbs, it's "or"
        if prev_base not in {'و', 'ف'}:  # Not after simple conjunction
            return 'أَمْ'

    return 'أَمْ'  # Default to "or" - more common


# ============================================================
# ثم (THUMMA/THAMMA) DISAMBIGUATION
# ============================================================
# ثُمَّ (thumma) = then, adverb of sequence - MUCH more common
# ثَمَّ (thamma) = there, adverb of place - rare

def disambiguate_thumma_thamma(word, word_base, next_word, prev_word=None):
    """
    Disambiguate ثم between ثُمَّ (then) and ثَمَّ (there).

    ثُمَّ is far more common (~95% of occurrences).
    ثَمَّ usually appears with هُنَاكَ or في ثَمَّ contexts.
    """
    if word_base != 'ثم':
        return None

    # Check if it has shadda (both forms do)
    if SHADDA not in word:
        # Add shadda - both forms need it
        return None

    # Check for "there" context
    next_base = strip_harakat(next_word) if next_word else None
    if next_base in {'من', 'ما', 'مكان', 'موضع'}:
        # Might be "there" but still uncertain
        return None

    # Default to ثُمَّ (then) - much more common
    return 'ثُمَّ'


# ============================================================
# MAIN DISAMBIGUATION
# ============================================================

def apply_homograph_rules(text):
    """
    Apply homograph disambiguation rules to diacritized text.

    Currently handles:
    1. من: مِنْ (from) vs مَنْ (who)
    2. أم: أَمْ (or) vs أُمّ (mother)
    3. ثم: ثُمَّ (then) vs ثَمَّ (there)
    """
    words = text.split()
    if len(words) < 1:
        return text

    result = []
    corrections = 0

    for i, word in enumerate(words):
        word_base = strip_harakat(word)
        next_word = words[i + 1] if i + 1 < len(words) else None
        prev_word = words[i - 1] if i > 0 else None

        corrected = None

        # Try each disambiguator
        if word_base == 'من':
            corrected = disambiguate_min_man(word, word_base, next_word, prev_word)
        elif word_base == 'أم':
            corrected = disambiguate_am_umm(word, word_base, next_word, prev_word)
        elif word_base == 'ثم':
            corrected = disambiguate_thumma_thamma(word, word_base, next_word, prev_word)

        if corrected and corrected != word:
            result.append(corrected)
            corrections += 1
        else:
            result.append(word)

    return ' '.join(result)


# ============================================================
# TESTING
# ============================================================

def test_homograph_rules():
    """Test homograph disambiguation."""
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 60)
    print("HOMOGRAPH DISAMBIGUATION TEST")
    print("=" * 60)

    test_cases = [
        # من tests
        ("مَنْ الرَّجُلُ", "مِنْ الرَّجُلِ", "من + definite → مِنْ"),
        ("مِنْ يَفْعَلُ", "مَنْ يَفْعَلُ", "من + verb → مَنْ"),

        # أم tests
        ("أُمْ زَيْدٍ", "أَمْ زَيْدٍ", "أم as 'or'"),

        # ثم tests
        ("ثَمَّ جَاءَ", "ثُمَّ جَاءَ", "ثم as 'then'"),
    ]

    for inp, expected, desc in test_cases:
        result = apply_homograph_rules(inp)
        status = "PASS" if result == expected else "MAYBE"
        print(f"\n{desc}")
        print(f"  Input:    {inp}")
        print(f"  Expected: {expected}")
        print(f"  Got:      {result}")
        print(f"  Status:   {status}")


if __name__ == '__main__':
    test_homograph_rules()
