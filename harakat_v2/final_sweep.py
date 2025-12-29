#!/usr/bin/env python3
"""
Final Sweep Corrector for Harakat V2

Post-processing corrections targeting the most common remaining errors:

1. أَنَّ vs أَنْ disambiguation (509 case errors)
   - أَنَّ (anna) → followed by noun (nominal sentence)
   - أَنْ (an) → followed by verb (subjunctive)

2. Similar patterns for إنَّ/إنْ, فإنَّ/فإنْ, etc.

These patterns account for ~500+ errors which could reduce DER by 0.1-0.15%
"""

import sys
import os
import re

if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except:
        pass

# Constants
HARAKAT = 'ًٌٍَُِّْ'
HARAKAT_SET = set(HARAKAT)
SHADDA = 'ّ'

# Arabic character sets for pattern detection
ARABIC_LETTERS = set('ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي')

# Common verb prefixes that indicate a following verb
VERB_PREFIXES = {'ي', 'ت', 'ن', 'أ'}  # Present tense prefixes

# Words that typically follow أَنَّ (nominal indicators) - CONSERVATIVE LIST
NOMINAL_STARTERS = {
    # Demonstratives (very high confidence)
    'ذلك', 'هذا', 'هذه', 'تلك', 'هؤلاء', 'ذاك', 'أولئك',
    # Common nominal sentence starters (high confidence)
    'كل', 'بعض', 'أهل', 'صاحب', 'غير',
    # Names/proper nouns (high confidence)
    'محمدا', 'أبا', 'أبو', 'ابن',
    # Indefinite nouns with tanween (high confidence - accusative objects)
    'رجلا', 'شيئا', 'أحدا', 'عبدا',
}

# Particles that strongly indicate أَنَّ (nominal context)
ANNA_CONTEXT_WORDS = {
    'اعلم', 'يعلم', 'علم', 'روي', 'ذكر', 'قال', 'نقل', 'ثبت',
    'ورد', 'جاء', 'بلغ', 'حدث', 'أخبر', 'روى', 'حكى',
    'الصحيح', 'الأصح', 'الأظهر', 'الراجح', 'المشهور', 'الصواب',
}

# Verbs that follow أَنْ (subjunctive context)
SUBJUNCTIVE_VERBS = {
    'يكون', 'تكون', 'نكون', 'أكون',
    'يقول', 'تقول', 'نقول', 'أقول',
    'يفعل', 'تفعل', 'نفعل', 'أفعل',
    'يجب', 'يجوز', 'يمكن', 'ينبغي',
    'يصح', 'يثبت', 'يلزم', 'يحل',
    'يأخذ', 'تأخذ', 'يدخل', 'تدخل',
    'يخرج', 'تخرج', 'يرجع', 'ترجع',
}


def strip_harakat(text):
    """Remove all diacritics."""
    return ''.join(c for c in text if c not in HARAKAT_SET)


def get_word_base(word):
    """Get undiacritized base of word."""
    return strip_harakat(word)


def has_shadda_on_nun(word):
    """Check if word has shadda on ن."""
    for i, c in enumerate(word):
        if c == 'ن':
            for j in range(i+1, len(word)):
                if word[j] == SHADDA:
                    return True
                if word[j] not in HARAKAT_SET:
                    break
    return False


def add_shadda_after_nun(word):
    """Add shadda after the ن in word."""
    result = []
    found_nun = False
    for c in word:
        result.append(c)
        if c == 'ن' and not found_nun:
            # Check if shadda already follows
            found_nun = True

    # Rebuild with shadda
    new_word = []
    for i, c in enumerate(word):
        new_word.append(c)
        if c == 'ن':
            # Check if next char is already shadda
            if i + 1 < len(word) and word[i+1] == SHADDA:
                continue
            # Add shadda after ن
            new_word.append(SHADDA)
            # Copy remaining
            new_word.extend(word[i+1:])
            break
    return ''.join(new_word)


def remove_shadda_from_nun(word):
    """Remove shadda after ن in word."""
    result = []
    skip_next_shadda = False
    for i, c in enumerate(word):
        if skip_next_shadda and c == SHADDA:
            skip_next_shadda = False
            continue
        result.append(c)
        if c == 'ن':
            skip_next_shadda = True
    return ''.join(result)


def is_verb_form(word_base):
    """Check if word looks like a verb based on pattern."""
    if not word_base:
        return False

    # Present tense verbs start with ي/ت/ن/أ
    if word_base[0] in VERB_PREFIXES and len(word_base) >= 3:
        return True

    # Check against known subjunctive verbs
    if word_base in SUBJUNCTIVE_VERBS:
        return True

    return False


def is_nominal_starter(word_base):
    """Check if word is a nominal sentence starter."""
    if not word_base:
        return False

    # Direct match
    if word_base in NOMINAL_STARTERS:
        return True

    # Words starting with ال (definite article) are typically nouns
    if word_base.startswith('ال') and len(word_base) > 3:
        return True

    return False


def should_have_shadda(target_base, next_word_base, prev_word_base=None):
    """
    Determine if أن/إن/etc should have shadda based on context.

    Returns:
        True: should have shadda (أَنَّ)
        False: should not have shadda (أَنْ)
        None: uncertain, keep original
    """
    if not next_word_base:
        return None

    # Strong signals for أَنَّ (nominal sentence follows)
    if is_nominal_starter(next_word_base):
        return True

    # Strong signals for أَنْ (verb follows in subjunctive)
    if is_verb_form(next_word_base):
        return False

    # Check for "لا" which often precedes subjunctive
    if next_word_base == 'لا':
        return False  # أَنْ لَا يَفْعَلَ

    # Context from previous word
    if prev_word_base:
        if prev_word_base in ANNA_CONTEXT_WORDS:
            return True

    # Default: uncertain
    return None


def apply_final_sweep(text):
    """
    Apply final sweep corrections to diacritized text.

    Targets:
    - أن/إن/فإن/بأن/وإن shadda disambiguation
    """
    words = text.split()
    if len(words) < 2:
        return text

    # Target patterns (base forms)
    target_bases = {'أن', 'إن', 'فإن', 'بأن', 'لأن', 'كأن', 'وأن', 'وإن'}

    result = []
    corrections = 0

    for i, word in enumerate(words):
        word_base = get_word_base(word)

        # Check if this is a target word
        if word_base in target_bases:
            # Get context
            next_base = get_word_base(words[i+1]) if i + 1 < len(words) else None
            prev_base = get_word_base(words[i-1]) if i > 0 else None

            # Determine if should have shadda
            should_shadda = should_have_shadda(word_base, next_base, prev_base)

            if should_shadda is not None:
                current_has_shadda = has_shadda_on_nun(word)

                if should_shadda and not current_has_shadda:
                    # Add shadda
                    word = add_shadda_after_nun(word)
                    corrections += 1
                elif not should_shadda and current_has_shadda:
                    # Remove shadda
                    word = remove_shadda_from_nun(word)
                    corrections += 1

        result.append(word)

    return ' '.join(result)


def test_corrections():
    """Test the correction logic."""

    # Test that shadda is added correctly
    print("Testing final sweep corrections:\n")

    # Test 1: Add shadda when nominal follows
    input1 = "أَنَ ذَلِكَ"  # أ + فتحة + ن + فتحة + space + ...
    result1 = apply_final_sweep(input1)
    has_shadda1 = has_shadda_on_nun(result1.split()[0])
    print(f"Test 1 - Add shadda (nominal follows):")
    print(f"  Input:  {input1}")
    print(f"  Output: {result1}")
    print(f"  Has shadda on nun: {has_shadda1}")
    print(f"  PASS" if has_shadda1 else f"  FAIL")
    print()

    # Test 2: Remove shadda when verb follows
    input2 = "أَنّ يَكُونَ"  # with shadda
    result2 = apply_final_sweep(input2)
    has_shadda2 = has_shadda_on_nun(result2.split()[0])
    print(f"Test 2 - Remove shadda (verb follows):")
    print(f"  Input:  {input2}")
    print(f"  Output: {result2}")
    print(f"  Has shadda on nun: {has_shadda2}")
    print(f"  PASS" if not has_shadda2 else f"  FAIL")
    print()

    # Test 3: Keep shadda (already correct, nominal)
    input3 = "أَنَّ رَجُلًا"
    result3 = apply_final_sweep(input3)
    has_shadda3 = has_shadda_on_nun(result3.split()[0])
    print(f"Test 3 - Keep shadda (nominal, already correct):")
    print(f"  Input:  {input3}")
    print(f"  Output: {result3}")
    print(f"  Has shadda on nun: {has_shadda3}")
    print(f"  PASS" if has_shadda3 else f"  FAIL")
    print()

    # Test 4: Keep no shadda (already correct, verb)
    input4 = "أَنْ يَقُولَ"
    result4 = apply_final_sweep(input4)
    has_shadda4 = has_shadda_on_nun(result4.split()[0])
    print(f"Test 4 - Keep no shadda (verb, already correct):")
    print(f"  Input:  {input4}")
    print(f"  Output: {result4}")
    print(f"  Has shadda on nun: {has_shadda4}")
    print(f"  PASS" if not has_shadda4 else f"  FAIL")
    print()

    # Test إن variants
    input5 = "إنَ اللَّهَ"
    result5 = apply_final_sweep(input5)
    has_shadda5 = has_shadda_on_nun(result5.split()[0])
    print(f"Test 5 - Add shadda to إن (nominal follows):")
    print(f"  Input:  {input5}")
    print(f"  Output: {result5}")
    print(f"  Has shadda on nun: {has_shadda5}")
    print(f"  PASS" if has_shadda5 else f"  FAIL")
    print()

    # Test فإن
    input6 = "فَإِنَ ذَلِكَ"
    result6 = apply_final_sweep(input6)
    has_shadda6 = has_shadda_on_nun(result6.split()[0])
    print(f"Test 6 - Add shadda to فإن (nominal follows):")
    print(f"  Input:  {input6}")
    print(f"  Output: {result6}")
    print(f"  Has shadda on nun: {has_shadda6}")
    print(f"  PASS" if has_shadda6 else f"  FAIL")
    print()

    # Test بأن
    input7 = "بِأَنَ الْأَمْرَ"
    result7 = apply_final_sweep(input7)
    has_shadda7 = has_shadda_on_nun(result7.split()[0])
    print(f"Test 7 - Add shadda to بأن (nominal follows):")
    print(f"  Input:  {input7}")
    print(f"  Output: {result7}")
    print(f"  Has shadda on nun: {has_shadda7}")
    print(f"  PASS" if has_shadda7 else f"  FAIL")
    print()


if __name__ == '__main__':
    test_corrections()
