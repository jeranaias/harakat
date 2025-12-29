#!/usr/bin/env python3
"""
Enhanced أَنَّ/أَنْ Disambiguation for V3

The V2 final_sweep handles basic cases but misses:
1. Nouns with accusative ending (ـَـهُ suffix)
2. Nouns with tanween fatha (ـًا)
3. Masdar (verbal noun) patterns

This module adds higher-recall rules for أَنَّ detection.
"""

import sys
import os
import re

if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')

# Constants
HARAKAT = 'ًٌٍَُِّْ'
HARAKAT_SET = set(HARAKAT)
SHADDA = 'ّ'
FATHA = 'َ'
DAMMA = 'ُ'
KASRA = 'ِ'
TANWEEN_FATH = 'ً'


def strip_harakat(text):
    """Remove diacritics."""
    return ''.join(c for c in text if c not in HARAKAT_SET)


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
    """Add shadda after the first ن in word."""
    result = []
    added = False
    i = 0
    while i < len(word):
        result.append(word[i])
        if word[i] == 'ن' and not added:
            # Check if shadda already follows
            if i + 1 < len(word) and word[i+1] == SHADDA:
                i += 1
                continue
            # Add shadda
            result.append(SHADDA)
            added = True
        i += 1
    return ''.join(result)


def is_accusative_noun(word, word_base):
    """
    Check if word appears to be a noun in accusative case.

    Accusative indicators:
    - Ends with ـَـهُ (fatha + ha + damma) - possessive suffix
    - Ends with ـًا (tanween fatha + alif)
    - Ends with ـَ (fatha) on non-verb pattern
    - فاعِل pattern nouns
    """
    if len(word_base) < 3:
        return False

    # Check for possessive suffix ending (like سَبَبَهُ)
    if word_base.endswith('ه') or word_base.endswith('ها') or word_base.endswith('هم') or word_base.endswith('هما'):
        # Check for fatha before the suffix
        # This indicates accusative or genitive possessive
        return True

    # Check for tanween fatha (accusative indefinite)
    if TANWEEN_FATH in word:
        return True

    # Check for accusative definite (fatha on last consonant before ه)
    # Pattern: consonant + fatha + هُ
    for i in range(len(word) - 2, 0, -1):
        if word[i] == FATHA and i + 1 < len(word) and word[i+1] == 'ه':
            return True
        if word[i] not in HARAKAT_SET:
            break

    # Check for فَاعِل pattern (active participle) - ends with fatha or kasra
    # Pattern: consonant + fatha + (final letter)
    if len(word) >= 3:
        # Check if word ends with fatha (accusative marker)
        for i in range(len(word) - 1, -1, -1):
            if word[i] == FATHA:
                return True
            if word[i] not in HARAKAT_SET:
                break

    return False


def is_masdar_pattern(word_base):
    """
    Check if word follows masdar (verbal noun) patterns.

    Common masdar patterns:
    - فَعْل (fa'l) - 3 letters with sukun
    - تَفْعِيل (taf'eel) - Form II masdar
    - إِفْعَال (if'aal) - Form IV masdar
    - اشْتِرَاط (ishtiraat) - Form VIII masdar
    - etc.
    """
    if len(word_base) < 4:
        return False

    # Form II masdar: تَفْعِيل (starts with ت, ends with يم/يل/ية/ير/يع/ين)
    # Distinguish from present tense verbs
    if word_base.startswith('ت') and len(word_base) >= 5:
        # Common Form II masdar endings
        masdar_endings = ('يم', 'يل', 'ية', 'ير', 'يع', 'ين', 'يب', 'يف', 'يد', 'يز', 'يس', 'يص', 'يق', 'يك')
        for ending in masdar_endings:
            if word_base.endswith(ending):
                return True

    # Form IV masdar: إِفْعَال (starts with إ or أ, ends with ال or اء)
    # More specific check to avoid matching pronouns like أم
    if word_base.startswith('إ') or word_base.startswith('أ'):
        if len(word_base) >= 5:
            # Must have typical masdar ending
            if word_base.endswith('ال') or word_base.endswith('اء') or word_base.endswith('ان'):
                return True

    # Form VIII masdar: افْتِعَال (starts with ا, has ت, ends with ال/اء/اط etc.)
    if word_base.startswith('ا') and 'ت' in word_base and len(word_base) >= 6:
        # Form VIII typically ends in اط, ال, اء, ار
        if word_base.endswith('اط') or word_base.endswith('ال') or word_base.endswith('اء') or word_base.endswith('ار'):
            return True

    return False


def is_present_tense_verb(word_base):
    """Check if word is likely a present tense verb (not masdar)."""
    if not word_base:
        return False

    # First check if it's a masdar pattern - masdars are NOT verbs
    if is_masdar_pattern(word_base):
        return False

    # Present tense verbs start with ي/ت/ن/أ
    # But masdars starting with ت often end in يم/يل/ية (already excluded above)
    if word_base[0] in 'يتنأ' and len(word_base) >= 3:
        # Additional check: typical verb endings vs masdar endings
        # Verbs often end with: ون، ين، ان، وا, etc. or just consonant
        # Masdars often end with: يم، يل، ية، اء، ان
        return True
    return False


def is_past_tense_verb(word, word_base):
    """
    Check if word is likely a past tense verb.

    Past tense verbs are indicated by:
    - فَعَلَ pattern (fatha on most letters)
    - Common past tense verb roots
    """
    if not word_base or len(word_base) < 3:
        return False

    # Very common past tense verbs (3-letter roots that are unambiguous)
    common_past_verbs = {
        'كان', 'قال', 'فعل', 'جاء', 'ذهب', 'خرج', 'دخل', 'رجع', 'مات',
        'أذن', 'علم', 'عمل', 'جعل', 'قصد', 'رأى', 'وجد', 'أخذ', 'قدم',
        'شاء', 'زاد', 'نزل', 'طلب', 'ترك', 'مضى', 'بدأ', 'حصل', 'صار',
        'وقع', 'بلغ', 'نال', 'ظهر', 'سمع', 'جلس', 'قتل', 'حكم', 'كتب',
        'نظر', 'أمر', 'قرأ', 'وصل', 'فتح', 'غفر', 'صنع', 'بعث', 'منع',
        'أتى', 'بقي', 'مضت', 'انتهى', 'حدث', 'اجتمع', 'افترق',
    }

    # Check for common past verbs with possessive endings stripped
    base_for_check = word_base
    for suffix in ['ها', 'هم', 'هما', 'ه']:
        if base_for_check.endswith(suffix) and len(base_for_check) > len(suffix) + 2:
            base_for_check = base_for_check[:-len(suffix)]
            break

    if base_for_check in common_past_verbs:
        return True

    # Check for feminine verb endings
    if word_base.endswith('ت') and len(word_base) >= 4:
        # Could be feminine past tense
        root = word_base[:-1]
        if root in common_past_verbs:
            return True

    # Check for plural past tense (وا)
    if word_base.endswith('وا') and len(word_base) >= 4:
        return True

    return False


def should_have_shadda_enhanced(target_base, next_word, prev_word=None):
    """
    Enhanced shadda detection for أن/إن variants.

    Returns True if should have shadda, False if not, None if uncertain.

    Key insight: إنْ (conditional, no shadda) is followed by verbs (past or present).
                 أنَّ/إنَّ (that, with shadda) is followed by nouns.
    """
    if not next_word:
        return None

    next_base = strip_harakat(next_word)

    # Skip punctuation
    if not next_base or all(c in '()،؛:.?!' for c in next_base):
        return None

    # Strong negative signal: ANY verb follows (present OR past)
    # إنْ + verb = conditional "if"
    # Check verbs FIRST before other positive signals
    if is_present_tense_verb(next_base):
        return False

    if is_past_tense_verb(next_word, next_base):
        return False

    # Strong negative signal: لا (negation of verb) or لم (negative past)
    if next_base in {'لا', 'لم'}:
        return False

    # Strong positive signal: definite noun (starts with ال)
    # This is checked early because definite nouns are clearly nominal
    if next_base.startswith('ال') and len(next_base) > 3:
        return True

    # Strong positive signal: prepositional phrases (clearly nominal context)
    nominal_indicators = {'له', 'لها', 'لهم', 'عليه', 'عليها', 'فيه', 'فيها', 'منه', 'منها'}
    if next_base in nominal_indicators:
        return True

    # Strong positive signal: masdar pattern follows
    if is_masdar_pattern(next_base):
        return True

    # Words ending with possessive: only consider if NOT a verb with object suffix
    # This is tricky because verbs can have object suffixes (قيدها = he restricted her)
    # Only mark as nominal if we're confident it's a NOUN with possessive
    # Skip this check - it's too error-prone

    # Strong positive signal: accusative noun follows (if not a verb)
    # But be conservative - only apply if clearly not a verb
    # Skip this check too - it's causing false positives

    return None


def apply_anna_fix(text):
    """
    Apply enhanced أَنَّ/أَنْ fix to diacritized text.

    This runs after the V2 final_sweep and catches additional cases.
    """
    words = text.split()
    if len(words) < 2:
        return text

    target_bases = {'أن', 'إن', 'فإن', 'بأن', 'لأن', 'كأن', 'وأن', 'وإن'}

    result = []
    corrections = 0

    for i, word in enumerate(words):
        word_base = strip_harakat(word)

        if word_base in target_bases:
            next_word = words[i+1] if i + 1 < len(words) else None
            prev_word = words[i-1] if i > 0 else None

            should_shadda = should_have_shadda_enhanced(word_base, next_word, prev_word)

            if should_shadda is True:
                current_has_shadda = has_shadda_on_nun(word)
                if not current_has_shadda:
                    word = add_shadda_after_nun(word)
                    corrections += 1

        result.append(word)

    return ' '.join(result)


def test_anna_fix():
    """Test anna fix."""
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except:
        pass

    print("=" * 60)
    print("ENHANCED ANNA FIX TEST")
    print("=" * 60)

    test_cases = [
        # Missing shadda cases from error analysis
        ("لِمَا أَنَ سَبَبَهُ لَا", "Accusative possessive (سببه)"),
        ("عَلَيْهِ أَنَ فَاقِدَ", "Noun in accusative (فاقد)"),
        ("لِأَنَ تَعْظِيمَ الصَّنَمِ", "Masdar (تعظيم)"),
        ("أَنَ اشْتِرَاطَ وَضْعِ", "Form VIII masdar (اشتراط)"),
        ("فَإِنَ فِيهَا دَوَابِّي", "Prepositional phrase"),
        ("أَنَ لَهُ وَلَاءٌ", "له follows"),

        # Should NOT add shadda
        ("أَنْ يَكُونَ كَذَا", "Verb follows - no shadda"),
        ("أَنْ لَا يَفْعَلَ", "لا + verb - no shadda"),
    ]

    for text, description in test_cases:
        result = apply_anna_fix(text)
        words = text.split()
        result_words = result.split()

        an_word_orig = None
        an_word_new = None
        for w1, w2 in zip(words, result_words):
            b = strip_harakat(w1)
            if b in {'أن', 'إن', 'فإن', 'بأن', 'لأن'}:
                an_word_orig = w1
                an_word_new = w2
                break

        orig_shadda = has_shadda_on_nun(an_word_orig) if an_word_orig else False
        new_shadda = has_shadda_on_nun(an_word_new) if an_word_new else False

        print(f"\n{description}")
        print(f"  Input:  {text}")
        print(f"  Output: {result}")
        if an_word_orig:
            change = "ADDED" if new_shadda and not orig_shadda else ("KEPT" if new_shadda else "NO SHADDA")
            print(f"  Shadda: {change}")


if __name__ == '__main__':
    test_anna_fix()
