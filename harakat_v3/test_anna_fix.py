#!/usr/bin/env python3
"""Test anna_fix module directly."""
import sys
import os
import io

def main():
    if sys.platform == 'win32':
        os.system('chcp 65001 > nul 2>&1')
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        except:
            pass

    # Add paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)

    from rules.anna_fix import (
        strip_harakat, is_masdar_pattern, is_present_tense_verb,
        is_accusative_noun, should_have_shadda_enhanced, apply_anna_fix
    )

    print("=" * 60)
    print("TESTING ANNA FIX COMPONENTS")
    print("=" * 60)

    # Test masdar patterns
    print("\n1. MASDAR PATTERN TESTS:")
    masdar_tests = [
        ('تعظيم', True, "Form II masdar"),
        ('تكبير', True, "Form II masdar"),
        ('تحليل', True, "Form II masdar"),
        ('اشتراط', True, "Form VIII masdar"),
        ('يكون', False, "Present verb, not masdar"),
        ('الكتاب', False, "Definite noun"),
    ]

    for word, expected, desc in masdar_tests:
        result = is_masdar_pattern(word)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] {word}: {result} (expected {expected}) - {desc}")

    # Test present tense verb detection
    print("\n2. PRESENT TENSE VERB TESTS:")
    verb_tests = [
        ('يكون', True, "Present 3rd masc"),
        ('تكون', True, "Present 2nd/3rd fem"),
        ('نكون', True, "Present 1st plural"),
        ('أكون', True, "Present 1st sing"),
        ('تعظيم', False, "Masdar, not verb"),
        ('الكتاب', False, "Definite noun"),
    ]

    for word, expected, desc in verb_tests:
        result = is_present_tense_verb(word)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] {word}: {result} (expected {expected}) - {desc}")

    # Test full should_have_shadda logic
    print("\n3. SHOULD HAVE SHADDA TESTS:")
    shadda_tests = [
        ('أن', 'تَعْظِيمَ', True, "Masdar follows - should have shadda"),
        ('أن', 'يَكُونَ', False, "Verb follows - no shadda"),
        ('أن', 'الصَّنَمِ', True, "Definite noun follows - should have shadda"),
        ('أن', 'لَا', False, "Negation follows - no shadda"),
        ('أن', 'سَبَبَهُ', True, "Possessive follows - should have shadda"),
    ]

    for target, next_word, expected, desc in shadda_tests:
        next_base = strip_harakat(next_word)
        result = should_have_shadda_enhanced(target, next_word, None)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] '{target}' + '{next_word}' (base: {next_base}): {result} (expected {expected}) - {desc}")

    # Test apply_anna_fix
    print("\n4. APPLY ANNA FIX TESTS:")
    fix_tests = [
        ("لِأَنَ تَعْظِيمَ الصَّنَمِ", "Add shadda for masdar"),
        ("أَنَ اشْتِرَاطَ وَضْعِ", "Add shadda for Form VIII masdar"),
        ("أَنْ يَكُونَ كَذَا", "Keep no shadda for verb"),
        ("أَنْ لَا يَفْعَلَ", "Keep no shadda for لا + verb"),
    ]

    for text, desc in fix_tests:
        result = apply_anna_fix(text)
        changed = text != result
        print(f"\n  {desc}:")
        print(f"    Input:  {text}")
        print(f"    Output: {result}")
        print(f"    Changed: {changed}")

if __name__ == '__main__':
    main()
