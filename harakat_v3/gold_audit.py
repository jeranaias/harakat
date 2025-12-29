#!/usr/bin/env python3
"""
Gold Standard Audit for Harakat V3

Systematically analyze the Tashkeela gold standard for inconsistencies.
These inconsistencies prevent accurate model improvement because the
model learns incorrect patterns that happen to match the noisy gold.

Key areas to audit:
1. Particle kasra (إلى، إلا، إذا، إذ، إن)
2. Sun letter assimilation (الشمس → الشَّمس)
3. Hamza spelling (أ vs ا)
4. Shadda consistency
5. Case ending patterns

Usage:
    python gold_audit.py                  # Full audit
    python gold_audit.py --fix            # Create normalized version
    python gold_audit.py --diff           # Show what would change
"""

import sys
import os
import argparse
from collections import defaultdict

if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')

harakat_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, harakat_dir)

# Constants
HARAKAT = 'ًٌٍَُِّْ'
HARAKAT_SET = set(HARAKAT)
FATHA = 'َ'
KASRA = 'ِ'
DAMMA = 'ُ'
SUKUN = 'ْ'
SHADDA = 'ّ'

# Sun letters (require shadda after ال)
SUN_LETTERS = set('تثدذرزسشصضطظلن')


def strip_harakat(text):
    return ''.join(c for c in text if c not in HARAKAT_SET)


# ============================================================
# ISSUE DETECTORS
# ============================================================

def check_particle_kasra(word, base):
    """
    Check if particle has correct kasra on hamza-below.
    Returns: (has_issue, description, fixed_word)
    """
    particles_needing_kasra = {'إلى', 'إلا', 'إذا', 'إذ', 'إن'}

    if base not in particles_needing_kasra:
        return False, None, word

    # Check if kasra present after إ
    has_kasra = False
    hamza_pos = None
    for i, c in enumerate(word):
        if c == 'إ':
            hamza_pos = i
            for j in range(i+1, min(i+3, len(word))):
                if word[j] == KASRA:
                    has_kasra = True
                    break
                if word[j] not in HARAKAT_SET:
                    break
            break

    if has_kasra:
        return False, None, word

    # Fix: add kasra after hamza
    if hamza_pos is not None:
        fixed = list(word)
        # Insert kasra after hamza
        insert_pos = hamza_pos + 1
        # Skip any existing vowel
        while insert_pos < len(fixed) and fixed[insert_pos] in HARAKAT_SET:
            if fixed[insert_pos] in {FATHA, DAMMA}:
                fixed.pop(insert_pos)  # Remove wrong vowel
            else:
                insert_pos += 1
        fixed.insert(hamza_pos + 1, KASRA)
        return True, f"Missing kasra on {base}", ''.join(fixed)

    return False, None, word


def check_sun_assimilation(word, base):
    """
    Check if word starting with ال has correct shadda on sun letter.
    Note: This checks for MISSING shadda, not words that already have shadda elsewhere.

    For sun letter assimilation:
    - الشمس → الشَّمس (shadda on ش)
    - الرحمن → الرَّحمن (shadda on ر)

    NOT applicable to:
    - الذي، التي (relative pronouns) - shadda is on ل, not on ذ/ت
    - Words that already have shadda on lam

    Returns: (has_issue, description, fixed_word)
    """
    if not base.startswith('ال') or len(base) < 3:
        return False, None, word

    sun_letter = base[2]
    if sun_letter not in SUN_LETTERS:
        return False, None, word

    # Skip relative pronouns (they have special pattern with shadda on lam)
    relative_pronouns = {'الذي', 'التي', 'الذين', 'اللذان', 'اللتان', 'اللاتي', 'اللائي'}
    if base in relative_pronouns:
        return False, None, word

    # Check if word already has shadda on lam (position 2 in base)
    # If so, it's a different pattern, not sun assimilation
    base_count = 0
    lam_pos = None
    sun_pos = None

    for i, c in enumerate(word):
        if c not in HARAKAT_SET:
            base_count += 1
            if base_count == 2:  # The lam
                lam_pos = i
            elif base_count == 3:  # The sun letter
                sun_pos = i
                break

    # Check if lam has shadda (special pattern like الّذي)
    if lam_pos is not None:
        for j in range(lam_pos+1, min(lam_pos+3, len(word))):
            if word[j] == SHADDA:
                return False, None, word  # Has shadda on lam, different pattern
            if word[j] not in HARAKAT_SET:
                break

    # Check if sun letter has shadda
    if sun_pos is not None:
        for j in range(sun_pos+1, min(sun_pos+3, len(word))):
            if word[j] == SHADDA:
                return False, None, word  # Has shadda, OK
            if word[j] not in HARAKAT_SET:
                break
        # Missing shadda on sun letter
        fixed = word[:sun_pos+1] + SHADDA + word[sun_pos+1:]
        return True, f"Missing shadda on sun letter {sun_letter}", fixed

    return False, None, word


def check_allah_pattern(word, base):
    """
    Check Allah (الله) has correct pattern.
    Should be: اللَّهِ/اللَّهُ/اللَّهَ (with shadda on second lam)
    """
    if base != 'الله':
        return False, None, word

    # Check if shadda present on second lam
    lam_count = 0
    for i, c in enumerate(word):
        if c == 'ل':
            lam_count += 1
            if lam_count == 2:  # Second lam
                for j in range(i+1, min(i+3, len(word))):
                    if word[j] == SHADDA:
                        return False, None, word  # Has shadda, OK
                    if word[j] not in HARAKAT_SET:
                        break
                # Missing shadda
                fixed = word[:i+1] + SHADDA + word[i+1:]
                return True, "الله missing shadda", fixed

    return False, None, word


# ============================================================
# MAIN AUDIT
# ============================================================

def audit_file(filepath, name, fix=False, diff=False):
    """Audit a gold standard file."""
    print(f"\n{'='*70}")
    print(f"AUDITING: {name}")
    print(f"{'='*70}")

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    issues = defaultdict(list)
    total_words = 0
    fixed_lines = []

    checkers = [
        ('particle_kasra', check_particle_kasra),
        ('sun_assimilation', check_sun_assimilation),
        ('allah', check_allah_pattern),
    ]

    for line_num, line in enumerate(lines):
        words = line.split()
        fixed_words = []

        for word in words:
            total_words += 1
            base = strip_harakat(word)
            current_word = word

            # Apply all checkers
            for check_name, checker in checkers:
                has_issue, desc, fixed = checker(current_word, base)
                if has_issue:
                    issues[check_name].append({
                        'line': line_num + 1,
                        'original': word,
                        'fixed': fixed,
                        'description': desc
                    })
                    current_word = fixed  # Chain fixes

            fixed_words.append(current_word)

        fixed_lines.append(' '.join(fixed_words))

    # Summary
    print(f"\nTotal words analyzed: {total_words:,}")
    print(f"\nIssues found:")
    print("-" * 50)

    total_issues = 0
    for check_name, issue_list in sorted(issues.items()):
        count = len(issue_list)
        total_issues += count
        pct = 100 * count / total_words
        print(f"  {check_name}: {count:,} ({pct:.3f}%)")

        # Show examples
        if diff and issue_list[:3]:
            print(f"    Examples:")
            for ex in issue_list[:3]:
                print(f"      Line {ex['line']}: {ex['original']} → {ex['fixed']}")
                print(f"        ({ex['description']})")

    print(f"\nTotal issues: {total_issues:,} ({100*total_issues/total_words:.2f}% of words)")

    # Potential DER impact
    # Rough estimate: each fix is ~1 diacritic position
    # Test set has ~443,641 diacritic positions
    if 'test' in name.lower():
        total_positions = 443641
    else:
        total_positions = 887282  # Estimate for val (2x test)

    potential_der_change = 100 * total_issues / total_positions
    print(f"\nPotential DER impact if gold was normalized: -{potential_der_change:.3f}%")

    # Write fixed file if requested
    if fix:
        output_path = filepath.replace('.txt', '_normalized.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(fixed_lines))
        print(f"\nNormalized file written to: {output_path}")

    return issues, fixed_lines


def main():
    parser = argparse.ArgumentParser(description='Audit gold standard for inconsistencies')
    parser.add_argument('--fix', action='store_true', help='Create normalized version')
    parser.add_argument('--diff', action='store_true', help='Show what would change')
    args = parser.parse_args()

    # Setup encoding for Windows
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 70)
    print("GOLD STANDARD AUDIT")
    print("=" * 70)
    print("\nChecking for systematic inconsistencies in Tashkeela gold data...")

    test_path = os.path.join(harakat_dir, '..', 'paper_eval', 'tashkeela_test.txt')
    val_path = os.path.join(harakat_dir, '..', 'paper_eval', 'tashkeela_val.txt')

    if os.path.exists(test_path):
        audit_file(test_path, "Test Set (2,500 lines)", fix=args.fix, diff=args.diff)

    if os.path.exists(val_path):
        audit_file(val_path, "Validation Set (5,000 lines)", fix=args.fix, diff=args.diff)

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
1. Create normalized gold standard with consistent diacritization
2. Re-evaluate V2 against normalized gold to get "true" DER
3. Consider re-training models on normalized data for better generalization
4. Keep original gold for comparison with published benchmarks
""")


if __name__ == '__main__':
    main()
