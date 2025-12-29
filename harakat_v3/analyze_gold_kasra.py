#!/usr/bin/env python3
"""
Analyze gold standard for kasra on hamza-below particles.

Check if the gold standard is consistent about kasra on إ in particles like إلى، إلا، إذا
"""

import sys
import os
from collections import defaultdict

if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')

harakat_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, harakat_dir)

HARAKAT = 'ًٌٍَُِّْ'
HARAKAT_SET = set(HARAKAT)
KASRA = 'ِ'


def strip_harakat(text):
    return ''.join(c for c in text if c not in HARAKAT_SET)


def has_kasra_after_hamza(word):
    """Check if word has kasra after hamza-below (إ)."""
    for i, c in enumerate(word):
        if c == 'إ':
            # Check next few chars for kasra
            for j in range(i+1, min(i+3, len(word))):
                if word[j] == KASRA:
                    return True
                if word[j] not in HARAKAT_SET:
                    return False
            return False
    return False


def analyze_file(filepath, name):
    """Analyze kasra patterns in gold data."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    particles = ['إلى', 'إلا', 'إذا', 'إذ', 'إن']
    stats = {p: {'with_kasra': 0, 'without_kasra': 0, 'examples_with': [], 'examples_without': []}
             for p in particles}

    for line in lines:
        words = line.split()
        for word in words:
            base = strip_harakat(word)
            if base in particles:
                has_kasra = has_kasra_after_hamza(word)
                if has_kasra:
                    stats[base]['with_kasra'] += 1
                    if len(stats[base]['examples_with']) < 3:
                        stats[base]['examples_with'].append(word)
                else:
                    stats[base]['without_kasra'] += 1
                    if len(stats[base]['examples_without']) < 3:
                        stats[base]['examples_without'].append(word)

    print(f"\n{name}")
    print("=" * 70)

    for p in particles:
        with_k = stats[p]['with_kasra']
        without_k = stats[p]['without_kasra']
        total = with_k + without_k
        if total == 0:
            continue

        pct_with = 100 * with_k / total

        print(f"\n{p}:")
        print(f"  With kasra:    {with_k:4d} ({pct_with:.1f}%)")
        print(f"  Without kasra: {without_k:4d} ({100-pct_with:.1f}%)")
        print(f"  Total:         {total:4d}")

        if stats[p]['examples_with']:
            print(f"  Examples with kasra:    {stats[p]['examples_with']}")
        if stats[p]['examples_without']:
            print(f"  Examples without kasra: {stats[p]['examples_without']}")


def main():
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 70)
    print("GOLD STANDARD KASRA ANALYSIS")
    print("=" * 70)
    print("\nChecking if gold data has kasra on hamza-below")
    print("in particles like: ila, illa, idha, idh, in")

    test_path = os.path.join(harakat_dir, '..', 'paper_eval', 'tashkeela_test.txt')
    val_path = os.path.join(harakat_dir, '..', 'paper_eval', 'tashkeela_val.txt')

    analyze_file(test_path, "Test Set (2,500 lines)")
    analyze_file(val_path, "Validation Set (5,000 lines)")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
If the gold standard is inconsistent (some with kasra, some without),
then our particle rule will HURT DER by "fixing" cases that were
already correct in V2's output (matching the inconsistent gold).

Options:
1. Only apply kasra fix when V2 output clearly lacks it AND gold usually has it
2. Don't apply kasra fix at all if gold is inconsistent
3. Accept that this is a data quality issue, not a model issue
""")


if __name__ == '__main__':
    main()
