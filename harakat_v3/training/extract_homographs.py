#!/usr/bin/env python3
"""
Extract Homograph Training Data from Tashkeela V2

This script extracts training examples for homograph disambiguation.
Each example contains:
- The homograph word (diacritized)
- Surrounding context (N words before/after)
- The target class (which diacritization variant)

Top homographs to disambiguate:
1. من: مِنْ (from/preposition) vs مَنْ (who/pronoun)
2. أم: أَمْ (or/conjunction) vs أُمّ (mother/noun)
3. ثم: ثُمَّ (then/sequence) vs ثَمَّ (there/location)

Output: JSON files with training examples for each homograph.
"""

import sys
import os
import json
from collections import defaultdict, Counter

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
    """Remove all diacritics."""
    return ''.join(c for c in text if c not in HARAKAT_SET)


def normalize_diacritics(word):
    """Normalize a diacritized word to its canonical form."""
    # Remove case endings (final vowel) for comparison
    base = strip_harakat(word)

    # Extract internal diacritics pattern
    result = []
    for c in word:
        result.append(c)
    return ''.join(result)


# ============================================================
# HOMOGRAPH DEFINITIONS
# ============================================================

HOMOGRAPHS = {
    'من': {
        'variants': {
            'مِنْ': 'preposition',  # from
            'مَنْ': 'pronoun',       # who
            'مِنَ': 'preposition',  # from (with fatha for ال)
        },
        'description': 'من (from) vs من (who)'
    },
    'أم': {
        'variants': {
            'أَمْ': 'conjunction',   # or
            'أُمّ': 'noun',          # mother
            'أُمَّ': 'noun',         # mother (accusative)
            'أُمِّ': 'noun',         # mother (genitive)
            'أُمُّ': 'noun',         # mother (nominative)
            # Add more case variants for mother
            'أُمٌّ': 'noun',         # mother (nominative with tanween)
        },
        'description': 'أم (or) vs أم (mother)'
    },
    'ثم': {
        'variants': {
            # All shadda forms - sequence is by far most common
            'ثُمَّ': 'sequence',     # then (temporal sequence)
            'ثَمَّ': 'location',     # there (location)
            'ثُمّ': 'sequence',      # then (no final vowel shown)
            'ثَمّ': 'location',      # there (no final vowel shown)
        },
        'description': 'ثم (then) vs ثم (there)'
    },
    'غير': {
        'variants': {
            'غَيْر': 'noun',         # other than
            'غَيْرَ': 'noun',        # other than (accusative)
            'غَيْرِ': 'noun',        # other than (genitive)
            'غَيْرُ': 'noun',        # other than (nominative)
            'غَيَّرَ': 'verb',       # changed (verb)
        },
        'description': 'غير (other) vs غيّر (changed)'
    },
    'علم': {
        'variants': {
            'عِلْم': 'noun_knowledge',   # knowledge
            'عَلِمَ': 'verb_active',     # he knew
            'عُلِمَ': 'verb_passive',    # it was known
            'عَلَم': 'noun_flag',        # flag
        },
        'description': 'علم (knowledge/knew/flag)'
    },
    'ملك': {
        'variants': {
            'مَلِك': 'noun_king',      # king
            'مُلْك': 'noun_property',  # property/kingdom
            'مَلَك': 'noun_angel',     # angel
            'مَلَكَ': 'verb',          # he owned
        },
        'description': 'ملك (king/property/owned)'
    },
    'قبل': {
        'variants': {
            'قَبْل': 'preposition',    # before
            'قَبْلَ': 'preposition',   # before (with fatha)
            'قَبْلِ': 'preposition',   # before (genitive construct)
            'قَبْلُ': 'preposition',   # before (nominative)
            'قِبَل': 'noun',           # direction/front
            'قِبَلِ': 'noun',          # direction (genitive)
        },
        'description': 'قبل (before/direction)'
    },
    'بعد': {
        'variants': {
            'بَعْد': 'preposition',    # after
            'بَعْدَ': 'preposition',   # after (with fatha)
            'بَعْدِ': 'preposition',   # after (genitive construct)
            'بَعْدُ': 'preposition',   # after (nominative)
            'بُعْد': 'noun',           # distance
            'بُعْدِ': 'noun',          # distance (genitive)
        },
        'description': 'بعد (after/distance)'
    },
    'ذكر': {
        'variants': {
            'ذَكَرَ': 'verb_active',   # he mentioned
            'ذُكِرَ': 'verb_passive',  # it was mentioned
            'ذِكْر': 'noun',           # mention/remembrance
            'ذِكْرِ': 'noun',          # mention (genitive)
            'ذِكْرَ': 'noun',          # mention (accusative)
            'ذِكْرُ': 'noun',          # mention (nominative)
            'ذَكَر': 'noun_male',      # male
            'ذَكَرٌ': 'noun_male',     # male (nominative)
            'ذَكَرٍ': 'noun_male',     # male (genitive)
            'ذَكَرًا': 'noun_male',    # male (accusative)
        },
        'description': 'ذكر (mentioned/remembrance/male)'
    },
    'سنة': {
        'variants': {
            'سَنَة': 'noun_year',      # year
            'سَنَةً': 'noun_year',     # year (accusative)
            'سَنَةٍ': 'noun_year',     # year (genitive)
            'سَنَةَ': 'noun_year',     # year (construct state)
            'سُنَّة': 'noun_tradition', # tradition/sunnah
            'سُنَّةً': 'noun_tradition', # tradition (accusative)
            'سُنَّةٍ': 'noun_tradition', # tradition (genitive)
            'سُنَّةِ': 'noun_tradition', # tradition (construct genitive)
        },
        'description': 'سنة (year/tradition)'
    },
}


def classify_homograph(word, word_base):
    """
    Classify a diacritized homograph into its variant class.

    Returns: (variant_key, class_name) or (None, 'unknown')
    """
    if word_base not in HOMOGRAPHS:
        return None, 'unknown'

    homograph_info = HOMOGRAPHS[word_base]

    # Try exact match first
    for variant, class_name in homograph_info['variants'].items():
        if word == variant:
            return variant, class_name

    # Try match ignoring case ending (final vowel)
    word_internal = word
    for variant, class_name in homograph_info['variants'].items():
        variant_base = strip_harakat(variant)
        if variant_base == word_base:
            # Compare internal diacritics
            word_chars = [(c, '') for c in word_base]
            word_idx = 0
            variant_idx = 0

            # Simple pattern matching
            # Extract diacritics after each consonant
            word_pattern = extract_pattern(word)
            variant_pattern = extract_pattern(variant)

            if word_pattern == variant_pattern:
                return variant, class_name

    return None, 'unknown'


def extract_pattern(word):
    """Extract consonant+diacritic pattern from a word."""
    pattern = []
    current_diacs = ''

    for c in word:
        if c in HARAKAT_SET:
            current_diacs += c
        else:
            if pattern:
                pattern[-1] = (pattern[-1][0], current_diacs)
            pattern.append((c, ''))
            current_diacs = ''

    if pattern:
        pattern[-1] = (pattern[-1][0], current_diacs)

    return pattern


def extract_examples_from_file(filepath, context_size=3):
    """
    Extract homograph examples from a diacritized file.

    Args:
        filepath: Path to diacritized text file
        context_size: Number of words before/after to include

    Returns:
        Dictionary mapping homograph base -> list of examples
    """
    examples = defaultdict(list)

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            words = line.split()

            for i, word in enumerate(words):
                word_base = strip_harakat(word)

                if word_base in HOMOGRAPHS:
                    # Get context
                    left_context = words[max(0, i - context_size):i]
                    right_context = words[i + 1:i + 1 + context_size]

                    # Classify the variant
                    variant, class_name = classify_homograph(word, word_base)

                    example = {
                        'word': word,
                        'word_base': word_base,
                        'left_context': left_context,
                        'right_context': right_context,
                        'variant': variant,
                        'class': class_name,
                        'line_num': line_num,
                        'full_line': line
                    }

                    examples[word_base].append(example)

    return examples


def analyze_homograph_distribution(examples):
    """Analyze the distribution of homograph variants."""
    print("\n" + "=" * 70)
    print("HOMOGRAPH DISTRIBUTION ANALYSIS")
    print("=" * 70)

    for base, ex_list in sorted(examples.items()):
        print(f"\n{base}: {len(ex_list)} occurrences")
        print(f"  Description: {HOMOGRAPHS[base]['description']}")

        # Count variants
        variant_counts = Counter()
        class_counts = Counter()
        unknown_examples = []

        for ex in ex_list:
            variant_counts[ex['variant']] += 1
            class_counts[ex['class']] += 1
            if ex['class'] == 'unknown':
                unknown_examples.append(ex)

        print(f"  By variant:")
        for variant, count in variant_counts.most_common():
            pct = 100 * count / len(ex_list)
            print(f"    {variant}: {count} ({pct:.1f}%)")

        print(f"  By class:")
        for cls, count in class_counts.most_common():
            pct = 100 * count / len(ex_list)
            print(f"    {cls}: {count} ({pct:.1f}%)")

        if unknown_examples:
            print(f"  Unknown variants (first 5):")
            for ex in unknown_examples[:5]:
                print(f"    {ex['word']} in: ...{' '.join(ex['left_context'][-2:])} [{ex['word']}] {' '.join(ex['right_context'][:2])}...")


def save_training_data(examples, output_dir):
    """Save training data for each homograph."""
    os.makedirs(output_dir, exist_ok=True)

    for base, ex_list in examples.items():
        # Filter out unknown
        known = [ex for ex in ex_list if ex['class'] != 'unknown']

        if len(known) < 10:
            print(f"  Skipping {base}: only {len(known)} known examples")
            continue

        output_file = os.path.join(output_dir, f"{base}_data.json")

        # Simplify for training
        training_data = []
        for ex in known:
            training_data.append({
                'left': [strip_harakat(w) for w in ex['left_context']],
                'right': [strip_harakat(w) for w in ex['right_context']],
                'class': ex['class'],
                'word': ex['word'],
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)

        print(f"  Saved {len(training_data)} examples to {output_file}")


def main():
    # Fix Windows console encoding
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 70)
    print("HOMOGRAPH TRAINING DATA EXTRACTION")
    print("=" * 70)

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    harakat_v3_dir = os.path.dirname(script_dir)
    harakat_dir = os.path.dirname(harakat_v3_dir)
    tashkeel_dir = os.path.dirname(harakat_dir)

    # Use V2 normalized gold for training
    train_file = os.path.join(tashkeel_dir, 'tashkeela_v2', 'tashkeela_train.txt')
    val_file = os.path.join(tashkeel_dir, 'tashkeela_v2', 'tashkeela_val.txt')

    # Output directory
    output_dir = os.path.join(harakat_v3_dir, 'training', 'homograph_data')

    print(f"\nInput files:")
    print(f"  Train: {train_file}")
    print(f"  Val: {val_file}")
    print(f"\nOutput: {output_dir}")

    # Check files exist
    if not os.path.exists(train_file):
        print(f"\nERROR: Training file not found: {train_file}")
        print("Please run tashkeela_v2/normalize.py first")
        return

    # Extract from training data
    print("\nExtracting from training data...")
    train_examples = extract_examples_from_file(train_file)

    # Analysis
    analyze_homograph_distribution(train_examples)

    # Save training data
    print("\nSaving training data...")
    save_training_data(train_examples, output_dir)

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
