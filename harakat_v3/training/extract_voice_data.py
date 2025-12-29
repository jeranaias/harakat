#!/usr/bin/env python3
"""
Extract Active/Passive Voice Training Data

Identifies verbs in Tashkeela V2 and extracts their voice (active/passive).

Active voice pattern: فَعَلَ (fa'ala) - fatha-fatha-fatha
Passive voice pattern: فُعِلَ (fu'ila) - damma-kasra-fatha

This covers Form I verbs. Other forms have similar patterns:
- Form II active: فَعَّلَ, passive: فُعِّلَ
- Form III active: فَاعَلَ, passive: فُوعِلَ
- Form IV active: أَفْعَلَ, passive: أُفْعِلَ
- etc.
"""

import sys
import os
import json
from collections import Counter, defaultdict

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
    """Remove diacritics."""
    return ''.join(c for c in text if c not in HARAKAT_SET)


def extract_diacritics(word):
    """Extract (consonant, diacritics) pairs from a word."""
    result = []
    current_diacs = ''

    for c in word:
        if c in HARAKAT_SET:
            current_diacs += c
        else:
            if result:
                result[-1] = (result[-1][0], current_diacs)
            result.append((c, ''))
            current_diacs = ''

    if result:
        result[-1] = (result[-1][0], current_diacs)

    return result


def detect_verb_voice(word, word_base):
    """
    Detect if a word is an active or passive verb.

    Returns: 'active', 'passive', or None (not a verb)
    """
    pairs = extract_diacritics(word)

    if len(pairs) < 3:
        return None

    # Get first three consonants and their diacritics
    c1, d1 = pairs[0]
    c2, d2 = pairs[1]
    c3, d3 = pairs[2]

    # Check for shadda (Form II, V, etc.)
    has_shadda_c2 = SHADDA in d2

    # Form I past tense patterns (3 consonants)
    if len(pairs) == 3 or (len(pairs) == 4 and pairs[3][0] in 'اوى'):
        # Active: فَعَلَ (fatha-fatha-fatha)
        if FATHA in d1 and FATHA in d2 and FATHA in d3:
            return 'active'

        # Active: فَعِلَ (fatha-kasra-fatha)
        if FATHA in d1 and KASRA in d2 and FATHA in d3:
            return 'active'

        # Active: فَعُلَ (fatha-damma-fatha) - rare
        if FATHA in d1 and DAMMA in d2 and FATHA in d3:
            return 'active'

        # Passive: فُعِلَ (damma-kasra-fatha)
        if DAMMA in d1 and KASRA in d2:
            return 'passive'

    # Form II patterns (with shadda on second consonant)
    if has_shadda_c2:
        # Active Form II: فَعَّلَ (fatha-shadda+fatha-fatha)
        if FATHA in d1 and FATHA in d2 and FATHA in d3:
            return 'active'

        # Passive Form II: فُعِّلَ (damma-shadda+kasra-fatha)
        if DAMMA in d1 and KASRA in d2:
            return 'passive'

    # Present tense (imperfect) patterns - يَفْعَلُ/يُفْعَلُ
    if c1 in 'يتنأ':  # Imperfect prefixes
        if len(pairs) >= 4:
            # Active: يَفْعَلُ (fatha-sukun-fatha-damma)
            if FATHA in d1 and SUKUN in d2:
                return 'active'

            # Active: يَفْعِلُ, يَفْعُلُ variations
            if FATHA in d1:
                return 'active'

            # Passive: يُفْعَلُ (damma-sukun-fatha-damma)
            if DAMMA in d1:
                return 'passive'

    return None


def extract_examples_from_file(filepath, context_size=3):
    """
    Extract verb voice examples from a diacritized file.
    """
    examples = []
    verb_counts = Counter()

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            words = line.split()

            for i, word in enumerate(words):
                word_base = strip_harakat(word)

                # Skip very short words
                if len(word_base) < 3:
                    continue

                voice = detect_verb_voice(word, word_base)
                if voice:
                    left_context = words[max(0, i - context_size):i]
                    right_context = words[i + 1:i + 1 + context_size]

                    example = {
                        'word': word,
                        'word_base': word_base,
                        'left': [strip_harakat(w) for w in left_context],
                        'right': [strip_harakat(w) for w in right_context],
                        'voice': voice,
                        'line_num': line_num,
                    }
                    examples.append(example)
                    verb_counts[voice] += 1

    return examples, verb_counts


def analyze_voice_distribution(examples):
    """Analyze the distribution of voice labels."""
    print("\n" + "=" * 70)
    print("VOICE DISTRIBUTION ANALYSIS")
    print("=" * 70)

    voice_counts = Counter(ex['voice'] for ex in examples)
    print(f"\nTotal verbs detected: {len(examples)}")
    for voice, count in voice_counts.most_common():
        pct = 100 * count / len(examples)
        print(f"  {voice}: {count} ({pct:.1f}%)")

    # Analyze by word base
    print("\nTop verb bases:")
    base_counts = defaultdict(lambda: Counter())
    for ex in examples:
        base_counts[ex['word_base']][ex['voice']] += 1

    # Find verbs that appear in both voices
    both_voices = []
    for base, voices in base_counts.items():
        if len(voices) > 1:
            both_voices.append((base, voices))

    both_voices.sort(key=lambda x: -sum(x[1].values()))

    print(f"\nVerbs with both active/passive forms ({len(both_voices)} found):")
    for base, voices in both_voices[:20]:
        active = voices.get('active', 0)
        passive = voices.get('passive', 0)
        total = active + passive
        print(f"  {base}: {active} active, {passive} passive ({total} total)")


def main():
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 70)
    print("ACTIVE/PASSIVE VOICE DATA EXTRACTION")
    print("=" * 70)

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    harakat_v3_dir = os.path.dirname(script_dir)
    harakat_dir = os.path.dirname(harakat_v3_dir)
    tashkeel_dir = os.path.dirname(harakat_dir)

    train_file = os.path.join(tashkeel_dir, 'tashkeela_v2', 'tashkeela_train.txt')
    output_dir = os.path.join(script_dir, 'voice_data')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nInput: {train_file}")
    print(f"Output: {output_dir}")

    if not os.path.exists(train_file):
        print(f"\nERROR: Training file not found: {train_file}")
        return

    # Extract from training data
    print("\nExtracting voice data...")
    examples, counts = extract_examples_from_file(train_file)

    # Analyze
    analyze_voice_distribution(examples)

    # Filter to only include verbs that appear in both voices
    base_voices = defaultdict(list)
    for ex in examples:
        base_voices[ex['word_base']].append(ex)

    ambiguous_examples = []
    for base, exs in base_voices.items():
        voices = set(ex['voice'] for ex in exs)
        if len(voices) > 1:
            ambiguous_examples.extend(exs)

    print(f"\nAmbiguous verbs (appear in both voices): {len(ambiguous_examples)}")

    # Save data
    output_file = os.path.join(output_dir, 'voice_data.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ambiguous_examples, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(ambiguous_examples)} examples to {output_file}")

    # Also save all data
    all_file = os.path.join(output_dir, 'voice_data_all.json')
    with open(all_file, 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(examples)} total examples to {all_file}")

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
