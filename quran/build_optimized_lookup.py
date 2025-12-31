#!/usr/bin/env python3
"""
Build optimized Quran lookup table.

This creates a compact lookup that:
1. Maps normalized word → most common diacritized form
2. Stores full ayahs for exact verse lookup
3. Stores phrase lookup for multi-word matching

The trade-off: we lose context-aware disambiguation for ambiguous words,
but this rarely matters in practice (2 errors out of 82,242 words).
"""

import json
import os
import re
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Arabic diacritics
HARAKAT = '\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065F\u0670'
HARAKAT_PATTERN = re.compile(f'[{HARAKAT}]')

# Uthmani normalization
UTHMANI_MAP = {
    'ٱ': 'ا', 'ٰ': 'ا', 'ۥ': '', 'ۦ': '', '۟': '', '۠': '', 'ۢ': '',
    '۪': '', '۫': '', '۬': '', 'ۭ': '', '۩': '', '۝': '', '۞': '', '\uFEFF': '',
}

# Key normalization for user input matching
KEY_NORMALIZATION = {
    'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ء': '', 'ى': 'ي', 'ة': 'ه', 'ؤ': 'و', 'ئ': 'ي',
}

def strip_diacritics(text):
    return HARAKAT_PATTERN.sub('', text)

def normalize_uthmani(text):
    for old, new in UTHMANI_MAP.items():
        text = text.replace(old, new)
    return text

def normalize_for_key(text):
    text = normalize_uthmani(text)
    text = strip_diacritics(text)
    for old, new in KEY_NORMALIZATION.items():
        text = text.replace(old, new)
    return text

def main():
    print("=" * 60)
    print("Building Optimized Quran Lookup")
    print("=" * 60)

    # Load Quran data
    quran_path = os.path.join(SCRIPT_DIR, 'quran_uthmani.json')
    with open(quran_path, 'r', encoding='utf-8') as f:
        quran_data = json.load(f)

    ayahs = quran_data['ayahs']
    print(f"Loaded {len(ayahs):,} ayahs")

    # Build lookups
    word_forms = defaultdict(lambda: defaultdict(int))  # key → {form → count}
    phrase_lookup = {}  # key_phrase → diac_phrase
    ayah_lookup = {}  # "surah:ayah" → original text

    for ayah in ayahs:
        surah = ayah['surah']
        ayah_num = ayah['ayah']
        text = ayah['text']

        # Store original ayah
        ayah_lookup[f"{surah}:{ayah_num}"] = text

        # Normalize for output (removes Uthmani markers, keeps hamzas)
        diac_text = normalize_uthmani(text)

        # Normalize for key (fully normalized for matching)
        key_text = normalize_for_key(text)

        # Phrase lookup
        phrase_lookup[key_text] = diac_text

        # Word forms with frequency
        diac_words = diac_text.split()
        key_words = key_text.split()

        for key_word, diac_word in zip(key_words, diac_words):
            if key_word:
                word_forms[key_word][diac_word] += 1

    # Build final word lookup (most common form for each word)
    word_lookup = {}
    for key_word, forms in word_forms.items():
        # Pick the most common form
        most_common = max(forms.items(), key=lambda x: x[1])
        word_lookup[key_word] = most_common[0]

    print(f"Unique words: {len(word_lookup):,}")
    print(f"Phrase entries: {len(phrase_lookup):,}")
    print(f"Ayah entries: {len(ayah_lookup):,}")

    # Count ambiguous words
    ambig = sum(1 for forms in word_forms.values() if len(forms) > 1)
    print(f"Ambiguous words: {ambig:,}")

    # Build output
    output = {
        'words': word_lookup,
        'phrases': phrase_lookup,
        'ayahs': ayah_lookup,
        'stats': {
            'total_ayahs': len(ayahs),
            'unique_words': len(word_lookup),
            'ambiguous_words': ambig,
        }
    }

    # Save
    out_path = os.path.join(SCRIPT_DIR, 'quran_lookup_optimized.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, separators=(',', ':'))

    size = os.path.getsize(out_path)
    print(f"\nSaved: {out_path}")
    print(f"Size: {size:,} bytes ({size/1024/1024:.2f} MB)")

    # Test compression
    import lzma
    with open(out_path, 'rb') as f:
        data = f.read()
    compressed = lzma.compress(data, preset=9)
    print(f"LZMA: {len(compressed):,} bytes ({len(compressed)/1024/1024:.2f} MB)")

if __name__ == '__main__':
    main()
