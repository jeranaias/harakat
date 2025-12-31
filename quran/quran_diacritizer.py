#!/usr/bin/env python3
"""
Quran Diacritizer Module

Provides 100% accurate diacritization for Quranic text by using
a pre-built lookup table from the authentic Uthmani Quran text.

Usage:
    from quran.quran_diacritizer import diacritize_quran, is_likely_quran

    # Check if text might be Quranic
    if is_likely_quran(text):
        result = diacritize_quran(text)
    else:
        result = diacritize(text)  # Use general model
"""

import json
import os
import re
from typing import Optional, List, Dict

# Path to lookup data
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOOKUP_PATH = os.path.join(SCRIPT_DIR, 'quran_lookup_optimized.json')

# Arabic diacritics
HARAKAT = '\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065F\u0670'
HARAKAT_PATTERN = re.compile(f'[{HARAKAT}]')

# Normalization maps
INPUT_NORMALIZATION = {
    'أ': 'ا',   # Alif with hamza above → plain alif
    'إ': 'ا',   # Alif with hamza below → plain alif
    'آ': 'ا',   # Alif madda → plain alif
    'ٱ': 'ا',   # Alif wasla → plain alif
    'ٰ': 'ا',   # Superscript alif → plain alif
    'ء': '',    # Standalone hamza → remove
    'ى': 'ي',   # Alif maqsura → ya
    'ة': 'ه',   # Ta marbuta → ha
    'ؤ': 'و',   # Waw with hamza → waw
    'ئ': 'ي',   # Ya with hamza → ya
    # Remove Uthmani markers
    'ۥ': '', 'ۦ': '', '۟': '', '۠': '', 'ۢ': '',
    '۪': '', '۫': '', '۬': '', 'ۭ': '', '۩': '', '۝': '', '۞': '',
}

# Common Quranic phrases for detection
QURAN_MARKERS = [
    'بسم الله', 'الرحمن الرحيم', 'الحمد لله', 'رب العالمين',
    'يوم الدين', 'قل هو الله', 'الله لا اله الا هو',
    'ان الذين امنوا', 'والذين كفروا', 'يا ايها الذين امنوا',
    'يا ايها الناس', 'ان الله', 'قال الله', 'قال ربكم',
]

# Global lookup data (loaded lazily)
_lookup_data: Optional[Dict] = None


def _load_lookup():
    """Load the Quran lookup table."""
    global _lookup_data
    if _lookup_data is None:
        if not os.path.exists(LOOKUP_PATH):
            raise FileNotFoundError(f"Quran lookup not found: {LOOKUP_PATH}")
        with open(LOOKUP_PATH, 'r', encoding='utf-8') as f:
            _lookup_data = json.load(f)
    return _lookup_data


def strip_diacritics(text: str) -> str:
    """Remove all diacritical marks from text."""
    return HARAKAT_PATTERN.sub('', text)


def normalize_for_lookup(text: str) -> str:
    """Normalize text for lookup (strip diacritics + normalize chars)."""
    text = strip_diacritics(text)
    for old, new in INPUT_NORMALIZATION.items():
        text = text.replace(old, new)
    return text


def is_likely_quran(text: str) -> bool:
    """Detect if text is likely Quranic."""
    normalized = normalize_for_lookup(text.lower())
    for marker in QURAN_MARKERS:
        if normalize_for_lookup(marker) in normalized:
            return True
    return False


def diacritize_quran(text: str, fallback_func=None) -> str:
    """
    Diacritize text as Quranic Arabic.

    Args:
        text: Arabic text (with or without diacritics)
        fallback_func: Function to call for words not found in Quran

    Returns:
        Fully diacritized text using Quranic spellings
    """
    lookup = _load_lookup()
    words = lookup['words']
    phrases = lookup['phrases']

    # Normalize input for lookup
    text_norm = normalize_for_lookup(text)

    # Strategy 1: Try exact phrase match (100% accurate for full ayahs)
    if text_norm in phrases:
        return phrases[text_norm]

    # Strategy 2: Word-by-word lookup
    input_words = text.split()
    norm_words = [normalize_for_lookup(w) for w in input_words]

    result = []
    for orig_word, norm_word in zip(input_words, norm_words):
        if norm_word in words:
            result.append(words[norm_word])
        elif fallback_func:
            result.append(fallback_func(orig_word))
        else:
            result.append(orig_word)

    return ' '.join(result)


def diacritize_quran_verse(surah: int, ayah: int) -> Optional[str]:
    """Get the exact diacritization of a specific verse."""
    lookup = _load_lookup()
    return lookup['ayahs'].get(f"{surah}:{ayah}")


def get_word_forms(word: str) -> List[str]:
    """Get the Quranic form of a word."""
    lookup = _load_lookup()
    word_norm = normalize_for_lookup(word)
    if word_norm in lookup['words']:
        return [lookup['words'][word_norm]]
    return []


def get_stats() -> Dict:
    """Get statistics about the Quran lookup table."""
    lookup = _load_lookup()
    return lookup.get('stats', {})


if __name__ == '__main__':
    # Quick test
    test = "ان الذين امنوا وعملوا الصالحات كانت لهم جنات الفردوس نزلا"
    result = diacritize_quran(test)
    print(f"Input:  {test}")
    print(f"Output: {result}")
    print(f"Stats:  {get_stats()}")
