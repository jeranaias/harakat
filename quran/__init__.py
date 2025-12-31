"""
Quran Diacritization Module

Provides 100% accurate diacritization for Quranic text.
"""

from .quran_diacritizer import (
    diacritize_quran,
    diacritize_quran_verse,
    is_likely_quran,
    get_word_forms,
    get_stats,
)

__all__ = [
    'diacritize_quran',
    'diacritize_quran_verse',
    'is_likely_quran',
    'get_word_forms',
    'get_stats',
]
