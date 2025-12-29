"""
Harakat V2 - Neural Enhanced Arabic Diacritization

A self-contained Arabic diacritization system achieving 2.23% DER.
Built on Harakat V1 (harakat_elite) with neural post-processing layers.

Pipeline:
    1. V1 Base (harakat_elite): Statistical diacritization (4.35% DER)
    2. Case V3 Ensemble: 3x BiLSTM+Attention for case endings
    3. Hybrid LSTM Corrector: Contrastive learning for internal vowels
    4. Final Sweep: Rule-based أَنَّ/أَنْ disambiguation (optional)

Results on Tashkeela test set (2,500 lines):
    - V1 Baseline:  4.35% DER (with case), 2.27% DER (without case)
    - V2 Complete:  2.29% DER (with case), 1.95% DER (without case)
    - V2 + Sweep:   2.23% DER (with case), 1.95% DER (without case)
    - Improvement:  49% relative reduction over V1

Usage:
    >>> from harakat_v2 import diacritize
    >>> result = diacritize("كتب الطالب الدرس")
    >>> print(result)  # كَتَبَ الطَّالِبُ الدَّرْسَ

    # With final sweep (optional, slightly better accuracy)
    >>> from harakat_v2 import diacritize_with_sweep
    >>> result = diacritize_with_sweep("أن الطالب كتب الدرس")

Author: Jesse Morgan (DLI Arabic Instructor)
License: MIT
"""

from .harakat_v2 import (
    HarakatV2,
    diacritize,
    get_diacritizer,
    strip_harakat,
)

from .final_sweep import apply_final_sweep

__version__ = '2.1.0'
__author__ = 'Jesse Morgan'


def diacritize_with_sweep(text):
    """
    Diacritize text using V2 pipeline + final sweep corrections.

    The final sweep applies rule-based corrections for أَنَّ/أَنْ disambiguation,
    reducing DER by an additional 0.06% (from 2.29% to 2.23%).
    """
    result = diacritize(text)
    return apply_final_sweep(result)


__all__ = [
    'HarakatV2',
    'diacritize',
    'diacritize_with_sweep',
    'get_diacritizer',
    'strip_harakat',
    'apply_final_sweep',
]
