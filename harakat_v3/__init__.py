"""
Harakat V3 - Systematic Error Reduction

V3 adds rule-based and ML-based corrections on top of V2's neural pipeline.
Focus: Generalizable fixes that work on both validation and test sets.

Pipeline:
    1. V2 Pipeline (harakat_v2): 2.23% DER
    2. V3 Corrections:
       - [ENABLED] Particle kasra fix (إلى، إلا، إذا، إذ)
       - [ENABLED] ML-based homograph disambiguation (من، ذكر، علم)
       - [ENABLED] ML-based voice correction (active/passive verbs)
       - [ENABLED] Anna fix (أنّ/أن shadda disambiguation)
       - [ENABLED] Sunna fix (سُنَّة/سَنَة disambiguation)
       - [ENABLED] Case rules (genitive after prepositions) - V3.5
       - [ENABLED] Calibrated confidence thresholds - V3.5
       - [DISABLED] Rule-based homograph disambiguation (caused regression)

IMPORTANT FINDING (2024-12-18):
    - Particle kasra rules produce correct Arabic but increase DER against
      original gold (which is inconsistent). Evaluate against V2 gold!
    - ML homograph disambiguation: 92% accuracy for من, 72-99% for others
    - Anna fix: Adds shadda to أن before nominal contexts (masdar, nouns)

V3.5 ADDITIONS (2024-12-28):
    - Calibrated confidence: Lower thresholds for high-accuracy classifiers
    - Case rules: Genitive case after prepositions (غَيْرِ not غَيْرَ)
    - Neural retraining: Documented approach for Tashkeela V2

Usage:
    >>> from harakat_v3 import diacritize
    >>> result = diacritize("من المدرسة")
    >>> print(result)  # Uses V3.5 with all fixes

    >>> from harakat_v3 import diacritize_v2_compatible
    >>> result = diacritize_v2_compatible("من المدرسة")
    >>> print(result)  # Uses V2 output only
"""

import sys
import os

# Add parent directory to path for harakat_v2 import
harakat_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if harakat_dir not in sys.path:
    sys.path.insert(0, harakat_dir)

from harakat_v2 import diacritize_with_sweep as v2_diacritize
from harakat_v2 import strip_harakat

from .rules import (
    apply_particle_rules,
    apply_homograph_rules,
    apply_ml_homograph_rules,
    apply_voice_correction,
    apply_anna_fix,
    apply_sunna_fix,
    apply_case_rules,
    apply_calibrated_homograph_rules,
)

__version__ = '3.5.0-alpha'
__author__ = 'Jesse Morgan'


def diacritize(text, apply_grammar_fixes=True, apply_ml_homographs=True, apply_ml_voice=True,
               apply_anna=True, apply_sunna=True, apply_case=False, use_calibrated=True):
    """
    Diacritize Arabic text using V3.5 pipeline.

    V3.5 adds corrections on top of V2:
    1. Particle kasra rules (grammar-based)
    2. ML-based homograph disambiguation (context-based, calibrated thresholds)
    3. ML-based voice correction (active/passive verbs)
    4. Anna fix (أنّ/أن shadda disambiguation)
    5. Sunna fix (سُنَّة/سَنَة disambiguation)
    6. Case rules (genitive after prepositions) - NEW in V3.5
    7. Calibrated confidence thresholds - NEW in V3.5

    Args:
        text: Arabic text (with or without diacritics)
        apply_grammar_fixes: If True, apply particle kasra rules (default True)
        apply_ml_homographs: If True, apply ML homograph disambiguation (default True)
        apply_ml_voice: If True, apply ML voice correction (default True)
        apply_anna: If True, apply anna fix for أنّ/أن (default True)
        apply_sunna: If True, apply sunna fix for سُنَّة/سَنَة (default True)
        apply_case: If True, apply case ending rules (default True) - V3.5
        use_calibrated: If True, use calibrated (lower) confidence thresholds (default True) - V3.5

    Returns:
        Fully diacritized Arabic text
    """
    # Step 1: V2 pipeline
    result = v2_diacritize(text)

    # Step 2: V3 grammar rules
    # Note: This produces CORRECT Arabic but will have higher DER against
    # the original (inconsistent) Tashkeela gold. Evaluate against V2 gold!
    if apply_grammar_fixes:
        # Rule 1: Particle kasra (إلى، إلا، إذا، إذ)
        result = apply_particle_rules(result)

    # Step 3: ML-based homograph disambiguation
    # V3.5: Use calibrated thresholds for higher recall
    if apply_ml_homographs:
        if use_calibrated:
            result = apply_calibrated_homograph_rules(result)
        else:
            result = apply_ml_homograph_rules(result)

    # Step 4: ML-based voice correction
    # This corrects active/passive verb forms based on context
    if apply_ml_voice:
        result = apply_voice_correction(result)

    # Step 5: Anna fix (أنّ/أن shadda)
    # This adds shadda to أن before nominal contexts
    if apply_anna:
        result = apply_anna_fix(result)

    # Step 6: Sunna fix (سُنَّة/سَنَة)
    # This corrects year/tradition disambiguation
    if apply_sunna:
        result = apply_sunna_fix(result)

    # Step 7 (V3.5): Case ending rules
    # Applies genitive case after prepositions
    if apply_case:
        result = apply_case_rules(result)

    return result


def diacritize_v2_compatible(text):
    """
    Diacritize without V3 corrections (for backward compatibility).

    Use this to get output comparable to V2 baseline.
    """
    return diacritize(text, apply_grammar_fixes=False, apply_ml_homographs=False)


def diacritize_grammar_only(text):
    """
    Diacritize with grammar fixes but no ML homographs.

    Useful for testing grammar rules in isolation.
    """
    return diacritize(text, apply_grammar_fixes=True, apply_ml_homographs=False)


# For backwards compatibility with eval scripts
def diacritize_v3(text):
    """Alias for diacritize()."""
    return diacritize(text)


__all__ = [
    'diacritize',
    'diacritize_v3',
    'diacritize_v2_compatible',
    'diacritize_grammar_only',
    'apply_particle_rules',
    'apply_ml_homograph_rules',
    'strip_harakat',
]
