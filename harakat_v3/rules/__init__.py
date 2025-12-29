"""
V3 Rules Module

Rule-based and ML-based corrections for Arabic diacritization.

Rules included:
1. Particle kasra (إلى، إلا، إذا، إذ) - Grammar-based
2. Homograph disambiguation (من، أم، ثم) - Context-based (rule-based, DISABLED)
3. ML-based homograph disambiguation - ML-based
4. ML-based voice correction - ML-based (active/passive verbs)
5. Anna fix (أنّ/أن shadda) - Rule-based for nominal contexts
6. Sunna fix (سُنَّة/سَنَة) - Rule-based for tradition vs year
7. Case ending rules (V3.5) - Grammar-based genitive after prepositions
8. Calibrated confidence (V3.5) - More aggressive ML thresholds
"""

from .particles import apply_particle_rules
from .homographs import apply_homograph_rules

# ML-based homograph disambiguation
try:
    from .homograph_ml import apply_ml_homograph_rules
except ImportError:
    def apply_ml_homograph_rules(text, **kwargs):
        return text

# ML-based voice correction
try:
    from .voice_ml import apply_voice_correction
except ImportError:
    def apply_voice_correction(text, **kwargs):
        return text

# Anna fix (أنّ/أن shadda disambiguation)
try:
    from .anna_fix import apply_anna_fix
except ImportError:
    def apply_anna_fix(text, **kwargs):
        return text

# Sunna fix (سُنَّة/سَنَة disambiguation)
try:
    from .sunna_fix import apply_sunna_fix
except ImportError:
    def apply_sunna_fix(text, **kwargs):
        return text

# V3.5: Case ending rules (genitive after prepositions)
try:
    from .case_rules import apply_case_rules
except ImportError:
    def apply_case_rules(text, **kwargs):
        return text

# V3.5: Calibrated confidence thresholds
try:
    from .confidence_calibrated import apply_calibrated_homograph_rules, get_calibrated_voice_threshold
except ImportError:
    def apply_calibrated_homograph_rules(text, **kwargs):
        return text
    def get_calibrated_voice_threshold(word_base):
        return 0.75

__all__ = [
    'apply_particle_rules',
    'apply_homograph_rules',
    'apply_ml_homograph_rules',
    'apply_voice_correction',
    'apply_anna_fix',
    'apply_sunna_fix',
    'apply_case_rules',
    'apply_calibrated_homograph_rules',
    'get_calibrated_voice_threshold',
]
