#!/usr/bin/env python3
"""
Confidence-Calibrated ML Corrections for V3.5

Strategy 3: More aggressive thresholds based on model accuracy.

Model Accuracies (from training):
- قبل: 99.4% → threshold 0.55 (was 0.90)
- من: 92.2% → threshold 0.60 (was 0.70)
- أم: 86.9% → threshold 0.70 (was 0.75)
- علم: 75.0% → threshold 0.75 (was 0.80)
- ذكر: 71.8% → threshold 0.75 (was 0.80)

Voice classifiers:
- يشترط: 89.1% → threshold 0.65
- يرد: 86.4% → threshold 0.65
- قبل: 86.2% → threshold 0.65
- others: 0.70 (was 0.75)
"""

import sys
import os

if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')

# Import the base modules
script_dir = os.path.dirname(os.path.abspath(__file__))
harakat_v3_dir = os.path.dirname(script_dir)
harakat_dir = os.path.dirname(harakat_v3_dir)

from .homograph_ml import load_models, predict_homograph, strip_harakat, CORRECTIONS

# Calibrated confidence thresholds (lowered based on model accuracy)
CALIBRATED_CONFIDENCE = {
    'من': 0.60,    # 92.2% accuracy - was 0.70
    'قبل': 0.55,   # 99.4% accuracy - was 0.90
    'أم': 0.70,    # 86.9% accuracy - was 0.75
    'علم': 0.75,   # 75.0% accuracy - was 0.80
    'ذكر': 0.75,   # 71.8% accuracy - was 0.80
}

# High-accuracy voice models get lower thresholds
VOICE_HIGH_ACCURACY = {
    'يشترط': 0.65,  # 89.1% accuracy
    'يرد': 0.65,    # 86.4% accuracy
    'قبل': 0.65,    # 86.2% accuracy
    'يقبل': 0.68,   # 81.7% accuracy
    'يحرم': 0.68,   # 80.2% accuracy
}


def apply_calibrated_homograph_rules(text):
    """
    Apply ML homograph rules with calibrated (lower) confidence thresholds.

    This is more aggressive than the default and should catch more errors.
    """
    models = load_models()
    if not models:
        return text

    words = text.split()
    if len(words) < 1:
        return text

    result = []
    corrections = 0

    for i, word in enumerate(words):
        word_base = strip_harakat(word)

        if word_base not in models:
            result.append(word)
            continue

        # Get context
        context_size = 3
        left_context = words[max(0, i - context_size):i]
        right_context = words[i + 1:i + 1 + context_size]

        # Predict with calibrated threshold
        predicted_class, confidence, corrected_form = predict_homograph(
            word_base, left_context, right_context
        )

        if predicted_class is None:
            result.append(word)
            continue

        # Use calibrated confidence threshold
        min_conf = CALIBRATED_CONFIDENCE.get(word_base, 0.70)
        if confidence < min_conf:
            result.append(word)
            continue

        # Apply correction if available
        if corrected_form:
            result.append(corrected_form)
            corrections += 1
        else:
            result.append(word)

    return ' '.join(result)


def get_calibrated_voice_threshold(word_base):
    """Get calibrated threshold for voice correction."""
    return VOICE_HIGH_ACCURACY.get(word_base, 0.70)


if __name__ == '__main__':
    print("Confidence Calibrated Module")
    print(f"Homograph thresholds: {CALIBRATED_CONFIDENCE}")
    print(f"Voice thresholds: {VOICE_HIGH_ACCURACY}")
