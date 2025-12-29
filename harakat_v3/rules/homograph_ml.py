#!/usr/bin/env python3
"""
ML-Based Homograph Disambiguation for V3

Uses trained sklearn classifiers to disambiguate homographs based on context.

Key homographs:
- من: مِنْ (from) vs مَنْ (who) - 92% accuracy
- قبل: قَبْل (before) vs قِبَل (direction) - 99% accuracy
- أم: أَمْ (or) vs أُمّ (mother) - 87% accuracy
- علم: عَلِمَ (knew) vs عُلِمَ (was known) - 75% accuracy
- ذكر: ذَكَرَ (mentioned) vs ذُكِرَ (was mentioned) vs ذِكْر (noun) - 72% accuracy
"""

import sys
import os
import pickle

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

# Lazy-loaded models
_MODELS = None
_MODELS_LOADED = False


def strip_harakat(text):
    """Remove diacritics."""
    return ''.join(c for c in text if c not in HARAKAT_SET)


def load_models():
    """Load trained homograph models."""
    global _MODELS, _MODELS_LOADED

    if _MODELS_LOADED:
        return _MODELS

    script_dir = os.path.dirname(os.path.abspath(__file__))
    harakat_v3_dir = os.path.dirname(script_dir)
    model_file = os.path.join(harakat_v3_dir, 'models', 'homograph_classifiers', 'all_homograph_models.pkl')

    if not os.path.exists(model_file):
        _MODELS = {}
        _MODELS_LOADED = True
        return _MODELS

    try:
        with open(model_file, 'rb') as f:
            _MODELS = pickle.load(f)
        _MODELS_LOADED = True
    except Exception as e:
        print(f"Warning: Could not load homograph models: {e}")
        _MODELS = {}
        _MODELS_LOADED = True

    return _MODELS


def create_context_string(left_context, right_context):
    """Create a context string for classification."""
    context_parts = []
    for i, w in enumerate(left_context):
        w_base = strip_harakat(w)
        context_parts.append(f"L{len(left_context)-i}_{w_base}")
    for i, w in enumerate(right_context):
        w_base = strip_harakat(w)
        context_parts.append(f"R{i+1}_{w_base}")
    return ' '.join(context_parts)


# Correction mappings: (base, predicted_class) -> corrected form
CORRECTIONS = {
    'من': {
        'preposition': 'مِنْ',
        'pronoun': 'مَنْ',
    },
    'قبل': {
        'preposition': None,  # Keep case ending, just verify internal
        'noun': None,         # Keep case ending
    },
    'أم': {
        'conjunction': 'أَمْ',
        'noun': None,  # Has multiple case forms, don't override
    },
    'علم': {
        'verb_active': 'عَلِمَ',
        'verb_passive': 'عُلِمَ',
    },
    'ذكر': {
        'verb_active': 'ذَكَرَ',
        'verb_passive': 'ذُكِرَ',
        'noun': None,       # Has case forms
        'noun_male': None,  # Has case forms
    },
}

# Minimum confidence to apply correction
MIN_CONFIDENCE = {
    'من': 0.70,    # High impact, be confident
    'قبل': 0.90,   # Very accurate model
    'أم': 0.75,    # Moderate confidence
    'علم': 0.80,   # Be conservative
    'ذكر': 0.80,   # 4-class, be conservative
}


def predict_homograph(word_base, left_context, right_context):
    """
    Predict the correct diacritization for a homograph.

    Args:
        word_base: Undiacritized word
        left_context: List of preceding words (diacritized)
        right_context: List of following words (diacritized)

    Returns:
        (predicted_class, confidence, corrected_form) or (None, 0, None)
    """
    models = load_models()

    if word_base not in models:
        return None, 0.0, None

    model_info = models[word_base]
    vectorizer = model_info['vectorizer']
    classifier = model_info['classifier']

    # Create context string
    context_str = create_context_string(left_context, right_context)

    try:
        X = vectorizer.transform([context_str])
        proba = classifier.predict_proba(X)[0]
        predicted_idx = proba.argmax()
        confidence = proba[predicted_idx]
        predicted_class = classifier.classes_[predicted_idx]
    except Exception as e:
        return None, 0.0, None

    # Get correction
    corrected_form = None
    if word_base in CORRECTIONS and predicted_class in CORRECTIONS[word_base]:
        corrected_form = CORRECTIONS[word_base][predicted_class]

    return str(predicted_class), float(confidence), corrected_form


def apply_ml_homograph_rules(text, min_confidence_override=None):
    """
    Apply ML-based homograph disambiguation to diacritized text.

    Args:
        text: Diacritized text
        min_confidence_override: Override default confidence thresholds

    Returns:
        Text with homograph corrections applied
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

        # Check if this is a homograph we handle
        if word_base not in models:
            result.append(word)
            continue

        # Get context
        context_size = 3
        left_context = words[max(0, i - context_size):i]
        right_context = words[i + 1:i + 1 + context_size]

        # Predict
        predicted_class, confidence, corrected_form = predict_homograph(
            word_base, left_context, right_context
        )

        if predicted_class is None:
            result.append(word)
            continue

        # Check confidence threshold
        min_conf = min_confidence_override or MIN_CONFIDENCE.get(word_base, 0.75)
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


def test_ml_homograph():
    """Test ML homograph disambiguation."""
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 60)
    print("ML HOMOGRAPH DISAMBIGUATION TEST")
    print("=" * 60)

    models = load_models()
    print(f"\nLoaded models: {list(models.keys())}")

    test_cases = [
        # من tests
        ("جَاءَ مَنْ الْبَيْتِ", "من after جاء, before البيت → مِنْ (from)"),
        ("مِنْ يَفْعَلُ هَذَا", "من before يفعل → مَنْ (who)"),

        # أم tests
        ("هَلْ أَنْتَ أُمْ هُوَ", "أم in question → أَمْ (or)"),

        # ذكر tests
        ("ثُمَّ ذَكَرَ الْمُصَنِّفُ", "ذكر after ثم → ذَكَرَ (mentioned)"),
        ("مِمَّا ذُكِرَ أَعْلَاهُ", "ذكر after مما → ذُكِرَ (was mentioned)"),
    ]

    for text, description in test_cases:
        print(f"\n{description}")
        print(f"  Input: {text}")
        result = apply_ml_homograph_rules(text)
        print(f"  Output: {result}")

        # Show predictions
        words = text.split()
        for i, word in enumerate(words):
            word_base = strip_harakat(word)
            if word_base in models:
                left = words[max(0, i-3):i]
                right = words[i+1:i+4]
                pred_class, conf, corr = predict_homograph(word_base, left, right)
                print(f"  -> {word_base}: {pred_class} ({conf:.2f}) -> {corr or 'keep'}")


if __name__ == '__main__':
    test_ml_homograph()
