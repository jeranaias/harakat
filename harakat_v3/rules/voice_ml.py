#!/usr/bin/env python3
"""
ML-Based Active/Passive Voice Disambiguation for V3

Uses trained classifiers to disambiguate verb voice.
Applies to verbs that commonly appear in both active and passive forms.
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
_VOICE_MODELS = None
_VOICE_MODELS_LOADED = False


def strip_harakat(text):
    """Remove diacritics."""
    return ''.join(c for c in text if c not in HARAKAT_SET)


def load_voice_models():
    """Load trained voice models."""
    global _VOICE_MODELS, _VOICE_MODELS_LOADED

    if _VOICE_MODELS_LOADED:
        return _VOICE_MODELS

    script_dir = os.path.dirname(os.path.abspath(__file__))
    harakat_v3_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(harakat_v3_dir, 'models', 'voice_classifier')

    _VOICE_MODELS = {
        'global': None,
        'by_word': {},
    }

    # Load global model
    global_file = os.path.join(models_dir, 'voice_classifier.pkl')
    if os.path.exists(global_file):
        try:
            with open(global_file, 'rb') as f:
                _VOICE_MODELS['global'] = pickle.load(f)
        except Exception as e:
            pass

    # Load word-specific models
    word_file = os.path.join(models_dir, 'voice_classifiers_by_word.pkl')
    if os.path.exists(word_file):
        try:
            with open(word_file, 'rb') as f:
                _VOICE_MODELS['by_word'] = pickle.load(f)
        except Exception as e:
            pass

    _VOICE_MODELS_LOADED = True
    return _VOICE_MODELS


def create_context_string(left_context, right_context, word_base):
    """Create context string for classification."""
    parts = [f"BASE_{word_base}"]
    for i, w in enumerate(left_context):
        w_base = strip_harakat(w) if any(c in HARAKAT_SET for c in w) else w
        parts.append(f"L{len(left_context)-i}_{w_base}")
    for i, w in enumerate(right_context):
        w_base = strip_harakat(w) if any(c in HARAKAT_SET for c in w) else w
        parts.append(f"R{i+1}_{w_base}")
    return ' '.join(parts)


def extract_diacritics(word):
    """Extract (consonant, diacritics) pairs."""
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


def is_potential_verb(word, word_base):
    """Check if word could be a verb we want to correct."""
    if len(word_base) < 3:
        return False

    pairs = extract_diacritics(word)
    if len(pairs) < 3:
        return False

    # Check for verb-like patterns
    c1, d1 = pairs[0]
    c2, d2 = pairs[1]
    c3, d3 = pairs[2] if len(pairs) > 2 else ('', '')

    # Past tense: فَعَلَ or فُعِلَ pattern
    if len(pairs) >= 3:
        # Active past: fatha on first letter
        if FATHA in d1 and FATHA in d2:
            return True
        # Passive past: damma on first letter
        if DAMMA in d1 and KASRA in d2:
            return True

    # Present tense: يَفْعَلُ or يُفْعَلُ pattern
    if c1 in 'يتنأ':
        if FATHA in d1 or DAMMA in d1:
            return True

    return False


def convert_to_active(word, word_base):
    """Convert a verb to active voice."""
    pairs = extract_diacritics(word)
    if len(pairs) < 3:
        return word

    c1, d1 = pairs[0]

    # Past tense passive → active: فُعِلَ → فَعَلَ
    if DAMMA in d1:
        # Replace damma with fatha on first consonant
        new_word = []
        for i, (c, d) in enumerate(pairs):
            if i == 0:
                # First consonant: damma → fatha
                new_d = d.replace(DAMMA, FATHA)
                new_word.append(c + new_d)
            elif i == 1:
                # Second consonant: kasra → fatha (for past) or keep shadda
                if SHADDA in d:
                    new_d = d.replace(KASRA, FATHA)
                else:
                    new_d = d.replace(KASRA, FATHA)
                new_word.append(c + new_d)
            else:
                new_word.append(c + d)
        return ''.join(new_word)

    # Present tense: يُفْعَلُ → يَفْعَلُ
    if c1 in 'يتنأ' and DAMMA in d1:
        new_word = []
        for i, (c, d) in enumerate(pairs):
            if i == 0:
                new_d = d.replace(DAMMA, FATHA)
                new_word.append(c + new_d)
            else:
                new_word.append(c + d)
        return ''.join(new_word)

    return word


def convert_to_passive(word, word_base):
    """Convert a verb to passive voice."""
    pairs = extract_diacritics(word)
    if len(pairs) < 3:
        return word

    c1, d1 = pairs[0]

    # Past tense active → passive: فَعَلَ → فُعِلَ
    if FATHA in d1 and c1 not in 'يتنأ':
        new_word = []
        for i, (c, d) in enumerate(pairs):
            if i == 0:
                # First consonant: fatha → damma
                new_d = d.replace(FATHA, DAMMA)
                new_word.append(c + new_d)
            elif i == 1:
                # Second consonant: fatha → kasra (keep shadda)
                if SHADDA in d:
                    new_d = d.replace(FATHA, KASRA)
                else:
                    new_d = d.replace(FATHA, KASRA)
                new_word.append(c + new_d)
            else:
                new_word.append(c + d)
        return ''.join(new_word)

    # Present tense: يَفْعَلُ → يُفْعَلُ
    if c1 in 'يتنأ' and FATHA in d1:
        new_word = []
        for i, (c, d) in enumerate(pairs):
            if i == 0:
                new_d = d.replace(FATHA, DAMMA)
                new_word.append(c + new_d)
            else:
                new_word.append(c + d)
        return ''.join(new_word)

    return word


def get_current_voice(word, word_base):
    """Detect current voice of a verb."""
    pairs = extract_diacritics(word)
    if len(pairs) < 3:
        return None

    c1, d1 = pairs[0]
    c2, d2 = pairs[1]

    # Check first consonant diacritics
    if c1 in 'يتنأ':
        # Present tense
        if FATHA in d1:
            return 'active'
        if DAMMA in d1:
            return 'passive'
    else:
        # Past tense
        if FATHA in d1 and FATHA in d2:
            return 'active'
        if FATHA in d1 and KASRA in d2:
            return 'active'
        if DAMMA in d1 and KASRA in d2:
            return 'passive'

    return None


def predict_voice(word_base, left_context, right_context):
    """
    Predict the correct voice for a verb.

    Returns: ('active'|'passive', confidence) or (None, 0)
    """
    models = load_voice_models()

    if not models['global'] and not models['by_word']:
        return None, 0.0

    # Try word-specific model first
    if word_base in models['by_word']:
        model = models['by_word'][word_base]
        vectorizer = model['vectorizer']
        classifier = model['classifier']

        # Create features without BASE_ for word-specific
        parts = []
        for i, w in enumerate(left_context):
            w_base = strip_harakat(w) if any(c in HARAKAT_SET for c in w) else w
            parts.append(f"L{len(left_context)-i}_{w_base}")
        for i, w in enumerate(right_context):
            w_base = strip_harakat(w) if any(c in HARAKAT_SET for c in w) else w
            parts.append(f"R{i+1}_{w_base}")
        context_str = ' '.join(parts)

        try:
            X = vectorizer.transform([context_str])
            proba = classifier.predict_proba(X)[0]
            pred_idx = proba.argmax()
            confidence = proba[pred_idx]
            pred_voice = classifier.classes_[pred_idx]
            return str(pred_voice), float(confidence)
        except:
            pass

    # Fall back to global model
    if models['global']:
        model = models['global']
        vectorizer = model['vectorizer']
        classifier = model['classifier']

        context_str = create_context_string(left_context, right_context, word_base)

        try:
            X = vectorizer.transform([context_str])
            proba = classifier.predict_proba(X)[0]
            pred_idx = proba.argmax()
            confidence = proba[pred_idx]
            pred_voice = classifier.classes_[pred_idx]
            return str(pred_voice), float(confidence)
        except:
            pass

    return None, 0.0


# Verbs we're confident about correcting
# These have high accuracy word-specific models
CONFIDENT_VERBS = {
    'ذكر', 'علم', 'يرد', 'يشترط', 'قبل', 'يحرم', 'يقبل', 'قدم', 'يقتل',
}

# Minimum confidence thresholds
MIN_CONFIDENCE = 0.75


def apply_voice_correction(text, min_confidence=MIN_CONFIDENCE):
    """
    Apply voice correction to diacritized text.

    Only corrects verbs where we have high confidence.
    """
    models = load_voice_models()
    if not models['global'] and not models['by_word']:
        return text

    words = text.split()
    if len(words) < 1:
        return text

    result = []
    corrections = 0

    for i, word in enumerate(words):
        word_base = strip_harakat(word)

        # Only correct verbs we're confident about
        if word_base not in models['by_word']:
            result.append(word)
            continue

        # Check if it's a verb pattern
        if not is_potential_verb(word, word_base):
            result.append(word)
            continue

        # Get context
        context_size = 3
        left_context = words[max(0, i - context_size):i]
        right_context = words[i + 1:i + 1 + context_size]

        # Predict voice
        pred_voice, confidence = predict_voice(word_base, left_context, right_context)

        if pred_voice is None or confidence < min_confidence:
            result.append(word)
            continue

        # Get current voice
        current_voice = get_current_voice(word, word_base)

        if current_voice is None or current_voice == pred_voice:
            result.append(word)
            continue

        # Apply correction
        if pred_voice == 'active':
            corrected = convert_to_active(word, word_base)
        else:
            corrected = convert_to_passive(word, word_base)

        if corrected != word:
            result.append(corrected)
            corrections += 1
        else:
            result.append(word)

    return ' '.join(result)


def test_voice_correction():
    """Test voice correction."""
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 60)
    print("VOICE CORRECTION TEST")
    print("=" * 60)

    models = load_voice_models()
    print(f"\nLoaded models:")
    print(f"  Global: {'Yes' if models['global'] else 'No'}")
    print(f"  Word-specific: {len(models['by_word'])} verbs")

    test_cases = [
        # ذكر tests
        ("ثُمَّ ذَكَرَ الْمُصَنِّفُ", "ذكر after ثم → active"),
        ("مِمَّا ذُكِرَ أَعْلَاهُ", "ذكر after مما → passive"),
    ]

    for text, description in test_cases:
        print(f"\n{description}")
        print(f"  Input:  {text}")
        result = apply_voice_correction(text)
        print(f"  Output: {result}")

        # Show prediction
        words = text.split()
        for i, word in enumerate(words):
            word_base = strip_harakat(word)
            if word_base in models['by_word']:
                left = words[max(0, i-3):i]
                right = words[i+1:i+4]
                pred, conf = predict_voice(word_base, left, right)
                current = get_current_voice(word, word_base)
                print(f"  -> {word_base}: current={current}, predicted={pred} ({conf:.2f})")


if __name__ == '__main__':
    test_voice_correction()
