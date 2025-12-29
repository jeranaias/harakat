#!/usr/bin/env python3
"""
Train Homograph Disambiguation Classifier

Uses a simple bag-of-words approach with context features.
For each homograph, trains a separate classifier.

This is intentionally simple to avoid overfitting:
- Bag of words from surrounding context
- No deep learning (sklearn classifiers)
- Focus on high-precision rules
"""

import sys
import os
import json
import pickle
from collections import Counter

if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Constants
HARAKAT = 'ًٌٍَُِّْ'
HARAKAT_SET = set(HARAKAT)


def strip_harakat(text):
    """Remove diacritics."""
    return ''.join(c for c in text if c not in HARAKAT_SET)


def create_context_string(example, undiacritize=True):
    """Create a context string from an example."""
    left = example.get('left', [])
    right = example.get('right', [])

    if undiacritize:
        left = [strip_harakat(w) for w in left]
        right = [strip_harakat(w) for w in right]

    # Use position markers
    context_parts = []
    for i, w in enumerate(left):
        context_parts.append(f"L{len(left)-i}_{w}")
    for i, w in enumerate(right):
        context_parts.append(f"R{i+1}_{w}")

    return ' '.join(context_parts)


def train_classifier(data_file, min_class_samples=20):
    """
    Train a classifier for one homograph.

    Args:
        data_file: Path to JSON data file
        min_class_samples: Minimum samples per class to train

    Returns:
        Trained model dict or None if insufficient data
    """
    print(f"\nTraining on: {os.path.basename(data_file)}")

    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if len(data) < 50:
        print(f"  Skipping: only {len(data)} examples (need 50+)")
        return None

    # Filter classes with enough samples
    class_counts = Counter(ex['class'] for ex in data)
    valid_classes = {cls for cls, cnt in class_counts.items() if cnt >= min_class_samples}

    if len(valid_classes) < 2:
        print(f"  Skipping: only {len(valid_classes)} valid classes")
        return None

    # Filter data to valid classes
    filtered_data = [ex for ex in data if ex['class'] in valid_classes]
    print(f"  Classes: {dict(Counter(ex['class'] for ex in filtered_data))}")

    # Create features
    contexts = [create_context_string(ex) for ex in filtered_data]
    labels = [ex['class'] for ex in filtered_data]

    # Vectorize
    vectorizer = CountVectorizer(
        min_df=3,
        max_df=0.9,
        ngram_range=(1, 2),
        max_features=5000
    )

    try:
        X = vectorizer.fit_transform(contexts)
    except ValueError as e:
        print(f"  Vectorization error: {e}")
        return None

    y = np.array(labels)

    # Train/test split
    if len(np.unique(y)) < 2:
        print(f"  Skipping: only one class after filtering")
        return None

    # Cross-validation score
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0)

    try:
        scores = cross_val_score(clf, X, y, cv=min(5, len(y) // 10), scoring='accuracy')
        print(f"  Cross-val accuracy: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
    except Exception as e:
        print(f"  CV error: {e}")
        scores = np.array([0.5])

    # Train final model
    clf.fit(X, y)

    # Get feature importance
    if hasattr(clf, 'coef_'):
        feature_names = vectorizer.get_feature_names_out()
        for i, cls in enumerate(clf.classes_):
            if len(clf.classes_) == 2 and i == 0:
                continue  # Binary case
            if len(clf.coef_.shape) > 1:
                coef = clf.coef_[i] if i < len(clf.coef_) else clf.coef_[0]
            else:
                coef = clf.coef_
            top_idx = np.argsort(coef)[-5:]
            top_features = [(feature_names[j], coef[j]) for j in top_idx]
            print(f"  Top features for {cls}: {top_features}")

    return {
        'vectorizer': vectorizer,
        'classifier': clf,
        'classes': list(clf.classes_),
        'accuracy': float(scores.mean()),
        'examples': len(filtered_data),
    }


def main():
    # Fix Windows encoding
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 70)
    print("HOMOGRAPH CLASSIFIER TRAINING")
    print("=" * 70)

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'homograph_data')
    output_dir = os.path.join(os.path.dirname(script_dir), 'models', 'homograph_classifiers')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Train classifiers for each homograph
    models = {}
    for filename in os.listdir(data_dir):
        if not filename.endswith('_data.json'):
            continue

        homograph = filename.replace('_data.json', '')
        data_file = os.path.join(data_dir, filename)

        model = train_classifier(data_file)
        if model:
            models[homograph] = model

            # Save individual model
            model_file = os.path.join(output_dir, f"{homograph}_model.pkl")
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            print(f"  Saved: {model_file}")

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)

    for homograph, model in sorted(models.items(), key=lambda x: -x[1]['accuracy']):
        print(f"  {homograph}: {model['accuracy']:.1%} accuracy on {model['examples']} examples")
        print(f"    Classes: {model['classes']}")

    # Save combined models
    combined_file = os.path.join(output_dir, 'all_homograph_models.pkl')
    with open(combined_file, 'wb') as f:
        pickle.dump(models, f)
    print(f"\nSaved combined models: {combined_file}")


if __name__ == '__main__':
    main()
