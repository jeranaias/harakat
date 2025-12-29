#!/usr/bin/env python3
"""
Train Active/Passive Voice Classifier

Uses context features to predict verb voice.
Focus on verbs that commonly appear in both forms.
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
from sklearn.metrics import classification_report

# Constants
HARAKAT = 'ًٌٍَُِّْ'
HARAKAT_SET = set(HARAKAT)


def strip_harakat(text):
    return ''.join(c for c in text if c not in HARAKAT_SET)


def create_context_string(example):
    """Create context features."""
    left = example.get('left', [])
    right = example.get('right', [])

    parts = []
    # Position-based features
    for i, w in enumerate(left):
        parts.append(f"L{len(left)-i}_{w}")
    for i, w in enumerate(right):
        parts.append(f"R{i+1}_{w}")

    return ' '.join(parts)


def train_voice_classifier(data_file, min_samples=50):
    """Train a voice classifier on ambiguous verbs."""
    print(f"\nLoading data from {data_file}...")

    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} examples")

    # Group by word base
    base_examples = {}
    for ex in data:
        base = ex['word_base']
        if base not in base_examples:
            base_examples[base] = []
        base_examples[base].append(ex)

    # Filter to bases with enough examples of each voice
    good_bases = []
    for base, exs in base_examples.items():
        voice_counts = Counter(ex['voice'] for ex in exs)
        if voice_counts.get('active', 0) >= min_samples and voice_counts.get('passive', 0) >= min_samples:
            good_bases.append(base)

    print(f"\nBases with >{min_samples} of each voice: {len(good_bases)}")
    for base in good_bases[:10]:
        exs = base_examples[base]
        vc = Counter(ex['voice'] for ex in exs)
        print(f"  {base}: {vc['active']} active, {vc['passive']} passive")

    # Create features for each base
    models = {}
    for base in good_bases:
        exs = base_examples[base]

        contexts = [create_context_string(ex) for ex in exs]
        labels = [ex['voice'] for ex in exs]

        # Vectorize
        vectorizer = CountVectorizer(
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),
            max_features=3000
        )

        try:
            X = vectorizer.fit_transform(contexts)
        except ValueError:
            continue

        y = np.array(labels)

        # Train
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')

        try:
            scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
            acc = scores.mean()
        except:
            acc = 0.5

        clf.fit(X, y)

        models[base] = {
            'vectorizer': vectorizer,
            'classifier': clf,
            'accuracy': float(acc),
            'examples': len(exs),
        }

        print(f"  {base}: {acc:.1%} accuracy ({len(exs)} examples)")

    return models


def train_global_classifier(data_file):
    """Train a single global classifier for all verbs."""
    print("\n" + "=" * 70)
    print("TRAINING GLOBAL VOICE CLASSIFIER")
    print("=" * 70)

    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"\nLoaded {len(data)} examples")

    # Balance the dataset
    active = [ex for ex in data if ex['voice'] == 'active']
    passive = [ex for ex in data if ex['voice'] == 'passive']

    print(f"Active: {len(active)}, Passive: {len(passive)}")

    # Downsample to balance
    min_size = min(len(active), len(passive))
    np.random.seed(42)
    active_sample = list(np.random.choice(active, min_size, replace=False))
    passive_sample = list(np.random.choice(passive, min_size, replace=False))
    balanced = active_sample + passive_sample
    np.random.shuffle(balanced)

    print(f"Balanced dataset: {len(balanced)} examples")

    # Create features
    def create_features(ex):
        """Create features including word base."""
        parts = [f"BASE_{ex['word_base']}"]
        for i, w in enumerate(ex.get('left', [])):
            parts.append(f"L{len(ex.get('left', []))-i}_{w}")
        for i, w in enumerate(ex.get('right', [])):
            parts.append(f"R{i+1}_{w}")
        return ' '.join(parts)

    contexts = [create_features(ex) for ex in balanced]
    labels = [ex['voice'] for ex in balanced]

    # Vectorize
    print("\nVectorizing...")
    vectorizer = CountVectorizer(
        min_df=5,
        max_df=0.95,
        ngram_range=(1, 2),
        max_features=10000
    )

    X = vectorizer.fit_transform(contexts)
    y = np.array(labels)

    print(f"Feature matrix: {X.shape}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    print("\nTraining classifier...")
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("\nTest set evaluation:")
    print(classification_report(y_test, y_pred))

    # Cross-validation
    print("\nCross-validation:")
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print(f"  Accuracy: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

    # Feature importance
    print("\nTop features for PASSIVE voice:")
    feature_names = vectorizer.get_feature_names_out()
    coef = clf.coef_[0]
    top_passive_idx = np.argsort(coef)[-10:]
    for idx in top_passive_idx[::-1]:
        print(f"  {feature_names[idx]}: {coef[idx]:.3f}")

    print("\nTop features for ACTIVE voice:")
    top_active_idx = np.argsort(coef)[:10]
    for idx in top_active_idx:
        print(f"  {feature_names[idx]}: {coef[idx]:.3f}")

    return {
        'vectorizer': vectorizer,
        'classifier': clf,
        'accuracy': float(scores.mean()),
    }


def main():
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 70)
    print("VOICE CLASSIFIER TRAINING")
    print("=" * 70)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, 'voice_data', 'voice_data.json')
    output_dir = os.path.join(os.path.dirname(script_dir), 'models', 'voice_classifier')
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(data_file):
        print(f"\nERROR: Data file not found: {data_file}")
        print("Run extract_voice_data.py first")
        return

    # Train global classifier
    global_model = train_global_classifier(data_file)

    # Save
    output_file = os.path.join(output_dir, 'voice_classifier.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(global_model, f)
    print(f"\nSaved global model to {output_file}")

    # Train per-word classifiers for top verbs
    print("\n" + "=" * 70)
    print("TRAINING PER-WORD CLASSIFIERS")
    print("=" * 70)

    word_models = train_voice_classifier(data_file, min_samples=100)

    # Save
    word_output_file = os.path.join(output_dir, 'voice_classifiers_by_word.pkl')
    with open(word_output_file, 'wb') as f:
        pickle.dump(word_models, f)
    print(f"\nSaved {len(word_models)} word-specific models to {word_output_file}")

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"\nGlobal classifier: {global_model['accuracy']:.1%} accuracy")
    print(f"Word-specific classifiers: {len(word_models)}")

    if word_models:
        avg_acc = np.mean([m['accuracy'] for m in word_models.values()])
        print(f"Average word-specific accuracy: {avg_acc:.1%}")


if __name__ == '__main__':
    main()
