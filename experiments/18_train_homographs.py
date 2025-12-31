#!/usr/bin/env python3
"""
Experiment 18: Train New Homograph Classifiers

Train classifiers for:
1. ثم - ثُمَّ (then, conjunction) vs ثَمَّ (there, adverb)
2. أم - أُمّ (mother) vs أَمْ (or, question particle)
"""
import sys
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
import numpy as np

HARAKAT = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652')

def strip_h(text):
    return ''.join(c for c in text if c not in HARAKAT)


def extract_training_data(corpus_file, target_word, class_patterns):
    """
    Extract training examples for a homograph.

    class_patterns: dict mapping class_name -> list of diacritic patterns that indicate this class
    """
    with open(corpus_file, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    examples = []  # (context_string, class_label)

    for line in lines:
        words = line.split()
        for i, word in enumerate(words):
            base = strip_h(word)
            if base != target_word:
                continue

            # Determine class from diacritics
            word_class = None
            for class_name, patterns in class_patterns.items():
                for pattern in patterns:
                    if pattern in word:
                        word_class = class_name
                        break
                if word_class:
                    break

            if not word_class:
                continue

            # Extract context
            left = [strip_h(w) for w in words[max(0, i-3):i]]
            right = [strip_h(w) for w in words[i+1:i+4]]

            # Create context string
            context_parts = []
            for j, w in enumerate(left):
                context_parts.append(f"L{len(left)-j}_{w}")
            for j, w in enumerate(right):
                context_parts.append(f"R{j+1}_{w}")

            context_str = ' '.join(context_parts)
            examples.append((context_str, word_class, word))

    return examples


def train_classifier(examples, word_name):
    """Train and evaluate a classifier."""
    if len(examples) < 10:
        print(f"  Not enough examples for {word_name}: {len(examples)}")
        return None

    # Split by class
    by_class = {}
    for ctx, cls, word in examples:
        by_class[cls] = by_class.get(cls, [])
        by_class[cls].append((ctx, word))

    print(f"\n  Class distribution for {word_name}:")
    for cls, items in sorted(by_class.items()):
        print(f"    {cls}: {len(items)} examples")
        # Show sample
        for ctx, word in items[:2]:
            print(f"      e.g. {word}")

    # Check if we have enough of each class
    min_class_size = min(len(items) for items in by_class.values())
    if min_class_size < 5:
        print(f"  Warning: smallest class has only {min_class_size} examples")

    # Prepare data
    X_text = [ctx for ctx, cls, word in examples]
    y = [cls for ctx, cls, word in examples]

    # Vectorize
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500)
    X = vectorizer.fit_transform(X_text)

    # Train with calibration
    base_clf = LogisticRegression(max_iter=1000, C=1.0)

    # Cross-validation
    if len(set(y)) > 1 and min_class_size >= 3:
        cv_folds = min(5, min_class_size)
        scores = cross_val_score(base_clf, X, y, cv=cv_folds)
        print(f"  Cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

    # Train final model with calibration
    clf = CalibratedClassifierCV(base_clf, cv=3 if min_class_size >= 3 else 2)
    clf.fit(X, y)

    return {
        'vectorizer': vectorizer,
        'classifier': clf,
        'classes': list(set(y)),
    }


def main():
    train_file = r'c:\tashkeel\tashkeela_v2\tashkeela_train.txt'
    output_dir = r'c:\tashkeel\harakat\harakat_v3\models\homograph_classifiers'

    # ثم classifier: ثُمَّ (then) vs ثَمَّ (there)
    print("=" * 60)
    print("Training ثم classifier (then vs there)")
    print("=" * 60)

    thumma_patterns = {
        'thumma': ['\u062b\u064f\u0645\u0651', '\u062b\u064f\u0645\u0651\u064e'],  # ثُمَّ (then) - damma on tha
        'thamma': ['\u062b\u064e\u0645\u0651', '\u062b\u064e\u0645\u0651\u064e'],  # ثَمَّ (there) - fatha on tha
    }

    thumma_examples = extract_training_data(train_file, 'ثم', thumma_patterns)
    print(f"Found {len(thumma_examples)} examples for ثم")

    thumma_model = train_classifier(thumma_examples, 'ثم')

    if thumma_model:
        model_file = os.path.join(output_dir, 'thumma_classifier.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(thumma_model, f)
        print(f"  Saved to {model_file}")

    # أم classifier: أُمّ (mother) vs أَمْ (or)
    print("\n" + "=" * 60)
    print("Training أم classifier (mother vs or)")
    print("=" * 60)

    umm_patterns = {
        'umm': ['\u0623\u064f\u0645\u0651', '\u0623\u064f\u0645\u0650\u0651'],  # أُمّ (mother) - damma+shadda
        'am': ['\u0623\u064e\u0645\u0652', '\u0623\u064e\u0645'],  # أَمْ (or) - fatha, maybe sukun
    }

    umm_examples = extract_training_data(train_file, 'أم', umm_patterns)
    print(f"Found {len(umm_examples)} examples for أم")

    umm_model = train_classifier(umm_examples, 'أم')

    if umm_model:
        model_file = os.path.join(output_dir, 'umm_classifier.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(umm_model, f)
        print(f"  Saved to {model_file}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
