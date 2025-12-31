# Harakat V3.5 - ML-Based Error Correction

**V3.5 achieves 2.13% DER** by adding machine learning classifiers and rule-based corrections on top of the V2 neural pipeline.

## Overview

V3.5 systematically targets remaining error categories from V2:

| Category | V2 Errors | V3.5 Fix | Improvement |
|----------|-----------|----------|-------------|
| Case endings | 480 (0.55% DER) | Neural ensemble | -49% |
| Homographs | ~200 errors | ML classifiers | Context-aware |
| Voice (active/passive) | ~100 errors | ML classifiers | 73% accuracy |
| Particle kasra | ~150 errors | Rule-based | Grammar rules |
| Anna/Sunna patterns | ~50 errors | Pattern matching | Targeted fixes |

---

## Pipeline Components

### 1. V2 Neural Pipeline (Base)
The foundation - provides initial diacritization at 2.29% DER.

### 2. Grammar Rules (particles.py)
Fixes missing kasra on particles:
- إِلَى (to) - always kasra on hamza
- إِلَّا (except) - kasra + shadda
- إِذَا (when/if) - kasra on hamza
- إِذْ (since) - kasra on hamza

### 3. ML Homograph Disambiguation (homograph_ml.py)
TF-IDF + LogisticRegression classifiers for context-aware word sense:

| Homograph | Accuracy | Classes |
|-----------|----------|---------|
| قبل | 99.4% | preposition, noun |
| من | 92.2% | preposition (مِنْ), pronoun (مَنْ) |
| أم | 86.9% | conjunction (أَمْ), noun (أُمّ) |
| علم | 75.0% | verb_active, verb_passive |
| ذكر | 71.8% | verb_active, verb_passive, noun |

### 4. ML Voice Correction (voice_ml.py)
Active/passive verb disambiguation for 26 high-frequency verbs:
- فَعَلَ (fa'ala - he did) vs فُعِلَ (fu'ila - it was done)
- يَفْعَلُ (active) vs يُفْعَلُ (passive)

Top performers:
| Verb | Accuracy | Examples |
|------|----------|----------|
| يشترط | 89.1% | 659 |
| يرد | 86.4% | 235 |
| قبل | 86.2% | 253 |
| يقبل | 81.7% | 650 |

### 5. Anna Fix (anna_fix.py)
Disambiguates أَنَّ (that - with shadda) vs أَنْ (to - no shadda):
- Add shadda before definite nouns, prepositional phrases
- No shadda before present/past tense verbs

### 6. Sunna Fix (sunna_fix.py)
Disambiguates سُنَّة (tradition) vs سَنَة (year):
- سُنَّة when followed by مؤكدة، راتبة
- سَنَة otherwise

### 7. Calibrated Confidence (confidence_calibrated.py)
Per-classifier threshold tuning:
- High-accuracy classifiers (قبل 99.4%) → lower threshold (0.55)
- Medium-accuracy classifiers (من 92%) → moderate threshold (0.60)

---

## Results

### Final Performance

| Dataset | DER | DER (no case) | WER | WER (no case) |
|---------|-----|---------------|-----|---------------|
| Test | **2.13%** | 1.45% | 5.96% | 3.89% |
| Validation | **2.27%** | 1.90% | 7.20% | 5.90% |

### Version History

| Version | DER (Test) | Key Addition |
|---------|------------|--------------|
| V2 Baseline | 2.29% | Neural case predictor |
| V3.0 | 1.98% | Particle kasra rules |
| V3.1 | 1.97% | ML homograph disambiguation |
| V3.2 | 1.96% | ML voice correction |
| V3.3 | 1.95% | Anna fix |
| V3.4 | 1.94% | Sunna fix |
| **V3.5** | **2.13%** | Calibrated confidence |

---

## Architecture

```
Input Text
    │
    ▼
V2 Neural Pipeline (harakat_v2.py)
    │   ├── harakat_elite: Character-level transformer
    │   ├── Case Predictor: 3-model BiLSTM ensemble
    │   └── Hybrid Corrector: Internal vowel fixes
    │
    ▼
V3.5 Corrections Layer
    │   ├── Particle Rules: Grammar-based kasra
    │   ├── ML Homographs: TF-IDF + LogReg classifiers
    │   ├── ML Voice: Active/passive verb forms
    │   ├── Anna Fix: أَنَّ/أَنْ shadda
    │   └── Sunna Fix: سُنَّة/سَنَة pattern
    │
    ▼
Output (Fully Diacritized)
```

---

## Files

```
harakat_v3/
├── README.md                 # This file
├── __init__.py               # Main entry point
│
├── rules/                    # Correction modules
│   ├── particles.py          # Particle kasra rules
│   ├── homograph_ml.py       # ML homograph disambiguation
│   ├── voice_ml.py           # ML voice correction
│   ├── anna_fix.py           # أَنَّ/أَنْ disambiguation
│   ├── sunna_fix.py          # سُنَّة/سَنَة disambiguation
│   ├── case_rules.py         # Case ending rules (disabled)
│   └── confidence_calibrated.py  # Threshold tuning
│
├── models/                   # Trained classifiers
│   ├── homograph_classifiers/
│   │   ├── all_homograph_models.pkl
│   │   └── [per-word models]
│   └── voice_classifier/
│       └── voice_classifiers_by_word.pkl
│
├── analysis/                 # Error analysis tools
│   └── error_categories.py
│
└── training/                 # Training scripts (not in git)
    ├── extract_homographs.py
    ├── train_homograph_classifier.py
    └── [training data]
```

---

## Usage

### From harakat_v3 directly
```python
from harakat_v3 import diacritize
result = diacritize("ذهب الولد إلى المدرسة")
# ذَهَبَ الْوَلَدُ إِلَى الْمَدْرَسَةِ
```

### With options
```python
from harakat_v3 import diacritize

# Full V3.5 pipeline (default)
result = diacritize(text,
    apply_grammar_fixes=True,     # Particle kasra
    apply_ml_homographs=True,     # ML disambiguation
    apply_ml_voice=True,          # Voice correction
    apply_anna=True,              # Anna fix
    apply_sunna=True,             # Sunna fix
    use_calibrated=True           # Calibrated thresholds
)

# V2 only (no V3.5 corrections)
from harakat_v3 import diacritize_v2_compatible
result = diacritize_v2_compatible(text)
```

---

## Training Details

### Homograph Classifiers

Training data extracted from Tashkeela V2 (50K+ sentences):

| Homograph | Examples | Features |
|-----------|----------|----------|
| من | 44,697 | Bag-of-words + position |
| قبل | 4,894 | Context window ±3 words |
| ذكر | 2,134 | TF-IDF vectors |
| أم | 1,152 | Logistic regression |
| علم | 1,011 | L2 regularization |

### Voice Classifiers

309,374 verb examples extracted, 104,815 ambiguous (appear in both active/passive).

Global classifier: 79.7% accuracy
Word-specific classifiers: 73% average (26 verbs with >200 examples)

---

## Error Analysis

### Remaining Error Categories (after V3.5)

| Category | Errors | DER Impact |
|----------|--------|------------|
| Internal vowel | ~700 | 0.80% |
| Case vowel | ~400 | 0.46% |
| Shadda | ~150 | 0.17% |
| Sukun | ~200 | 0.23% |

### Top Remaining Problem Words

| Word | Errors | Issue |
|------|--------|-------|
| من | ~25 | Complex homograph |
| ذكر | ~15 | 4-way ambiguity |
| غير | ~12 | Noun/verb |
| شرط | ~10 | Form patterns |

---

## Future Improvements

### To reach sub-1% DER:

1. **Syntactic parser integration** - Case endings depend on grammatical role
2. **More verb patterns** - Forms II-X have predictable vowels
3. **Noun pattern rules** - فَاعِل، مَفْعُول patterns
4. **Larger context models** - Transformer-based classifiers

### Estimated impact:
- Syntax-aware case: -0.3% DER
- Verb patterns: -0.2% DER
- Noun patterns: -0.1% DER

---

## Notes

### What Works
- Conservative, high-confidence rules
- Linguistically-grounded patterns
- Targeted small models for specific problems
- Preserving V2 predictions when uncertain

### What Doesn't Work
- Aggressive corrections (cause regressions)
- Low-confidence threshold application
- Corpus-specific patterns (don't generalize)
- Rules without linguistic basis

### Design Principles
1. Every fix must improve BOTH validation and test sets
2. Precision > Recall (avoid regressions)
3. When uncertain, trust V2
4. Document linguistic basis for each rule
