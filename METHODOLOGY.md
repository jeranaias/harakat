# Harakat Project Methodology

**A Comprehensive Guide to Arabic Diacritization Development**

This document captures our complete methodology, experimental procedures, evaluation protocols, and lessons learned throughout the Harakat project. It serves as the authoritative reference for understanding what we've built, why we made certain decisions, and how to continue development.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Evaluation Protocol](#evaluation-protocol)
3. [Data Resources](#data-resources)
4. [Architecture Evolution](#architecture-evolution)
5. [Experimental Procedures](#experimental-procedures)
6. [Lessons Learned](#lessons-learned)
7. [Generalization Guidelines](#generalization-guidelines)
8. [Code Organization](#code-organization)
9. [Reproducibility Checklist](#reproducibility-checklist)

---

## Project Overview

### Goal
Build the most accurate Arabic diacritization system possible while maintaining:
- Self-contained deployment (single file, no external dependencies beyond PyTorch)
- Reasonable inference speed (~5 lines/second acceptable)
- Minimal model size (<10MB compressed)

### Metrics

**Primary**: DER (Diacritic Error Rate)
- Calculated per base character position
- Each Arabic letter can have 0-2 diacritics (vowel + optional shadda)
- Formula: `DER = errors / total_diacritic_positions`

**Secondary**:
- WER (Word Error Rate): Any error in word = word error
- DER without case: Excludes final position (case ending) errors

### Current State (as of 2024-12-18)

| Version | DER (with case) | DER (no case) | WER |
|---------|-----------------|---------------|-----|
| V1 (harakat_elite) | 4.35% | 2.27% | 11.82% |
| V2 (neural pipeline) | 2.29% | 1.95% | 6.44% |
| V2 + Final Sweep | 2.23% | 1.95% | 6.23% |
| V3 Target | <1.50% | <1.30% | <4.00% |

---

## Evaluation Protocol

### CRITICAL: Consistent Evaluation

All DER measurements MUST use the same evaluation code and dataset to be comparable.

### Standard Test Set

**File**: `tashkeela_test.txt` (2,500 lines)
**Location**: `c:\tashkeel\paper_eval\tashkeela_test.txt` (canonical)
**Copies**: Also in `harakat\benchmarks\` for convenience

### Validation Set (CRITICAL FOR GENERALIZATION)

**File**: `tashkeela_val.txt` (5,000 lines)
**Purpose**: True measure of generalization

**IMPORTANT FINDING (2024-12-18)**:
```
V2 + Sweep Results:
- Test set:       2.23% DER
- Validation set: 3.05% DER
- Gap:            -0.82%
```

The test set shows significantly better performance than validation. This means:
1. **Always evaluate on BOTH sets** before merging any change
2. **Validation is the true measure** of how well the model generalizes
3. If test improves but validation doesn't, the change is overfitting
4. V3 improvements should target validation DER, not just test DER

**DO NOT** evaluate final metrics on:
- `tashkeela_train.txt` - training data, never for evaluation
- Any subset of test data
- Test set alone without validation check

### DER Calculation Code

The canonical DER calculation is in `paper_eval/evaluate_all_models.py`. Key functions:

```python
HARAKAT = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670')

def extract_diacritic_sequence(word):
    """Extract base chars and their associated diacritics."""
    base_chars = []
    diacritics = []
    current_diacritics = []

    for c in word:
        if c in HARAKAT:
            current_diacritics.append(c)
        else:
            if base_chars:
                diacritics.append(''.join(sorted(current_diacritics)))
            base_chars.append(c)
            current_diacritics = []

    if base_chars:
        diacritics.append(''.join(sorted(current_diacritics)))

    return base_chars, diacritics
```

**Important details**:
1. Diacritics are sorted before comparison (handles shadda+vowel order)
2. Each base character gets one diacritic slot (may be empty string)
3. Multi-diacritic combinations (shadda+vowel) are treated as single unit
4. Word alignment handles tokenization differences

### Evaluation Procedure

```python
# Standard evaluation loop
for gold_line in test_lines:
    undiac = strip_harakat(gold_line)
    pred_line = model.diacritize(undiac)

    # Word-by-word comparison with alignment
    # ... (see full code in paper_eval/)
```

### What to Report

Always report:
1. DER with case endings
2. DER without case endings
3. Total errors / total positions
4. Test set used (should always be tashkeela_test.txt, 2500 lines)

---

## Data Resources

### Tashkeela Corpus

**Source**: Tashkeela project (classical Arabic texts)
**Contents**: Quran, Hadith, classical poetry, literature

| Split | Lines | Purpose |
|-------|-------|---------|
| Train | ~150,000 | Model training |
| Validation | 5,000 | Hyperparameter tuning |
| Test | 2,500 | Final evaluation only |

**Location**: `c:\tashkeel\` (root level)

### Vocabulary Files

- `harakat_lzma.bin`: V1 compressed vocabulary (3.1 MB)
- Located with `harakat.py` in `harakat/` folder

### Model Checkpoints

Training checkpoints stored in:
- `neural model sweep/models/` - raw training outputs
- `harakat/harakat_v2/archive/models/` - production models

---

## Architecture Evolution

### V1: Statistical Model (harakat_elite)

**Approach**: N-gram language models + morphological analysis

**Components**:
1. 1-5 gram character language models
2. DAWG-based morphological analyzer
3. Context-aware disambiguation
4. Viterbi decoding for optimal path

**Strengths**:
- Very fast (~280 lines/second)
- Small model size (3.1 MB compressed)
- Good on common patterns

**Weaknesses**:
- Poor on grammatical case endings (requires syntax)
- Can't learn new patterns without retraining vocabulary
- No sentence-level context

**Result**: 4.35% DER

### V2: Neural Enhancement Pipeline

**Key Insight**: Don't replace V1, enhance it. V1 is already good at common patterns.

**Architecture**:
```
Input → V1 Base → Case V3 Ensemble → Hybrid LSTM → Output
         4.35%        ↓                    ↓
                  Fix case endings    Fix internal vowels
                     ~2.5%              ~2.29%
```

#### Stage 1: V1 Base
- Same as standalone V1
- Provides strong baseline predictions

#### Stage 2: Case V3 Ensemble
**Purpose**: Fix case ending (final position) errors

**Architecture** (each of 3 models):
- Character embedding: 64-dim
- Word BiLSTM: 128 hidden
- Context BiLSTM: 2-layer, 128 hidden
- Multi-head attention: 4 heads
- Classifier: 256 → 128 → 64 → 8 classes

**Training**:
- Data: 90% of Tashkeela (~150K sentences)
- Epochs: 30
- Batch size: 64
- Learning rate: 1e-3 with ReduceLROnPlateau

**Ensemble**: Weighted soft voting (0.35, 0.35, 0.30)

**Result**: ~97.5% accuracy on case endings, ~48% error reduction

#### Stage 3: Hybrid LSTM Corrector
**Purpose**: Fix remaining internal vowel errors

**Architecture**:
- Dual-head design:
  - Change detector: Should this position change?
  - Correction predictor: What should it be?
- Word encoder: BiLSTM (96 hidden)
- Context encoder: BiLSTM (96 hidden)
- V1 diacritic embedding (learned)

**Key Innovation**: Contrastive learning - trained on V1's specific mistakes

**Adaptive thresholding**:
- High confidence (>0.75): Always apply
- Medium (0.55-0.75): Apply if correction confident (>0.7)
- Low (<0.55): Keep V1's prediction

**Result**: Additional ~12% error reduction on internal vowels

#### Stage 4: Final Sweep (V2.1)
**Purpose**: Rule-based أَنَّ/أَنْ disambiguation

**Approach**: Syntactic rules based on following word
- أَنَّ (anna) → followed by noun (nominal sentence)
- أَنْ (an) → followed by verb (subjunctive)

**Result**: 266 additional errors fixed, 2.29% → 2.23% DER

### Why This Architecture Works

1. **Preserve what works**: V1 is already 95.65% accurate
2. **Targeted fixes**: Each stage addresses specific error types
3. **Conservative corrections**: When uncertain, keep original
4. **Ensemble reduces variance**: Multiple models agree = higher confidence
5. **Small models beat large**: BiLSTM >> Transformer for this data size

---

## Experimental Procedures

### Before Any Experiment

1. **Establish baseline**: Run current best model on test set
2. **Document hypothesis**: What do you expect to improve?
3. **Define success criteria**: What DER improvement is meaningful?
4. **Check for regressions**: Will this break other cases?

### Running an Experiment

```markdown
## Experiment: [Name]
**Date**: YYYY-MM-DD
**Hypothesis**: [What you're testing]
**Approach**: [How you're testing it]

### Setup
- Baseline DER: X.XX%
- Test set: tashkeela_test.txt (2500 lines)
- Code version: [git hash or description]

### Changes Made
[Describe exact code changes]

### Results
- DER before: X.XX% (N errors)
- DER after: X.XX% (N errors)
- Errors fixed: N
- Regressions introduced: N

### Analysis
[What worked, what didn't, why]

### Decision
[ ] Merge - improvement with no regressions
[ ] Iterate - promising but needs refinement
[ ] Abandon - doesn't work or causes regressions
```

### After Each Experiment

1. Update `PROGRESS.md` with results
2. If successful, update DER history table
3. Archive failed experiments (document why they failed)
4. Commit working changes with clear message

### Model Training Protocol

1. **Always use same train/val/test split**
2. **Log all hyperparameters**
3. **Save checkpoints every N epochs**
4. **Evaluate on validation set during training**
5. **Final evaluation ONLY on test set**
6. **Never tune on test set**

### Quantization Protocol

```python
# Standard quantization for deployment
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear, nn.LSTM, nn.Conv1d},
    dtype=torch.qint8
)
```

Then LZMA compress: `lzma.compress(data, preset=9)`

---

## Lessons Learned

### What Worked

#### 1. BiLSTM > Transformer for Small Data
- Training data: ~2M examples (150K sentences × ~13 words)
- Transformer overfit badly, only +0.016% improvement
- BiLSTM with attention achieved ~48% error reduction
- **Lesson**: Match model capacity to data size

#### 2. Ensemble Reduces Variance
- Single model: Inconsistent across runs
- 3-model ensemble: Stable, better generalization
- Weighted voting outperformed majority voting
- **Lesson**: Multiple small models > one large model

#### 3. Conservative Thresholding
- Aggressive corrections caused regressions
- Adaptive thresholds preserve good predictions
- High confidence only: Minimal regressions
- **Lesson**: When uncertain, trust the baseline

#### 4. Contrastive Learning for Correction
- Training on V1's specific mistakes
- Model learns "what V1 gets wrong"
- More efficient than general diacritization
- **Lesson**: Target the actual error distribution

#### 5. Rule-Based Post-Processing
- Some patterns are 100% deterministic
- Rules are faster and more reliable than neural
- أَنَّ/أَنْ disambiguation: Pure syntactic rule
- **Lesson**: Use rules when you have certainty

### What Didn't Work

#### 1. Full Retraining
- Retraining V1 vocabulary: Minimal gain
- V1's patterns already well-optimized
- **Lesson**: Don't reinvent what works

#### 2. Single Threshold
- One threshold for all corrections: Regressions
- Different word types need different confidence
- **Lesson**: Context-dependent thresholds

#### 3. Transformer for Correction
- Character-level transformer for post-correction
- Overfit on training data
- Only +0.016% on test set
- **Lesson**: Simpler models generalize better

#### 4. End-to-End Neural
- Replacing V1 entirely with neural model
- Lost the strong baseline
- Worse overall than hybrid approach
- **Lesson**: Enhance, don't replace

### Key Insights

1. **Error analysis first**: Know exactly what's wrong before fixing
2. **Measure everything**: Small changes can have big effects
3. **Test incrementally**: One change at a time
4. **Keep the baseline**: Always compare to previous best
5. **Document failures**: Failed experiments teach more than successes

---

## Generalization Guidelines

### Before Adding Any Rule or Model

Ask these questions:

#### 1. Does it generalize beyond test set?
- Test on validation set too
- Check for overfitting to test patterns
- Consider: Would this work on unseen text?

#### 2. What's the precision/recall tradeoff?
- High precision (few false positives) = safe to add
- High recall (catches all cases) = may cause regressions
- **Prefer precision over recall**

#### 3. Is it linguistically principled?
- Based on Arabic grammar rules = generalizable
- Based on corpus statistics = may be corpus-specific
- Document the linguistic basis

#### 4. What's the failure mode?
- When this rule/model is wrong, what happens?
- Is the failure graceful (preserves baseline)?
- Can we detect when it's uncertain?

### Rule-Based Fix Criteria

A rule should only be added if:

1. **Deterministic**: Same input always → same output
2. **Linguistically grounded**: Based on Arabic grammar
3. **High precision**: >95% correct on test cases
4. **No context bleeding**: Doesn't affect unrelated words
5. **Documented**: Linguistic explanation included

Example of GOOD rule:
```python
# إِلَى always has kasra on hamza - this is standard Arabic orthography
# Source: Any Arabic grammar reference
if word_base == 'إلى':
    return fix_initial_kasra(word)
```

Example of BAD rule:
```python
# "ذكر" is usually "ذَكَرَ" in our corpus
# Problem: Could be ذِكْر (noun) or ذُكِرَ (passive) depending on context
if word_base == 'ذكر':
    return 'ذَكَرَ'  # WRONG - overfits to corpus
```

### Model-Based Fix Criteria

A model should only be added if:

1. **Clear problem definition**: What error type does it fix?
2. **Sufficient training data**: >10,000 examples for the pattern
3. **Validation performance**: >90% accuracy on held-out data
4. **No regression on baseline**: Test full pipeline before/after
5. **Reasonable confidence calibration**: High confidence = high accuracy

### Testing for Generalization

```python
def test_generalization(fix_function):
    """Test that a fix generalizes properly."""

    # Test on validation set (not used for development)
    val_before = evaluate(model, val_set)
    val_after = evaluate(model + fix, val_set)

    # Test on test set
    test_before = evaluate(model, test_set)
    test_after = evaluate(model + fix, test_set)

    # Check for overfitting
    val_improvement = val_before - val_after
    test_improvement = test_before - test_after

    if test_improvement > val_improvement * 1.5:
        print("WARNING: May be overfitting to test set")

    # Check for regressions
    if test_after > test_before:
        print("REGRESSION: Fix makes things worse")
        return False

    return True
```

---

## Code Organization

### Repository Structure

```
tashkeel/
├── harakat/                    # Production code
│   ├── harakat.py              # V1 module
│   ├── harakat_lzma.bin        # V1 data
│   ├── harakat_v2/             # V2 module
│   │   ├── harakat_v2.py       # Self-contained V2 (5.5 MB)
│   │   ├── final_sweep.py      # Rule-based post-processor
│   │   ├── __init__.py         # Package interface
│   │   ├── README.md           # V2 documentation
│   │   └── archive/            # Development artifacts
│   ├── harakat_v3/             # V3 development
│   │   ├── PLAN.md             # V3 roadmap
│   │   ├── PROGRESS.md         # Progress tracking
│   │   ├── analysis/           # Error analysis scripts
│   │   ├── models/             # Trained models
│   │   ├── rules/              # Rule implementations
│   │   └── training/           # Training scripts
│   └── benchmarks/             # Test data
├── paper_eval/                 # Evaluation scripts
│   ├── evaluate_all_models.py  # Canonical evaluation
│   └── tashkeela_test.txt      # Test set (2500 lines)
├── neural model sweep/         # Neural experiments
│   └── archive/                # Archived experiments
├── METHODOLOGY.md              # This document
├── tashkeela_train.txt         # Training data
├── tashkeela_val.txt           # Validation data
└── tashkeela_test.txt          # Test data (copy)
```

### File Naming Conventions

- `*_v1.py`, `*_v2.py`: Version-specific implementations
- `train_*.py`: Training scripts
- `eval_*.py`: Evaluation scripts
- `analyze_*.py`: Analysis scripts
- `*.pt`: PyTorch model checkpoints
- `*.pt.lzma`: Compressed model checkpoints
- `*_int8.pt`: Quantized models

### Code Style

- Python 3.8+ compatible
- Type hints for public functions
- Docstrings for all modules and classes
- UTF-8 encoding throughout
- Windows compatibility (handle chcp 65001)

---

## Reproducibility Checklist

### Before Starting New Work

- [ ] Pull latest code
- [ ] Verify baseline DER matches documented value
- [ ] Check test set is correct (2500 lines)
- [ ] Document starting point in PROGRESS.md

### During Development

- [ ] One change at a time
- [ ] Test after each change
- [ ] Log all hyperparameters
- [ ] Save intermediate checkpoints
- [ ] Document unexpected results

### Before Merging Changes

- [ ] Full evaluation on test set
- [ ] Compare to baseline (no regressions)
- [ ] Test on validation set (generalization check)
- [ ] Update PROGRESS.md with results
- [ ] Update README if user-facing changes
- [ ] Clear commit message explaining change

### For Model Training

- [ ] Document exact data split used
- [ ] Log random seeds if applicable
- [ ] Save training curves
- [ ] Note hardware used (affects reproducibility)
- [ ] Version all dependencies

---

## Quick Reference

### Evaluate Current Model

```bash
cd c:\tashkeel\harakat\harakat_v2
python -c "
from harakat_v2 import diacritize_with_sweep
# ... evaluation code
"
```

### Run Error Analysis

```bash
cd c:\tashkeel\harakat\harakat_v3\analysis
python error_categories.py
```

### Key Files

| Purpose | Location |
|---------|----------|
| V2 module | `harakat/harakat_v2/harakat_v2.py` |
| Evaluation | `paper_eval/evaluate_all_models.py` |
| Test set | `paper_eval/tashkeela_test.txt` |
| V3 plan | `harakat/harakat_v3/PLAN.md` |
| This doc | `harakat/METHODOLOGY.md` |

### Useful Commands

```python
# Quick DER check
from harakat_v2 import diacritize_with_sweep, strip_harakat
result = diacritize_with_sweep("كتب الطالب الدرس")

# Strip diacritics
clean = strip_harakat("كَتَبَ")

# Check if word has shadda
has_shadda = 'ّ' in word
```

---

*Last updated: 2024-12-18*
*Maintainer: Jesse Morgan*
