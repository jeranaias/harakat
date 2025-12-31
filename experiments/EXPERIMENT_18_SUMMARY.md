# Experiment 18: Homograph Classifiers & Pattern Expansion

## Summary
Extended homograph disambiguation with new ML classifiers and bug fixes.

## Changes Made

### 1. New Homograph Classifiers

#### ثم Classifier (then/there)
- **File**: `harakat_v3/models/homograph_classifiers/thumma_classifier.pkl`
- **Classes**:
  - `thumma` (ثُمَّ) = "then" (conjunction, 94% of cases)
  - `thamma` (ثَمَّ) = "there" (adverb, 6% of cases)
- **Training**: 9,039 examples from tashkeela_train.txt
- **Accuracy**: 97.8% cross-validation
- **Status**: DISABLED - severe class imbalance makes it unusable

#### أم Classifier (mother/or)
- **File**: `harakat_v3/models/homograph_classifiers/umm_classifier.pkl`
- **Classes**:
  - `am` (أَمْ) = "or" (question particle, 55% of cases)
  - `umm` (أُمّ) = "mother" (noun, 45% of cases)
- **Training**: 2,051 examples from tashkeela_train.txt
- **Accuracy**: 93.0% cross-validation
- **Status**: WORKING - with special handling for 'umm' case

### 2. Bug Fixes Found During Testing

#### SAFE_THAMMA_PREV Patterns (REVERTED)
The original pattern analysis was WRONG. These patterns:
```python
SAFE_THAMMA_PREV = {
    '(', 'ومن', ':', 'عليه', 'قال', ...
}
```
Were intended to detect ثَمَّ (there) but actually appear before BOTH ثُمَّ and ثَمَّ.
- **Result**: Caused 78 REGRESSIONS instead of fixes
- **Action**: DISABLED `apply_safe_thumma_patterns()` in diacritize()

#### أم Classifier Output Not Applied
The classifier correctly predicted 'umm' (mother) with 99.6% confidence, but:
- `CORRECTIONS['أم']['umm'] = None` meant no fix was applied
- V2's wrong أَمُ stayed instead of being corrected to أُمّ

**Fix**: Added special handling in `apply_ml_homograph_rules()`:
```python
elif word_base == 'أم' and predicted_class == 'umm':
    # Fix internal vowels: أَم → أُمّ, preserve case ending
    fixed = word.replace('أَ', 'أُ', 1)  # fatha→damma on alif
    # Add shadda on meem if not present
    ...
```

#### Masdar Detection Missing Suffix Stripping
`is_masdar_pattern()` didn't strip suffixes before checking patterns:
- `تصديق` was correctly detected as masdar
- `تصديقها` (with suffix) was NOT detected

**Fix**: Added suffix stripping at start of `is_masdar_pattern()`

#### Common Nouns Wrongly Detected as Verbs
`أحدهما` ("one of them") was flagged as present tense verb because:
- Starts with أ
- Has 6+ letters

**Fix**: Added `'أحد', 'أحدهما', 'أحدهم'` to `common_nouns` set

### 3. Expanded Shadda Patterns (أنّ/إنّ)

Added 30 new patterns to `ml_verified_shadda_patterns`:
- High frequency: ذلك (40x), هذا (31x), الله (18x), النبي (18x), له (18x)
- Medium frequency: الأصل, رجلا, المراد, هذه, الحكم, etc.

### 4. Common Phrase Overrides

Added word overrides for critical phrases:
```python
('بسم', 'الله'): 'بِسْمِ',  # bismillah - sukun on س
('إله', 'إلا'): 'إِلَهَ',   # la ilaha illa allah
```

## Files Modified

1. **harakat.py**:
   - `load_models()` - Added loading for new classifiers
   - `CORRECTIONS` - Added ثم, أم corrections
   - `MIN_CONFIDENCE` - Added thresholds
   - `apply_safe_thumma_patterns()` - DISABLED (caused regressions)
   - `apply_ml_homograph_rules()` - Added special أم handling
   - `is_masdar_pattern()` - Added suffix stripping
   - `is_present_tense_verb()` - Added أحد to common_nouns
   - `ml_verified_shadda_patterns` - Expanded with 30 new patterns

2. **New Files**:
   - `harakat_v3/models/homograph_classifiers/thumma_classifier.pkl`
   - `harakat_v3/models/homograph_classifiers/umm_classifier.pkl`
   - `experiments/18_train_homographs.py`

## Results

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| DER | 2.16% | 2.13% | -0.03% |
| Errors | 9,586 | 9,465 | -121 |

### Breakdown of Fixes
| Fix | Errors Fixed |
|-----|--------------|
| Disable broken ثم patterns | 78 |
| Fix أم classifier integration | 39 |
| Fix masdar/verb detection | 4 |
| **Total** | **121** |

## Key Lessons Learned

1. **Pattern analysis can be misleading**: Patterns that appear "100% consistent" in small samples may still cause regressions when applied globally.

2. **Classifier bias from class imbalance**: The ثم classifier (94% vs 6%) is useless because it always predicts the majority class.

3. **CORRECTIONS with None values do nothing**: When a classifier correctly predicts a class but `CORRECTIONS[word][class] = None`, no fix is applied.

4. **Suffix handling is critical**: Functions like `is_masdar_pattern()` must strip suffixes before pattern matching.

## Common Phrases Now Correct

```
بسم الله الرحمن الرحيم → بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ ✓
لا إله إلا الله → لَا إِلَهَ إِلَّا اللَّهُ ✓
كانت أم سلمة → كَانَتْ أُمُّ سَلَمَةَ ✓
هذا أم ذاك → هَذَا أَمْ ذَاكَ ✓
```

## Train/Test Separation
- Training: tashkeela_train.txt (49,999 lines)
- Testing: tashkeela_test.txt (2,499 lines)
- All classifiers trained on train set, evaluated on test set
