# DER Improvement Log

## Current Status (2024-12-31) - UPDATED

### Latest Measurements
| Configuration | DER | Errors | Positions |
|--------------|-----|--------|-----------|
| V2 only | 3.02% | 13,510 | 447,604 |
| V3.5 baseline | 2.58% | 11,557 | 447,604 |
| V3.5 + fixes (prev) | 2.53% | 11,345 | 447,604 |
| **V3.5 + all fixes** | **2.53%** | **11,312** | 447,604 |

### Session Progress: -245 errors fixed total (+33 in DER reduction experiments)
| Fix | Errors Fixed | DER Change |
|-----|--------------|------------|
| ML model path fix | -70 | 2.58% → 2.57% |
| Anna shadda removal | -7 | 2.57% → 2.56% |
| Verb list expansion | -34 | 2.56% → 2.56% |
| Present tense verb fix | -13 | 2.56% → 2.55% |
| Word override dictionary | -88 | 2.55% → 2.53% |

### Word Override Dictionary (New)
Pre-V2 override that bypasses the neural model entirely for known patterns:
| Pattern | Gold | Fixes |
|---------|------|-------|
| إن + لم | إِنْ | 49 |
| ألا + ترى | أَلَا | 15 |
| رضى + الله | رَضِىَ | 11 |
| وإن + ) | وَإِنْ | 9 |
| + others | | 14 |

---

## Previous Baseline Measurements
| Configuration | DER | Errors | Positions |
|--------------|-----|--------|-----------|
| V2 only | 3.02% | 13,510 | 447,604 |
| V3.5 (full) | 2.58% | 11,557 | 447,604 |

### Targets
- **Immediate**: Sub-2.0% DER (<8,952 errors)
- **Final**: Sub-1.5% DER (<6,714 errors)

### What V3.5 Corrections Do
| Stage | Errors Fixed | DER Impact |
|-------|--------------|------------|
| Particle rules | ~1,953 | -0.44% |
| Anna fix | ~29 | -0.01% |
| ML homographs | ~6 | -0.01% |
| ML voice | minimal | ~0% |
| Sunna fix | minimal | ~0% |

**Key Insight**: Particle rules provide 97% of the V3.5 improvement!

---

## Clarification: The 1.889% Number

The "1.889%" mentioned earlier was **INTERNAL VOWEL DER ONLY**, not full DER.

From `neural model sweep/RESULTS_SUMMARY.md`:
- V1 internal vowel DER: 2.25%
- LSTM V10 internal vowel DER: 1.894%
- Improvement: +0.354% (901 errors fixed out of 254,876 internal positions)

This is NOT comparable to our full DER measurement which includes case endings.

---

## Error Breakdown (500 line sample)

### By Position Type
| Position | Errors | % of Total |
|----------|--------|------------|
| Internal | ~1,400 | 60% |
| Final (case) | ~900 | 40% |

### By Error Type (Top 15)
1. internal:damma->fatha: 196
2. internal:fatha->kasra: 191
3. internal:kasra->fatha: 178
4. internal:fatha->damma: 170
5. final:fatha->damma: 92
6. internal:sukun->fatha: 91
7. internal:fatha->sukun: 74
8. final:damma->fatha: 65
9. internal:missing_fatha: 54
10. final:damma->kasra: 54

### By Word (Top 15)
1. أن: 33 errors
2. من: 31 errors
3. وأن: 24 errors
4. أذن: 19 errors
5. فإن: 16 errors
6. ذكر: 16 errors
7. إن: 14 errors
8. غير: 14 errors
9. أم: 14 errors
10. فلأن: 12 errors

---

## Failed Attempts

### Session 2024-12-30 (Current)

#### Possessive/Accusative Shadda Indicators
- Result: +10 errors (regression)
- Attempted: Add shadda indicators for nouns with possessive suffix and accusative tanween
- Reverted immediately

#### Lower Homograph Thresholds
- Result: +48 errors (regression)
- Attempted: Lower confidence thresholds from 0.75 to 0.50 for علم/ذكر
- Cause: More regressions than fixes; V2 already handles most cases

### Previous Sessions

#### Retrained LSTM V10
- Result: DER increased from 2.58% to 2.61% (regression)
- Cause: Model trained on different data distribution

#### New ML Classifiers
- Result: More regressions than fixes (net -7 errors)
- Cause: V2 already handles most cases; new classifiers fought existing logic

---

## Key Findings (Session 2024-12-30)

### What Worked
1. **ML Model Path Fix** - Models weren't loading due to wrong path (biggest win: -70 errors)
2. **Bidirectional Shadda Fix** - Now REMOVES shadda when should_shadda=False (not just adds)
3. **Expanded Past Tense Verb List** - Added Form II-X verbs (خالف, جرى, رفع, etc.)
4. **Better Present Tense Detection** - Exclude common nouns (أهل, أقسام, نفقة)
5. **Demonstrative Pronouns** - Added as positive shadda indicators (هذا, هذه, etc.)

### What Did NOT Work
1. **Lower ML Thresholds** - V2 is already good; lowering thresholds causes more harm
2. **Possessive/Accusative Rules** - Too many edge cases, net regression
3. **Sun Letter Shadda** - Only 7 errors in 500 lines (not a significant source)

### Remaining Challenges
1. **Internal Vowel Confusion** - ~1,500 errors from fatha↔damma↔kasra confusion
2. **Voice Ambiguity** - حمل (28.6%), قتل (57.9%), وجد (57.1%) accuracy
3. **Case Endings** - ~700 errors from final position vowels

---

## Updated Path Forward

### Realistic Assessment
- Current: 2.55% (11,433 errors)
- To reach sub-2% need: ~2,481 fewer errors (27% reduction)
- Most remaining errors are internal vowel/voice confusion where rule-based fixes cause regressions

### Potential Approaches (not yet attempted)
1. **Selective Word Dictionaries** - Hardcode common words that are consistently wrong
2. **LSTM V10 Internal Only** - Only use LSTM for internal vowels, not case endings
3. **Retrain ML on Current Errors** - Train new classifiers specifically on V2 error cases
4. **Hybrid Rules** - Only apply rules when V2 confidence is low

---

## Commands for Testing

```bash
# Quick DER test
cd c:/tashkeel/harakat && py -3 benchmarks/quick_der_test.py

# Test specific configuration
cd c:/tashkeel/harakat && py -3 -c "
from harakat import diacritize
# ... test code
"
```

---

## Notes

- Test file: `tashkeela_v2/tashkeela_test.txt` (2,499 lines)
- Archived benchmark (different): `harakat/benchmarks/tashkeela_test.txt` (2,500 lines)
- These files have different diacritization standards!
