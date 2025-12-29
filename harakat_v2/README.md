# Harakat V2 - Neural Enhanced Arabic Diacritization

**2.23% DER** - A 49% improvement over V1's 4.35% DER

Harakat V2 builds on the statistical foundation of Harakat V1 (harakat_elite) by adding neural post-processing layers that dramatically improve accuracy, particularly for grammatical case endings (i'rab).

---

## Results

Evaluated on Tashkeela test set (2,500 lines):

| Model | DER (with case) | DER (without case) | WER |
|-------|-----------------|--------------------|----|
| V1 (harakat_elite) | 4.35% | 2.27% | 11.82% |
| V2 Neural Pipeline | 2.29% | 1.95% | 6.44% |
| **V2 + Final Sweep** | **2.23%** | **1.95%** | **6.23%** |
| **Total Improvement** | **48.7%** | **14.1%** | **47.3%** |

---

## Architecture

Harakat V2 uses a four-stage pipeline:

```
Input Text (undiacritized)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: V1 Base (harakat_elite)                           │
│  ─────────────────────────────────                          │
│  Statistical model using:                                   │
│  • N-gram language models (1-5 grams)                       │
│  • Morphological analysis with DAWG                         │
│  • Context-aware disambiguation                             │
│  • 3.14 MB LZMA-compressed vocabulary                       │
│                                                             │
│  Output: 4.35% DER (good base, weak on case endings)        │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: Case V3 Ensemble                                  │
│  ─────────────────────────────                              │
│  3x BiLSTM models with attention for case ending prediction │
│                                                             │
│  Architecture (each model):                                 │
│  • Character embedding (64-dim)                             │
│  • Word BiLSTM (128 hidden)                                 │
│  • Context BiLSTM (2-layer, 128 hidden)                     │
│  • Multi-head attention (4 heads)                           │
│  • Classifier: 256 → 128 → 64 → 8 classes                   │
│                                                             │
│  Ensemble: Weighted soft voting (0.35, 0.35, 0.30)          │
│  Output: Corrected case endings (ُ ِ َ ٌ ٍ ً ْ)              │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: Hybrid LSTM Corrector                             │
│  ──────────────────────────────                             │
│  Contrastive learning model for internal vowel correction   │
│                                                             │
│  Architecture:                                              │
│  • Dual-head design:                                        │
│    - Change detector (should this position change?)         │
│    - Correction predictor (what should it be?)              │
│  • Word encoder: BiLSTM (96 hidden)                         │
│  • Context encoder: BiLSTM (96 hidden)                      │
│  • V1 diacritic embedding (learned)                         │
│                                                             │
│  Adaptive thresholding:                                     │
│  • High confidence (>0.75): Always apply correction         │
│  • Medium (0.55-0.75): Apply if correction confident (>0.7) │
│  • Low (<0.55): Keep V1's prediction                        │
│                                                             │
│  Output: Neural corrected diacritization (2.29% DER)        │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4: Final Sweep (Optional)                            │
│  ───────────────────────────────                            │
│  Rule-based أَنَّ/أَنْ disambiguation                          │
│                                                             │
│  Key insight: أَنَّ (anna) vs أَنْ (an)                        │
│  • أَنَّ → followed by noun (nominal sentence)               │
│  • أَنْ → followed by verb (subjunctive)                     │
│                                                             │
│  Targets: أن, إن, فإن, بأن, وإن, لأن                         │
│  ~266 errors corrected, ~3% relative improvement            │
│                                                             │
│  Output: Final diacritization (2.23% DER)                   │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
    Output Text (fully diacritized)
```

---

## Why This Works

### The Problem with V1

V1 (harakat_elite) is a powerful statistical model that excels at common patterns but struggles with:

1. **Grammatical case endings (i'rab)** - Arabic marks grammatical function with word-final vowels. These depend on sentence-level syntax which statistical models can't fully capture:
   - Nominative (subject): ُ (damma) or ٌ (tanween damma)
   - Accusative (object): َ (fatha) or ً (tanween fatha)
   - Genitive (possession): ِ (kasra) or ٍ (tanween kasra)

2. **Context-dependent internal vowels** - Some vowels depend on syntactic context or semantic meaning that require deeper analysis.

### How V2 Solves This

**Case V3 Ensemble** specifically targets case endings:
- Trained on 90% of Tashkeela corpus (~150K sentences)
- Uses sentence context (±4 words) to predict grammatical case
- Ensemble of 3 models reduces variance and improves robustness
- ~97.5% accuracy on case ending prediction
- Achieves ~48% error reduction on case endings alone

**Hybrid LSTM Corrector** fixes remaining internal vowel errors:
- Uses contrastive learning to identify which V1 predictions are likely wrong
- Only corrects when confident, preserving V1's correct predictions
- Adaptive thresholds prevent over-correction (conservative approach)
- Achieves additional ~12% error reduction on internal vowels

### Key Insight

**BiLSTM >> Transformer for this task.** Despite transformers' popularity, the simpler BiLSTM architecture worked significantly better because:
1. Training data is limited (~2M examples)
2. Task is position-wise classification, not generation
3. Less overfitting with smaller model capacity
4. Faster inference on CPU

---

## Usage

### Simple API

```python
from harakat_v2 import diacritize

# Single text
result = diacritize("كتب الطالب الدرس")
print(result)  # كَتَبَ الطَّالِبُ الدَّرْسَ
```

### Class-based API

```python
from harakat_v2 import HarakatV2

# Initialize (loads all models - ~3 seconds first time)
h = HarakatV2()

# Diacritize
result = h.diacritize("ذهب الولد الى المدرسة")
print(result)  # ذَهَبَ الوَلَدُ إِلَى المَدْرَسَةِ
```

### Command Line

```bash
# Direct text
python harakat_v2.py "كتب الطالب الدرس"

# From file
python harakat_v2.py -f input.txt -o output.txt

# From stdin
echo "كتب الطالب الدرس" | python harakat_v2.py
```

---

## Self-Contained Design

**harakat_v2.py is a single 5.5 MB file containing everything:**

- V1 (harakat_elite) code - LZMA compressed and embedded
- Case V3 Ensemble (3 models) - Int8 quantized, LZMA compressed
- Hybrid LSTM Corrector - Int8 quantized, LZMA compressed
- All vocabularies and configurations

**No external model files needed.** Just `pip install torch numpy` and run.

---

## File Structure

```
harakat_v2/
├── harakat_v2.py      # Complete self-contained module (5.5 MB)
│                      # Everything embedded - just import and use
├── final_sweep.py     # Rule-based أَنَّ/أَنْ disambiguation
├── __init__.py        # Package interface
├── README.md          # This file
└── archive/           # Development files (not needed for usage)
    ├── build_complete_dist.py   # Script that built harakat_v2.py
    ├── build_dist.py            # Older build script
    ├── harakat_v2.py            # Original development version
    ├── dist/                    # Distribution builds
    ├── distributions/           # Alternative deployment formats
    └── models/                  # Raw model checkpoints
```

---

## Model Sizes

| Component | Raw Size | In harakat_v2.py | Compression |
|-----------|----------|------------------|-------------|
| V1 (harakat_elite) | ~8 MB | 3.14 MB | LZMA |
| Case V3 Ensemble (3x) | ~12 MB | ~1.3 MB | Int8 + LZMA |
| Hybrid LSTM | ~5 MB | ~0.5 MB | Int8 + LZMA |
| **Total** | ~25 MB | **5.5 MB** | **78% reduction** |

---

## Requirements

- Python 3.8+
- PyTorch 1.9+ (CPU only, no GPU needed)
- NumPy

```bash
pip install torch numpy
```

---

## Performance

| Metric | Value |
|--------|-------|
| Throughput | ~5 lines/second (full pipeline) |
| V1 stage alone | ~280 lines/second |
| Memory | ~200 MB peak |
| First load | ~3 seconds (model decompression) |

Note: V2 is slower than V1 due to neural inference, but 49% more accurate.

---

## Training Data

Models trained on Tashkeela corpus:
- **Training**: 90% (~150,000 sentences)
- **Validation**: 5,000 sentences
- **Test**: 2,500 sentences

The corpus covers classical Arabic texts including Quran, Hadith, poetry, and classical literature.

---

## Development History

### The Journey from 4.35% to 2.21%

**Phase 1: Case Ending Analysis**
- Discovered 62.7% of V1 errors were on case endings (i'rab)
- Root cause: Statistical model lacks syntactic understanding

**Phase 2: Case Model Development**
- Case V1: Simple BiLSTM → 85% accuracy
- Case V2: Added sentence context → 93% accuracy
- Case V3: Multi-head attention + 3-model ensemble → **97.5% accuracy**

**Phase 3: Internal Vowel Correction**
- Transformer attempt: Only +0.016% improvement (disappointing)
- LSTM contrastive approach: +0.35% improvement
- Adaptive thresholding: Additional +0.05% improvement
- **Final: +0.40% total improvement on internal vowels**

### What Didn't Work

| Approach | Result | Why |
|----------|--------|-----|
| Transformer corrector | +0.016% | Overfit on limited data |
| Single threshold | Regressions | Too aggressive |
| Full vocab retraining | Minimal gain | V1 vocab already good |

### What Worked

| Approach | Impact | Why |
|----------|--------|-----|
| Case V3 Ensemble | -2.14% DER | Syntactic patterns are learnable |
| Contrastive LSTM | Additional improvement | Learns V1's specific weaknesses |
| Adaptive thresholding | Prevents regressions | Conservative when uncertain |

---

## Remaining Errors

After V2, the main remaining error sources are:

1. **Shadda detection** - أَنَ vs أَنَّ (that vs indeed)
2. **Homographs** - مِن vs مَن (from vs who)
3. **Rare vocabulary** - Words not in training data
4. **Verb form ambiguity** - فَعَلَ vs فُعِلَ (active vs passive)

These represent harder linguistic challenges that may require larger models or external knowledge bases.

---

## License

MIT License - Free for academic and commercial use.

---

## Author

**Jesse Morgan**
DLI Arabic Instructor

Built with insights from Arabic linguistics and modern NLP techniques.
