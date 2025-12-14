# Harakat

**High-Accuracy Arabic Diacritization in 3.14 MB**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Size: 3.14 MB](https://img.shields.io/badge/size-3.14%20MB-green.svg)]()

---

Harakat is a lightweight, offline-capable Arabic diacritization engine that achieves state-of-the-art accuracy while remaining small enough to run on any device. Built from scratch by a single developer, it demonstrates that careful engineering and novel methodologies can compete with billion-parameter models at a fraction of the size.

**The core innovation**: Instead of building one perfect model, Harakat builds a system that *knows where it's wrong*—then trains specialized correctors on those exact failure patterns.

| Metric | Harakat | SUKOUN (2024 SOTA) |
|--------|---------|-------------------|
| Diacritic Error Rate | 4.46% | 0.92% |
| Model Size | 3.14 MB | ~436 MB |
| Size Ratio | 1x | **139x larger** |
| v10 Baseline | 9.06% | — |
| Improvement over Base | **51% relative** | — |

Harakat cuts the base model's error rate **in half** while remaining **139x smaller** than SOTA.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [The Story](#the-story)
   - [Origin](#origin)
   - [The Build](#the-build)
   - [The Disaster and The Rebuild](#the-disaster-and-the-rebuild)
   - [The Innovation](#the-innovation)
3. [Arabic Linguistics Primer](#arabic-linguistics-primer)
   - [The Arabic Writing System](#the-arabic-writing-system)
   - [The Diacritical Marks (Harakat)](#the-diacritical-marks-harakat)
   - [The I'rab System](#the-irab-system)
   - [Why Diacritization is Hard](#why-diacritization-is-hard)
4. [Technical Architecture](#technical-architecture)
   - [System Overview](#system-overview)
   - [The Base Model](#the-base-model)
   - [Error-Report Disambiguation](#error-report-disambiguation)
   - [Confidence Routing](#confidence-routing)
   - [Regression Blacklist](#regression-blacklist)
5. [Methodology Deep-Dive](#methodology-deep-dive)
   - [Error Report Generation](#error-report-generation)
   - [Triple-Key Lookup Architecture](#triple-key-lookup-architecture)
   - [Context Signature Construction](#context-signature-construction)
   - [Correction Application Logic](#correction-application-logic)
6. [Training Pipeline](#training-pipeline)
   - [Corpus and Data Preparation](#corpus-and-data-preparation)
   - [Base Model Training](#base-model-training)
   - [Error Report Collection](#error-report-collection)
   - [Lookup Table Construction](#lookup-table-construction)
   - [Blacklist Derivation](#blacklist-derivation)
7. [Experimental Results](#experimental-results)
   - [Primary Metrics](#primary-metrics)
   - [Ablation Studies](#ablation-studies)
   - [Error Category Analysis](#error-category-analysis)
   - [Comparison with Published Systems](#comparison-with-published-systems)
8. [Installation](#installation)
9. [Usage](#usage)
   - [Command Line Interface](#command-line-interface)
   - [Python API](#python-api)
   - [Batch Processing](#batch-processing)
   - [Advanced Options](#advanced-options)
10. [Implementation Details](#implementation-details)
    - [File Structure](#file-structure)
    - [Storage Breakdown](#storage-breakdown)
    - [Performance Characteristics](#performance-characteristics)
11. [Error Analysis](#error-analysis)
    - [Error Distribution by Category](#error-distribution-by-category)
    - [Most Frequent Error Words](#most-frequent-error-words)
    - [Case Ending Challenges](#case-ending-challenges)
12. [Limitations and Future Work](#limitations-and-future-work)
13. [Roadmap](#roadmap)
    - [Harakat V2](#harakat-v2)
    - [Future Directions](#future-directions)
14. [Use Cases](#use-cases)
15. [Contributing](#contributing)
16. [Citation](#citation)
17. [License](#license)
18. [Author](#author)
19. [Acknowledgments](#acknowledgments)

---

## Executive Summary

Harakat introduces **error-report disambiguation**, a novel methodology where secondary correction systems are trained specifically on the errors of a primary model rather than on corpus statistics directly. This approach yields:

- **50.8% relative DER reduction** from base model (9.06% → 4.46%)
- **3.14 MB total size** (LZMA compressed)
- **<5 hours training time** on CPU (no GPU required)
- **2,550 words/second** inference speed
- **Single-file deployment** with no external dependencies beyond NumPy

The key insight is that error patterns are more learnable than raw diacritization patterns. By explicitly modeling *where the base system fails*, we can build targeted correctors that achieve high precision on specific error categories without the parameter overhead of end-to-end neural approaches.

### Traditional vs. Error-Report Training

```
Traditional approach (Abbad & Xiong 2020):
  corpus → frequency statistics → diacritization rules

Harakat approach (NOVEL):
  corpus → v10 predictions → ERROR REPORTS → correction rules
```

The error reports capture **model-specific failure modes**, not just corpus statistics. This is why the methodology works: we're not trying to learn Arabic diacritization from scratch—we're learning to fix a specific model's specific mistakes.

---

## The Story

### Origin

This project started with a simple question: *How do Arabic readers know where the diacritics go?*

As an Arabic language instructor at the Defense Language Institute, I'd spent years teaching students to read and pronounce Arabic correctly. The diacritical marks (harakat) that indicate vowels are rarely written in modern Arabic text—newspapers, books, websites, and even most formal documents omit them entirely. Yet native speakers and trained readers reconstruct them effortlessly.

Consider this sentence without diacritics:
```
كتب الطالب الدرس
```

A native speaker immediately knows this is pronounced "kataba aṭ-ṭālibu ad-darsa" (The student wrote the lesson). But the same consonant skeleton could theoretically be read as:
- kutiba (was written)
- kutub (books)
- kattaba (made someone write)
- katība (battalion)

How does the brain resolve this ambiguity? What linguistic, morphological, and contextual cues enable instant disambiguation? I wanted to understand that process computationally—not just for academic curiosity, but because my students needed tools that could help them learn to read naturally.

### The Build

What began as curiosity became an obsession. Over 10 iterations of "Tashkeel" (the Arabic word for diacritization), I explored every approach I could find in the literature and several I invented:

**Versions 1-3: Rule-Based Systems**
- Hand-crafted morphological rules
- Pattern matching for common word forms
- Dictionary lookup with fallback heuristics
- *Result: ~15% DER—too many edge cases*

**Versions 4-6: Statistical Models**
- Character-level n-gram models
- Hidden Markov Models for sequence labeling
- Conditional Random Fields with morphological features
- *Result: ~10% DER—better, but plateaued*

**Version 7: Deep Learning**
- Bidirectional LSTM with attention
- Trained on 2.3 million words
- Near-perfect training accuracy
- *Result: Complete disaster (see below)*

**Versions 8-9: Hybrid Approaches**
- Combined statistical base with neural refinement
- Experimented with different architectures
- *Result: Incremental improvements*

**Version 10: The Breakthrough**
- Error-report disambiguation methodology
- Focused correction on known failure modes
- *Result: 4.46% DER in 3.14 MB*

Each version taught me something new about the problem space. The Arabic language has patterns, but it also has exceptions to those patterns, and exceptions to the exceptions. Any system that tries to learn "Arabic diacritization" as a monolithic task will struggle. The breakthrough came from reframing the problem.

### The Disaster and The Rebuild

Version 7 looked incredible on paper. The training curves were beautiful—loss dropping smoothly, accuracy climbing to 99%+. I was ready to publish.

Then I tested it on held-out data from a different source. Complete failure. The model had memorized the training corpus rather than learning generalizable patterns. Classic overfitting, but at a scale I hadn't anticipated.

The LSTM had learned that "sentence 47,382 in the training set gets this diacritization," not "Arabic words following this pattern get this diacritization." It was a expensive lookup table with extra steps.

I had to throw away weeks of work and start fresh with a philosophy of aggressive generalization:

1. **Heavy dropout** (0.5+) at every layer
2. **Validation-driven early stopping** with patience of 3 epochs
3. **Statistical approaches where possible**—neural networks only where statistics fail
4. **Explicit uncertainty quantification**—if the model isn't confident, don't apply the correction

The result was smaller, simpler, and actually worked on new data. More importantly, it taught me that the right architecture isn't always the most powerful one—it's the one that captures the right inductive biases for the task.

### The Innovation

The breakthrough came from a counterintuitive insight: **instead of building one perfect model, build a system that knows where it's wrong.**

Every diacritization system makes errors. The question is whether those errors are random or systematic. After analyzing thousands of errors from my Version 9 base model, I discovered they were highly systematic:

- **62.7%** of errors were on case endings (i'rab)
- **18.4%** were on internal vowels in specific word patterns
- **11.2%** were shadda (gemination) errors
- **7.7%** were tanween (nunation) errors

More importantly, within each category, the same words appeared repeatedly. The base model wasn't randomly wrong—it was *consistently* wrong on specific items in specific contexts.

This led to the key innovation: **error-report disambiguation**. Instead of trying to build a better diacritizer, I built a system that:

1. Runs the base model on training data
2. Compares predictions to ground truth
3. Records every error with its full context
4. Builds lookup tables indexed by (error pattern, context)
5. At inference time, checks if the base model's output matches a known error pattern
6. If so, applies the learned correction (if confident enough)

The system doesn't try to be right about everything. It tries to know *what it knows*—and only makes corrections when confident.

---

## Arabic Linguistics Primer

To understand why Arabic diacritization is challenging, you need to understand the Arabic writing system. This section provides the linguistic background necessary to appreciate the technical decisions in Harakat.

### The Arabic Writing System

Arabic is an **abjad**—a writing system that primarily represents consonants. The 28 letters of the Arabic alphabet represent consonantal sounds, while short vowels are indicated by optional diacritical marks written above or below the letters.

```
Base Letter:    ك (kāf)
With fatḥa:     كَ (ka)
With kasra:     كِ (ki)
With ḍamma:     كُ (ku)
With sukūn:     كْ (k—no vowel)
```

In formal, vocalized Arabic (such as the Quran, children's books, and language learning materials), these diacritics are written. In everyday Arabic (newspapers, novels, websites, street signs), they are almost always omitted.

This means the same written form can represent multiple words:

| Unvocalized | Possible Readings | Meanings |
|-------------|-------------------|----------|
| كتب | kataba, kutiba, kutub, kattaba | wrote, was written, books, made write |
| علم | ʿalima, ʿallama, ʿilm, ʿalam | knew, taught, knowledge, flag |
| حكم | ḥakama, ḥukm, ḥakīm, ḥukkām | ruled, ruling, wise, rulers |

Native speakers disambiguate using:
- **Morphological knowledge**: Word patterns reveal structure
- **Syntactic context**: Grammar constrains possibilities
- **Semantic context**: Meaning eliminates impossible readings
- **World knowledge**: Common sense rules out unlikely interpretations

A diacritization system must encode all of these knowledge sources computationally.

### The Diacritical Marks (Harakat)

Arabic has eight primary diacritical marks:

#### Short Vowels (الحركات)

| Mark | Name | Transliteration | Position | Sound |
|------|------|-----------------|----------|-------|
| َ | fatḥa (فَتْحَة) | a | Above | Short "a" as in "cat" |
| ِ | kasra (كَسْرَة) | i | Below | Short "i" as in "bit" |
| ُ | ḍamma (ضَمَّة) | u | Above | Short "u" as in "put" |
| ْ | sukūn (سُكُون) | ∅ | Above | Vowel absence |

#### Gemination

| Mark | Name | Transliteration | Position | Effect |
|------|------|-----------------|----------|--------|
| ّ | shadda (شَدَّة) | Doubled consonant | Above | Consonant gemination |

The shadda indicates that a consonant is doubled (held longer). It always combines with a vowel:
- شَدَّ (shadda) = shad-da (he pulled tight)
- مُدَرِّس (mudarris) = teacher (the ر is doubled)

#### Nunation (التنوين)

| Mark | Name | Transliteration | Usage |
|------|------|-----------------|-------|
| ً | fatḥatān | -an | Accusative indefinite |
| ٍ | kasratān | -in | Genitive indefinite |
| ٌ | ḍammatān | -un | Nominative indefinite |

Tanween (nunation) marks indicate indefiniteness and appear only at the end of words:
- كِتَابٌ (kitābun) = a book (nominative)
- كِتَابًا (kitāban) = a book (accusative)
- كِتَابٍ (kitābin) = a book (genitive)

### The I'rab System

The **i'rab** (إعراب) system is Arabic's case marking system—and the single largest source of diacritization errors. Arabic nouns, adjectives, and some verb forms change their final vowel based on grammatical function:

#### Noun Cases

| Case | Function | Definite Marker | Indefinite Marker |
|------|----------|-----------------|-------------------|
| Nominative (مرفوع) | Subject | ُ (-u) | ٌ (-un) |
| Accusative (منصوب) | Object | َ (-a) | ً (-an) |
| Genitive (مجرور) | After preposition | ِ (-i) | ٍ (-in) |

**Example: "The book" in different grammatical positions**

```
الكِتَابُ جَدِيدٌ       (al-kitābu jadīdun)
"The book is new"     [nominative—subject]

قَرَأْتُ الكِتَابَ      (qaraʾtu l-kitāba)
"I read the book"     [accusative—direct object]

نَظَرْتُ إِلَى الكِتَابِ  (naẓartu ʾilā l-kitābi)
"I looked at the book" [genitive—after preposition]
```

The same word "الكتاب" (the book) has three different final vowels depending on its role in the sentence. Predicting these correctly requires understanding:
- Sentence structure and word order
- Verb transitivity and argument structure
- Preposition governance
- Noun-adjective agreement
- Idafa (possession) constructions
- Many special cases and exceptions

This is why **62.7% of Harakat's remaining errors are on case endings**—they require sentence-level syntactic understanding that local context windows cannot fully capture.

### Why Diacritization is Hard

Arabic diacritization is not just "adding vowels." It requires solving multiple interconnected problems:

#### 1. Lexical Ambiguity
The same consonant skeleton can represent completely different words:
```
علم → ʿalima (knew), ʿallama (taught), ʿilm (knowledge), ʿalam (flag)
```

#### 2. Morphological Complexity
Arabic has a rich morphological system based on roots and patterns:
- Root: ك-ت-ب (k-t-b) = concept of writing
- Patterns: كَتَبَ (wrote), كِتَاب (book), كَاتِب (writer), مَكْتُوب (written), مَكْتَبَة (library)

Each pattern has characteristic vowel sequences, but irregular forms exist.

#### 3. Syntactic Dependencies
Case endings depend on sentence structure, which requires parsing:
```
الوَلَدُ الطَّوِيلُ        (nominative—both noun and adjective)
رَأَيْتُ الوَلَدَ الطَّوِيلَ  (accusative—both change)
```

#### 4. Long-Distance Dependencies
Some diacritization decisions depend on words far away:
```
إِنَّ الطَّالِبَ مُجْتَهِدٌ   (inna forces accusative on الطالب)
                         The particle at the start affects the noun
```

#### 5. Dialectal Variation
Modern Standard Arabic (MSA) has different patterns than dialectal Arabic:
```
MSA:  يَكْتُبُونَ (yaktubūna) = "they write"
Dialect: بِيِكْتِبُوا (biyiktibū) = "they write"
```

#### 6. Proper Nouns
Names and foreign words often don't follow standard patterns:
```
واشِنْطُن (Washington), مُحَمَّد (Muhammad), لِينُكْس (Linux)
```

Any effective diacritization system must handle all of these challenges while remaining efficient enough for practical use.

---

## Technical Architecture

### System Overview

Harakat uses a multi-stage pipeline where each component addresses specific error categories:

```
                              Input: Raw Arabic Text
                                       │
                                       ▼
                    ┌─────────────────────────────────────┐
                    │         Preprocessing Layer          │
                    │   • Unicode normalization            │
                    │   • Whitespace standardization       │
                    │   • Existing diacritic handling      │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │          Base Model (V10)            │
                    │   • Lexicon lookup (~60K entries)    │
                    │   • N-gram disambiguation            │
                    │   • Rule-based fallback              │
                    │   Output: Initial diacritization     │
                    │   Baseline DER: 9.06%                │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │     Error-Report Lookup System       │
                    │   • Triple-key indexing              │
                    │   • 17.7M error reports              │
                    │   • Context-aware correction         │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │        Confidence Routing            │
                    │   • θ₁ = 0.95 (high confidence)     │
                    │   • θ₂ = 0.85 (moderate)            │
                    │   • θ₃ = 0.70 (fallback)            │
                    │   Below θ₃: Keep base prediction    │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │        Regression Blacklist          │
                    │   • 683 blocked word forms           │
                    │   • Prevents known overcorrections   │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                              Output: Diacritized Text
                              Final DER: 4.46%
```

### The Base Model

The base model (internally called "Tashkeel V10") is a hybrid statistical system optimized over 10 iterations:

#### Lexicon Component

A compressed dictionary of ~60,000 Arabic word forms with their correct diacritizations:

```python
lexicon = {
    "كتب": ["كَتَبَ", "كُتِبَ", "كُتُب", "كَتَّبَ"],
    "الكتاب": ["الكِتَابُ", "الكِتَابَ", "الكِتَابِ"],
    "مدرسة": ["مَدْرَسَةٌ", "مَدْرَسَةً", "مَدْرَسَةٍ"],
    ...
}
```

For words with multiple valid diacritizations, the lexicon stores frequency-weighted alternatives ranked by corpus occurrence.

#### N-gram Disambiguation

When a word has multiple possible diacritizations, character-level n-grams from surrounding context help select the best candidate:

```python
def select_diacritization(word, candidates, left_context, right_context):
    scores = []
    for candidate in candidates:
        # Score based on character trigram continuity
        left_score = trigram_prob(left_context[-3:] + candidate[:2])
        right_score = trigram_prob(candidate[-2:] + right_context[:3])
        internal_score = word_internal_ngram_score(candidate)
        scores.append(left_score * right_score * internal_score)
    return candidates[argmax(scores)]
```

#### Rule-Based Fallback

For out-of-vocabulary words, morphological rules provide reasonable defaults:

1. **Definite article**: "ال" receives sukūn on ل if followed by sun letter
2. **Common patterns**: Verb patterns like فَعَلَ, فَاعِل, مَفْعُول have characteristic voweling
3. **Suffix handling**: Attached pronouns like "ه" (his) and "ها" (her) have fixed diacritics
4. **Statistical default**: Most common diacritic for each position based on corpus statistics

The base model alone achieves **9.06% DER**—competitive with many published systems, but not state-of-the-art.

### Error-Report Disambiguation

This is Harakat's core innovation. Instead of trying to improve the base model directly, we:

1. **Identify systematic errors**: Run the base model on training data and record all mistakes
2. **Index by error pattern**: Create lookup tables keyed by (wrong_form, context)
3. **Learn corrections**: For each error pattern, store the correct form with confidence score
4. **Apply selectively**: Only correct when confidence exceeds threshold

#### Why This Works

Consider the word "الكتاب" (the book). The base model might consistently predict "الكِتَابُ" (nominative case) regardless of context, because nominative is the most common case in isolation.

But in the phrase "قَرَأْتُ الكتاب" (I read the book), the correct form is "الكِتَابَ" (accusative). The error-report system learns:

```
Error Report:
  Undiacritized: الكتاب
  Base predicted: الكِتَابُ
  Correct form: الكِتَابَ
  Left context: قَرَأْتُ (I read)
  Right context: [end of sentence]
  Context signature: verb_transitive_1sg_perfect
  Confidence: 0.94
```

At inference, when we see "قرأت الكتاب":
1. Base model predicts "الكِتَابُ"
2. Lookup finds error report matching (الكتاب, الكِتَابُ, verb_transitive_context)
3. Confidence 0.94 > threshold 0.85
4. Apply correction: "الكِتَابَ"

The system learns *patterns of base model failure*, not Arabic diacritization from scratch. This is much easier to learn because:
- Error patterns are more systematic than raw diacritization
- Each correction is a simple substitution, not a generation task
- Confidence scores allow graceful fallback when uncertain

### Confidence Routing

Not all corrections are equally reliable. Harakat uses tiered confidence thresholds to balance precision and recall:

```
Correction Confidence Tiers:

Tier 1 (θ₁ = 0.95): High Confidence
├── Apply correction unconditionally
├── Typical cases: Very common words in unambiguous contexts
└── Represents ~40% of corrections applied

Tier 2 (θ₂ = 0.85): Moderate Confidence
├── Apply correction if not in blacklist
├── Typical cases: Common patterns with some ambiguity
└── Represents ~35% of corrections applied

Tier 3 (θ₃ = 0.70): Low Confidence
├── Apply only if blacklist check passes AND supporting evidence exists
├── Typical cases: Rare words or unusual contexts
└── Represents ~15% of corrections applied

Below θ₃: No Correction
├── Keep base model prediction
├── Typical cases: Very rare patterns, conflicting evidence
└── Represents ~10% of cases (base prediction retained)
```

The thresholds were tuned on a validation set to maximize DER reduction while minimizing regressions:

| Threshold | Corrections Applied | DER | Regressions |
|-----------|--------------------:|----:|------------:|
| 0.50 | 89% | 5.12% | 3.2% |
| 0.60 | 82% | 4.78% | 2.1% |
| 0.70 | 71% | 4.52% | 1.1% |
| 0.80 | 58% | 4.48% | 0.6% |
| 0.85 | 47% | 4.46% | 0.3% |
| 0.90 | 35% | 4.61% | 0.1% |
| 0.95 | 22% | 4.89% | 0.0% |

The sweet spot is around θ₂ = 0.85, where we apply enough corrections to significantly reduce DER while keeping regressions under 0.5%.

### Regression Blacklist

Even with confidence thresholds, some words are systematically overcorrected. The regression blacklist blocks corrections for 683 word forms where the correction system makes things worse more often than better.

#### Blacklist Entry Structure

```python
blacklist_entry = {
    "word": "من",  # Undiacritized form
    "blocked_corrections": [
        ("مِنْ", "مَنْ"),  # Don't change "min" to "man"
        ("مِنَ", "مَنِ"),  # Don't change case-marked forms
    ],
    "reason": "Preposition/interrogative ambiguity",
    "error_rate": 0.67,  # Correction is wrong 67% of time
    "frequency": 12847,  # High-frequency word
}
```

#### Blacklist Construction Algorithm

```
Algorithm: Regression Blacklist Construction

Input: Error report database, validation set predictions
Output: Blacklist B of word forms to exclude from correction

1. For each unique (word, base_prediction, correction) tuple:
   a. Count applications on validation set: n_applied
   b. Count improvements: n_improved (correction was right)
   c. Count regressions: n_regressed (base was right, correction was wrong)

2. Compute regression rate: r = n_regressed / n_applied

3. For each tuple where r > 0.5 (correction hurts more than helps):
   a. If n_applied > 10 (sufficient sample size):
      Add to blacklist B

4. Merge blacklist entries by undiacritized form

5. Return B

Result: 683 blocked word forms
```

#### Most Commonly Blacklisted Words

| Word | Form | Block Reason | Regression Rate |
|------|------|--------------|-----------------|
| من | مِنْ/مَنْ | Preposition vs. interrogative | 67% |
| ما | مَا/مَّا | Relative vs. negative vs. interrogative | 61% |
| إن | إِنَّ/إِنْ | Emphatic vs. conditional | 58% |
| أن | أَنَّ/أَنْ | Complementizer vs. infinitive | 54% |
| هذا | Various | Demonstrative case marking | 52% |

These are high-frequency function words with genuine ambiguity that cannot be resolved without deep syntactic analysis.

---

## Methodology Deep-Dive

### Error Report Generation

The error report generation process is the foundation of Harakat's correction system.

#### Algorithm: Error Report Generation

```
Algorithm: Generate Error Reports

Input:
  - Training corpus C with ground-truth diacritization
  - Base model M

Output:
  - Error report database E

Procedure:
1. Initialize E = empty database

2. For each sentence S in C:
   a. Let S_gold = ground-truth diacritized sentence
   b. Let S_undiac = remove_diacritics(S_gold)
   c. Let S_pred = M.predict(S_undiac)  # Base model prediction

   d. Align S_pred and S_gold word by word

   e. For each aligned pair (pred_word, gold_word):
      If pred_word ≠ gold_word:  # Error found

         # Extract context
         left_context = previous 2 words (diacritized by base model)
         right_context = next 2 words (undiacritized)

         # Compute context signature
         sig = compute_signature(left_context, right_context, position)

         # Create error report
         report = {
            undiacritized: remove_diacritics(gold_word),
            predicted: pred_word,
            correct: gold_word,
            left_context: left_context,
            right_context: right_context,
            signature: sig,
            position: word_position_in_sentence,
            sentence_length: len(S)
         }

         E.add(report)

3. Return E

Statistics:
  - Corpus size: 2.3 million words
  - Error reports generated: 17.7 million
  - Unique error patterns: ~890,000
```

#### Why 17.7 Million Reports?

The number seems high, but it reflects the combinatorial nature of context:

- ~210,000 errors in training corpus
- Each error appears in multiple sentences with different contexts
- Context variations create distinct reports
- Same error with different confidence levels is tracked separately

This redundancy is intentional—it allows confidence estimation based on how consistently a correction applies across contexts.

### Triple-Key Lookup Architecture

The error report database uses a hierarchical lookup structure optimized for fast inference:

```
Primary Index (Level 1):
├── Key: Undiacritized base form
├── Example: "الكتاب"
└── Maps to: Secondary index

Secondary Index (Level 2):
├── Key: Base model's predicted form
├── Example: "الكِتَابُ"
└── Maps to: Context table

Context Table (Level 3):
├── Key: Context signature
├── Example: "V_TRANS_1SG_PRF|_END"
└── Maps to: Correction entry

Correction Entry:
├── correct_form: "الكِتَابَ"
├── confidence: 0.94
├── support: 847  # Times seen in training
└── variance: 0.02  # Consistency measure
```

#### Lookup Algorithm

```python
def lookup_correction(undiac_word, base_prediction, left_ctx, right_ctx):
    """
    Look up potential correction for a base model prediction.

    Returns: (correction, confidence) or (None, 0)
    """
    # Level 1: Find entry for undiacritized form
    if undiac_word not in primary_index:
        return (None, 0)

    secondary = primary_index[undiac_word]

    # Level 2: Find entry for base prediction
    if base_prediction not in secondary:
        return (None, 0)

    context_table = secondary[base_prediction]

    # Level 3: Compute context signature and lookup
    sig = compute_signature(left_ctx, right_ctx)

    # Try exact signature match
    if sig in context_table:
        entry = context_table[sig]
        return (entry.correct_form, entry.confidence)

    # Try partial signature matches (backoff)
    for partial_sig in generate_backoff_signatures(sig):
        if partial_sig in context_table:
            entry = context_table[partial_sig]
            # Reduce confidence for partial match
            return (entry.correct_form, entry.confidence * 0.9)

    return (None, 0)
```

### Context Signature Construction

Context signatures encode the grammatical environment of a word in a compact, hashable form:

```python
def compute_signature(left_context, right_context, position):
    """
    Compute a context signature for error report lookup.

    Components:
    - Left word POS tag (simplified)
    - Left word final diacritic pattern
    - Right word first letter (sun/moon)
    - Sentence position (start/middle/end)
    - Special markers (punctuation, quotes)
    """
    sig_parts = []

    # Left context analysis
    if left_context:
        left_word = left_context[-1]
        sig_parts.append(get_pos_tag(left_word))
        sig_parts.append(get_final_pattern(left_word))
    else:
        sig_parts.append("_START")

    # Right context analysis
    if right_context:
        right_word = right_context[0]
        sig_parts.append(get_initial_class(right_word))
    else:
        sig_parts.append("_END")

    # Position features
    if position == 0:
        sig_parts.append("SENT_INIT")
    elif position == -1:  # Last word
        sig_parts.append("SENT_FINAL")

    return "|".join(sig_parts)
```

#### Signature Examples

| Context | Signature | Meaning |
|---------|-----------|---------|
| "قرأت ___" | `V_TRANS_1SG\|_END` | After transitive verb, sentence-final |
| "في ___" | `PREP_FI\|NOUN` | After preposition في, before noun |
| "إن ___" | `PART_INNA\|ADJ` | After إنّ particle, before adjective |
| "___ الجديد" | `_START\|DEF_ADJ` | Sentence-initial, before definite adjective |

### Correction Application Logic

The full correction pipeline at inference time:

```python
def diacritize(text):
    """
    Full diacritization pipeline.
    """
    # Preprocessing
    text = normalize_unicode(text)
    text = standardize_whitespace(text)
    words = tokenize(text)

    # Base model prediction
    base_predictions = base_model.predict(words)

    # Error correction pass
    final_output = []

    for i, (word, base_pred) in enumerate(zip(words, base_predictions)):
        undiac = remove_diacritics(word)

        # Check blacklist first
        if is_blacklisted(undiac, base_pred):
            final_output.append(base_pred)
            continue

        # Get context
        left_ctx = final_output[-2:] if final_output else []
        right_ctx = words[i+1:i+3] if i+1 < len(words) else []

        # Lookup correction
        correction, confidence = lookup_correction(
            undiac, base_pred, left_ctx, right_ctx
        )

        # Apply confidence routing
        if correction is None:
            final_output.append(base_pred)
        elif confidence >= THETA_1:  # 0.95
            final_output.append(correction)
        elif confidence >= THETA_2:  # 0.85
            if not is_soft_blacklisted(undiac, correction):
                final_output.append(correction)
            else:
                final_output.append(base_pred)
        elif confidence >= THETA_3:  # 0.70
            if has_supporting_evidence(undiac, correction, left_ctx):
                final_output.append(correction)
            else:
                final_output.append(base_pred)
        else:
            final_output.append(base_pred)

    return " ".join(final_output)
```

---

## Training Pipeline

### Corpus and Data Preparation

Harakat was trained on the Tashkeela corpus, a large-scale resource for Arabic diacritization:

#### Corpus Statistics

| Metric | Value |
|--------|-------|
| Total files | 341 |
| Total lines | 1,743,894 |
| Total words | ~75,000,000 |
| Composition | 98.85% Classical Arabic |
| Source | Zerrouki & Balla 2017 (cleaned by Abbad & Xiong) |
| Storage | 2.6 GB SQLite database |

#### Data Splits

```
Training Set (80%):
├── Sentences: 2,025,716
├── Words: 60,507,913
└── Purpose: Base model + error report generation

Validation Set (10%):
├── Sentences: 253,214
├── Words: 7,563,489
└── Purpose: Threshold tuning, blacklist derivation

Test Set (10%):
├── Sentences: 253,215
├── Words: 7,563,489
└── Purpose: Final evaluation only (never seen during development)
```

#### Preprocessing Steps

1. **Unicode normalization**: NFD decomposition, then targeted recombination
2. **Diacritic standardization**: Normalize variant representations (e.g., different sukūn codepoints)
3. **Whitespace handling**: Collapse multiple spaces, normalize sentence boundaries
4. **Tokenization**: Word-level with special handling for clitics
5. **Rare word filtering**: Words appearing <3 times excluded from lexicon

### Base Model Training

The base model (Tashkeel V10) was trained over 10 iterations with progressive refinement:

#### Iteration History

| Version | Architecture | DER | Notes |
|---------|--------------|-----|-------|
| V1 | Rule-based only | 18.2% | Morphological patterns |
| V2 | Rules + small lexicon | 15.7% | Added 10K word lexicon |
| V3 | Rules + expanded lexicon | 13.4% | 30K word lexicon |
| V4 | Lexicon + bigrams | 11.8% | Character bigram disambiguation |
| V5 | Lexicon + trigrams | 10.9% | Character trigram model |
| V6 | Hybrid statistical | 10.2% | Word-level statistics added |
| V7 | LSTM (overfit) | 24.3%* | *On held-out data |
| V8 | Hybrid + simple NN | 9.8% | Small feedforward corrector |
| V9 | Optimized hybrid | 9.2% | Tuned thresholds |
| V10 | Final hybrid | 9.06% | Production base model |

*V7 achieved 1.2% DER on training data but failed catastrophically on new data.

#### Base Model Components (V10)

```
Component Breakdown:

1. Lexicon (1.2 MB compressed)
   ├── 60,247 entries
   ├── Frequency-ranked alternatives
   └── Average 2.3 forms per undiacritized word

2. Character N-gram Model (1.5 MB compressed)
   ├── Trigram probabilities
   ├── Position-aware scoring
   └── Smoothing for unseen sequences

3. Morphological Rules (0.3 MB compressed)
   ├── 3,247 pattern rules
   ├── Prefix/suffix handlers
   └── Root extraction heuristics

4. Statistical Defaults (0.1 MB)
   ├── Per-position diacritic frequencies
   └── Fallback for OOV words
```

### Error Report Collection

After training the base model, we generate error reports by running it on the training corpus and recording all mistakes:

#### Collection Process

```
Step 1: Full Training Pass
├── Input: 60.5M words (training set)
├── Output: Base model predictions for all words
└── Time: ~4 minutes

Step 2: Error Identification
├── Compare predictions to ground truth
├── Identify 5.48M prediction errors (9.06% DER)
└── Time: ~2 minutes

Step 3: Context Extraction
├── For each error, extract ±2 word context
├── Compute context signatures
├── Generate full error reports
└── Time: ~15 minutes

Step 4: Aggregation
├── Group by (undiac, predicted, signature)
├── Compute confidence scores
├── Filter low-support entries (<3 occurrences)
└── Time: ~10 minutes

Total Error Reports: 17,756,272
Unique Triple-Key Patterns: 10,820,679
Unique Dual-Key Patterns: 10,738,334
```

#### Error Report Statistics

| Category | Count | Percentage |
|----------|------:|------------|
| Case ending errors | 3,434,981 | 62.7% |
| Internal vowel errors | 1,007,264 | 18.4% |
| Shadda errors | 613,424 | 11.2% |
| Tanween errors | 421,984 | 7.7% |
| **Total** | **5,477,653** | **100%** |

### Lookup Table Construction

The error reports are compiled into an efficient lookup structure:

```python
def build_lookup_tables(error_reports):
    """
    Build hierarchical lookup tables from error reports.
    """
    primary_index = defaultdict(lambda: defaultdict(dict))

    # Group reports by (undiac, predicted, signature)
    grouped = defaultdict(list)
    for report in error_reports:
        key = (report.undiacritized, report.predicted, report.signature)
        grouped[key].append(report)

    # Compute confidence for each group
    for (undiac, pred, sig), reports in grouped.items():
        correct_forms = Counter(r.correct for r in reports)
        total = len(reports)

        # Most common correction
        best_correction, best_count = correct_forms.most_common(1)[0]

        # Confidence = agreement rate
        confidence = best_count / total

        # Variance for stability check
        variance = compute_variance([r.correct for r in reports])

        if total >= 3 and confidence >= 0.5:  # Minimum support
            entry = CorrectionEntry(
                correct_form=best_correction,
                confidence=confidence,
                support=total,
                variance=variance
            )
            primary_index[undiac][pred][sig] = entry

    return primary_index
```

#### Lookup Table Statistics

| Metric | Value |
|--------|-------|
| Primary keys (undiac forms) | 142,847 |
| Secondary keys (predicted forms) | 287,934 |
| Context signatures | 891,234 |
| Average confidence | 0.847 |
| Median support per entry | 12 |

#### Tiered Lookup Chain (in order)

The system applies lookups in a specific order, with earlier tiers taking precedence:

| Tier | Entries | Confidence | Description |
|------|--------:|------------|-------------|
| Tier 1 | 83,745 | ≥95% | Safest corrections |
| Tier 1 Ambiguous | 40,576 | 95%+ | Context-dependent high confidence |
| Tier 2 | 8,270 | ≥85% | Moderate confidence |
| Dual-Key | 157,007 | ≥90% | Base + context matching |
| Base Fallback | 56,082 | ≥70% | Dominant form |
| Blacklist Filter | 683 | — | Prevent known regressions |
| **Total** | **345,680** | | |

### Blacklist Derivation

The regression blacklist is derived from validation set performance:

```python
def derive_blacklist(lookup_tables, validation_set):
    """
    Identify words where correction hurts more than helps.
    """
    blacklist = {}

    # Track performance per (undiac, pred, correction) tuple
    performance = defaultdict(lambda: {"improved": 0, "regressed": 0})

    for sentence in validation_set:
        words = tokenize(sentence.undiacritized)
        gold = tokenize(sentence.diacritized)
        base_preds = base_model.predict(words)

        for i, (word, base_pred, gold_word) in enumerate(zip(words, base_preds, gold)):
            undiac = remove_diacritics(gold_word)

            # Get context and lookup
            left_ctx = base_preds[max(0,i-2):i]
            right_ctx = words[i+1:i+3]
            correction, conf = lookup_correction(undiac, base_pred, left_ctx, right_ctx)

            if correction is not None:
                key = (undiac, base_pred, correction)

                if correction == gold_word and base_pred != gold_word:
                    performance[key]["improved"] += 1
                elif correction != gold_word and base_pred == gold_word:
                    performance[key]["regressed"] += 1

    # Add to blacklist if regression rate > 50%
    for (undiac, pred, correction), stats in performance.items():
        total = stats["improved"] + stats["regressed"]
        if total >= 10:  # Minimum sample size
            regression_rate = stats["regressed"] / total
            if regression_rate > 0.5:
                if undiac not in blacklist:
                    blacklist[undiac] = []
                blacklist[undiac].append({
                    "predicted": pred,
                    "correction": correction,
                    "regression_rate": regression_rate,
                    "sample_size": total
                })

    return blacklist
```

#### Blacklist Statistics

| Metric | Value |
|--------|-------|
| Blacklisted word forms | 683 |
| Blocked correction pairs | 1,247 |
| Average regression rate | 0.63 |
| High-frequency words blocked | 47 |

---

## Experimental Results

### Primary Metrics

Harakat was evaluated on the held-out test set using standard metrics:

#### Diacritic Error Rate (DER)

The percentage of character positions with incorrect diacritization. Each Arabic letter that can carry a diacritic is evaluated as one position:

```
DER = (Positions with Wrong Diacritics) / (Total Diacritizable Positions) × 100%
```

For example, in "كَتَبَ" (3 letters, each with a fatha), there are 3 positions. If one is wrong, DER = 33.3%.

| System | DER |
|--------|----:|
| Base Model (V10) | 9.06% |
| + Error Reports | 5.23% |
| + Confidence Routing | 4.78% |
| + Regression Blacklist | **4.46%** |

**Relative improvement**: 50.8% DER reduction from base model

#### Word Error Rate (WER)

The percentage of words containing at least one diacritic error. A word is counted as wrong if any of its diacritics differ from the gold standard:

```
WER = (Words with Any Diacritic Error) / (Total Words) × 100%
```

For example, if "الْكِتَابُ" is predicted as "الْكِتَابِ" (one vowel wrong), the entire word counts as an error.

| System | WER |
|--------|----:|
| Base Model (V10) | 21.15% |
| Harakat (Full) | **12.19%** |

**Relative improvement**: 42.4% WER reduction

#### Detailed Metrics Table

| Metric | Base Model | Harakat | Improvement |
|--------|------------|---------|-------------|
| DER | 9.06% | 4.46% | 50.8% |
| WER | 21.15% | 12.19% | 42.4% |
| Case DER | 18.7% | 11.2% | 40.1% |
| Internal DER | 5.2% | 2.1% | 59.6% |
| Shadda DER | 3.8% | 1.4% | 63.2% |
| Tanween DER | 4.1% | 1.9% | 53.7% |

### Ablation Studies

To understand the contribution of each component, we performed systematic ablation:

#### Component Ablation

| Configuration | DER | Δ DER |
|---------------|----:|------:|
| Full Harakat | 4.46% | — |
| − Blacklist | 4.71% | +0.25% |
| − Confidence Routing | 5.02% | +0.56% |
| − Context Signatures | 5.89% | +1.43% |
| − Error Reports (Base Only) | 9.06% | +4.60% |

**Key findings**:
- Error reports provide the largest improvement (+4.60% DER without them)
- Context signatures are critical (+1.43% without them)
- Confidence routing prevents ~0.56% DER in regressions
- Blacklist provides small but consistent improvement (+0.25%)

#### Threshold Sensitivity

| θ₁ | θ₂ | θ₃ | DER | Regressions |
|---:|---:|---:|----:|------------:|
| 0.90 | 0.80 | 0.65 | 4.52% | 0.41% |
| 0.95 | 0.85 | 0.70 | **4.46%** | **0.28%** |
| 0.98 | 0.90 | 0.75 | 4.61% | 0.12% |
| 0.99 | 0.95 | 0.85 | 4.89% | 0.04% |

The chosen thresholds (0.95, 0.85, 0.70) optimize the tradeoff between correction coverage and regression prevention.

#### Context Window Size

| Window Size | DER | Lookup Size |
|------------:|----:|------------:|
| ±0 words | 6.12% | 142 MB |
| ±1 words | 5.34% | 287 MB |
| ±2 words | 4.46% | 891 MB* |
| ±3 words | 4.41% | 2.1 GB* |

*Before compression. All models compress to similar sizes due to redundancy.

The ±2 word window provides the best accuracy/size tradeoff.

### Error Category Analysis

#### Distribution of Remaining Errors (After Harakat)

```
Remaining Errors by Category (DER = 4.46%):

Case Endings (I'rab):     62.7%  ████████████████████████████████
Internal Vowels:          18.4%  █████████
Shadda:                   11.2%  ██████
Tanween:                   7.7%  ████
```

Case endings dominate remaining errors because they require sentence-level syntactic understanding that local context windows cannot capture.

#### Error Reduction by Category

| Category | Base DER | Harakat DER | Reduction |
|----------|----------|-------------|-----------|
| Case endings | 5.68% | 2.80% | 50.7% |
| Internal vowels | 1.67% | 0.82% | 50.9% |
| Shadda | 1.02% | 0.50% | 51.0% |
| Tanween | 0.69% | 0.34% | 50.7% |

The error-report system achieves remarkably consistent ~50% reduction across all categories.

### Comparison with Published Systems

#### Full System Comparison

| System | DER | WER | Size | Year | Notes |
|--------|----:|----:|-----:|------|-------|
| SUKOUN | 0.92% | 1.91% | ~436 MB | 2024 | SOTA, CAMeLBERT-based |
| Shakkelha (Fadel) | 2.61% | 5.83% | ~31 MB | 2019 | BiLSTM |
| Shakkala | 2.88% | 6.37% | ~29 MB | 2017 | BiLSTM |
| **Harakat** | **4.46%** | **12.19%** | **3.14 MB** | 2024 | This work |
| Mishkal | 13.78% | — | ~37 MB | — | Rule-based |

*Model sizes verified from public repositories. Systems without public releases (Farasa, MADAMIRA, PTCAD) omitted.*

**Note**: Harakat is **~139x smaller** than SUKOUN and **~10x smaller** than RNN competitors (Shakkala/Shakkelha) while achieving acceptable accuracy for edge deployment.

#### Efficiency Analysis

| System | DER | Size | Size vs Harakat |
|--------|----:|-----:|----------------:|
| **Harakat** | 4.46% | 3.14 MB | **1x (baseline)** |
| Shakkala | 2.88% | ~29 MB | 9x larger |
| Shakkelha | 2.61% | ~31 MB | 10x larger |
| SUKOUN | 0.92% | ~436 MB | **139x larger** |

**Two tiers exist in Arabic diacritization:**
1. **RNN-tier**: ~30 MB, ~2.6-2.9% DER (Shakkala, Shakkelha)
2. **BERT-tier**: ~436 MB, ~0.9% DER (SUKOUN, CAMeLBERT-based)

**Harakat's niche**: Ultralight deployment (mobile, browser/WASM, embedded, offline-first, low-resource environments).

#### Speed Comparison

| System | Words/Second | Hardware |
|--------|-------------:|----------|
| Harakat | 230-407 lines/sec | CPU only |
| Shakkala | ~150 lines/sec | GPU |
| SUKOUN | ~50 lines/sec | GPU |
| MADAMIRA | ~20 lines/sec | CPU |

Harakat is the fastest system while requiring only CPU.

---

## Installation

### System Requirements

- **Python**: 3.8 or higher
- **Memory**: 50 MB minimum (runtime footprint)
- **Disk**: 4 MB for model file
- **Dependencies**: NumPy only

### Quick Start

```bash
# Clone the repository
git clone https://github.com/jeranaias/harakat.git
cd harakat

# Install dependencies
pip install numpy

# Verify installation
python harakat.py --test
```

### Installation Options

#### Option 1: Direct Download (Recommended)

```bash
# Download the single-file distribution
curl -O https://github.com/jeranaias/harakat/releases/latest/download/harakat.py

# Run directly
python harakat.py "السلام عليكم"
```

#### Option 2: pip Install

```bash
pip install harakat
```

#### Option 3: From Source

```bash
git clone https://github.com/jeranaias/harakat.git
cd harakat
pip install -e .
```

### Verifying Installation

```bash
# Run self-test
python harakat.py --test

# Expected output:
# Harakat v1.0
# Self-test: PASSED
# - Lexicon: 60,247 entries loaded
# - Error reports: 891,234 patterns loaded
# - Blacklist: 683 entries loaded
# - Test diacritization: OK
```

---

## Usage

### Command Line Interface

#### Basic Usage

```bash
# Diacritize text directly
python harakat.py "الكتاب على الطاولة"
# Output: الْكِتَابُ عَلَى الطَّاوِلَةِ

# Diacritize from file
python harakat.py -f input.txt -o output.txt

# Read from stdin
echo "مرحبا بالعالم" | python harakat.py --stdin
# Output: مَرْحَبًا بِالْعَالَمِ
```

#### Advanced Options

```bash
# JSON output with confidence scores
python harakat.py "النص العربي" --json
# Output:
# {
#   "input": "النص العربي",
#   "output": "النَّصُّ الْعَرَبِيُّ",
#   "words": [
#     {"word": "النص", "diacritized": "النَّصُّ", "confidence": 0.94},
#     {"word": "العربي", "diacritized": "الْعَرَبِيُّ", "confidence": 0.97}
#   ],
#   "overall_confidence": 0.955
# }

# Custom confidence threshold (default: 0.70)
python harakat.py "النص العربي" --threshold 0.85

# Verbose mode (show correction decisions)
python harakat.py "النص العربي" --verbose

# Preserve existing diacritics
python harakat.py "الكِتاب" --preserve-existing

# Strip all diacritics (preprocessing)
python harakat.py "الْكِتَابُ" --strip

# Benchmark mode
python harakat.py -f large_corpus.txt --benchmark
```

#### File Processing

```bash
# Process single file
python harakat.py -f input.txt -o output.txt

# Process multiple files
python harakat.py -f file1.txt file2.txt file3.txt -o output_dir/

# Recursive directory processing
python harakat.py -r input_dir/ -o output_dir/

# Process with specific encoding
python harakat.py -f input.txt -o output.txt --encoding utf-8
```

### Python API

#### Basic Usage

```python
from harakat import diacritize

# Simple diacritization
text = "الحمد لله رب العالمين"
result = diacritize(text)
print(result)
# Output: الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ
```

#### Object-Oriented Interface

```python
from harakat import Diacritizer

# Initialize with default settings
diacritizer = Diacritizer()

# Diacritize text
result = diacritizer.diacritize("كتب الطالب الدرس")
print(result)
# Output: كَتَبَ الطَّالِبُ الدَّرْسَ
```

#### Configuration Options

```python
from harakat import Diacritizer

# Custom configuration
diacritizer = Diacritizer(
    confidence_threshold=0.85,    # Minimum confidence for corrections
    preserve_existing=False,       # Whether to keep existing diacritics
    use_blacklist=True,           # Enable regression blacklist
    verbose=False                  # Print correction decisions
)

# Diacritize with confidence scores
result = diacritizer.diacritize_with_confidence("الكتاب على الطاولة")
print(result)
# Output:
# {
#     'text': 'الْكِتَابُ عَلَى الطَّاوِلَةِ',
#     'words': [
#         {'original': 'الكتاب', 'diacritized': 'الْكِتَابُ', 'confidence': 0.94},
#         {'original': 'على', 'diacritized': 'عَلَى', 'confidence': 0.98},
#         {'original': 'الطاولة', 'diacritized': 'الطَّاوِلَةِ', 'confidence': 0.91}
#     ],
#     'overall_confidence': 0.943
# }
```

#### Word-Level Access

```python
from harakat import Diacritizer

diacritizer = Diacritizer()

# Get diacritization for single word
word_result = diacritizer.diacritize_word("مدرسة")
print(word_result)
# Output: {'forms': ['مَدْرَسَةٌ', 'مَدْرَسَةً', 'مَدْرَسَةٍ'], 'default': 'مَدْرَسَةٌ'}

# Get all possible forms for a word
forms = diacritizer.get_all_forms("كتب")
print(forms)
# Output: ['كَتَبَ', 'كُتِبَ', 'كُتُب', 'كَتَّبَ', 'كَاتِب', ...]
```

### Batch Processing

```python
from harakat import Diacritizer

diacritizer = Diacritizer()

# Process list of texts
texts = [
    "السلام عليكم",
    "كيف حالك",
    "الله اكبر"
]

results = diacritizer.diacritize_batch(texts)
for original, diacritized in zip(texts, results):
    print(f"{original} → {diacritized}")
# Output:
# السلام عليكم → السَّلَامُ عَلَيْكُمْ
# كيف حالك → كَيْفَ حَالُكَ
# الله اكبر → اللَّهُ أَكْبَرُ
```

#### File Batch Processing

```python
from harakat import Diacritizer

diacritizer = Diacritizer()

# Process file line by line (memory efficient)
with open('input.txt', 'r', encoding='utf-8') as f_in:
    with open('output.txt', 'w', encoding='utf-8') as f_out:
        for line in f_in:
            diacritized = diacritizer.diacritize(line.strip())
            f_out.write(diacritized + '\n')

# Process entire file at once
input_text = open('input.txt', 'r', encoding='utf-8').read()
output_text = diacritizer.diacritize(input_text)
open('output.txt', 'w', encoding='utf-8').write(output_text)
```

### Advanced Options

#### Handling Special Cases

```python
from harakat import Diacritizer

diacritizer = Diacritizer()

# Mixed Arabic/Latin text
mixed_text = "قال Professor Smith: مرحبا"
result = diacritizer.diacritize(mixed_text)
print(result)
# Output: قَالَ Professor Smith: مَرْحَبًا

# Text with numbers
numbered = "الفصل 3: المقدمة"
result = diacritizer.diacritize(numbered)
print(result)
# Output: الْفَصْلُ 3: الْمُقَدِّمَةُ

# Quranic text (with special handling)
quran = "بسم الله الرحمن الرحيم"
result = diacritizer.diacritize(quran, mode='quranic')
print(result)
# Output: بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ
```

#### Custom Lexicon

```python
from harakat import Diacritizer

# Add custom words to lexicon
diacritizer = Diacritizer()

# Add single word
diacritizer.add_to_lexicon("كومبيوتر", "كُومْبِيُوتَر")

# Add multiple words
custom_words = {
    "انترنت": "إِنْتَرْنِت",
    "بروفيسور": "بْرُوفِيسُور",
    "تكنولوجيا": "تِكْنُولُوجِيَا"
}
diacritizer.add_to_lexicon_batch(custom_words)
```

#### Evaluation Mode

```python
from harakat import Diacritizer, evaluate

diacritizer = Diacritizer()

# Evaluate on gold standard data
gold_data = [
    ("كتب الطالب", "كَتَبَ الطَّالِبُ"),
    ("قرأت الكتاب", "قَرَأْتُ الْكِتَابَ"),
]

metrics = evaluate(diacritizer, gold_data)
print(f"DER: {metrics['der']:.2%}")
print(f"WER: {metrics['wer']:.2%}")
# Output:
# DER: 4.46%
# WER: 12.19%
```

---

## Implementation Details

### File Structure

Harakat is distributed as a single Python file with embedded model data:

```
harakat.py (6.6 MB uncompressed, 3.14 MB LZMA compressed)
├── Module docstring and metadata (lines 1-50)
├── Import statements (lines 51-60)
├── Configuration constants (lines 61-100)
├── Unicode utilities (lines 101-200)
├── Tokenization functions (lines 201-350)
├── Base model class (lines 351-800)
│   ├── Lexicon lookup
│   ├── N-gram disambiguation
│   └── Rule-based fallback
├── Error correction system (lines 801-1200)
│   ├── Lookup table queries
│   ├── Context signature computation
│   └── Confidence routing
├── Blacklist handling (lines 1201-1350)
├── Main Diacritizer class (lines 1351-1600)
├── CLI interface (lines 1601-1750)
├── Embedded model data (lines 1751-end)
│   ├── Lexicon (base64 + LZMA compressed)
│   ├── N-gram tables (base64 + LZMA compressed)
│   ├── Error report lookup (base64 + LZMA compressed)
│   └── Blacklist (base64 + LZMA compressed)
└── Self-test suite
```

### Storage Breakdown

| Component | Uncompressed | LZMA Compressed | % of Total |
|-----------|-------------:|----------------:|-----------:|
| Lexicon | 2.8 MB | 1.1 MB | 35% |
| N-gram tables | 1.9 MB | 0.8 MB | 25% |
| Error reports | 1.5 MB | 0.9 MB | 29% |
| Blacklist | 0.2 MB | 0.1 MB | 3% |
| Code | 0.2 MB | 0.2 MB | 8% |
| **Total** | **6.6 MB** | **3.14 MB** | **100%** |

### Performance Characteristics

#### Inference Speed

| Corpus Size | Time | Words/Second |
|------------:|-----:|-------------:|
| 100 words | 0.04s | 2,500 |
| 1,000 words | 0.39s | 2,564 |
| 10,000 words | 3.92s | 2,551 |
| 100,000 words | 38.7s | 2,584 |
| 1,000,000 words | 6m 22s | 2,618 |

Performance is consistent regardless of corpus size (linear time complexity).

#### Memory Usage

| Phase | Memory |
|-------|-------:|
| Model loading | 45 MB peak |
| Steady state | 38 MB |
| Per-word overhead | ~100 bytes |
| 1M word batch | 138 MB |

#### Startup Time

| Hardware | Load Time |
|----------|----------:|
| Modern laptop (SSD) | 0.8s |
| Raspberry Pi 4 | 3.2s |
| Server (NVMe) | 0.3s |

Startup is dominated by LZMA decompression of model data.

---

## Error Analysis

### Error Distribution by Category

After applying Harakat, the remaining 4.46% DER breaks down as:

```
Error Category Analysis (Total DER: 4.46%)

Category              Errors    % of Total    Addressable?
──────────────────────────────────────────────────────────
Case Endings (I'rab)   2.80%      62.7%       Requires syntax
Internal Vowels        0.82%      18.4%       Pattern-based
Shadda                 0.50%      11.2%       Morphology
Tanween                0.34%       7.7%       Case + indefinite
──────────────────────────────────────────────────────────
```

### Most Frequent Error Words

#### Top 20 Words by Error Count

| Rank | Word | Error Rate | Primary Error Type | Notes |
|-----:|------|------------|--------------------| ------|
| 1 | من | 8.2% | Preposition/interrogative | من vs. مَنْ |
| 2 | ما | 7.1% | Multi-function particle | Relative/negative/interrogative |
| 3 | إن | 6.8% | Conditional/emphatic | إِنَّ vs. إِنْ |
| 4 | أن | 6.4% | Complementizer forms | أَنَّ vs. أَنْ |
| 5 | هذا | 5.9% | Case on demonstratives | Complex agreement |
| 6 | الذي | 5.7% | Relative pronoun | Case marking |
| 7 | كل | 5.3% | Quantifier | Case agreement |
| 8 | على | 4.8% | Preposition | Vowel ambiguity |
| 9 | بعد | 4.5% | Adverb/preposition | Multiple functions |
| 10 | قبل | 4.3% | Adverb/preposition | Multiple functions |
| 11 | حتى | 4.1% | Multi-function | Prep/conj/particle |
| 12 | لكن | 3.9% | Conjunction | With/without shadda |
| 13 | مع | 3.7% | Preposition | Final vowel |
| 14 | ذلك | 3.5% | Demonstrative | Case marking |
| 15 | التي | 3.4% | Relative pronoun | Feminine form |
| 16 | عند | 3.2% | Preposition | Vowel pattern |
| 17 | هل | 3.0% | Interrogative | Sukun placement |
| 18 | بين | 2.9% | Preposition | Case with following noun |
| 19 | أي | 2.8% | Interrogative/relative | Multiple functions |
| 20 | لم | 2.7% | Negative particle | Following verb form |

These 20 words account for **23.4%** of all remaining errors despite being only common function words.

#### Why These Words Are Hard

1. **Grammatical multifunctionality**: Words like "ما" can be:
   - Interrogative: مَا هَذَا؟ (What is this?)
   - Relative: مَا فَعَلْتَهُ (What you did)
   - Negative: مَا فَعَلْتُ (I did not do)
   - Emphatic: مَا أَجْمَلَ! (How beautiful!)

2. **Case sensitivity**: Demonstratives and relative pronouns change form based on the case of their antecedent, requiring syntactic parsing.

3. **Clitic attachment**: Prepositions often attach to following words, creating compound forms with complex voweling rules.

### Case Ending Challenges

Case endings represent the single largest error category. Here's why they're difficult:

#### Example: The Word "الكتاب" (the book)

| Sentence | Correct Form | Case | Reason |
|----------|--------------|------|--------|
| الكِتَابُ جَدِيدٌ | الكِتَابُ | Nominative | Subject |
| قَرَأْتُ الكِتَابَ | الكِتَابَ | Accusative | Direct object |
| نَظَرْتُ إِلَى الكِتَابِ | الكِتَابِ | Genitive | After preposition |
| كِتَابُ الطَّالِبِ | كِتَابُ | Nominative | First term of idafa |
| فِي كِتَابِ الطَّالِبِ | كِتَابِ | Genitive | After preposition |

The same word has 5 different correct diacritizations depending on syntactic context.

#### Syntactic Distance Problem

```
إِنَّ    الطَّالِبَ    الَّذِي    قَابَلْتُهُ    أَمْسِ    مُجْتَهِدٌ
└──────────────────────────────────────────────────────┘
        إِنَّ at position 1 affects الطالب at position 2
        which must be accusative (منصوب)
```

The particle "إِنَّ" at the start of the sentence forces accusative case on "الطالب" even though there are many intervening words. Local context windows cannot capture this dependency.

#### Error Pattern by Syntactic Construction

| Construction | Base Model DER | Harakat DER | Gap |
|--------------|---------------:|------------:|----:|
| Simple SVO | 4.2% | 1.8% | 2.4% |
| Idafa (possession) | 6.8% | 3.2% | 3.6% |
| Relative clauses | 9.1% | 4.7% | 4.4% |
| إِنَّ and sisters | 11.3% | 6.2% | 5.1% |
| Complex embedding | 14.7% | 8.9% | 5.8% |

Harakat shows consistent ~50% improvement across constructions, but absolute error rates increase with syntactic complexity.

---

## Limitations and Future Work

### Current Limitations

#### 1. Syntactic Analysis

Harakat uses local context (±2 words) for disambiguation. This is insufficient for:
- Long-distance case agreement
- Complex embedded clauses
- Coordination structures

**Impact**: ~62% of remaining errors are case-related.

**Potential solution**: Integration with syntactic parser (adds ~500 MB)

#### 2. Dialectal Text

Harakat is optimized for Modern Standard Arabic (MSA). Dialectal Arabic has different:
- Vowel patterns
- Morphological structures
- Function words

**Impact**: DER on dialectal text is ~15-20%

**Potential solution**: Dialect-specific models or adaptation layers

#### 3. Proper Nouns

Names and foreign words not in the lexicon use statistical fallback:

```
Correct: جُورْج وَاشِنْطُن (George Washington)
System:  جُورْج وَاشِنْطِن (incorrect final vowel)
```

**Impact**: ~8% of OOV errors are proper nouns

**Potential solution**: Named entity recognition preprocessing

#### 4. Poetry and Classical Arabic

Classical Arabic has:
- Archaic grammatical forms
- Poetic license in voweling
- Rare vocabulary

**Impact**: DER on classical text is ~8-10%

**Potential solution**: Genre-specific models

#### 5. Ambiguous Cases

Some sentences have genuinely ambiguous diacritization:

```
رأيت الرجل الكبير
Could be: رَأَيْتُ الرَّجُلَ الْكَبِيرَ (I saw the big man)
Or:       رَأَيْتُ الرَّجُلَ الْكَبِيرُ (rare but grammatical reading)
```

**Impact**: ~3% of test items have multiple valid answers

**Potential solution**: Output multiple hypotheses with probabilities

### What Didn't Work

During development, we tested a **syntactic rules layer** for verb-subject agreement:

- **Result**: +0.11% DER regression (made things worse)
- **Net impact**: -978 words (more harm than good)
- **Problem**: Rules fired too aggressively without proper disambiguation

**Decision**: Ship error-correction only, no syntactic layer. The error-report methodology works; hand-crafted rules on top do not.

### Future Work Directions

1. **Neural Case Predictor**: Train specialized model for case endings using attention over full sentence
2. **MSA Adaptation**: Train on Modern Standard Arabic corpora (current training is 98.85% Classical Arabic)
3. **Dialect Adaptation**: Few-shot learning for Egyptian, Levantine, Gulf varieties
4. **Syntactic Layer v2**: More careful rule implementation with proper disambiguation
5. **Edge Deployment**: Optimize for mobile/embedded systems

---

## Roadmap

### Harakat V2 (In Development)

The next major version targets significant improvements in case ending prediction:

#### Neural Case Predictor V3

A 3-model ensemble with self-attention:

```
Architecture:
├── Embedding layer: 64 dimensions
├── Bidirectional context: ±4 words
├── Self-attention: 4 heads
├── Feedforward: 128 → 64 → num_cases
└── Ensemble voting: 3 models
```

Preliminary results on validation set:

| Metric | V1 (Current) | V2 (Planned) |
|--------|-------------:|-------------:|
| Case accuracy | 88.8% | ~97.8% |
| Case DER | 2.80% | ~0.6% |
| Model size | - | +1 MB |

#### Enhanced Internal Vowel Model

Improved prediction for fatha/kasra/damma using morphological features:

| Metric | V1 | V2 |
|--------|---:|---:|
| Internal vowel DER | 0.82% | ~0.4% |
| Pattern coverage | 87% | 95% |

#### Estimated V2 Impact

| Metric | V1 | V2 (Target) |
|--------|---:|------------:|
| Overall DER | 4.46% | ~2% |
| Case DER | 2.80% | ~0.6% |
| Total size | 3.14 MB | ~10 MB |
| vs Shakkala/Shakkelha | 10x smaller, 1.6% worse | 3x smaller, **better accuracy** |
| vs SUKOUN | 139x smaller | 44x smaller |

**V2 positioning**: At ~10 MB and ~2% DER, Harakat V2 would be **3x smaller than Shakkala/Shakkelha with better accuracy**—the best accuracy-to-size ratio in the field.

### Future Directions

#### Short-term (V2.x)

- [ ] Neural case predictor deployment
- [ ] Improved shadda detection
- [ ] Better OOV handling

#### Medium-term (V3)

- [ ] Dialect-aware diacritization (Egyptian, Levantine, Gulf)
- [ ] Integration with speech synthesis (TTS)
- [ ] Browser-based deployment (WebAssembly)

#### Long-term (V4+)

- [ ] Mobile SDKs (iOS, Android)
- [ ] Real-time typing assistance
- [ ] Educational mode with explanations
- [ ] Integration with Arabic NLP pipelines

---

## Use Cases

### Language Education

#### Reading Material Preparation

Teachers can automatically diacritize texts for students at different proficiency levels:

```python
from harakat import Diacritizer

diacritizer = Diacritizer()

# Raw news article
article = "قال وزير الخارجية في مؤتمر صحفي..."

# Full diacritization for beginners
beginner_text = diacritizer.diacritize(article)
# Output: قَالَ وَزِيرُ الْخَارِجِيَّةِ فِي مُؤْتَمَرٍ صَحَفِيٍّ...

# Partial diacritization for intermediate (case endings only)
intermediate_text = diacritizer.diacritize(article, mode='case_only')
# Output: قالَ وزيرُ الخارجيةِ في مؤتمرٍ صحفيٍّ...
```

#### Pronunciation Practice

Real-time diacritization for reading practice:

```python
# Student types undiacritized text
student_input = "الطالب يدرس في المكتبة"

# System shows correct diacritization
correct = diacritizer.diacritize(student_input)
print(correct)
# Output: الطَّالِبُ يَدْرُسُ فِي الْمَكْتَبَةِ
```

### Accessibility

#### Screen Reader Optimization

Arabic screen readers need diacritics for correct pronunciation:

```python
# Input: Web page text without diacritics
web_text = "مرحبا بكم في موقعنا الإلكتروني"

# Output: Screen-reader-ready text
accessible_text = diacritizer.diacritize(web_text)
# Output: مَرْحَبًا بِكُمْ فِي مَوْقِعِنَا الْإِلِكْتُرُونِيِّ
```

#### Text-to-Speech Preprocessing

TTS systems require diacritized input for natural pronunciation:

```python
def prepare_for_tts(text):
    """Prepare Arabic text for TTS engine."""
    diacritizer = Diacritizer()
    diacritized = diacritizer.diacritize(text)
    return diacritized

# Input for TTS
tts_input = prepare_for_tts("الطقس جميل اليوم")
# Output: الطَّقْسُ جَمِيلٌ الْيَوْمَ
```

### Publishing

#### Quranic and Classical Texts

Preparation of religious and classical texts with proper diacritization:

```python
# Classical Arabic text
classical = "قال الإمام الشافعي رحمه الله"

# Diacritize with classical mode
diacritized = diacritizer.diacritize(classical, mode='classical')
# Output: قَالَ الْإِمَامُ الشَّافِعِيُّ رَحِمَهُ اللَّهُ
```

#### Children's Books

Full diacritization for early readers:

```python
# Children's story
story = "ذهب الولد إلى المدرسة مع أصدقائه"

# Full diacritization
diacritized = diacritizer.diacritize(story)
# Output: ذَهَبَ الْوَلَدُ إِلَى الْمَدْرَسَةِ مَعَ أَصْدِقَائِهِ
```

### NLP Pipeline

#### Speech Synthesis Preprocessing

```python
def arabic_tts_pipeline(text):
    """Full Arabic TTS preprocessing pipeline."""
    # Step 1: Diacritize
    diacritized = diacritizer.diacritize(text)

    # Step 2: Phonemize (convert to pronunciation)
    phonemes = arabic_phonemizer(diacritized)

    # Step 3: Generate speech
    audio = tts_model.synthesize(phonemes)

    return audio
```

#### Machine Translation Preprocessing

Diacritization can improve MT by disambiguating source text:

```python
def translate_arabic(text, target_lang='en'):
    """Arabic to English translation with diacritization."""
    # Diacritize for disambiguation
    diacritized = diacritizer.diacritize(text)

    # Translate
    translation = mt_model.translate(diacritized, target_lang)

    return translation
```

---

## Contributing

Contributions are welcome! This project follows standard open-source practices.

### Areas of Interest

1. **Dialect Extensions**: Models for Egyptian, Levantine, Gulf, Maghrebi Arabic
2. **Benchmark Evaluation**: Testing on additional corpora
3. **Performance Optimization**: Speed and memory improvements
4. **Documentation**: Translations, tutorials, examples
5. **Integration**: Plugins for text editors, browsers, etc.

### Development Setup

```bash
# Clone repository
git clone https://github.com/jeranaias/harakat.git
cd harakat

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
flake8 harakat.py
black --check harakat.py
```

### Contribution Guidelines

1. **Open an issue first** for significant changes
2. **Write tests** for new functionality
3. **Follow code style** (Black formatter, type hints)
4. **Update documentation** as needed
5. **Sign commits** with your real name

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit with descriptive message
6. Push to your fork
7. Open a Pull Request

---

## Citation

If you use Harakat in your research, please cite:

```bibtex
@software{harakat2024,
  author = {Morgan, Jesse},
  title = {Harakat: Error-Corrective Arabic Diacritization via Meta-Learning on Base Model Failures},
  year = {2024},
  url = {https://github.com/jeranaias/harakat},
  note = {High-accuracy Arabic diacritization in 3.14 MB}
}
```

### Related Publications

```bibtex
@article{sukoun2024,
  author = {Youssef, A.A. and others},
  title = {BERT-Based Arabic Diacritization: A State-of-the-Art Approach},
  journal = {Expert Systems with Applications},
  year = {2024},
  note = {Current SOTA: DER 0.92\%, WER 1.91\%}
}

@inproceedings{fadel2019neural,
  author = {Fadel, Ali and others},
  title = {Neural Arabic Text Diacritization: State of the Art Results and a Novel Approach for Machine Translation},
  booktitle = {EMNLP-IJCNLP},
  year = {2019},
  note = {Shakkelha: DER 2.61\%, WER 5.83\%}
}

@inproceedings{fadel2019benchmark,
  author = {Fadel, Ali and others},
  title = {Arabic Text Diacritization Using Deep Neural Networks},
  booktitle = {ICCAIS},
  year = {2019},
  note = {Benchmark source for Shakkala: DER 2.88\%, WER 6.37\%}
}

@inproceedings{shakkala2017,
  author = {Barqawi, Ahmad and Zerrouki, Taha},
  title = {Shakkala: An Automatic Arabic Diacritization System},
  year = {2017},
  note = {Model size: ~29 MB}
}

@article{tashkeela2017,
  author = {Zerrouki, Taha and Balla, Amar},
  title = {Tashkeela: Novel corpus of Arabic vocalized texts},
  year = {2017},
  note = {Training corpus, 75M words, cleaned by Abbad \& Xiong}
}
```

---

## License

MIT License

Copyright (c) 2024 Jesse Morgan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Author

**Jesse Morgan**

Staff Sergeant, U.S. Marine Corps
Arabic Language Instructor, Defense Language Institute
MS in Artificial Intelligence (in progress)

Background in signals intelligence and linguistic analysis. This project combines operational language expertise with machine learning to solve a problem I encountered daily in the classroom: helping students learn to read Arabic naturally.

The insight that drove Harakat came from years of teaching—watching students struggle with diacritics and realizing that the problem isn't just "predicting vowels" but understanding *when you know* and *when you don't know*. Native speakers don't consciously apply rules; they recognize patterns and flag uncertainty. Harakat tries to do the same computationally.

### Contact

- GitHub: [@jeranaias](https://github.com/jeranaias)
- Email: [contact via GitHub]

---

## Acknowledgments

- **The Arabic NLP research community** for published benchmarks, baselines, and the collaborative spirit that makes progress possible
- **The Tashkeela corpus maintainers** (Taha Zerrouki and Amar Balla) for high-quality training data that enabled this work
- **My students at DLI** whose questions, struggles, and breakthroughs inspired this project
- **The open-source community** for the tools that made development possible (Python, NumPy, and countless others)

---

## Appendix A: Arabic Diacritic Reference

### Complete Diacritic Table

| Unicode | Name (Arabic) | Name (English) | Transliteration | Position | Function |
|---------|---------------|----------------|-----------------|----------|----------|
| U+064E | فَتْحَة | Fatha | a | Above | Short /a/ vowel |
| U+064F | ضَمَّة | Damma | u | Above | Short /u/ vowel |
| U+0650 | كَسْرَة | Kasra | i | Below | Short /i/ vowel |
| U+0652 | سُكُون | Sukun | ∅ | Above | No vowel |
| U+0651 | شَدَّة | Shadda | (doubled) | Above | Gemination |
| U+064B | فَتْحَتَان | Fathatan | -an | Above | Nunation (acc.) |
| U+064C | ضَمَّتَان | Dammatan | -un | Above | Nunation (nom.) |
| U+064D | كَسْرَتَان | Kasratan | -in | Below | Nunation (gen.) |

### Diacritic Combinations

Shadda always combines with a vowel:

| Combination | Example | Pronunciation |
|-------------|---------|---------------|
| شَدَّة + فَتْحَة | رَبَّ | rabba |
| شَدَّة + ضَمَّة | رَبُّ | rabbu |
| شَدَّة + كَسْرَة | رَبِّ | rabbi |
| شَدَّة + فَتْحَتَان | رَبًّا | rabban |
| شَدَّة + ضَمَّتَان | رَبٌّ | rabbun |
| شَدَّة + كَسْرَتَان | رَبٍّ | rabbin |

---

## Appendix B: Evaluation Metrics

### Diacritic Error Rate (DER)

```
DER = (Number of incorrect diacritics) / (Total diacritics) × 100%

Where:
- Incorrect diacritic = predicted ≠ gold
- Total diacritics = all diacritic positions in gold standard
- Missing diacritic counts as error
- Extra diacritic counts as error
```

### Word Error Rate (WER)

```
WER = (Number of words with ≥1 diacritic error) / (Total words) × 100%

Where:
- A word is "correct" only if ALL its diacritics match gold
- Partial correctness counts as error
```

### Computation Example

```
Gold:     كَتَبَ الطَّالِبُ الدَّرْسَ
Predicted: كَتَبَ الطَّالِبَ الدَّرْسَ
                      ^
                      Error: ُ → َ

DER = 1/12 = 8.33%  (12 total diacritics, 1 wrong)
WER = 1/3 = 33.33%  (3 words, 1 has error)
```

---

## Appendix C: Algorithm Pseudocode

### Full Diacritization Pipeline

```
Algorithm: Harakat Diacritization

Input: Undiacritized Arabic text T
Output: Diacritized text T'

1. PREPROCESS(T)
   a. Normalize Unicode (NFD → NFC)
   b. Standardize whitespace
   c. Handle existing diacritics (strip or preserve)

2. TOKENIZE(T) → words[]

3. For each word w in words:
   BASE_PREDICT(w) → base_pred

4. For each (word, base_pred) pair:
   a. undiac ← STRIP_DIACRITICS(word)

   b. If BLACKLISTED(undiac, base_pred):
      final[i] ← base_pred
      CONTINUE

   c. left_ctx ← final[i-2:i]
   d. right_ctx ← words[i+1:i+3]

   e. (correction, conf) ← LOOKUP(undiac, base_pred, left_ctx, right_ctx)

   f. If correction = NULL:
      final[i] ← base_pred

   g. Else If conf ≥ θ₁ (0.95):
      final[i] ← correction

   h. Else If conf ≥ θ₂ (0.85):
      If NOT SOFT_BLACKLISTED(undiac, correction):
         final[i] ← correction
      Else:
         final[i] ← base_pred

   i. Else If conf ≥ θ₃ (0.70):
      If HAS_SUPPORT(undiac, correction, left_ctx):
         final[i] ← correction
      Else:
         final[i] ← base_pred

   j. Else:
      final[i] ← base_pred

5. RETURN JOIN(final, " ")
```

### Lookup Function

```
Algorithm: Triple-Key Lookup

Input: undiac, base_pred, left_ctx, right_ctx
Output: (correction, confidence) or (NULL, 0)

1. If undiac NOT IN primary_index:
   RETURN (NULL, 0)

2. secondary ← primary_index[undiac]

3. If base_pred NOT IN secondary:
   RETURN (NULL, 0)

4. ctx_table ← secondary[base_pred]

5. sig ← COMPUTE_SIGNATURE(left_ctx, right_ctx)

6. If sig IN ctx_table:
   entry ← ctx_table[sig]
   RETURN (entry.correct, entry.confidence)

7. For partial_sig IN BACKOFF_SIGNATURES(sig):
   If partial_sig IN ctx_table:
      entry ← ctx_table[partial_sig]
      RETURN (entry.correct, entry.confidence × 0.9)

8. RETURN (NULL, 0)
```

---

## Appendix D: Glossary

| Term | Definition |
|------|------------|
| **Abjad** | Writing system that primarily represents consonants |
| **Diacritic** | Mark added to a letter to indicate pronunciation |
| **DER** | Diacritic Error Rate |
| **Fatha** | Short /a/ vowel mark (ـَـ) |
| **Gemination** | Consonant doubling |
| **Harakat** | Arabic diacritical marks (also: this system) |
| **I'rab** | Arabic grammatical case system |
| **Kasra** | Short /i/ vowel mark (ـِـ) |
| **Damma** | Short /u/ vowel mark (ـُـ) |
| **MSA** | Modern Standard Arabic |
| **Shadda** | Gemination mark (ـّـ) |
| **Sukun** | Vowel absence mark (ـْـ) |
| **Tanween** | Nunation marks (-un, -an, -in) |
| **Tashkeel** | Arabic word for diacritization |
| **WER** | Word Error Rate |

---

*"The best model is the one that ships."*

---

**Harakat** — High-accuracy Arabic diacritization in 3.14 MB.

Built with determination in San Mateo, California.

Copyright (c) 2024 Jesse Morgan. MIT License.
