# Quran Diacritization Module

This module provides **99.9976% accurate** diacritization for Quranic text using a pre-built lookup table from the authentic Uthmani Quran.

## Features

- **Phrase-level matching**: Full ayah text is matched exactly when possible
- **Word-level fallback**: Individual words are diacritized using the most common Quranic form
- **Verse lookup**: Retrieve the exact diacritization of any specific verse by surah:ayah
- **Auto-detection**: Automatically detect Quranic text patterns

## Usage

```python
from quran import diacritize_quran, is_likely_quran, diacritize_quran_verse

# Diacritize Quranic text
text = "ان الذين امنوا وعملوا الصالحات كانت لهم جنات الفردوس نزلا"
result = diacritize_quran(text)
# إِنَّ الَّذِينَ ءَامَنُوا وَعَمِلُوا الصَّالِحَاتِ كَانَتْ لَهُمْ جَنَّاتُ الْفِرْدَوْسِ نُزُلًا

# Check if text is Quranic
if is_likely_quran(text):
    result = diacritize_quran(text)

# Get a specific verse
verse = diacritize_quran_verse(1, 1)  # Al-Fatiha, verse 1
# بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ
```

## CLI Usage

```bash
python harakat.py --quran "بسم الله الرحمن الرحيم"
```

## Accuracy

- **99.9976%** word-level accuracy on the full Quran (82,242 words)
- Only 2 ambiguous cases where different ayahs share the same undiacritized text
- Both "errors" are valid Quranic diacritizations from different surahs

## Files

| File | Description | Size |
|------|-------------|------|
| `quran_lookup_optimized.json` | Compact lookup table | 3.87 MB (0.51 MB LZMA) |
| `quran_uthmani.json` | Full Quran source data | 2.3 MB |
| `quran_diacritizer.py` | Main diacritization module | ~5 KB |
| `build_optimized_lookup.py` | Build script for lookup table | ~3 KB |

## Data Source

The Quran text is sourced from [alquran.cloud](https://alquran.cloud/) using the Uthmani script edition (`quran-uthmani`), which provides:

- 6,236 ayahs (verses)
- 114 surahs (chapters)
- Full tashkeel (diacritical marks)
- Authentic Uthmani orthography

## How It Works

1. **Phrase Matching**: First tries to match the entire input as a known ayah
2. **Word Lookup**: Falls back to word-by-word diacritization using the most common form
3. **Fallback**: Unknown words can be passed to a fallback function (e.g., the general diacritizer)

### Normalization

Input text is normalized before lookup to handle common variations:
- `أ إ آ` → `ا` (hamza variations to plain alif)
- `ء` → removed (standalone hamza)
- `ة` → `ه` (ta marbuta to ha)
- `ى` → `ي` (alif maqsura to ya)
- Uthmani markers removed (small waw, small ya, etc.)

## Rebuilding the Lookup

To rebuild the lookup table from the Quran source:

```bash
cd quran
python build_optimized_lookup.py
```

This requires `quran_uthmani.json` to be present (downloaded separately).

## Integration with Harakat

The Quran module integrates with the main Harakat diacritizer:

```python
from harakat import diacritize

# Auto-detect Quran mode
result = diacritize(text)  # Uses Quran lookup if detected

# Force Quran mode
result = diacritize(text, quran_mode=True)

# Disable Quran mode
result = diacritize(text, quran_mode=False)
```
