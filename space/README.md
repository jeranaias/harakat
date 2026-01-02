---
title: Harakat - Arabic Diacritization
emoji: 🔤
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
short_description: High-accuracy Arabic diacritization (2.29% DER, 99.997% Quran)
---

# Harakat - Arabic Diacritization

**High-Accuracy Arabic Text Diacritization (Tashkeel)**

![Demo](https://raw.githubusercontent.com/jeranaias/harakat/main/docs/demo.gif)

## Performance

| Metric | Value |
|--------|-------|
| Diacritic Error Rate (DER) | **2.29%** |
| DER (without case endings) | **1.53%** |
| Word Error Rate (WER) | **6.37%** |
| Quran Accuracy | **99.997%** |
| Model Size | **6.7 MB** |
| vs SOTA | **62x smaller** |

## Features

- **Automatic Quran Detection**: Quranic phrases auto-detected with near-perfect accuracy
- **Lightweight**: Single-file model, runs anywhere
- **Production-Ready**: REST API with batch processing support
- **Bilingual Interface**: English & Arabic UI

## Usage

Paste any Arabic text and click Submit to add diacritical marks.

**Examples:**
- `بسم الله الرحمن الرحيم` → `بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ`
- `العلم نور والجهل ظلام` → `الْعِلْمُ نُورٌ وَالْجَهْلُ ظَلَامٌ`

## API

### REST API (Recommended)

```python
import requests

response = requests.post(
    "https://jcmguy-harakat.hf.space/api/diacritize",
    json={"text": "بسم الله الرحمن الرحيم"}
)
print(response.json()["output"])
# بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ
```

### Gradio API

```javascript
// Step 1: Submit request
const call = await fetch('https://jcmguy-harakat.hf.space/call/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ data: ['مرحبا بالعالم'] })
});
const { event_id } = await call.json();

// Step 2: Get result
const result = await fetch(`/call/predict/${event_id}`);
// Returns: data: ["مَرْحَبًا بِالْعَالَمِ"]
```

## API Documentation

- **Swagger UI**: [/docs](https://jcmguy-harakat.hf.space/docs)
- **ReDoc**: [/redoc](https://jcmguy-harakat.hf.space/redoc)

## Links

- [GitHub Repository](https://github.com/jeranaias/harakat)
- [Documentation](https://jeranaias.github.io/harakat)
- [Google Colab](https://colab.research.google.com/github/jeranaias/harakat/blob/main/examples/harakat_demo.ipynb)

---

**Author**: Jesse Morgan | **License**: MIT
