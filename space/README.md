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
short_description: High-accuracy Arabic diacritization (1.71% DER)
---

# Harakat - Arabic Diacritization

**High-Accuracy Arabic Text Diacritization**

- **1.71% DER** on Tashkeela benchmark
- **6 MB** total model size
- **73x smaller** than state-of-the-art

## Usage

Paste any Arabic text and click Submit to add diacritical marks (tashkeel/harakat).

## API

POST to `/api/diacritize` with JSON body:
```json
{"text": "مرحبا بالعالم"}
```

Response:
```json
{"success": true, "input": "مرحبا بالعالم", "output": "مَرْحَبًا بِالْعَالَمِ"}
```

## Links

- [GitHub Repository](https://github.com/jeranaias/harakat)
- [Documentation](https://github.com/jeranaias/harakat#readme)
