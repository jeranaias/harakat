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
short_description: High-accuracy Arabic diacritization (2.13% DER)
---

# Harakat - Arabic Diacritization

**High-Accuracy Arabic Text Diacritization**

- **2.13% DER** on Tashkeela benchmark
- **6 MB** total model size
- **73x smaller** than state-of-the-art

## Usage

Paste any Arabic text and click Submit to add diacritical marks (tashkeel/harakat).

## API

Use Gradio's API format:

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

## Links

- [GitHub Repository](https://github.com/jeranaias/harakat)
- [Documentation](https://github.com/jeranaias/harakat#readme)
