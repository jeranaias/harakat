# Elite Upgrade Plan

Transform Harakat into a showcase-worthy project for resume/portfolio.

## Status Tracker

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Add more badges to README | Pending | HF Space, Demo, Downloads, Build |
| 2 | GitHub Actions CI | Pending | Test on push, Python 3.8-3.12 |
| 3 | PyPI package | Pending | `pip install harakat` |
| 4 | Google Colab notebook | Pending | One-click interactive demo |
| 5 | GIF demo in README | Pending | Screen recording of demo |
| 6 | Contributing.md | Pending | Contribution guidelines |
| 7 | Changelog.md | Pending | Version history |
| 8 | Docker image | Pending | Containerized deployment |
| 9 | OpenAPI docs | Pending | API documentation |
| 10 | Test suite + coverage | Pending | pytest + coverage badge |

---

## Task 1: Add More Badges

**Target badges:**
```markdown
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DER: 2.29%](https://img.shields.io/badge/DER-2.29%25-brightgreen.svg)]()
[![Quran: 99.997%](https://img.shields.io/badge/Quran-99.997%25-gold.svg)]()
[![Model Size](https://img.shields.io/badge/Size-6.7MB-blue.svg)]()
[![Demo](https://img.shields.io/badge/Demo-Live-success.svg)](https://jeranaias.github.io/harakat/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Space-orange.svg)](https://huggingface.co/spaces/jcmguy/harakat)
[![CI](https://github.com/jeranaias/harakat/actions/workflows/ci.yml/badge.svg)](https://github.com/jeranaias/harakat/actions)
[![PyPI](https://img.shields.io/pypi/v/harakat.svg)](https://pypi.org/project/harakat/)
```

---

## Task 2: GitHub Actions CI

**File: `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']

    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest pytest-cov
    - name: Run tests
      run: pytest tests/ -v --cov=harakat --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      if: matrix.python-version == '3.11'
```

---

## Task 3: PyPI Package

**Update `setup.py`:**
- Verify metadata is complete
- Add classifiers
- Test with `pip install -e .`
- Build: `python -m build`
- Upload: `twine upload dist/*`

**Required files:**
- setup.py (update)
- pyproject.toml (create)
- MANIFEST.in (create)

---

## Task 4: Google Colab Notebook

**File: `examples/harakat_demo.ipynb`**

Contents:
1. One-click install cell
2. Basic usage examples
3. Quran mode demo
4. Batch processing example
5. Performance benchmark

Add badge to README:
```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeranaias/harakat/blob/main/examples/harakat_demo.ipynb)
```

---

## Task 5: GIF Demo

**Create animated demo showing:**
1. Paste Arabic text
2. Click diacritize
3. See result

Tools: ScreenToGif, LICEcap, or terminal recording with asciinema

Add to README after title.

---

## Task 6: Contributing.md

**Contents:**
- How to report bugs
- How to request features
- Development setup
- Code style guidelines
- Pull request process
- Code of conduct reference

---

## Task 7: Changelog.md

**Format:** Keep a Changelog (keepachangelog.com)

```markdown
# Changelog

## [3.5.0] - 2024-12-29
### Added
- Quran mode with 99.997% accuracy
- Embedded Quran lookup table

### Changed
- Model size: 6 MB → 6.7 MB (includes Quran)

## [3.0.0] - 2024-XX-XX
### Added
- ML homograph disambiguation
- Voice correction system
...
```

---

## Task 8: Docker Image

**File: `Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY harakat.py requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["python", "-m", "harakat", "--serve"]
```

**File: `docker-compose.yml`** for easy local testing

---

## Task 9: OpenAPI Docs

**Enhance HF Space API with:**
- Swagger UI at /docs
- OpenAPI schema at /openapi.json
- Request/response examples

FastAPI already provides this - just needs documentation.

---

## Task 10: Test Suite + Coverage

**File: `tests/test_harakat.py`**

```python
import pytest
from harakat import diacritize

def test_basic_diacritization():
    result = diacritize("مرحبا")
    assert "َ" in result or "ْ" in result

def test_quran_mode():
    result = diacritize("بسم الله الرحمن الرحيم", quran_mode=True)
    assert "بِسْمِ" in result

def test_empty_input():
    assert diacritize("") == ""

def test_non_arabic():
    result = diacritize("Hello World")
    assert result == "Hello World"
```

---

## Execution Order

1. **Badges** (immediate visual impact)
2. **GitHub Actions** (enables CI badge)
3. **Test suite** (needed for CI)
4. **PyPI** (major credibility)
5. **Colab notebook** (interactive demo)
6. **Contributing.md + Changelog.md** (professionalism)
7. **Docker** (deployment option)
8. **OpenAPI docs** (API polish)
9. **GIF demo** (last - needs screen recording)

---

## Success Criteria

- [ ] All badges render correctly
- [ ] CI passes on all Python versions
- [ ] `pip install harakat` works
- [ ] Colab notebook runs without errors
- [ ] Docker container builds and runs
- [ ] Test coverage > 80%
- [ ] All docs are professional quality
