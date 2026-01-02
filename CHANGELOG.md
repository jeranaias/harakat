# Changelog

All notable changes to Harakat will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.5.1] - 2025-01-01

### Added
- Docker support with multi-stage build (`Dockerfile`, `docker-compose.yml`)
- OpenAPI documentation with Swagger UI at `/docs` and ReDoc at `/redoc`
- Batch diacritization API endpoint (`/api/diacritize/batch`)
- Terminal demo script and GIF recording tools
- CONTRIBUTING.md with contribution guidelines

### Fixed
- Famous proverb "من جد وجد ومن زرع حصد" now correctly diacritized
- Fixed من/مَنْ (who) vs مِنْ (from) disambiguation in proverb context

## [3.5.0] - 2024-12-29

### Added
- **Quran Mode**: 99.997% accuracy on Quranic text (82,240/82,242 words correct)
- Automatic detection of Quranic phrases
- Embedded Uthmani Quran lookup table (~0.5 MB LZMA compressed)
- `--quran` CLI flag for explicit Quran mode
- `quran_mode` parameter in Python API
- Google Colab interactive notebook
- Comprehensive test suite (23 tests)
- GitHub Actions CI (Python 3.8-3.12)
- Modern Python packaging (pyproject.toml)

### Changed
- Model size: 6 MB → 6.7 MB (includes Quran lookup)
- Improved DER (no case): 1.95% → 1.53% (22% improvement)
- Updated documentation with accurate metrics

### Fixed
- Consistent diacritic statistics across all documentation
- Various minor bug fixes

## [3.0.0] - 2024-12-15

### Added
- ML homograph disambiguation (50+ classifiers)
- Voice correction system (active/passive)
- Calibrated confidence thresholds per classifier
- TF-IDF context features for disambiguation

### Changed
- DER: 2.29% (maintained from V2)
- Improved internal vowel accuracy

## [2.0.0] - 2024-11-01

### Added
- Neural case predictor (BiLSTM + Attention ensemble)
- 97.4% case ending accuracy
- Hybrid correction system

### Changed
- DER: 4.46% → 2.29% (49% reduction)
- Model size: 3.14 MB → ~6 MB

## [1.0.0] - 2024-10-01

### Added
- Initial release
- Error-report disambiguation methodology
- Triple-key lookup architecture
- Confidence routing system
- Regression blacklist

### Performance
- DER: 4.46%
- WER: 12.19%
- Model size: 3.14 MB
- 77% DER reduction from base model (9.06%)

---

## Version Comparison

| Version | DER | DER (no case) | WER | Size | Key Feature |
|---------|-----|---------------|-----|------|-------------|
| 3.5.0 | 2.29% | 1.53% | 6.37% | 6.7 MB | Quran mode |
| 3.0.0 | 2.29% | 1.95% | 6.44% | ~6 MB | ML classifiers |
| 2.0.0 | 2.29% | — | 6.44% | ~6 MB | Neural case |
| 1.0.0 | 4.46% | — | 12.19% | 3.14 MB | Initial release |

---

## Upgrade Guide

### From 3.0.0 to 3.5.0

No breaking changes. New features:

```python
# Quran mode (new)
result = diacritize(text, quran_mode=True)

# Auto-detection (default behavior)
result = diacritize(text)  # Detects Quran automatically
```

### From 2.0.0 to 3.0.0

No breaking changes. ML corrections are applied automatically.

### From 1.0.0 to 2.0.0

No breaking changes. Neural case prediction is applied automatically.

---

[3.5.0]: https://github.com/jeranaias/harakat/compare/v3.0.0...v3.5.0
[3.0.0]: https://github.com/jeranaias/harakat/compare/v2.0.0...v3.0.0
[2.0.0]: https://github.com/jeranaias/harakat/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/jeranaias/harakat/releases/tag/v1.0.0
