"""
Test suite for Harakat Arabic Diacritization.

Run with: pytest tests/ -v
"""
import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harakat import diacritize, __version__


class TestBasicDiacritization:
    """Test basic diacritization functionality."""

    def test_simple_word(self):
        """Test diacritization of a simple word."""
        result = diacritize("مرحبا")
        # Should contain at least some diacritics
        assert any(c in result for c in '\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652')

    def test_sentence(self):
        """Test diacritization of a full sentence."""
        result = diacritize("ذهب الولد الى المدرسة")
        assert len(result) > len("ذهب الولد الى المدرسة")  # Should be longer with diacritics

    def test_preserves_spaces(self):
        """Test that spaces are preserved."""
        result = diacritize("كلمة اخرى")
        assert " " in result

    def test_multiple_sentences(self):
        """Test multiple sentences."""
        result = diacritize("هذا كتاب. هذا قلم.")
        assert "." in result  # Punctuation preserved


class TestEdgeCases:
    """Test edge cases and special inputs."""

    def test_empty_string(self):
        """Test empty input."""
        result = diacritize("")
        assert result == ""

    def test_whitespace_only(self):
        """Test whitespace-only input."""
        result = diacritize("   ")
        assert result.strip() == ""

    def test_non_arabic(self):
        """Test non-Arabic text passes through."""
        result = diacritize("Hello World")
        assert result == "Hello World"

    def test_mixed_arabic_english(self):
        """Test mixed Arabic and English."""
        result = diacritize("مرحبا Hello")
        assert "Hello" in result

    def test_numbers(self):
        """Test Arabic text with numbers."""
        result = diacritize("عام 2024")
        assert "2024" in result

    def test_punctuation(self):
        """Test punctuation is preserved."""
        result = diacritize("مرحبا!")
        assert "!" in result


class TestQuranMode:
    """Test Quran-specific diacritization."""

    def test_basmala(self):
        """Test Basmala diacritization."""
        result = diacritize("بسم الله الرحمن الرحيم", quran_mode=True)
        # Should have correct Quranic diacritization
        assert "بِسْمِ" in result or "بِسْم" in result

    def test_fatiha(self):
        """Test Al-Fatiha opening."""
        result = diacritize("الحمد لله رب العالمين", quran_mode=True)
        assert "الْحَمْدُ" in result or "الحَمْد" in result

    def test_quran_mode_false(self):
        """Test explicit non-Quran mode."""
        result = diacritize("بسم الله الرحمن الرحيم", quran_mode=False)
        # Should still work, just use general diacritization
        assert len(result) > 0

    def test_quran_auto_detect(self):
        """Test auto-detection of Quranic text."""
        # Common Quranic phrase should be auto-detected
        result = diacritize("قل هو الله احد")
        assert len(result) > len("قل هو الله احد")


class TestUnicodeHandling:
    """Test Unicode normalization and handling."""

    def test_already_diacritized(self):
        """Test input that already has diacritics."""
        result = diacritize("مَرْحَبًا")
        # Should handle gracefully
        assert len(result) > 0

    def test_hamza_variants(self):
        """Test different hamza forms."""
        result1 = diacritize("أحمد")
        result2 = diacritize("احمد")
        # Both should produce valid output
        assert len(result1) > 0
        assert len(result2) > 0

    def test_ta_marbuta(self):
        """Test ta marbuta handling."""
        result = diacritize("مدرسة")
        assert "ة" in result or "ه" in result


class TestPerformance:
    """Test performance characteristics."""

    def test_long_text(self):
        """Test handling of longer text."""
        long_text = "هذا نص طويل. " * 100
        result = diacritize(long_text)
        assert len(result) > 0

    def test_single_character(self):
        """Test single character input."""
        result = diacritize("ا")
        assert len(result) >= 1


class TestVersion:
    """Test version information."""

    def test_version_exists(self):
        """Test that version is defined."""
        assert __version__ is not None

    def test_version_format(self):
        """Test version format."""
        parts = __version__.split('.')
        assert len(parts) >= 2  # At least major.minor


class TestConsistency:
    """Test output consistency."""

    def test_deterministic(self):
        """Test that same input gives same output."""
        text = "الكتاب على الطاولة"
        result1 = diacritize(text)
        result2 = diacritize(text)
        assert result1 == result2

    def test_word_boundaries(self):
        """Test word boundary preservation."""
        result = diacritize("كلمة كلمة كلمة")
        words = result.split()
        assert len(words) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
