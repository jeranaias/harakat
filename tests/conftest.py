"""
Pytest configuration for Harakat tests.
"""
import pytest
import sys
import os

# Ensure harakat module is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_texts():
    """Sample Arabic texts for testing."""
    return {
        'simple': 'مرحبا',
        'sentence': 'ذهب الولد الى المدرسة',
        'quran': 'بسم الله الرحمن الرحيم',
        'mixed': 'مرحبا Hello World',
        'numbers': 'عام 2024 ميلادي',
    }


@pytest.fixture
def quran_verses():
    """Sample Quran verses for testing."""
    return {
        'basmala': 'بسم الله الرحمن الرحيم',
        'fatiha_1': 'الحمد لله رب العالمين',
        'ikhlas': 'قل هو الله احد',
    }
