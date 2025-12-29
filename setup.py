#!/usr/bin/env python3
"""
Harakat - Arabic Diacritization System

Install with: pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="harakat",
    version="3.5.0",
    author="Jesse Morgan",
    author_email="jeranaias@gmail.com",
    description="High-accuracy Arabic diacritization (1.71% DER)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jeranaias/harakat",
    py_modules=["harakat", "harakat_v1"],
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Natural Language :: Arabic",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "harakat=harakat:main",
        ],
    },
    keywords="arabic diacritization nlp text-processing harakat tashkeel",
    project_urls={
        "Bug Reports": "https://github.com/jeranaias/harakat/issues",
        "Source": "https://github.com/jeranaias/harakat",
    },
)
