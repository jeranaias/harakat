#!/usr/bin/env python3
"""
Basic usage example for Harakat Arabic diacritization.
"""

import sys
sys.path.insert(0, '..')

from harakat import diacritize

# Example 1: Simple diacritization
text = "كتب الطالب الدرس"
result = diacritize(text)
print(f"Input:  {text}")
print(f"Output: {result}")
print()

# Example 2: Common phrases
phrases = [
    "السلام عليكم",
    "الحمد لله رب العالمين",
    "بسم الله الرحمن الرحيم",
    "كيف حالك",
    "اهلا وسهلا",
]

print("Common Phrases:")
print("-" * 50)
for phrase in phrases:
    diacritized = diacritize(phrase)
    print(f"{phrase}")
    print(f"  -> {diacritized}")
    print()

# Example 3: Longer text
paragraph = """
العلم نور والجهل ظلام. من طلب العلم وجب عليه الصبر والمثابرة.
قال الحكماء ان العلم في الصغر كالنقش على الحجر.
"""

print("Paragraph Example:")
print("-" * 50)
print("Input:")
print(paragraph)
print("\nOutput:")
print(diacritize(paragraph))
