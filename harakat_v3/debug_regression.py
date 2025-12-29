#!/usr/bin/env python3
"""Debug regression cases."""

import sys
import os

if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')

harakat_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, harakat_dir)

from harakat_v2 import diacritize_with_sweep as v2_diacritize
from harakat_v3.rules import apply_particle_rules

HARAKAT = 'ًٌٍَُِّْ'

def strip_harakat(text):
    return ''.join(c for c in text if c not in HARAKAT)

# Regression case
gold = "وَيَرُدُّ عَلَيْهِ أَنَّ فَاقِدَ الطَّهُورَيْنِ وَنَحْوَهُ لَيْسَ لَهُ صَلَاةٌ إلَّا بِالتَّيَمُّمِ"
undiac = strip_harakat(gold)

v2_out = v2_diacritize(undiac)
v3_out = apply_particle_rules(v2_out)

print("Regression analysis:")
print("=" * 70)
print(f"Gold: {gold}")
print(f"V2:   {v2_out}")
print(f"V3:   {v3_out}")
print()

# Word by word
gold_words = gold.split()
v2_words = v2_out.split()
v3_words = v3_out.split()

print("Word by word comparison:")
print("-" * 70)
for i, (g, v2, v3) in enumerate(zip(gold_words, v2_words, v3_words)):
    base = strip_harakat(g)
    marker = ""
    if v2 != v3:
        if g == v2:
            marker = " <- V2 correct, V3 WRONG"
        elif g == v3:
            marker = " <- V3 correct, V2 wrong"
        else:
            marker = " <- BOTH wrong, but different"
    elif v2 != g:
        marker = " <- both wrong"

    if v2 != v3 or v2 != g:
        print(f"Word {i}: base='{base}'")
        print(f"  Gold: {g}")
        print(f"  V2:   {v2}")
        print(f"  V3:   {v3}{marker}")
        print()
