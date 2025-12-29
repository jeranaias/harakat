#!/usr/bin/env python3
"""Debug particle rule issues."""

import sys
import os

if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')

harakat_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, harakat_dir)

from harakat_v2 import diacritize_with_sweep as v2_diacritize
from harakat_v3.rules import apply_particle_rules

# Test specific examples from error analysis
test_cases = [
    # Gold reference from test set
    "إِلَى الْإِسْلَامِ",
    "إِلَى قُلُوبِهِمْ",
    "إِلَى الْقَلْبِ",
]

print("=" * 70)
print("DEBUG: Particle Rule Impact")
print("=" * 70)

for gold in test_cases:
    # Strip diacritics to simulate input
    undiac = ''.join(c for c in gold if c not in 'ًٌٍَُِّْ')

    # V2 output
    v2_out = v2_diacritize(undiac)

    # V3 (with particle rules)
    v3_out = apply_particle_rules(v2_out)

    print(f"\nGold:     {gold}")
    print(f"V2 out:   {v2_out}")
    print(f"V3 out:   {v3_out}")
    print(f"V2==Gold: {v2_out == gold}")
    print(f"V3==Gold: {v3_out == gold}")

    # Check character by character
    if v3_out != gold:
        print("  Differences:")
        for i, (g, v) in enumerate(zip(gold, v3_out)):
            if g != v:
                print(f"    Position {i}: gold='{g}' (U+{ord(g):04X}) vs v3='{v}' (U+{ord(v):04X})")
