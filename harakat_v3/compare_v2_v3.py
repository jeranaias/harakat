#!/usr/bin/env python3
"""Compare V2 and V3 outputs directly."""

import sys
import os

if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')

harakat_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, harakat_dir)

from harakat_v2 import diacritize_with_sweep as v2_diacritize
from harakat_v3 import diacritize as v3_diacritize

# Constants
HARAKAT = 'ًٌٍَُِّْ'

def strip_harakat(text):
    return ''.join(c for c in text if c not in HARAKAT)

# Load test file
test_path = os.path.join(harakat_dir, '..', 'paper_eval', 'tashkeela_test.txt')
with open(test_path, 'r', encoding='utf-8') as f:
    lines = [l.strip() for l in f if l.strip()][:100]  # First 100 lines

print("Comparing V2 vs V3 on first 100 test lines...")
print("=" * 70)

v2_better = 0
v3_better = 0
same = 0
diffs = []

for gold_line in lines:
    undiac = strip_harakat(gold_line)

    v2_out = v2_diacritize(undiac)
    v3_out = v3_diacritize(undiac)

    if v2_out == v3_out:
        same += 1
    else:
        # Compare to gold
        v2_errors = sum(1 for a, b in zip(v2_out, gold_line) if a != b)
        v3_errors = sum(1 for a, b in zip(v3_out, gold_line) if a != b)

        if v2_errors < v3_errors:
            v2_better += 1
            if len(diffs) < 5:
                diffs.append(('v2_better', gold_line, v2_out, v3_out))
        elif v3_errors < v2_errors:
            v3_better += 1
            if len(diffs) < 5:
                diffs.append(('v3_better', gold_line, v2_out, v3_out))
        else:
            same += 1  # Same error count but different output

print(f"Same output: {same}")
print(f"V2 better:   {v2_better}")
print(f"V3 better:   {v3_better}")
print()

for diff_type, gold, v2, v3 in diffs[:3]:
    print(f"\n{diff_type}:")
    print(f"  Gold: {gold[:80]}...")
    print(f"  V2:   {v2[:80]}...")
    print(f"  V3:   {v3[:80]}...")
