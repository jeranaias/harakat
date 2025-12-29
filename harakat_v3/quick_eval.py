#!/usr/bin/env python3
"""Quick V3.4 evaluation."""
import sys
import os
import io

script_dir = os.path.dirname(os.path.abspath(__file__))
harakat_dir = os.path.dirname(script_dir)
sys.path.insert(0, harakat_dir)

from harakat_v3 import diacritize

def main():
    if sys.platform == 'win32':
        os.system('chcp 65001 > nul 2>&1')
    if not isinstance(sys.stdout, io.TextIOWrapper):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        except:
            pass

    HARAKAT = set('ًٌٍَُِّْٰ')

    def strip_harakat(text):
        return ''.join(c for c in text if c not in HARAKAT)

    def calc_der(pred_lines, gold_lines):
        total_errors = 0
        total_positions = 0

        for pred, gold in zip(pred_lines, gold_lines):
            pred_words = pred.split()
            gold_words = gold.split()

            pi, gi = 0, 0
            while pi < len(pred_words) and gi < len(gold_words):
                pw, gw = pred_words[pi], gold_words[gi]
                pb, gb = strip_harakat(pw), strip_harakat(gw)

                if pb == gb:
                    pred_seq, gold_seq = [], []
                    curr_p, curr_g = '', ''

                    for c in pw:
                        if c in HARAKAT:
                            curr_p += c
                        else:
                            pred_seq.append(curr_p)
                            curr_p = ''
                    pred_seq.append(curr_p)

                    for c in gw:
                        if c in HARAKAT:
                            curr_g += c
                        else:
                            gold_seq.append(curr_g)
                            curr_g = ''
                    gold_seq.append(curr_g)

                    if len(pred_seq) == len(gold_seq):
                        for p, g in zip(pred_seq, gold_seq):
                            total_positions += 1
                            if p != g:
                                total_errors += 1

                    pi += 1
                    gi += 1
                else:
                    pi += 1
                    gi += 1

        return total_errors, total_positions

    tashkeel_dir = os.path.dirname(harakat_dir)
    v2_test_path = os.path.join(tashkeel_dir, 'tashkeela_v2', 'tashkeela_test.txt')

    print("=" * 50)
    print("V3.4 QUICK EVALUATION")
    print("=" * 50)

    with open(v2_test_path, 'r', encoding='utf-8') as f:
        gold_lines = [l.strip() for l in f if l.strip()][:500]

    print(f"\nEvaluating {len(gold_lines)} lines...")

    preds = []
    for i, line in enumerate(gold_lines):
        undiac = strip_harakat(line)
        pred = diacritize(undiac)
        preds.append(pred)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(gold_lines)}...")

    errors, positions = calc_der(preds, gold_lines)
    der = 100 * errors / positions if positions > 0 else 0

    print(f"\nResult: {der:.2f}% DER ({errors:,} errors / {positions:,} positions)")

if __name__ == '__main__':
    main()
