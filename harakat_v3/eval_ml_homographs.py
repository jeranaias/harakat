#!/usr/bin/env python3
"""
Evaluate V3 Pipeline Components

Compare DER with different V3 configurations.
"""

import sys
import os
import time

if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')

script_dir = os.path.dirname(os.path.abspath(__file__))
harakat_dir = os.path.dirname(script_dir)
sys.path.insert(0, harakat_dir)

from harakat_v3 import diacritize, diacritize_grammar_only, diacritize_v2_compatible

# Create functions for different configs
def diacritize_no_voice(text):
    return diacritize(text, apply_grammar_fixes=True, apply_ml_homographs=True, apply_ml_voice=False, apply_anna=False)

def diacritize_no_anna(text):
    return diacritize(text, apply_grammar_fixes=True, apply_ml_homographs=True, apply_ml_voice=True, apply_anna=False)

HARAKAT = set('ًٌٍَُِّْٰ')


def strip_harakat(text):
    return ''.join(c for c in text if c not in HARAKAT)


def calc_der(pred_lines, gold_lines):
    """Calculate DER."""
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


def main():
    # Don't wrap stdout if already wrapped by harakat_v2
    import io
    if not isinstance(sys.stdout, io.TextIOWrapper):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        except:
            pass

    print("=" * 70)
    print("ML HOMOGRAPH DISAMBIGUATION EVALUATION")
    print("=" * 70)

    tashkeel_dir = os.path.dirname(harakat_dir)
    v2_test_path = os.path.join(tashkeel_dir, 'tashkeela_v2', 'tashkeela_test.txt')
    v2_val_path = os.path.join(tashkeel_dir, 'tashkeela_v2', 'tashkeela_val.txt')

    for name, gold_path, limit in [
        ('Test', v2_test_path, 500),
        ('Validation', v2_val_path, 500),
    ]:
        if not os.path.exists(gold_path):
            print(f"\nSkipping {name}: file not found")
            continue

        print(f"\n{'='*70}")
        print(f"{name} Set ({limit} lines)")
        print(f"{'='*70}")

        with open(gold_path, 'r', encoding='utf-8') as f:
            gold_lines = [l.strip() for l in f if l.strip()][:limit]

        # Generate predictions with different configurations
        configs = [
            ('V2 only (no V3)', diacritize_v2_compatible),
            ('V3 grammar only', diacritize_grammar_only),
            ('V3 + ML homographs', diacritize_no_voice),
            ('V3 + voice (no anna)', diacritize_no_anna),
            ('V3 full (+ anna)', diacritize),
        ]

        results = {}
        for config_name, diacritize_fn in configs:
            print(f"\n{config_name}:")
            start = time.time()
            preds = []
            for i, line in enumerate(gold_lines):
                undiac = strip_harakat(line)
                pred = diacritize_fn(undiac)
                preds.append(pred)
                if (i + 1) % 100 == 0:
                    print(f"  {i+1}/{len(gold_lines)}...")

            elapsed = time.time() - start
            errors, positions = calc_der(preds, gold_lines)
            der = 100 * errors / positions if positions > 0 else 0

            results[config_name] = (der, errors, positions)
            print(f"  DER: {der:.2f}% ({errors:,} errors / {positions:,} positions)")
            print(f"  Time: {elapsed:.1f}s ({len(gold_lines)/elapsed:.1f} lines/sec)")

        # Summary
        print(f"\n{name} Set Summary:")
        print("-" * 50)
        for config_name, (der, err, pos) in results.items():
            print(f"  {config_name:<25} {der:.2f}% DER")

        # Impact calculation
        if 'V3 grammar only' in results and 'V3 + ML homographs' in results:
            grammar_der = results['V3 grammar only'][0]
            homo_der = results['V3 + ML homographs'][0]
            improvement = grammar_der - homo_der
            print(f"\n  ML homograph impact: {improvement:+.2f}% DER")

        if 'V3 + ML homographs' in results and 'V3 + voice (no anna)' in results:
            homo_der = results['V3 + ML homographs'][0]
            voice_der = results['V3 + voice (no anna)'][0]
            improvement = homo_der - voice_der
            print(f"  ML voice impact: {improvement:+.2f}% DER")

        if 'V3 + voice (no anna)' in results and 'V3 full (+ anna)' in results:
            voice_der = results['V3 + voice (no anna)'][0]
            full_der = results['V3 full (+ anna)'][0]
            improvement = voice_der - full_der
            print(f"  Anna fix impact: {improvement:+.2f}% DER")

        if 'V2 only (no V3)' in results and 'V3 full (+ anna)' in results:
            v2_der = results['V2 only (no V3)'][0]
            full_der = results['V3 full (+ anna)'][0]
            improvement = v2_der - full_der
            print(f"\n  TOTAL V3 improvement: {improvement:+.2f}% DER")


if __name__ == '__main__':
    main()
