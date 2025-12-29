#!/usr/bin/env python3
"""
Build single-file distribution of Harakat V3.5.

Creates a self-contained harakat.py with:
1. V2 system (harakat_elite + case predictor + hybrid corrector) - from harakat_v2.py
2. V3.5 rules (particles, anna, sunna, case)
3. V3.5 ML models (homograph classifiers, voice classifier) - embedded as base64

Result: Single .py file with everything embedded.
"""

import os
import sys
import re
import pickle
import lzma
import base64
from pathlib import Path

script_dir = Path(__file__).parent

def main():
    print("=" * 70)
    print("Building Harakat V3.5 Single-File Distribution")
    print("=" * 70)

    # =========================================================================
    # STEP 1: Load V2 system (already self-contained)
    # =========================================================================
    print("\n[1/5] Loading V2 system (harakat_v2.py)...")

    v2_path = script_dir / "harakat_v2" / "harakat_v2.py"
    if not v2_path.exists():
        print(f"ERROR: {v2_path} not found!")
        sys.exit(1)

    with open(v2_path, 'r', encoding='utf-8') as f:
        v2_code = f.read()

    print(f"   V2 code: {len(v2_code):,} bytes")

    # =========================================================================
    # STEP 2: Load V3 rules
    # =========================================================================
    print("\n[2/5] Loading V3.5 rules...")

    rules_dir = script_dir / "harakat_v3" / "rules"
    rule_files = [
        "particles.py",
        "anna_fix.py",
        "sunna_fix.py",
        "case_rules.py",
        "homograph_ml.py",
        "voice_ml.py",
        "confidence_calibrated.py",
    ]

    rules_code = {}
    for rule_file in rule_files:
        rule_path = rules_dir / rule_file
        if rule_path.exists():
            with open(rule_path, 'r', encoding='utf-8') as f:
                rules_code[rule_file] = f.read()
            print(f"   {rule_file}: {len(rules_code[rule_file]):,} bytes")
        else:
            print(f"   WARNING: {rule_file} not found")

    # =========================================================================
    # STEP 3: Embed V3 ML models
    # =========================================================================
    print("\n[3/5] Embedding V3.5 ML models...")

    models_dir = script_dir / "harakat_v3" / "models"

    # Homograph models
    homograph_model_path = models_dir / "homograph_classifiers" / "all_homograph_models.pkl"
    if homograph_model_path.exists():
        with open(homograph_model_path, 'rb') as f:
            homograph_data = f.read()
        homograph_compressed = lzma.compress(homograph_data)
        homograph_b64 = base64.b64encode(homograph_compressed).decode('ascii')
        print(f"   Homograph models: {len(homograph_data):,} bytes -> {len(homograph_compressed):,} bytes (LZMA)")
    else:
        homograph_b64 = None
        print("   WARNING: homograph models not found")

    # Voice classifier
    voice_model_path = models_dir / "voice_classifier" / "voice_classifiers_by_word.pkl"
    if voice_model_path.exists():
        with open(voice_model_path, 'rb') as f:
            voice_data = f.read()
        voice_compressed = lzma.compress(voice_data)
        voice_b64 = base64.b64encode(voice_compressed).decode('ascii')
        print(f"   Voice models: {len(voice_data):,} bytes -> {len(voice_compressed):,} bytes (LZMA)")
    else:
        voice_b64 = None
        print("   WARNING: voice models not found")

    # =========================================================================
    # STEP 4: Generate combined code
    # =========================================================================
    print("\n[4/5] Generating combined code...")

    combined_code = f'''#!/usr/bin/env python3
"""
Harakat V3.5 - Arabic Diacritization System (Single-File Distribution)

Complete Arabic text diacritization with:
- V2 Neural Pipeline (harakat_elite + case predictor + hybrid corrector)
- V3.5 Rule-based corrections (particles, anna, sunna)
- V3.5 ML-based corrections (homograph disambiguation, voice correction)

Performance: 1.71% DER on Tashkeela test set

Usage:
    from harakat import diacritize
    result = diacritize("الكتاب على الطاولة")

    # Command line:
    python harakat.py "الكتاب على الطاولة"
    python harakat.py -f input.txt -o output.txt

Author: Jesse Morgan
License: MIT
"""

__version__ = "3.5.0"
__author__ = "Jesse Morgan"

import sys
import os
import re
import pickle
import lzma
import base64
from io import BytesIO

# =============================================================================
# V3.5 Embedded ML Models
# =============================================================================

_HOMOGRAPH_MODELS_B64 = """{homograph_b64 if homograph_b64 else ''}"""

_VOICE_MODELS_B64 = """{voice_b64 if voice_b64 else ''}"""

_homograph_models = None
_voice_models = None

def _load_homograph_models():
    global _homograph_models
    if _homograph_models is None and _HOMOGRAPH_MODELS_B64:
        compressed = base64.b64decode(_HOMOGRAPH_MODELS_B64)
        data = lzma.decompress(compressed)
        _homograph_models = pickle.loads(data)
    return _homograph_models

def _load_voice_models():
    global _voice_models
    if _voice_models is None and _VOICE_MODELS_B64:
        compressed = base64.b64decode(_VOICE_MODELS_B64)
        data = lzma.decompress(compressed)
        _voice_models = pickle.loads(data)
    return _voice_models

'''

    # Add V2 code (remove its if __name__ == '__main__' block and rename diacritize to v2_diacritize)
    v2_code_stripped = re.sub(
        r"\nif __name__ == '__main__':.*$",
        '',
        v2_code,
        flags=re.DOTALL
    )
    # Rename V2's diacritize function to v2_diacritize to avoid conflict
    v2_code_stripped = re.sub(r'\ndef diacritize\(', '\ndef v2_diacritize(', v2_code_stripped)

    # Wrap V2 code in a module namespace
    combined_code += '''
# =============================================================================
# V2 Neural Pipeline (Embedded)
# =============================================================================

'''
    combined_code += v2_code_stripped

    # Add V3 rules
    combined_code += '''

# =============================================================================
# V3.5 Rule-Based Corrections
# =============================================================================

'''

    def strip_test_code(code):
        """Remove test functions and if __name__ == '__main__' blocks."""
        # Remove test functions (def test_*)
        code = re.sub(r'\ndef test_\w+\([^)]*\):.*?(?=\ndef |\nclass |\nif __name__|$)', '', code, flags=re.DOTALL)
        # Remove if __name__ == '__main__' blocks
        code = re.sub(r"\nif __name__ == ['\"]__main__['\"]:\s*\n.*$", '', code, flags=re.DOTALL)
        return code

    # Add particle rules
    if "particles.py" in rules_code:
        particle_code = rules_code["particles.py"]
        # Remove imports that are already included
        particle_code = re.sub(r'^import re\n', '', particle_code, flags=re.MULTILINE)
        particle_code = re.sub(r'^from \. import.*\n', '', particle_code, flags=re.MULTILINE)
        particle_code = strip_test_code(particle_code)
        combined_code += f'''
# --- particles.py ---
{particle_code}
'''

    # Add anna fix
    if "anna_fix.py" in rules_code:
        anna_code = rules_code["anna_fix.py"]
        anna_code = re.sub(r'^import re\n', '', anna_code, flags=re.MULTILINE)
        anna_code = strip_test_code(anna_code)
        combined_code += f'''
# --- anna_fix.py ---
{anna_code}
'''

    # Add sunna fix
    if "sunna_fix.py" in rules_code:
        sunna_code = rules_code["sunna_fix.py"]
        sunna_code = re.sub(r'^import re\n', '', sunna_code, flags=re.MULTILINE)
        sunna_code = strip_test_code(sunna_code)
        combined_code += f'''
# --- sunna_fix.py ---
{sunna_code}
'''

    # Add case rules
    if "case_rules.py" in rules_code:
        case_code = rules_code["case_rules.py"]
        case_code = re.sub(r'^import re\n', '', case_code, flags=re.MULTILINE)
        case_code = strip_test_code(case_code)
        combined_code += f'''
# --- case_rules.py ---
{case_code}
'''

    # Add ML homograph rules
    if "homograph_ml.py" in rules_code:
        homo_code = rules_code["homograph_ml.py"]
        # Update to use embedded models
        homo_code = re.sub(r'^import pickle\n', '', homo_code, flags=re.MULTILINE)
        homo_code = re.sub(r'^from pathlib import Path\n', '', homo_code, flags=re.MULTILINE)
        homo_code = re.sub(
            r'def _load_models\(\):.*?return _models',
            'def _load_models():\\n    return _load_homograph_models()',
            homo_code,
            flags=re.DOTALL
        )
        homo_code = strip_test_code(homo_code)
        combined_code += f'''
# --- homograph_ml.py (using embedded models) ---
{homo_code}
'''

    # Add voice ML rules
    if "voice_ml.py" in rules_code:
        voice_code = rules_code["voice_ml.py"]
        voice_code = re.sub(r'^import pickle\n', '', voice_code, flags=re.MULTILINE)
        voice_code = re.sub(r'^from pathlib import Path\n', '', voice_code, flags=re.MULTILINE)
        voice_code = re.sub(
            r'def _load_voice_models\(\):.*?return _models',
            'def _load_voice_models_internal():\\n    return _load_voice_models()',
            voice_code,
            flags=re.DOTALL
        )
        voice_code = strip_test_code(voice_code)
        combined_code += f'''
# --- voice_ml.py (using embedded models) ---
{voice_code}
'''

    # Add confidence calibrated rules
    if "confidence_calibrated.py" in rules_code:
        calib_code = rules_code["confidence_calibrated.py"]
        # Remove relative imports (from . import and from .module import)
        calib_code = re.sub(r'^from \.\w* import.*\n', '', calib_code, flags=re.MULTILINE)
        calib_code = strip_test_code(calib_code)
        combined_code += f'''
# --- confidence_calibrated.py ---
{calib_code}
'''

    # Add main diacritize function
    combined_code += '''

# =============================================================================
# V3.5 Main API
# =============================================================================

def diacritize(text, apply_particles=True, apply_anna=True, apply_sunna=True,
               apply_ml_homographs=True, apply_ml_voice=True, use_calibrated=True):
    """
    Diacritize Arabic text using V3.5 pipeline.

    V3.5 adds corrections on top of V2:
    1. Particle kasra rules (grammar-based)
    2. ML-based homograph disambiguation (calibrated thresholds)
    3. ML-based voice correction (active/passive verbs)
    4. Anna fix (أنّ/أن shadda disambiguation)
    5. Sunna fix (سُنَّة/سَنَة disambiguation)

    Args:
        text: Arabic text (with or without diacritics)
        apply_particles: Apply particle kasra rules (default True)
        apply_anna: Apply anna fix (default True)
        apply_sunna: Apply sunna fix (default True)
        apply_ml_homographs: Apply ML homograph disambiguation (default True)
        apply_ml_voice: Apply ML voice correction (default True)
        use_calibrated: Use calibrated confidence thresholds (default True)

    Returns:
        Fully diacritized Arabic text
    """
    # Step 1: V2 pipeline
    result = v2_diacritize(text)

    # Step 2: Particle rules
    if apply_particles:
        try:
            result = apply_particle_rules(result)
        except Exception:
            pass

    # Step 3: ML homograph disambiguation
    if apply_ml_homographs:
        try:
            if use_calibrated:
                result = apply_calibrated_homograph_rules(result)
            else:
                result = apply_ml_homograph_rules(result)
        except Exception:
            pass  # Skip if models not available

    # Step 4: ML voice correction
    if apply_ml_voice:
        try:
            result = apply_voice_correction(result)
        except Exception:
            pass

    # Step 5: Anna fix
    if apply_anna:
        try:
            result = apply_anna_fix(result)
        except Exception:
            pass

    # Step 6: Sunna fix
    if apply_sunna:
        try:
            result = apply_sunna_fix(result)
        except Exception:
            pass

    return result


def diacritize_v2_only(text):
    """Diacritize using V2 only (no V3.5 corrections)."""
    return v2_diacritize(text)


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Harakat V3.5 - Arabic Diacritization System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=\'''
Examples:
  %(prog)s "الكتاب على الطاولة"
  %(prog)s -f input.txt -o output.txt
  echo "مرحبا" | %(prog)s --stdin
        \'''
    )
    parser.add_argument('text', nargs='?', help='Text to diacritize')
    parser.add_argument('-f', '--file', help='Input file')
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('--stdin', action='store_true', help='Read from stdin')
    parser.add_argument('--v2-only', action='store_true', help='Use V2 only (no V3.5 corrections)')
    parser.add_argument('--version', action='version', version=f'Harakat {__version__}')

    args = parser.parse_args()

    # Get input text
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    elif args.text:
        text = args.text
    elif args.stdin or not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.print_help()
        sys.exit(1)

    # Diacritize
    if args.v2_only:
        result = diacritize_v2_only(text.strip())
    else:
        result = diacritize(text.strip())

    # Output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"Output written to: {args.output}")
    else:
        try:
            sys.stdout.buffer.write((result + '\\n').encode('utf-8'))
        except (AttributeError, UnicodeEncodeError):
            print(result)
'''

    # Fix escaped quotes in epilog only
    combined_code = combined_code.replace("\\'''", "'''")

    # =========================================================================
    # STEP 5: Write output
    # =========================================================================
    print("\n[5/5] Writing output...")

    output_path = script_dir / "harakat_v35_embedded.py"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(combined_code)

    file_size = os.path.getsize(output_path)
    print(f"   Output: {output_path}")
    print(f"   Size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")

    print("\n" + "=" * 70)
    print("Build complete!")
    print("=" * 70)
    print(f"\nTo test: py {output_path} \"مرحبا بالعالم\"")


if __name__ == '__main__':
    main()
