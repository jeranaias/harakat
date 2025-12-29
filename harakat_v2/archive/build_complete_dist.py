#!/usr/bin/env python3
"""
Build COMPLETE single-file distribution of Harakat V2.

This script creates a truly self-contained harakat_v2_complete.py that includes:
1. V1 (harakat_elite) - lexicon + rules (embedded)
2. Case V3 Ensemble - neural case endings (quantized + embedded)
3. Hybrid LSTM Corrector - internal vowel correction (quantized + embedded)

The result is a single .py file with NO external dependencies except PyTorch.
"""

import os
import sys
import torch
import lzma
import base64
import io
from pathlib import Path

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from harakat_v2 import CasePredictorV3, ContrastiveModel, HYBRID_VOCAB


def quantize_tensor(tensor, bits=8):
    """Quantize tensor to int8."""
    tensor = tensor.float()
    min_val = tensor.min().item()
    max_val = tensor.max().item()

    if max_val == min_val:
        return torch.zeros_like(tensor, dtype=torch.int8), 1.0, 0

    qmin, qmax = -128, 127
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = int(round(qmin - min_val / scale))
    zero_point = max(qmin, min(qmax, zero_point))
    quantized = torch.round((tensor / scale) + zero_point).clamp(qmin, qmax).to(torch.int8)

    return quantized, scale, zero_point


def quantize_state_dict(state_dict):
    """Quantize all tensors in state dict."""
    quantized, scales, zero_points = {}, {}, {}

    for name, tensor in state_dict.items():
        if tensor.dtype == torch.float32 or tensor.dtype == torch.float16:
            q, s, z = quantize_tensor(tensor)
            quantized[name] = q
            scales[name] = s
            zero_points[name] = z
        else:
            quantized[name] = tensor
            scales[name] = None
            zero_points[name] = None

    return quantized, scales, zero_points


def pack_quantized_model(quantized, scales, zero_points):
    """Pack quantized model into bytes."""
    buffer = io.BytesIO()
    torch.save({
        'quantized': quantized,
        'scales': scales,
        'zero_points': zero_points
    }, buffer)
    return buffer.getvalue()


def main():
    print("=" * 70)
    print("Building COMPLETE Harakat V2 Single-File Distribution")
    print("=" * 70)

    models_dir = script_dir / 'models'
    dist_dir = script_dir / 'dist'
    dist_dir.mkdir(exist_ok=True)

    # =========================================================================
    # STEP 1: Load harakat_elite.py (V1)
    # =========================================================================
    print("\n[1/5] Loading harakat_elite (V1 base model)...")

    harakat_elite_paths = [
        Path("c:/tashkeel/harakat_elite_full/harakat_elite.py"),
        Path("c:/tashkeel/paper_eval/harakat_elite.py"),
        script_dir.parent.parent / "harakat_elite_full" / "harakat_elite.py",
    ]

    harakat_elite_path = None
    for p in harakat_elite_paths:
        if p.exists():
            harakat_elite_path = p
            break

    if not harakat_elite_path:
        print("  ERROR: Could not find harakat_elite.py")
        print("  Searched:", harakat_elite_paths)
        sys.exit(1)

    with open(harakat_elite_path, 'r', encoding='utf-8') as f:
        harakat_elite_code = f.read()

    # Compress the V1 code
    v1_compressed = lzma.compress(harakat_elite_code.encode('utf-8'), preset=9)
    v1_b64 = base64.b64encode(v1_compressed).decode('ascii')
    print(f"  Source: {harakat_elite_path}")
    print(f"  Original: {len(harakat_elite_code)/1024:.1f} KB")
    print(f"  Compressed: {len(v1_compressed)/1024:.1f} KB")
    print(f"  Base64: {len(v1_b64)/1024:.1f} KB")

    # =========================================================================
    # STEP 2: Load and quantize Case V3 Ensemble
    # =========================================================================
    print("\n[2/5] Loading Case V3 Ensemble...")

    case_dir = models_dir / 'case_v3'
    info = torch.load(case_dir / 'ensemble_info.pt', weights_only=False)

    case_models_data = []
    for i in range(1, info['num_models'] + 1):
        ckpt = torch.load(case_dir / f'case_v3_model{i}.pt', weights_only=False)
        state_dict = ckpt['model_state']
        q, s, z = quantize_state_dict(state_dict)
        packed = pack_quantized_model(q, s, z)
        case_models_data.append(packed)
        print(f"  Model {i}: {len(packed)/1024:.1f} KB quantized")

    case_config = {
        'num_models': info['num_models'],
        'vocab_size': info['vocab_size'],
        'embed_dim': info['embed_dim'],
        'hidden_dim': info['hidden_dim'],
        'num_classes': info['num_classes'],
        'max_context': info['max_context']
    }

    case_bundle = {'config': case_config, 'models': case_models_data}
    case_buffer = io.BytesIO()
    torch.save(case_bundle, case_buffer)
    case_bytes = case_buffer.getvalue()
    case_compressed = lzma.compress(case_bytes, preset=9)
    case_b64 = base64.b64encode(case_compressed).decode('ascii')
    print(f"  Total: {len(case_bytes)/1024:.1f} KB -> {len(case_compressed)/1024:.1f} KB compressed")

    # =========================================================================
    # STEP 3: Load and quantize Hybrid LSTM
    # =========================================================================
    print("\n[3/5] Loading Hybrid LSTM Corrector...")

    hybrid_path = models_dir / 'contrastive_lstm_v10.pt'
    hybrid_ckpt = torch.load(hybrid_path, weights_only=False)
    hybrid_config = hybrid_ckpt.get('config', {'embed_dim': 64, 'hidden_dim': 96})

    q, s, z = quantize_state_dict(hybrid_ckpt['model_state'])
    hybrid_packed = pack_quantized_model(q, s, z)
    print(f"  Quantized: {len(hybrid_packed)/1024:.1f} KB")

    hybrid_bundle = {'config': hybrid_config, 'model': hybrid_packed}
    hybrid_buffer = io.BytesIO()
    torch.save(hybrid_bundle, hybrid_buffer)
    hybrid_bytes = hybrid_buffer.getvalue()
    hybrid_compressed = lzma.compress(hybrid_bytes, preset=9)
    hybrid_b64 = base64.b64encode(hybrid_compressed).decode('ascii')
    print(f"  Compressed: {len(hybrid_compressed)/1024:.1f} KB")

    # =========================================================================
    # STEP 4: Generate single-file Python script
    # =========================================================================
    print("\n[4/5] Generating harakat_v2_complete.py...")

    template = generate_complete_template(v1_b64, case_b64, hybrid_b64, case_config, hybrid_config)

    output_path = dist_dir / 'harakat_v2_complete.py'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template)

    file_size = output_path.stat().st_size
    print(f"  Created: {output_path}")
    print(f"  File size: {file_size/1024/1024:.2f} MB")

    # =========================================================================
    # STEP 5: Verify it works
    # =========================================================================
    print("\n[5/5] Verifying...")

    sys.path.insert(0, str(dist_dir))
    import harakat_v2_complete as h2c

    test_input = "كتب الطالب الدرس"
    result = h2c.diacritize(test_input)
    print(f"  Input:  {test_input}")
    print(f"  Output: {result}")

    has_diacritics = any('\u064B' <= c <= '\u0655' for c in result)
    if has_diacritics:
        print("  Status: SUCCESS - Diacritics present!")
    else:
        print("  Status: WARNING - No diacritics in output")

    print("\n" + "=" * 70)
    print("BUILD COMPLETE")
    print("=" * 70)
    print(f"\nFully self-contained distribution: {output_path}")
    print(f"Size: {file_size/1024/1024:.2f} MB")
    print("\nThis file has ZERO external dependencies (except PyTorch).")
    print("It embeds V1 (harakat_elite) + Case V3 + Hybrid LSTM.")


def generate_complete_template(v1_b64, case_b64, hybrid_b64, case_config, hybrid_config):
    """Generate the complete single-file distribution."""

    template = '''#!/usr/bin/env python3
"""
Harakat V2 Complete - Fully Self-Contained Arabic Diacritization
=================================================================

This is a COMPLETE single-file distribution with ALL models embedded:
- V1 (harakat_elite): Lexicon + rules + error correction
- Case V3 Ensemble: Neural case ending prediction
- Hybrid LSTM: Internal vowel correction

Pipeline: V1 → Case V3 → Hybrid LSTM → 2.29% DER

Usage:
    from harakat_v2_complete import diacritize
    result = diacritize("كتب الطالب الدرس")
    print(result)  # كَتَبَ الطَّالِبُ الدَّرْسَ

Requirements:
    - PyTorch 1.9+
    - NO other dependencies!

Author: Jesse Morgan
License: MIT
"""

import os
import sys
import io
import lzma
import base64
import types

# Ensure UTF-8 output on Windows
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# EMBEDDED DATA
# =============================================================================

_V1_CODE_B64 = """
''' + v1_b64 + '''
"""

_CASE_V3_DATA = """
''' + case_b64 + '''
"""

_HYBRID_LSTM_DATA = """
''' + hybrid_b64 + '''
"""

_CASE_CONFIG = ''' + repr(case_config) + '''
_HYBRID_CONFIG = ''' + repr(hybrid_config) + '''


# =============================================================================
# V1 MODULE LOADER
# =============================================================================

_v1_module = None

def _load_v1_module():
    """Load harakat_elite from embedded code."""
    global _v1_module
    if _v1_module is not None:
        return _v1_module

    # Decompress and decode V1 code
    v1_compressed = base64.b64decode(_V1_CODE_B64.strip())
    v1_code = lzma.decompress(v1_compressed).decode('utf-8')

    # Create module and register it
    _v1_module = types.ModuleType('harakat_elite')
    _v1_module.__file__ = '<embedded>'
    sys.modules['harakat_elite'] = _v1_module

    # Execute code in module namespace
    exec(compile(v1_code, '<harakat_elite>', 'exec'), _v1_module.__dict__)

    return _v1_module


# =============================================================================
# CONSTANTS
# =============================================================================

HARAKAT = '\\u064b\\u064c\\u064d\\u064e\\u064f\\u0650\\u0651\\u0652'
HARAKAT_SET = frozenset(HARAKAT)
ARABIC_CHARS = 'ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي'
ALL_CHARS = ARABIC_CHARS + HARAKAT + ' '

CASE_CHAR_TO_IDX = {c: i+1 for i, c in enumerate(ALL_CHARS)}
CASE_CHAR_TO_IDX['<PAD>'] = 0
CASE_CHAR_TO_IDX['<UNK>'] = len(ALL_CHARS) + 1

FINAL_CLASSES = ['', '\\u064f', '\\u0650', '\\u064e', '\\u064c', '\\u064d', '\\u064b', '\\u0652']

HYBRID_CHARS = 'ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىيال'
HYBRID_CHAR_TO_IDX = {'<PAD>': 0, '<UNK>': 1, ' ': 2}
for c in HYBRID_CHARS:
    HYBRID_CHAR_TO_IDX[c] = len(HYBRID_CHAR_TO_IDX)
HYBRID_VOCAB = len(HYBRID_CHAR_TO_IDX) + 5

DIAC_CLASSES = ['', '\\u064e', '\\u064f', '\\u0650', '\\u064b', '\\u064c', '\\u064d', '\\u0652']
DIAC_TO_IDX = {d: i for i, d in enumerate(DIAC_CLASSES)}
IDX_TO_DIAC = {i: d for d, i in DIAC_TO_IDX.items()}


def strip_harakat(text: str) -> str:
    return ''.join(c for c in text if c not in HARAKAT_SET)


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class CasePredictorV3(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.word_lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.ctx_lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True,
                                 num_layers=2, dropout=0.1)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, word, context):
        word_emb = self.embed(word)
        _, (word_h, _) = self.word_lstm(word_emb)
        word_vec = torch.cat([word_h[0], word_h[1]], dim=-1)
        ctx_emb = self.embed(context)
        ctx_out, _ = self.ctx_lstm(ctx_emb)
        word_query = word_vec.unsqueeze(1)
        attended, _ = self.attention(word_query, ctx_out, ctx_out)
        attended = self.layer_norm(attended.squeeze(1))
        combined = torch.cat([word_vec, attended], dim=-1)
        return self.classifier(combined)


class ContrastiveModel(nn.Module):
    def __init__(self, vocab_size=45, embed_dim=64, hidden_dim=96, num_diacs=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.diac_embed = nn.Embedding(num_diacs, 16)
        self.ctx_lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.word_lstm = nn.LSTM(embed_dim + 16 + hidden_dim*2, hidden_dim,
                                  bidirectional=True, batch_first=True)
        self.change_head = nn.Linear(hidden_dim * 2, 1)
        self.correction_head = nn.Linear(hidden_dim * 2, num_diacs)

    def forward(self, word, v1_diacs, context):
        batch, seq_len = word.size()
        ctx_emb = self.embed(context)
        _, (ctx_h, _) = self.ctx_lstm(ctx_emb)
        ctx_vec = torch.cat([ctx_h[0], ctx_h[1]], dim=-1).unsqueeze(1).expand(-1, seq_len, -1)
        word_emb = self.embed(word)
        v1_emb = self.diac_embed(v1_diacs)
        combined = torch.cat([word_emb, v1_emb, ctx_vec], dim=-1)
        out, _ = self.word_lstm(combined)
        out = out[:, :-1, :]
        return self.change_head(out).squeeze(-1), self.correction_head(out)


# =============================================================================
# QUANTIZATION HELPERS
# =============================================================================

def dequantize_tensor(quantized, scale, zero_point):
    if scale is None:
        return quantized
    return (quantized.float() - zero_point) * scale


def load_quantized_state_dict(packed_bytes):
    buffer = io.BytesIO(packed_bytes)
    data = torch.load(buffer, weights_only=False)
    state_dict = {}
    for name in data['quantized']:
        q = data['quantized'][name]
        s = data['scales'][name]
        z = data['zero_points'][name]
        state_dict[name] = dequantize_tensor(q, s, z)
    return state_dict


def load_embedded_models():
    case_compressed = base64.b64decode(_CASE_V3_DATA.strip())
    case_bytes = lzma.decompress(case_compressed)
    case_bundle = torch.load(io.BytesIO(case_bytes), weights_only=False)

    hybrid_compressed = base64.b64decode(_HYBRID_LSTM_DATA.strip())
    hybrid_bytes = lzma.decompress(hybrid_compressed)
    hybrid_bundle = torch.load(io.BytesIO(hybrid_bytes), weights_only=False)

    return case_bundle, hybrid_bundle


# =============================================================================
# CASE V3 ENSEMBLE
# =============================================================================

class CaseV3Ensemble:
    def __init__(self, bundle):
        config = bundle['config']
        self.models = []
        for packed in bundle['models']:
            model = CasePredictorV3(
                vocab_size=config['vocab_size'],
                embed_dim=config['embed_dim'],
                hidden_dim=config['hidden_dim'],
                num_classes=config['num_classes']
            )
            state_dict = load_quantized_state_dict(packed)
            model.load_state_dict(state_dict)
            model.eval()
            self.models.append(model)
        self.max_word = 30
        self.max_ctx = config['max_context']

    def encode(self, text: str, max_len: int) -> list:
        ids = [CASE_CHAR_TO_IDX.get(c, CASE_CHAR_TO_IDX['<UNK>']) for c in text[:max_len]]
        return ids + [0] * (max_len - len(ids))

    @torch.no_grad()
    def predict_batch(self, words: list, contexts: list) -> list:
        if not words:
            return []
        batch_size = len(words)
        word_batch = torch.tensor([self.encode(w, self.max_word) for w in words], dtype=torch.long)
        ctx_batch = torch.tensor([self.encode(c, self.max_ctx) for c in contexts], dtype=torch.long)
        all_probs = torch.zeros(batch_size, len(FINAL_CLASSES))
        for model in self.models:
            logits = model(word_batch, ctx_batch)
            all_probs += F.softmax(logits, dim=-1)
        preds = all_probs.argmax(dim=1).tolist()
        return [FINAL_CLASSES[p] for p in preds]


# =============================================================================
# HYBRID LSTM CORRECTOR
# =============================================================================

class HybridCorrector:
    def __init__(self, bundle, high_thresh=0.75, low_thresh=0.55, corr_conf_thresh=0.7):
        self.device = torch.device('cpu')
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.corr_conf_thresh = corr_conf_thresh
        config = bundle['config']
        state_dict = load_quantized_state_dict(bundle['model'])
        self.model = ContrastiveModel(
            vocab_size=HYBRID_VOCAB,
            embed_dim=config.get('embed_dim', 64),
            hidden_dim=config.get('hidden_dim', 96)
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _encode_word(self, word: str, max_len: int = 20) -> list:
        return [min(HYBRID_CHAR_TO_IDX.get(c, 1), HYBRID_VOCAB-1)
                for c in word[:max_len]] + [0] * max(0, max_len - len(word))

    def _encode_ctx(self, ctx: str, max_len: int = 80) -> list:
        return [min(HYBRID_CHAR_TO_IDX.get(c, 1), HYBRID_VOCAB-1)
                for c in ctx[:max_len]] + [0] * max(0, max_len - len(ctx))

    def _extract_internal(self, word: str) -> tuple:
        base, diacs, curr = [], [], []
        for c in word:
            if c in HARAKAT_SET:
                curr.append(c)
            else:
                if base:
                    primary = next((d for d in curr if d != '\\u0651'), '')
                    diacs.append(primary)
                base.append(c)
                curr = []
        if base:
            primary = next((d for d in curr if d != '\\u0651'), '')
            diacs.append(primary)
        return ''.join(base), diacs[:-1] if len(diacs) > 1 else []

    def _reconstruct_word(self, base: str, internal_diacs: list, original_word: str) -> str:
        orig_diacs = []
        curr_diacs = []
        char_idx = -1
        for c in original_word:
            if c in HARAKAT_SET:
                curr_diacs.append(c)
            else:
                if char_idx >= 0:
                    orig_diacs.append(curr_diacs)
                curr_diacs = []
                char_idx += 1
        if char_idx >= 0:
            orig_diacs.append(curr_diacs)

        result = []
        for i, char in enumerate(base):
            result.append(char)
            if i < len(internal_diacs):
                new_diac = internal_diacs[i]
                if i < len(orig_diacs) and '\\u0651' in orig_diacs[i]:
                    if new_diac:
                        result.append('\\u0651')
                        result.append(new_diac)
                    else:
                        result.append('\\u0651')
                elif new_diac:
                    result.append(new_diac)
            elif i < len(orig_diacs):
                for d in orig_diacs[i]:
                    result.append(d)
        return ''.join(result)

    def correct_word(self, word: str, context_words: list) -> str:
        base = strip_harakat(word)
        _, v1_internal = self._extract_internal(word)
        if len(v1_internal) == 0:
            return word
        word_enc = torch.tensor([self._encode_word(base)], device=self.device)
        v1_diacs = [DIAC_TO_IDX.get(d, 0) for d in v1_internal] + [0] * (20 - len(v1_internal))
        v1_diacs_enc = torch.tensor([v1_diacs], device=self.device)
        ctx = ' '.join(context_words)
        ctx_enc = torch.tensor([self._encode_ctx(ctx)], device=self.device)
        with torch.no_grad():
            change_logits, corr_logits = self.model(word_enc, v1_diacs_enc, ctx_enc)
        new_internal = list(v1_internal)
        for pos in range(len(v1_internal)):
            change_prob = torch.sigmoid(change_logits[0, pos]).item()
            corr_probs = torch.softmax(corr_logits[0, pos], dim=-1)
            corr_conf = corr_probs.max().item()
            pred = corr_probs.argmax().item()
            if change_prob > self.high_thresh:
                new_internal[pos] = IDX_TO_DIAC.get(pred, '')
            elif change_prob > self.low_thresh and corr_conf > self.corr_conf_thresh:
                new_internal[pos] = IDX_TO_DIAC.get(pred, '')
        return self._reconstruct_word(base, new_internal, word)

    def correct(self, text: str, undiacritized_text: str = None) -> str:
        if undiacritized_text is None:
            undiacritized_text = strip_harakat(text)
        words = text.split()
        undiac_words = undiacritized_text.split()
        corrected_words = []
        for i, word in enumerate(words):
            start = max(0, i - 3)
            end = min(len(undiac_words), i + 4)
            context = undiac_words[start:i] + undiac_words[i+1:end]
            corrected = self.correct_word(word, context)
            corrected_words.append(corrected)
        return ' '.join(corrected_words)


# =============================================================================
# HARAKAT V2 COMPLETE
# =============================================================================

class HarakatV2:
    """
    Harakat V2: Complete Neural Enhanced Arabic Diacritization

    Fully self-contained - all models embedded.
    Achieves 2.29% DER on Tashkeela test set.
    """

    def __init__(self):
        # Load V1 from embedded code
        v1_module = _load_v1_module()
        self.v1_diacritize = v1_module.diacritize

        # Load neural models
        case_bundle, hybrid_bundle = load_embedded_models()
        self.case_ensemble = CaseV3Ensemble(case_bundle)
        self.hybrid_corrector = HybridCorrector(hybrid_bundle)

    def _apply_case_endings(self, v1_output: str) -> str:
        words = v1_output.split()
        if not words:
            return v1_output
        word_texts, contexts = [], []
        for i, word in enumerate(words):
            start = max(0, i - 4)
            end = min(len(words), i + 5)
            ctx = ' '.join(words[start:end])
            word_texts.append(word)
            contexts.append(ctx)
        case_preds = self.case_ensemble.predict_batch(word_texts, contexts)
        result = []
        for word, case in zip(words, case_preds):
            chars = list(word)
            last_base = -1
            for i in range(len(chars) - 1, -1, -1):
                if chars[i] not in HARAKAT_SET:
                    last_base = i
                    break
            if last_base >= 0:
                new_word = ''.join(chars[:last_base + 1])
                for c in chars[last_base + 1:]:
                    if c == '\\u0651':
                        new_word += '\\u0651'
                new_word += case
                result.append(new_word)
            else:
                result.append(word + case)
        return ' '.join(result)

    def diacritize(self, text: str) -> str:
        v1_output = self.v1_diacritize(text)
        case_output = self._apply_case_endings(v1_output)
        final_output = self.hybrid_corrector.correct(case_output, text)
        return final_output


# =============================================================================
# CONVENIENCE API
# =============================================================================

_diacritizer = None

def get_diacritizer() -> HarakatV2:
    global _diacritizer
    if _diacritizer is None:
        _diacritizer = HarakatV2()
    return _diacritizer

def diacritize(text: str) -> str:
    """
    Diacritize Arabic text using Harakat V2.

    This is the main entry point. Fully self-contained, no external dependencies.

    Example:
        >>> from harakat_v2_complete import diacritize
        >>> result = diacritize("كتب الطالب الدرس")
        >>> print(result)  # كَتَبَ الطَّالِبُ الدَّرْسَ
    """
    return get_diacritizer().diacritize(text)


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Harakat V2 Complete - Arabic Diacritization')
    parser.add_argument('text', nargs='?', help='Text to diacritize')
    parser.add_argument('-f', '--file', help='Input file')
    parser.add_argument('-o', '--output', help='Output file')
    args = parser.parse_args()

    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    elif args.text:
        text = args.text
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.print_help()
        sys.exit(1)

    result = diacritize(text.strip())

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result)
    else:
        print(result)
'''

    return template


if __name__ == '__main__':
    main()
