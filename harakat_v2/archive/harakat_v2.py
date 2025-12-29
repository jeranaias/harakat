#!/usr/bin/env python3
"""
Harakat V2 - Neural Enhanced Arabic Diacritization
===================================================

Building on Harakat V1 (4.46% DER), V2 adds neural post-processing
to achieve 2.29% DER - a 49% relative improvement.

Pipeline:
  1. V1 (harakat_elite): Base diacritization (4.46% DER)
  2. Case V3 Ensemble: LSTM + attention for case endings (-> 2.55% DER)
  3. Hybrid LSTM Corrector: Internal vowel correction (-> 2.29% DER)

Results on Tashkeela test set (2,500 lines):
  - V1 Baseline:           4.458% DER (with case), 2.306% DER (without case)
  - V1 + Case V3:          2.549% DER (with case), 2.306% DER (without case)
  - V1 + Case V3 + Hybrid: 2.287% DER (with case), 1.943% DER (without case)

Model Sizes:
  - V1 Base:        3.14 MB (LZMA compressed)
  - Case V3 (3x):   ~4.0 MB
  - Hybrid LSTM:    ~1.6 MB
  - Total:          ~8.7 MB

Author: Jesse Morgan (DLI Arabic Instructor)
License: MIT
"""

import os
import sys

# Ensure UTF-8 output on Windows
if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


# =============================================================================
# CONSTANTS
# =============================================================================

HARAKAT = '\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652'
HARAKAT_SET = frozenset(HARAKAT)
ARABIC_CHARS = 'ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي'
ALL_CHARS = ARABIC_CHARS + HARAKAT + ' '

# Case model vocab
CASE_CHAR_TO_IDX = {c: i+1 for i, c in enumerate(ALL_CHARS)}
CASE_CHAR_TO_IDX['<PAD>'] = 0
CASE_CHAR_TO_IDX['<UNK>'] = len(ALL_CHARS) + 1

FINAL_CLASSES = ['', '\u064f', '\u0650', '\u064e', '\u064c', '\u064d', '\u064b', '\u0652']

# Hybrid LSTM vocab
HYBRID_CHARS = 'ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىيال'
HYBRID_CHAR_TO_IDX = {'<PAD>': 0, '<UNK>': 1, ' ': 2}
for c in HYBRID_CHARS:
    HYBRID_CHAR_TO_IDX[c] = len(HYBRID_CHAR_TO_IDX)
HYBRID_VOCAB = len(HYBRID_CHAR_TO_IDX) + 5

DIAC_CLASSES = ['', '\u064e', '\u064f', '\u0650', '\u064b', '\u064c', '\u064d', '\u0652']
DIAC_TO_IDX = {d: i for i, d in enumerate(DIAC_CLASSES)}
IDX_TO_DIAC = {i: d for d, i in DIAC_TO_IDX.items()}


def strip_harakat(text: str) -> str:
    """Remove all diacritical marks from text."""
    return ''.join(c for c in text if c not in HARAKAT_SET)


# =============================================================================
# CASE V3 ENSEMBLE MODEL
# =============================================================================

class CasePredictorV3(nn.Module):
    """BiLSTM with attention for case ending prediction."""

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


class CaseV3Ensemble:
    """Ensemble of Case V3 models with batched inference."""

    def __init__(self, models_dir: Path):
        info = torch.load(models_dir / 'ensemble_info.pt', weights_only=False)
        self.models = []

        for i in range(1, info['num_models'] + 1):
            model = CasePredictorV3(
                vocab_size=info['vocab_size'],
                embed_dim=info['embed_dim'],
                hidden_dim=info['hidden_dim'],
                num_classes=info['num_classes']
            )
            ckpt = torch.load(models_dir / f'case_v3_model{i}.pt', weights_only=False)
            model.load_state_dict(ckpt['model_state'])
            model.eval()
            self.models.append(model)

        self.max_word = 30
        self.max_ctx = info['max_context']

    def encode(self, text: str, max_len: int) -> list:
        ids = [CASE_CHAR_TO_IDX.get(c, CASE_CHAR_TO_IDX['<UNK>']) for c in text[:max_len]]
        return ids + [0] * (max_len - len(ids))

    @torch.no_grad()
    def predict_batch(self, words: list, contexts: list) -> list:
        """Predict case endings for batch of words."""
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

class ContrastiveModel(nn.Module):
    """LSTM-based model for internal vowel correction."""

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

        change_logits = self.change_head(out).squeeze(-1)
        correction_logits = self.correction_head(out)
        return change_logits, correction_logits


class HybridCorrector:
    """
    Hybrid corrector with adaptive thresholding.

    Uses multiple confidence levels:
    - High confidence (>0.75): always apply correction
    - Medium confidence (0.55-0.75): apply only if correction is also confident (>0.7)
    - Low confidence (<0.55): keep V1's prediction
    """

    def __init__(self, model_path: Path, high_thresh=0.75, low_thresh=0.55, corr_conf_thresh=0.7):
        self.device = torch.device('cpu')
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.corr_conf_thresh = corr_conf_thresh

        checkpoint = torch.load(model_path, weights_only=False)
        config = checkpoint.get('config', {'embed_dim': 64, 'hidden_dim': 96})

        self.model = ContrastiveModel(
            vocab_size=HYBRID_VOCAB,
            embed_dim=config.get('embed_dim', 64),
            hidden_dim=config.get('hidden_dim', 96)
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

    def _encode_word(self, word: str, max_len: int = 20) -> list:
        return [min(HYBRID_CHAR_TO_IDX.get(c, 1), HYBRID_VOCAB-1)
                for c in word[:max_len]] + [0] * max(0, max_len - len(word))

    def _encode_ctx(self, ctx: str, max_len: int = 80) -> list:
        return [min(HYBRID_CHAR_TO_IDX.get(c, 1), HYBRID_VOCAB-1)
                for c in ctx[:max_len]] + [0] * max(0, max_len - len(ctx))

    def _extract_internal(self, word: str) -> tuple:
        """Extract base characters and internal diacritics (excluding final)."""
        base, diacs, curr = [], [], []
        for c in word:
            if c in HARAKAT_SET:
                curr.append(c)
            else:
                if base:
                    primary = next((d for d in curr if d != '\u0651'), '')
                    diacs.append(primary)
                base.append(c)
                curr = []
        if base:
            primary = next((d for d in curr if d != '\u0651'), '')
            diacs.append(primary)
        return ''.join(base), diacs[:-1] if len(diacs) > 1 else []

    def _reconstruct_word(self, base: str, internal_diacs: list, original_word: str) -> str:
        """Reconstruct word with new internal diacritics, preserving shaddas and final."""
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
                if i < len(orig_diacs) and '\u0651' in orig_diacs[i]:
                    if new_diac:
                        result.append('\u0651')
                        result.append(new_diac)
                    else:
                        result.append('\u0651')
                elif new_diac:
                    result.append(new_diac)
            elif i < len(orig_diacs):
                for d in orig_diacs[i]:
                    result.append(d)

        return ''.join(result)

    def correct_word(self, word: str, context_words: list) -> str:
        """Correct internal vowels with adaptive thresholding."""
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
        """Correct internal vowels in text."""
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
# HARAKAT V2 DIACRITIZER
# =============================================================================

class HarakatV2:
    """
    Harakat V2: Neural Enhanced Arabic Diacritization

    Pipeline:
      1. V1 (harakat_elite): Base diacritization
      2. Case V3 Ensemble: Case ending correction
      3. Hybrid LSTM: Internal vowel correction

    Achieves 2.29% DER on Tashkeela test set.
    """

    def __init__(self, models_dir: str = None, v1_module=None):
        """
        Initialize Harakat V2.

        Args:
            models_dir: Path to models folder (default: ./models)
            v1_module: Optional pre-imported harakat_elite module
        """
        if models_dir is None:
            models_dir = Path(__file__).parent / 'models'
        else:
            models_dir = Path(models_dir)

        # Load V1 (harakat_elite)
        if v1_module is not None:
            self.v1_diacritize = v1_module.diacritize
        else:
            # Try to find and import harakat_elite
            self._setup_v1_import()

        # Load Case V3 Ensemble
        case_dir = models_dir / 'case_v3'
        self.case_ensemble = CaseV3Ensemble(case_dir)

        # Load Hybrid LSTM Corrector
        hybrid_path = models_dir / 'contrastive_lstm_v10.pt'
        self.hybrid_corrector = HybridCorrector(
            hybrid_path,
            high_thresh=0.75,
            low_thresh=0.55,
            corr_conf_thresh=0.7
        )

    def _setup_v1_import(self):
        """Set up import path for harakat_elite."""
        # Check common locations
        possible_paths = [
            Path(__file__).parent.parent.parent / 'harakat_elite_full',
            Path(__file__).parent.parent / 'harakat_elite_full',
            Path(__file__).parent / 'harakat_elite_full',
        ]

        for p in possible_paths:
            if (p / 'harakat_elite.py').exists():
                sys.path.insert(0, str(p))
                break

        try:
            from harakat_elite import diacritize
            self.v1_diacritize = diacritize
        except ImportError as e:
            raise ImportError(
                "Could not import harakat_elite. Please ensure harakat_elite.py is accessible. "
                f"Tried paths: {possible_paths}. Error: {e}"
            )

    def _apply_case_endings(self, v1_output: str) -> str:
        """Apply case endings using Case V3 Ensemble."""
        words = v1_output.split()
        if not words:
            return v1_output

        # Prepare batch data
        word_texts = []
        contexts = []
        for i, word in enumerate(words):
            start = max(0, i - 4)
            end = min(len(words), i + 5)
            ctx = ' '.join(words[start:end])
            word_texts.append(word)
            contexts.append(ctx)

        # Get case predictions
        case_preds = self.case_ensemble.predict_batch(word_texts, contexts)

        # Apply predictions
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
                # Preserve shadda if present
                for c in chars[last_base + 1:]:
                    if c == '\u0651':
                        new_word += '\u0651'
                new_word += case
                result.append(new_word)
            else:
                result.append(word + case)

        return ' '.join(result)

    def diacritize(self, text: str) -> str:
        """
        Diacritize Arabic text using the full V2 pipeline.

        Args:
            text: Undiacritized Arabic text

        Returns:
            Fully diacritized text
        """
        # Step 1: V1 base diacritization
        v1_output = self.v1_diacritize(text)

        # Step 2: Case V3 ensemble correction
        case_output = self._apply_case_endings(v1_output)

        # Step 3: Hybrid LSTM internal vowel correction
        final_output = self.hybrid_corrector.correct(case_output, text)

        return final_output


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_diacritizer = None

def get_diacritizer(models_dir: str = None) -> HarakatV2:
    """Get or create the singleton diacritizer instance."""
    global _diacritizer
    if _diacritizer is None:
        _diacritizer = HarakatV2(models_dir)
    return _diacritizer

def diacritize(text: str) -> str:
    """
    Diacritize Arabic text using Harakat V2.

    This is the main entry point for diacritization.

    Args:
        text: Undiacritized Arabic text

    Returns:
        Fully diacritized text

    Example:
        >>> from harakat_v2 import diacritize
        >>> result = diacritize("كتب الطالب الدرس")
        >>> print(result)
        كَتَبَ الطَّالِبُ الدَّرْسَ
    """
    return get_diacritizer().diacritize(text)


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for Harakat V2."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Harakat V2 - Neural Enhanced Arabic Diacritization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python harakat_v2.py "كتب الطالب الدرس"
  python harakat_v2.py -f input.txt -o output.txt
  echo "مرحبا بالعالم" | python harakat_v2.py
        """
    )
    parser.add_argument('text', nargs='?', help='Text to diacritize')
    parser.add_argument('-f', '--file', help='Input file')
    parser.add_argument('-o', '--output', help='Output file (default: stdout)')
    parser.add_argument('--models', help='Path to models directory')

    args = parser.parse_args()

    # Initialize diacritizer
    diacritizer = HarakatV2(args.models)

    # Get input
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    elif args.text:
        text = args.text
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.print_help()
        return

    # Diacritize
    result = diacritizer.diacritize(text.strip())

    # Output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result)
    else:
        print(result)


if __name__ == '__main__':
    main()
