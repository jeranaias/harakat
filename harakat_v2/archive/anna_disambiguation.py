#!/usr/bin/env python3
"""
أَنَّ vs أَنْ Disambiguation Model

The most common remaining error in Harakat V2 is confusing:
- أَنَّ (anna) - "that" with shadda, used with nominal sentences
- أَنْ (an) - "that" without shadda, used with subjunctive verbs

This model uses sentence context to predict whether ان words should have shadda.

Target words: أن, إن, بأن, فإن, لأن, كأن, وأن, وإن

Training approach:
1. Extract all sentences containing these words from Tashkeela
2. Train BiLSTM classifier on sentence context
3. Integrate as post-processor in V2 pipeline
"""

import sys
import os

if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import json
import re

# Constants
HARAKAT = 'ًٌٍَُِّْ'
HARAKAT_SET = set(HARAKAT)
SHADDA = 'ّ'

# Target patterns - words that can be أَنَّ or أَنْ
# Base forms (without diacritics) that we need to disambiguate
TARGET_BASES = {'ان', 'بان', 'فان', 'لان', 'كان', 'وان'}

# Character vocabulary
ARABIC_CHARS = 'ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىيًٌٍَُِّْ '
CHAR_TO_IDX = {c: i+1 for i, c in enumerate(ARABIC_CHARS)}
CHAR_TO_IDX['<PAD>'] = 0
CHAR_TO_IDX['<UNK>'] = len(ARABIC_CHARS) + 1
VOCAB_SIZE = len(ARABIC_CHARS) + 2


def strip_harakat(text):
    return ''.join(c for c in text if c not in HARAKAT_SET)


def has_shadda_on_nun(word):
    """Check if word has shadda on the final ن."""
    # Find position of ن
    for i, c in enumerate(word):
        if c == 'ن':
            # Check following chars for shadda
            for j in range(i+1, len(word)):
                if word[j] == SHADDA:
                    return True
                if word[j] not in HARAKAT_SET:
                    break
    return False


class AnnaSentenceDataset(Dataset):
    """Dataset for anna/an disambiguation at sentence level."""

    def __init__(self, samples, max_len=150):
        self.samples = samples
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def encode(self, text):
        ids = [CHAR_TO_IDX.get(c, CHAR_TO_IDX['<UNK>']) for c in text[:self.max_len]]
        pad_len = self.max_len - len(ids)
        return ids + [0] * pad_len

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'context': torch.tensor(self.encode(sample['context']), dtype=torch.long),
            'word_pos': torch.tensor(sample['word_pos'], dtype=torch.long),
            'label': torch.tensor(sample['has_shadda'], dtype=torch.float),
        }


class AnnaDisambiguator(nn.Module):
    """
    BiLSTM model for أَنَّ vs أَنْ disambiguation.

    Uses sentence context to predict if target word should have shadda.
    """

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=48, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True, num_layers=2, dropout=dropout)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, context, word_pos=None):
        emb = self.embed(context)
        lstm_out, _ = self.lstm(emb)

        # Attention over context
        attn_weights = F.softmax(self.attention(lstm_out).squeeze(-1), dim=-1)
        attended = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)

        logits = self.classifier(attended).squeeze(-1)
        return logits

    def predict(self, context, threshold=0.5):
        self.eval()
        with torch.no_grad():
            logits = self.forward(context)
            probs = torch.sigmoid(logits)
            return (probs > threshold).long(), probs


def extract_training_data(corpus_path, max_samples=100000):
    """Extract training samples from diacritized corpus."""
    samples = []

    print(f"Extracting training data from {corpus_path}...")

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if len(samples) >= max_samples:
                break

            line = line.strip()
            if not line:
                continue

            words = line.split()

            for i, word in enumerate(words):
                base = strip_harakat(word)

                # Check if this is a target word
                if base in TARGET_BASES or base.startswith('ا') and base.endswith('ن') and len(base) <= 4:
                    # Check if base matches our patterns
                    if not any(base == t or base.endswith(t) for t in ['ان']):
                        continue

                    # Get context (surrounding words)
                    start = max(0, i - 8)
                    end = min(len(words), i + 9)
                    context = ' '.join(words[start:end])

                    # Determine label
                    has_shadda = has_shadda_on_nun(word)

                    samples.append({
                        'context': strip_harakat(context),  # Undiacritized context
                        'word': word,
                        'word_base': base,
                        'word_pos': i - start,
                        'has_shadda': has_shadda,
                    })

            if (line_num + 1) % 10000 == 0:
                print(f"  {line_num + 1} lines, {len(samples)} samples")

    # Balance dataset
    with_shadda = [s for s in samples if s['has_shadda']]
    without_shadda = [s for s in samples if not s['has_shadda']]

    print(f"Total samples: {len(samples)}")
    print(f"  With shadda: {len(with_shadda)}")
    print(f"  Without shadda: {len(without_shadda)}")

    # Undersample majority class
    min_count = min(len(with_shadda), len(without_shadda))
    import random
    random.shuffle(with_shadda)
    random.shuffle(without_shadda)

    balanced = with_shadda[:min_count] + without_shadda[:min_count]
    random.shuffle(balanced)

    print(f"Balanced samples: {len(balanced)}")

    return balanced


def train_model(train_data, val_data, output_dir, epochs=20, batch_size=64, lr=1e-3):
    """Train the disambiguation model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nTraining on {device}")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    train_dataset = AnnaSentenceDataset(train_data)
    val_dataset = AnnaSentenceDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = AnnaDisambiguator().to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    best_acc = 0
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            context = batch['context'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(context)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                context = batch['context'].to(device)
                labels = batch['label'].to(device)

                logits = model(context)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:2d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch + 1,
            }, os.path.join(output_dir, 'anna_disambiguator.pt'))

    print(f"\nBest validation accuracy: {best_acc:.4f}")
    return model


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', default='../../tashkeela_train.txt')
    parser.add_argument('--output', default='models/anna')
    parser.add_argument('--max-samples', type=int, default=100000)
    parser.add_argument('--epochs', type=int, default=20)

    args = parser.parse_args()

    samples = extract_training_data(args.corpus, args.max_samples)

    # Split
    val_size = int(len(samples) * 0.1)
    val_data = samples[:val_size]
    train_data = samples[val_size:]

    model = train_model(train_data, val_data, args.output, epochs=args.epochs)

    print("\nDone!")


if __name__ == '__main__':
    main()
