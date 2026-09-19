"""
Character-level dataset for language model training.

Builds a vocabulary from the text, encodes everything to integers,
and serves up (input, target) pairs for next-token prediction.

Next-token prediction: given characters at positions 0..N-1,
predict characters at positions 1..N. That's the training signal
for a language model — learn to guess what comes next.
"""

import numpy as np
import json
from pathlib import Path


class CharDataset:

    def __init__(self, text: str = None, vocab: dict = None):
        if text is not None:
            # Build vocabulary from text
            chars       = sorted(set(text))
            self.vocab  = {ch: i for i, ch in enumerate(chars)}
            self.ivocab = {i: ch for ch, i in self.vocab.items()}
            self.data   = np.array([self.vocab[c] for c in text],
                                   dtype=np.int32)
        elif vocab is not None:
            self.vocab  = vocab
            self.ivocab = {i: ch for ch, i in vocab.items()}
            self.data   = None
        else:
            raise ValueError("Provide either text or vocab")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> np.ndarray:
        return np.array([self.vocab.get(c, 0) for c in text], dtype=np.int32)

    def decode(self, ids) -> str:
        return ''.join(self.ivocab.get(int(i), '?') for i in ids)

    def get_batch(self, batch_size: int, seq_len: int):
        """
        Sample a random batch of (input, target) pairs.

        input:  token ids at positions [i   : i+seq_len]
        target: token ids at positions [i+1 : i+seq_len+1]

        Shape of each: (batch_size, seq_len)
        """
        max_start = len(self.data) - seq_len - 1
        starts    = np.random.randint(0, max_start, size=batch_size)

        X = np.stack([self.data[s:s+seq_len]   for s in starts])
        Y = np.stack([self.data[s+1:s+seq_len+1] for s in starts])

        return X.astype(np.float32), Y.astype(np.float32)

    def save_vocab(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.vocab, f)
        print(f"Vocab ({self.vocab_size} chars) saved to {path}")

    @classmethod
    def from_vocab_file(cls, vocab_path: str, data: np.ndarray = None):
        with open(vocab_path) as f:
            vocab = json.load(f)
        ds      = cls(vocab=vocab)
        ds.data = data
        return ds

    @classmethod
    def from_file(cls, text_path: str, max_chars: int = None):
        print(f"Loading {text_path}...")
        with open(text_path, encoding='utf-8') as f:
            text = f.read(max_chars) if max_chars else f.read()

        # Filter to printable ASCII only — keeps vocab small and clean
        allowed = set(
            'abcdefghijklmnopqrstuvwxyz'
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            '0123456789'
            ' .,!?\'"-:;()\n'
        )
        text = ''.join(c for c in text if c in allowed)

        print(f"  {len(text):,} characters, building vocab...")
        ds = cls(text=text)
        print(f"  Vocab size: {ds.vocab_size} characters")
        return ds
