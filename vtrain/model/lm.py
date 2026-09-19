"""
Small character-level language model.
Embedding → N transformer blocks → output projection → logits
"""

import numpy as np
from vtrain.tensor            import Tensor
from vtrain.model.linear      import Linear
from vtrain.model.transformer import TransformerBlock
import vtrain.functional as F


class CharEmbedding:
    """
    Lookup table: integer token id → embedding vector.
    Each character gets its own learned vector of size d_model.
    """

    def __init__(self, vocab_size: int, d_model: int):
        scale       = np.sqrt(1.0 / d_model)
        self.weight = Tensor(
            np.random.randn(vocab_size, d_model).astype(np.float32) * scale,
            name='embedding'
        )
        self.vocab_size = vocab_size
        self.d_model    = d_model

    def forward(self, token_ids: np.ndarray) -> Tensor:
        """
        token_ids: (batch, seq_len) integer array
        Returns:   (batch * seq_len, d_model) — flattened for transformer
        """
        flat_ids = token_ids.flatten().astype(np.int32)
        data     = self.weight.data[flat_ids]   # index rows

        out       = Tensor(data.copy())
        out._prev = {self.weight}
        ids_ref   = flat_ids
        w_ref     = self.weight

        def _backward():
            if w_ref.requires_grad:
                # Scatter gradients back to embedding rows
                np.add.at(w_ref.grad, ids_ref, out.grad)

        out._backward = _backward
        return out

    def parameters(self):
        return [self.weight]


class SmallLM:
    """
    Small character-level language model.

    Architecture:
        token_ids → Embedding → TransformerBlock x n_layers
                  → Linear (d_model → vocab_size) → logits

    Logits are raw scores — cross-entropy loss applies softmax internally.
    """

    def __init__(self, vocab_size: int, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 2):
        self.vocab_size = vocab_size
        self.d_model    = d_model

        self.embedding = CharEmbedding(vocab_size, d_model)
        self.blocks    = [
            TransformerBlock(d_model=d_model, n_heads=n_heads)
            for _ in range(n_layers)
        ]
        self.head = Linear(d_model, vocab_size)

    def forward(self, mgr, token_ids: np.ndarray) -> Tensor:
        """
        token_ids: (batch, seq_len) integer array
        Returns:   (batch * seq_len, vocab_size) logits
        """
        x = self.embedding.forward(token_ids)   # (B*T, d_model)

        for block in self.blocks:
            x = block.forward(mgr, x)

        logits = self.head.forward(mgr, x)      # (B*T, vocab_size)
        return logits

    def __call__(self, mgr, token_ids):
        return self.forward(mgr, token_ids)

    def parameters(self):
        params = self.embedding.parameters()
        for block in self.blocks:
            params += block.parameters()
        params += self.head.parameters()
        return params

    def generate(self, mgr, seed_text: str, dataset,
                 n_chars: int = 200, temperature: float = 1.0) -> str:
        """
        Generate text by sampling one character at a time.

        seed_text:   starting string
        n_chars:     how many new characters to generate
        temperature: > 1 = more random, < 1 = more conservative
        """
        ids = dataset.encode(seed_text).tolist()

        for _ in range(n_chars):
            # Use last 64 chars as context window
            ctx      = np.array(ids[-64:], dtype=np.float32).reshape(1, -1)
            logits   = self.forward(mgr, ctx)   # (seq_len, vocab_size)

            # Take logits for the last position only
            last     = logits.data[-1]           # (vocab_size,)

            # Temperature scaling then softmax for sampling
            scaled   = last / temperature
            scaled  -= scaled.max()
            probs    = np.exp(scaled)
            probs   /= probs.sum()

            # Sample from distribution
            next_id  = np.random.choice(len(probs), p=probs)
            ids.append(next_id)

        return dataset.decode(ids[len(seed_text):])
