import kp
import numpy as np
from vtrain.ops.matmul import matmul
from vtrain.ops.softmax import softmax
from vtrain.ops.elementwise import mul


def attention(mgr: kp.Manager,
              Q: np.ndarray,
              K: np.ndarray,
              V: np.ndarray,
              mask: np.ndarray = None) -> np.ndarray:
    """
    Scaled dot-product attention.

    Q shape: (seq_len, d_k)
    K shape: (seq_len, d_k)
    V shape: (seq_len, d_v)

    Returns: (seq_len, d_v)

    What's happening step by step:
    1. Q @ Kᵀ  — how much does each query "match" each key
    2. Scale    — stops the dot products growing huge as d_k increases
    3. Softmax  — turn scores into probabilities (attention weights)
    4. Weights @ V — weighted sum of values
    """
    d_k = Q.shape[-1]

    # Transpose K on CPU for now — (seq_len, d_k) → (d_k, seq_len)
    # We'll move this to a GPU shader in the optimization pass
    K_T = np.ascontiguousarray(K.T)

    # Step 1: raw attention scores — (seq_len, seq_len)
    scores = matmul(mgr, Q, K_T)

    # Step 2: scale — prevents softmax from getting a single
    # overwhelming winner when d_k is large
    # 1/sqrt(d_k) as a (1,1) broadcast via elementwise mul
    scale_factor = np.full(scores.shape,
                           1.0 / np.sqrt(d_k),
                           dtype=np.float32)
    scores = mul(mgr, scores, scale_factor)

    # Step 3: optional mask — used in decoder to prevent attending
    # to future tokens. Masked positions get set to -inf so softmax
    # makes them zero probability.
    if mask is not None:
        scores = scores + mask  # mask contains 0 or -1e9

    # Step 4: softmax over last dim — (seq_len, seq_len)
    weights = softmax(mgr, scores)

    # Step 5: weighted sum of values — (seq_len, d_v)
    output = matmul(mgr, weights, V)

    return output


def multihead_attention(mgr: kp.Manager,
                        X: np.ndarray,
                        W_q: np.ndarray,
                        W_k: np.ndarray,
                        W_v: np.ndarray,
                        W_o: np.ndarray,
                        n_heads: int) -> np.ndarray:
    """
    Multi-head attention — runs attention in parallel across n_heads
    subspaces, then concatenates and projects back.

    X shape:   (seq_len, d_model)
    W_q/k/v:  (d_model, d_model)
    W_o:      (d_model, d_model)

    Each head gets d_model/n_heads dimensions to work in.
    """
    seq_len, d_model = X.shape
    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    d_k = d_model // n_heads

    # Project input into Q, K, V spaces
    Q = matmul(mgr, X, W_q)  # (seq_len, d_model)
    K = matmul(mgr, X, W_k)
    V = matmul(mgr, X, W_v)

    # Split into heads — each head gets a slice of the embedding dim
    # Reshape: (seq_len, d_model) → (n_heads, seq_len, d_k)
    Q = Q.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    K = K.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    V = V.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)

    # Run attention independently per head
    head_outputs = []
    for h in range(n_heads):
        head_out = attention(mgr, Q[h], K[h], V[h])
        head_outputs.append(head_out)

    # Concatenate heads back: (n_heads, seq_len, d_k) → (seq_len, d_model)
    concat = np.concatenate(head_outputs, axis=-1)

    # Final projection
    output = matmul(mgr, concat, W_o)

    return output
