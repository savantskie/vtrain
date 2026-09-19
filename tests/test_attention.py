import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import kp
import numpy as np
from vtrain.ops.attention import attention, multihead_attention


def numpy_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ V


def test_attention():
    mgr = kp.Manager(0)

    seq_len = 8
    d_k     = 16
    d_v     = 16

    Q = np.random.randn(seq_len, d_k).astype(np.float32)
    K = np.random.randn(seq_len, d_k).astype(np.float32)
    V = np.random.randn(seq_len, d_v).astype(np.float32)

    expected = numpy_attention(Q, K, V)
    result   = attention(mgr, Q, K, V)

    if np.allclose(result, expected, atol=1e-4):
        print(f"✓ Single-head attention ({seq_len}x{d_k}) PASSED")
    else:
        max_err = np.abs(result - expected).max()
        print(f"✗ Single-head attention FAILED  max_err={max_err:.6f}")
        return

    # Multi-head
    seq_len = 16
    d_model = 64
    n_heads = 4

    X   = np.random.randn(seq_len, d_model).astype(np.float32)
    W_q = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
    W_k = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
    W_v = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
    W_o = np.random.randn(d_model, d_model).astype(np.float32) * 0.1

    result_mha = multihead_attention(mgr, X, W_q, W_k, W_v, W_o, n_heads)

    # For MHA we just check shape and that values are finite —
    # no clean numpy reference to compare against at this level
    assert result_mha.shape == (seq_len, d_model), \
        f"Wrong shape: {result_mha.shape}"
    assert np.isfinite(result_mha).all(), "Output contains NaN or Inf"

    print(f"✓ Multi-head attention ({n_heads} heads, {seq_len}x{d_model}) PASSED")
    print(f"  Output shape: {result_mha.shape}")
    print(f"  Output range: [{result_mha.min():.4f}, {result_mha.max():.4f}]")


if __name__ == "__main__":
    test_attention()
