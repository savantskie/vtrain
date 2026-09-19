import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import kp
import numpy as np
from vtrain.ops.layernorm import layernorm

def numpy_layernorm(X, gamma, beta, eps=1e-5):
    mean = X.mean(axis=-1, keepdims=True)
    var  = X.var(axis=-1,  keepdims=True)
    return gamma * (X - mean) / np.sqrt(var + eps) + beta

def test_layernorm():
    mgr = kp.Manager(0)

    # Small known case first
    X = np.array([[1.0, 2.0, 3.0, 4.0],
                  [4.0, 3.0, 2.0, 1.0]], dtype=np.float32)
    gamma = np.ones(4,  dtype=np.float32)
    beta  = np.zeros(4, dtype=np.float32)

    expected = numpy_layernorm(X, gamma, beta)
    result   = layernorm(mgr, X, gamma, beta)

    print(f"Expected:\n{expected}")
    print(f"Got:\n{result}")

    if np.allclose(result, expected, atol=1e-4):
        print("\n✓ 2x4 PASSED")
    else:
        print("\n✗ 2x4 FAILED")
        print(f"  Max err: {np.abs(result - expected).max():.6f}")
        return

    # Realistic size — what you'd see between transformer layers
    X2    = np.random.randn(32, 512).astype(np.float32)
    gamma2 = np.random.rand(512).astype(np.float32) + 0.5
    beta2  = np.random.randn(512).astype(np.float32) * 0.1

    expected2 = numpy_layernorm(X2, gamma2, beta2)
    result2   = layernorm(mgr, X2, gamma2, beta2)

    if np.allclose(result2, expected2, atol=1e-4):
        print("✓ 32x512 with learned gamma/beta PASSED")
    else:
        max_err = np.abs(result2 - expected2).max()
        print(f"✗ 32x512 FAILED  max_err={max_err:.6f}")

if __name__ == "__main__":
    test_layernorm()
