import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import kp
import numpy as np
from vtrain.ops.softmax import softmax

def numpy_softmax(X):
    X = X - X.max(axis=-1, keepdims=True)  # same stable trick
    e = np.exp(X)
    return e / e.sum(axis=-1, keepdims=True)

def test_softmax():
    mgr = kp.Manager(0)

    # Known small case
    X = np.array([[1.0, 2.0, 3.0, 4.0],
                  [4.0, 3.0, 2.0, 1.0]], dtype=np.float32)

    expected = numpy_softmax(X)
    result   = softmax(mgr, X)

    print(f"Expected:\n{expected}")
    print(f"Got:\n{result}")
    print(f"Row sums (should be 1.0): {result.sum(axis=-1)}")

    if np.allclose(result, expected, atol=1e-5):
        print("\n✓ 2x4 PASSED")
    else:
        print("\n✗ 2x4 FAILED")
        print(f"  Max err: {np.abs(result - expected).max():.6f}")
        return

    # Realistic attention score size — 32 heads, 512 sequence length
    X2 = np.random.randn(32, 512).astype(np.float32)
    expected2 = numpy_softmax(X2)
    result2   = softmax(mgr, X2)

    row_sums = result2.sum(axis=-1)

    if np.allclose(result2, expected2, atol=1e-5) and np.allclose(row_sums, 1.0, atol=1e-5):
        print("✓ 32x512 PASSED")
    else:
        max_err = np.abs(result2 - expected2).max()
        print(f"✗ 32x512 FAILED  max_err={max_err:.6f}")
        print(f"  Row sum range: {row_sums.min():.6f} – {row_sums.max():.6f}")

if __name__ == "__main__":
    test_softmax()
