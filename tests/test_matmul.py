import sys
from pathlib import Path

# Make sure Python can find our vtrain package
sys.path.insert(0, str(Path(__file__).parent.parent))

import kp
import numpy as np
from vtrain.ops.matmul import matmul

def test_matmul():
    mgr = kp.Manager(0)  # GPU0, first MI50

    # Small obvious case first: 2x2 matrices
    # Makes it easy to spot exactly what went wrong if it does
    A = np.array([[1, 2],
                  [3, 4]], dtype=np.float32)

    B = np.array([[5, 6],
                  [7, 8]], dtype=np.float32)

    # What numpy says the answer should be
    expected = A @ B

    # What our shader says
    result = matmul(mgr, A, B)

    print(f"A:\n{A}")
    print(f"B:\n{B}")
    print(f"Expected:\n{expected}")
    print(f"Got:\n{result}")

    if np.allclose(result, expected, atol=1e-4):
        print("\n✓ 2x2 PASSED")
    else:
        print("\n✗ 2x2 FAILED")
        return

    # Slightly bigger — tests that workgroup boundary math is right
    A2 = np.random.rand(64, 128).astype(np.float32)
    B2 = np.random.rand(128, 64).astype(np.float32)

    expected2 = A2 @ B2
    result2   = matmul(mgr, A2, B2)

    if np.allclose(result2, expected2, atol=1e-3):
        print("✓ 64x128 @ 128x64 PASSED")
    else:
        print("✗ 64x128 @ 128x64 FAILED")
        max_err = np.abs(result2 - expected2).max()
        print(f"  Max error: {max_err}")

if __name__ == "__main__":
    test_matmul()
