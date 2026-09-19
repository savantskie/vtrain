import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import kp
import numpy as np
from vtrain.ops.transpose import transpose


def test_transpose():
    mgr = kp.Manager(0)

    # Square
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]], dtype=np.float32)

    result   = transpose(mgr, A)
    expected = A.T

    if np.allclose(result, expected):
        print("✓ 3x3 square PASSED")
    else:
        print(f"✗ 3x3 square FAILED")
        print(f"  Expected:\n{expected}")
        print(f"  Got:\n{result}")
        return

    # Non-square — this is the important one for matmul backward
    B = np.random.randn(64, 128).astype(np.float32)
    result2   = transpose(mgr, B)
    expected2 = B.T

    if np.allclose(result2, expected2):
        print("✓ 64x128 → 128x64 PASSED")
    else:
        max_err = np.abs(result2 - expected2).max()
        print(f"✗ 64x128 FAILED  max_err={max_err:.6f}")
        return

    # Double transpose should give back the original
    result3 = transpose(mgr, result2)
    if np.allclose(result3, B):
        print("✓ Double transpose roundtrip PASSED")
    else:
        print("✗ Double transpose roundtrip FAILED")


if __name__ == "__main__":
    test_transpose()
