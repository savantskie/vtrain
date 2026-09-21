import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import kp
import numpy as np
from vtrain.ops.relu import relu

def test_relu():
    mgr = kp.Manager(0)

    A = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    expected = np.maximum(0, A)
    result = relu(mgr, A)

    print(f"Input:    {A}")
    print(f"Expected: {expected}")
    print(f"Got:      {result}")

    if np.allclose(result, expected):
        print("\n✓ 1D vector PASSED")
    else:
        print("\n✗ 1D vector FAILED")
        return

    B = np.random.randn(64, 128).astype(np.float32)
    expected2 = np.maximum(0, B)
    result2 = relu(mgr, B)

    if np.allclose(result2, expected2):
        print("✓ 64x128 matrix PASSED")
    else:
        print("✗ 64x128 matrix FAILED")
        max_err = np.abs(result2 - expected2).max()
        print(f"  Max error: {max_err}")

if __name__ == "__main__":
    test_relu()