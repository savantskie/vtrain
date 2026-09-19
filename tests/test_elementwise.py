import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import kp
import numpy as np
from vtrain.ops.elementwise import relu, sigmoid, tanh_act, gelu, add, sub, mul, div

def test_all():
    mgr = kp.Manager(0)

    A = np.random.randn(64, 128).astype(np.float32)
    B = np.random.randn(64, 128).astype(np.float32) + 0.1  # offset avoids div-by-zero

    tests = [
        ("relu",    lambda: relu(mgr, A),     lambda: np.maximum(0, A)),
        ("sigmoid", lambda: sigmoid(mgr, A),  lambda: 1 / (1 + np.exp(-A))),
        ("tanh",    lambda: tanh_act(mgr, A), lambda: np.tanh(A)),
        ("gelu",    lambda: gelu(mgr, A),     lambda: 0.5 * A * (1 + np.tanh(
                                                  np.sqrt(2 / np.pi) * (A + 0.044715 * A**3)))),
        ("add",     lambda: add(mgr, A, B),   lambda: A + B),
        ("sub",     lambda: sub(mgr, A, B),   lambda: A - B),
        ("mul",     lambda: mul(mgr, A, B),   lambda: A * B),
        ("div",     lambda: div(mgr, A, B),   lambda: A / B),
    ]

    all_passed = True
    for name, gpu_fn, cpu_fn in tests:
        result   = gpu_fn()
        expected = cpu_fn()
        if np.allclose(result, expected, atol=1e-4):
            print(f"✓ {name}")
        else:
            max_err = np.abs(result - expected).max()
            print(f"✗ {name}  max_err={max_err:.6f}")
            all_passed = False

    print("\nAll passed." if all_passed else "\nSome ops failed.")

if __name__ == "__main__":
    test_all()
