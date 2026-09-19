import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import kp
import numpy as np
from vtrain.tensor   import Tensor
from vtrain.grad_check import grad_check
import vtrain.functional as F


def test_autograd():
    mgr = kp.Manager(0)

    print("── matmul ──────────────────────────────")
    def fn_matmul(inputs):
        return F.matmul(mgr, inputs[0], inputs[1])

    A = Tensor(np.random.randn(4, 8).astype(np.float32) * 0.1)
    B = Tensor(np.random.randn(8, 4).astype(np.float32) * 0.1)
    grad_check(fn_matmul, [A, B], name="matmul")

    print("\n── relu ────────────────────────────────")
    def fn_relu(inputs):
        return F.relu(mgr, inputs[0])

    X = Tensor(np.random.randn(4, 4).astype(np.float32))
    grad_check(fn_relu, [X], name="relu")

    print("\n── sigmoid ─────────────────────────────")
    def fn_sigmoid(inputs):
        return F.sigmoid(mgr, inputs[0])

    X = Tensor(np.random.randn(4, 4).astype(np.float32))
    grad_check(fn_sigmoid, [X], name="sigmoid")

    print("\n── tanh ────────────────────────────────")
    def fn_tanh(inputs):
        return F.tanh(mgr, inputs[0])

    X = Tensor(np.random.randn(4, 4).astype(np.float32))
    grad_check(fn_tanh, [X], name="tanh")

    print("\n── mul ─────────────────────────────────")
    def fn_mul(inputs):
        return F.mul(mgr, inputs[0], inputs[1])

    A = Tensor(np.random.randn(4, 4).astype(np.float32))
    B = Tensor(np.random.randn(4, 4).astype(np.float32) + 0.5)
    grad_check(fn_mul, [A, B], name="mul")

    print("\n── add ─────────────────────────────────")
    def fn_add(inputs):
        return F.add(mgr, inputs[0], inputs[1])

    A = Tensor(np.random.randn(4, 4).astype(np.float32))
    B = Tensor(np.random.randn(4, 4).astype(np.float32))
    grad_check(fn_add, [A, B], name="add")

    print("\n── softmax ─────────────────────────────")
    def fn_softmax(inputs):
        return F.softmax(mgr, inputs[0])

    X = Tensor(np.random.randn(4, 8).astype(np.float32))
    grad_check(fn_softmax, [X], name="softmax")

    print("\n── layernorm ───────────────────────────")
    def fn_layernorm(inputs):
        return F.layernorm(mgr, inputs[0], inputs[1], inputs[2])

    X     = Tensor(np.random.randn(4, 8).astype(np.float32))
    gamma = Tensor(np.ones(8,  dtype=np.float32))
    beta  = Tensor(np.zeros(8, dtype=np.float32))
    grad_check(fn_layernorm, [X, gamma, beta], name="layernorm")

    print("\n── chained ops ─────────────────────────")
    def fn_chain(inputs):
        h = F.matmul(mgr, inputs[0], inputs[1])
        h = F.relu(mgr, h)
        h = F.softmax(mgr, h)
        return h

    A = Tensor(np.random.randn(4, 8).astype(np.float32) * 0.1)
    B = Tensor(np.random.randn(8, 4).astype(np.float32) * 0.1)
    grad_check(fn_chain, [A, B], name="chain")


if __name__ == "__main__":
    test_autograd()
