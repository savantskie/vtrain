import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import kp
import numpy as np
from vtrain.tensor     import Tensor
from vtrain.grad_check import grad_check
from vtrain.loss       import mse_loss, cross_entropy_loss, one_hot
import vtrain.functional as F


def test_losses():
    mgr = kp.Manager(0)

    print("── MSE loss ────────────────────────────")

    pred   = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    target = Tensor(np.array([[1.5, 1.5], [2.5, 4.5]], dtype=np.float32),
                    requires_grad=False)

    loss = mse_loss(pred, target)
    expected_loss = np.mean((pred.data - target.data) ** 2)

    print(f"  Loss value: {loss.data:.6f}  expected: {expected_loss:.6f}")
    assert np.isclose(loss.data, expected_loss, atol=1e-6), "MSE value wrong"
    print("  ✓ MSE value correct")

    # Grad check
    pred   = Tensor(np.random.randn(4, 4).astype(np.float32))
    target = Tensor(np.random.randn(4, 4).astype(np.float32),
                    requires_grad=False)

    def fn_mse(inputs):
        return mse_loss(inputs[0], target)

    grad_check(fn_mse, [pred], name="mse")

    print("\n── Cross-entropy loss ──────────────────")

    # 4 samples, 3 classes
    logits  = np.random.randn(4, 3).astype(np.float32)
    probs   = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs  /= probs.sum(axis=-1, keepdims=True)

    labels  = np.array([0, 1, 2, 1])
    oh      = one_hot(labels, 3)

    pred    = Tensor(probs)
    target  = Tensor(oh, requires_grad=False)
    loss    = cross_entropy_loss(pred, target)

    expected_ce = -np.mean(np.sum(oh * np.log(np.clip(probs, 1e-7, 1.0)), axis=-1))
    print(f"  Loss value: {loss.data:.6f}  expected: {expected_ce:.6f}")
    assert np.isclose(loss.data, expected_ce, atol=1e-5), "CE value wrong"
    print("  ✓ Cross-entropy value correct")

    def fn_ce(inputs):
        return cross_entropy_loss(inputs[0], target)

    pred = Tensor(np.abs(np.random.randn(4, 3).astype(np.float32)) + 0.1)
    pred.data /= pred.data.sum(axis=-1, keepdims=True)  # make valid probs
    grad_check(fn_ce, [pred], name="cross_entropy")

    print("\n── End-to-end graph ────────────────────")
    # matmul → relu → softmax → cross-entropy
    # Smallest possible meaningful forward pass + loss

    X      = Tensor(np.random.randn(4, 8).astype(np.float32) * 0.1)
    W      = Tensor(np.random.randn(8, 3).astype(np.float32) * 0.1)
    labels = np.array([0, 1, 2, 0])
    oh     = Tensor(one_hot(labels, 3), requires_grad=False)

    h    = F.matmul(mgr, X, W)
    h    = F.relu(mgr, h)
    h    = F.softmax(mgr, h)
    loss = cross_entropy_loss(h, oh)

    print(f"  Forward loss: {loss.data:.6f}")
    loss.backward()

    print(f"  W.grad norm:  {np.linalg.norm(W.grad):.6f}")
    print(f"  X.grad norm:  {np.linalg.norm(X.grad):.6f}")

    assert np.isfinite(W.grad).all(), "W gradients contain NaN/Inf"
    assert np.isfinite(X.grad).all(), "X gradients contain NaN/Inf"
    print("  ✓ Gradients finite and flowing through full graph")


if __name__ == "__main__":
    test_losses()
