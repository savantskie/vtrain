import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import kp
import numpy as np
from vtrain.tensor  import Tensor
from vtrain.loss    import mse_loss
from vtrain.optim   import SGD, Adam
import vtrain.functional as F


def test_optimizer(name, optim_class, lr, steps=200):
    """
    Train a single linear layer (matmul) to fit y = X @ W_true.
    If the optimizer is working, loss should drop consistently.
    """
    mgr = kp.Manager(0)

    np.random.seed(42)
    W_true = np.random.randn(4, 2).astype(np.float32)
    X_data = np.random.randn(8, 4).astype(np.float32)
    Y_data = X_data @ W_true   # ground truth

    X = Tensor(X_data, requires_grad=False)
    Y = Tensor(Y_data, requires_grad=False)
    W = Tensor(np.random.randn(4, 2).astype(np.float32) * 0.1)

    opt = optim_class([W], lr=lr)

    first_loss = None
    for step in range(steps):
        opt.zero_grad()

        pred = F.matmul(mgr, X, W)
        loss = mse_loss(pred, Y)
        loss.backward()
        opt.step()

        if step == 0:
            first_loss = loss.data.item()
        if (step + 1) % 40 == 0:
            print(f"  step {step+1:3d}  loss={loss.data.item():.6f}")

    final_loss = loss.data.item()
    improved   = final_loss < first_loss * 0.5  # expect 2x improvement

    if improved:
        print(f"  ✓ {name}: loss {first_loss:.4f} → {final_loss:.6f}")
    else:
        print(f"  ✗ {name}: loss didn't improve enough "
              f"({first_loss:.4f} → {final_loss:.6f})")

    return improved


if __name__ == "__main__":
    print("── SGD ─────────────────────────────────")
    test_optimizer("SGD",  SGD,  lr=0.01)

    print("\n── Adam ────────────────────────────────")
    test_optimizer("Adam", Adam, lr=0.01)
