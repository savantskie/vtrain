import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import kp
import numpy as np
from vtrain.tensor           import Tensor
from vtrain.loss             import mse_loss
from vtrain.optim            import Adam
from vtrain.model.linear     import Linear, FeedForward
from vtrain.model.transformer import TransformerBlock
import vtrain.functional as F


def test_linear_training():
    """Train a 2-layer MLP to fit y = x^2."""
    print("── MLP training: y = x² ────────────────")
    mgr = kp.Manager(0)
    np.random.seed(0)

    x_data = np.linspace(-1, 1, 32).reshape(-1, 1).astype(np.float32)
    y_data = (x_data ** 2).astype(np.float32)

    fc1 = Linear(1, 16)
    fc2 = Linear(16, 1)
    params = fc1.parameters() + fc2.parameters()
    opt = Adam(params, lr=0.01)

    first_loss = None
    for step in range(100):
        opt.zero_grad()

        X = Tensor(x_data, requires_grad=False)
        Y = Tensor(y_data, requires_grad=False)

        h    = fc1.forward(mgr, X)
        h    = F.relu(mgr, h)
        pred = fc2.forward(mgr, h)
        loss = mse_loss(pred, Y)
        loss.backward()
        opt.step()

        if step == 0:
            first_loss = loss.data.item()
        if (step + 1) % 25 == 0:
            print(f"  step {step+1:3d}  loss={loss.data.item():.6f}")

    final_loss = loss.data.item()
    if final_loss < first_loss * 0.1:
        print(f"  ✓ MLP: {first_loss:.4f} → {final_loss:.6f}")
    else:
        print(f"  ✗ MLP didn't converge ({first_loss:.4f} → {final_loss:.6f})")


def test_transformer_shapes():
    """Verify TransformerBlock produces the right output shape and finite values."""
    print("\n── TransformerBlock shape check ────────")
    mgr = kp.Manager(0)
    np.random.seed(0)

    seq_len = 8
    d_model = 32
    n_heads = 4

    block = TransformerBlock(d_model=d_model, n_heads=n_heads)
    X     = Tensor(np.random.randn(seq_len, d_model).astype(np.float32) * 0.1)

    out = block.forward(mgr, X)

    shape_ok  = out.data.shape == (seq_len, d_model)
    finite_ok = np.isfinite(out.data).all()

    print(f"  Input:  {X.data.shape}")
    print(f"  Output: {out.data.shape}")
    print(f"  Params: {len(block.parameters())}")

    if shape_ok and finite_ok:
        print("  ✓ Shape and finite check PASSED")
    else:
        print(f"  ✗ shape_ok={shape_ok}  finite_ok={finite_ok}")
        return

    # Quick backward pass — just verify gradients flow without errors
    loss = mse_loss(out, Tensor(np.zeros_like(out.data), requires_grad=False))
    loss.backward()

    grad_ok = all(
        p.grad is not None and np.isfinite(p.grad).all()
        for p in block.parameters()
    )
    print(f"  ✓ Gradients finite through full block" if grad_ok
          else "  ✗ Gradient issue in block")


if __name__ == "__main__":
    test_linear_training()
    test_transformer_shapes()
