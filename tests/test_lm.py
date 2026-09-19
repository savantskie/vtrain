import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import kp
import numpy as np
from vtrain.data.dataset  import CharDataset
from vtrain.model.lm      import SmallLM
from vtrain.loss          import cross_entropy_loss, one_hot
from vtrain.optim         import Adam
from vtrain.tensor        import Tensor


def test_lm():
    print("── LM shape check ──────────────────────")
    mgr = kp.Manager(0)
    np.random.seed(0)

    # Tiny dataset from a sample string
    text = "the quick brown fox jumps over the lazy dog. " * 100
    ds   = CharDataset(text=text)

    print(f"  Vocab size: {ds.vocab_size}")
    print(f"  Data size:  {len(ds.data):,} chars")

    model = SmallLM(
        vocab_size = ds.vocab_size,
        d_model    = 32,
        n_heads    = 2,
        n_layers   = 1
    )
    print(f"  Parameters: {len(model.parameters())}")

    # Forward pass
    X, Y = ds.get_batch(batch_size=2, seq_len=8)
    logits = model.forward(mgr, X)

    expected_shape = (2 * 8, ds.vocab_size)
    assert logits.data.shape == expected_shape, \
        f"Wrong shape: {logits.data.shape} vs {expected_shape}"
    assert np.isfinite(logits.data).all(), "Logits contain NaN/Inf"
    print(f"  ✓ Forward pass shape: {logits.data.shape}")

    # Loss + backward
    Y_flat = Y.flatten().astype(np.int32)
    Y_oh   = Tensor(
        np.eye(ds.vocab_size, dtype=np.float32)[Y_flat],
        requires_grad=False
    )

    # Softmax logits before cross entropy
    import vtrain.functional as F
    probs = F.softmax(mgr, logits)
    loss  = cross_entropy_loss(probs, Y_oh)

    print(f"  Initial loss: {loss.data.item():.4f}  "
          f"(expected ~{np.log(ds.vocab_size):.2f} for random init)")

    loss.backward()

    grad_ok = all(
        p.grad is not None and np.isfinite(p.grad).all()
        for p in model.parameters()
    )
    print(f"  ✓ Gradients finite" if grad_ok else "  ✗ Gradient issue")

    # Quick 20-step training loop to verify loss moves
    print("\n── LM 20-step smoke test ───────────────")
    opt = Adam(model.parameters(), lr=0.01)

    first_loss = None
    for step in range(20):
        opt.zero_grad()

        X, Y    = ds.get_batch(batch_size=4, seq_len=8)
        logits  = model.forward(mgr, X)
        probs   = F.softmax(mgr, logits)
        Y_flat  = Y.flatten().astype(np.int32)
        Y_oh    = Tensor(
            np.eye(ds.vocab_size, dtype=np.float32)[Y_flat],
            requires_grad=False
        )
        loss = cross_entropy_loss(probs, Y_oh)
        loss.backward()
        opt.step()

        if step == 0:
            first_loss = loss.data.item()
        if (step + 1) % 5 == 0:
            print(f"  step {step+1:2d}  loss={loss.data.item():.4f}")

    final_loss = loss.data.item()
    if final_loss < first_loss:
        print(f"  ✓ Loss moving: {first_loss:.4f} → {final_loss:.4f}")
    else:
        print(f"  ✗ Loss not improving ({first_loss:.4f} → {final_loss:.4f})")

    # Generation test
    print("\n── Generation test ─────────────────────")
    sample = model.generate(mgr, "the ", ds, n_chars=50, temperature=0.8)
    print(f"  Seed: 'the '")
    print(f"  Generated: '{sample}'")
    print("  (garbage expected — model is untrained)")


if __name__ == "__main__":
    test_lm()
