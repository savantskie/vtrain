import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import kp
import numpy as np
from vtrain.tensor            import Tensor
from vtrain.loss              import mse_loss
from vtrain.optim             import Adam
from vtrain.model.linear      import Linear
from vtrain.model.checkpoint  import save, load, params_from_block
from vtrain.train             import Trainer
import vtrain.functional as F
import tempfile
import os


def test_checkpoint():
    print("── Save / Load ──────────────────────────")
    mgr = kp.Manager(0)
    np.random.seed(0)

    fc = Linear(4, 2)
    p  = params_from_block(fc, prefix="fc")

    with tempfile.TemporaryDirectory() as tmp:
        save(p, tmp)

        # Corrupt the weights
        original_W = fc.W.data.copy()
        fc.W.data[:] = 999.0

        # Reload
        load(p, tmp)

        if np.allclose(fc.W.data, original_W):
            print("  ✓ Save/load roundtrip correct")
        else:
            print("  ✗ Weights don't match after reload")
            return

    print("\n── Trainer fit ─────────────────────────")

    x = np.random.randn(64, 4).astype(np.float32)
    y = np.random.randn(64, 2).astype(np.float32)

    fc  = Linear(4, 2)
    opt = Adam(fc.parameters(), lr=0.01)

    def model_fn(mgr, X):
        return fc.forward(mgr, X)

    trainer = Trainer(model_fn, opt, mse_loss, mgr)
    history = trainer.fit(x, y, epochs=5, batch_size=16, log_every=20)

    if history[-1] < history[0]:
        print(f"  ✓ Trainer: loss {history[0]:.4f} → {history[-1]:.6f}")
    else:
        print(f"  ✗ Trainer: loss didn't improve")


if __name__ == "__main__":
    test_checkpoint()
