import numpy as np
import time
from pathlib import Path
from vtrain.tensor import Tensor
from vtrain.model.checkpoint import save, params_from_block


class Trainer:
    """
    Reusable training loop with logging and checkpointing.

    Usage:
        trainer = Trainer(model, optimizer, loss_fn, mgr)
        trainer.fit(X, Y, epochs=100, batch_size=16)
    """

    def __init__(self, model, optimizer, loss_fn, mgr,
                 checkpoint_dir: str = None,
                 checkpoint_every: int = 100):
        self.model            = model
        self.optimizer        = optimizer
        self.loss_fn          = loss_fn
        self.mgr              = mgr
        self.checkpoint_dir   = Path(checkpoint_dir) if checkpoint_dir else None
        self.checkpoint_every = checkpoint_every
        self.history          = []   # loss per step
        self.step             = 0

    def _forward(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        X = Tensor(X_batch, requires_grad=False)
        Y = Tensor(Y_batch, requires_grad=False)
        pred = self.model(self.mgr, X)
        loss = self.loss_fn(pred, Y)
        return loss

    def fit(self, X: np.ndarray, Y: np.ndarray,
            epochs: int = 1, batch_size: int = None,
            log_every: int = 10):
        """
        Train for a number of epochs.

        X, Y:       full dataset as numpy arrays
        epochs:     passes over the full dataset
        batch_size: None = use full dataset each step
        log_every:  print loss every N steps
        """
        n_samples = X.shape[0]
        if batch_size is None:
            batch_size = n_samples

        t_start = time.time()

        for epoch in range(epochs):
            # Shuffle each epoch
            idx = np.random.permutation(n_samples)
            X_s, Y_s = X[idx], Y[idx]

            for start in range(0, n_samples, batch_size):
                X_batch = X_s[start:start + batch_size]
                Y_batch = Y_s[start:start + batch_size]

                self.optimizer.zero_grad()
                loss = self._forward(X_batch, Y_batch)
                loss.backward()
                self.optimizer.step()

                loss_val = float(loss.data)
                self.history.append(loss_val)
                self.step += 1

                if self.step % log_every == 0:
                    elapsed = time.time() - t_start
                    print(f"  step {self.step:4d}  "
                          f"epoch {epoch+1:3d}  "
                          f"loss={loss_val:.6f}  "
                          f"elapsed={elapsed:.1f}s")

                if (self.checkpoint_dir is not None and
                        self.step % self.checkpoint_every == 0):
                    self._save_checkpoint()

        print(f"\nTraining complete — {self.step} steps, "
              f"final loss={self.history[-1]:.6f}")
        return self.history

    def _save_checkpoint(self):
        ckpt_path = self.checkpoint_dir / f"step_{self.step:06d}"
        p = params_from_block(self.model, prefix="model")
        save(p, str(ckpt_path))

    def resume(self, checkpoint_path: str):
        """Load weights from a checkpoint and set step counter."""
        from vtrain.model.checkpoint import load
        p = params_from_block(self.model, prefix="model")
        load(p, checkpoint_path)
        # Extract step from folder name if it follows our convention
        name = Path(checkpoint_path).name
        if name.startswith("step_"):
            self.step = int(name.split("_")[1])
            print(f"Resumed from step {self.step}")
