import numpy as np
from vtrain.tensor import Tensor


class SGD:
    """
    Stochastic Gradient Descent.

    weight = weight - lr * gradient

    That's it. The 'stochastic' part just means we compute gradients
    on a batch (subset) of data rather than the whole dataset at once —
    that's handled in the training loop, not here.
    """

    def __init__(self, params: list, lr: float = 0.01):
        """
        params: list of Tensors to update (your model's weights)
        lr:     learning rate — how big a step to take each update
        """
        self.params = params
        self.lr     = lr

    def state_dict(self) -> dict:
        """Return optimizer state for checkpointing."""
        return {"lr": self.lr}

    def load_state_dict(self, state: dict):
        """Restore optimizer state from checkpoint."""
        self.lr = state["lr"]

    def step(self):
        """Apply one gradient update to all parameters."""
        for p in self.params:
            if p.requires_grad and p.grad is not None:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        """Reset all gradients to zero. Call before each forward pass."""
        for p in self.params:
            if p.grad is not None:
                p.grad[:] = 0.0


class Adam:
    """
    Adam optimizer — Adaptive Moment Estimation.

    Keeps a running average of gradients (m) and squared gradients (v),
    uses them to adapt the learning rate per parameter.
    Converges faster than SGD in practice, especially early in training.

    Hyperparameters:
        lr:    learning rate (default 0.001 — much smaller than SGD)
        beta1: decay rate for gradient average (default 0.9)
        beta2: decay rate for squared gradient average (default 0.999)
        eps:   prevents division by zero (default 1e-8)
    """

    def __init__(self, params: list, lr: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8):
        self.params = params
        self.lr     = lr
        self.beta1  = beta1
        self.beta2  = beta2
        self.eps    = eps
        self.t      = 0   # step counter — used for bias correction

        # One momentum buffer per parameter, initialized to zero
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

    def state_dict(self) -> dict:
        """Return optimizer state for checkpointing."""
        return {
            "m":     [arr.copy() for arr in self.m],
            "v":     [arr.copy() for arr in self.v],
            "t":     self.t,
            "lr":    self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps":   self.eps,
        }

    def load_state_dict(self, state: dict):
        """Restore optimizer state from checkpoint."""
        for i in range(len(self.m)):
            self.m[i][:] = state["m"][i]
            self.v[i][:] = state["v"][i]
        self.t     = state["t"]
        self.lr    = state["lr"]
        self.beta1 = state["beta1"]
        self.beta2 = state["beta2"]
        self.eps   = state["eps"]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if not p.requires_grad or p.grad is None:
                continue

            g = p.grad

            # Update biased moment estimates
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2

            # Bias correction — early steps are pulled toward zero
            # without this, m and v start near zero and underestimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad[:] = 0.0
