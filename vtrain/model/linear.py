import numpy as np
from vtrain.tensor import Tensor
import vtrain.functional as F


class Linear:
    """
    Fully connected layer: output = X @ W + b
    W shape: (in_features, out_features)
    b shape: (out_features,) — broadcast across batch
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        # Xavier initialization — keeps activations stable at the start
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.W = Tensor(
            np.random.randn(in_features, out_features).astype(np.float32) * scale,
            name=f"W({in_features}x{out_features})"
        )
        self.b = Tensor(
            np.zeros(out_features, dtype=np.float32),
            name=f"b({out_features})"
        ) if bias else None

    def forward(self, mgr, X: Tensor) -> Tensor:
        out = F.matmul(mgr, X, self.W)

        if self.b is not None:
            batch        = X.data.shape[0]
            b_tiled_data = np.tile(self.b.data, (batch, 1))
            result       = Tensor(out.data + b_tiled_data)
            result._prev = {out, self.b}
            b_ref        = self.b

            def _backward():
                if out.requires_grad:
                    out.grad += result.grad
                if b_ref.requires_grad:
                    # Bias gradient is sum over batch dimension
                    b_ref.grad += result.grad.sum(axis=0)

            result._backward = _backward
            return result

        return out

    def parameters(self):
        params = [self.W]
        if self.b is not None:
            params.append(self.b)
        return params

    def __call__(self, mgr, X):
        return self.forward(mgr, X)


class FeedForward:
    """Two linear layers with GELU — the FFN inside a transformer block."""

    def __init__(self, d_model: int, d_ff: int = None):
        if d_ff is None:
            d_ff = 4 * d_model
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)

    def forward(self, mgr, X: Tensor) -> Tensor:
        h = self.fc1.forward(mgr, X)
        h = F.gelu(mgr, h)
        h = self.fc2.forward(mgr, h)
        return h

    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()

    def __call__(self, mgr, X):
        return self.forward(mgr, X)
