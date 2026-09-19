import numpy as np
import kp
from vtrain.tensor import Tensor
from vtrain.ops.matmul    import matmul    as _gpu_matmul
from vtrain.ops.transpose import transpose as _gpu_transpose
from vtrain.ops.elementwise import (
    relu     as _gpu_relu,
    sigmoid  as _gpu_sigmoid,
    tanh_act as _gpu_tanh,
    gelu     as _gpu_gelu,
    add      as _gpu_add,
    sub      as _gpu_sub,
    mul      as _gpu_mul,
    div      as _gpu_div,
)
from vtrain.ops.layernorm import layernorm as _gpu_layernorm
from vtrain.ops.softmax   import softmax   as _gpu_softmax


# ── Matmul ────────────────────────────────────────────────────────────────────

def matmul(mgr: kp.Manager, A: Tensor, B: Tensor) -> Tensor:
    out = Tensor(_gpu_matmul(mgr, A.data, B.data))
    out._prev = {A, B}

    def _backward():
        # C = A @ B
        # dL/dA = dL/dC @ B.T
        # dL/dB = A.T  @ dL/dC
        if A.requires_grad:
            B_T = _gpu_transpose(mgr, B.data)
            A.grad += _gpu_matmul(mgr, out.grad, B_T)
        if B.requires_grad:
            A_T = _gpu_transpose(mgr, A.data)
            B.grad += _gpu_matmul(mgr, A_T, out.grad)

    out._backward = _backward
    return out


# ── Unary elementwise ─────────────────────────────────────────────────────────

def relu(mgr: kp.Manager, X: Tensor) -> Tensor:
    out = Tensor(_gpu_relu(mgr, X.data))
    out._prev = {X}

    def _backward():
        # Gradient only flows where input was positive
        # (x > 0) gives 1.0 where active, 0.0 where not
        if X.requires_grad:
            mask = (X.data > 0).astype(np.float32)
            X.grad += out.grad * mask

    out._backward = _backward
    return out


def sigmoid(mgr: kp.Manager, X: Tensor) -> Tensor:
    s = _gpu_sigmoid(mgr, X.data)
    out = Tensor(s)
    out._prev = {X}

    def _backward():
        # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        if X.requires_grad:
            X.grad += out.grad * s * (1.0 - s)

    out._backward = _backward
    return out


def tanh(mgr: kp.Manager, X: Tensor) -> Tensor:
    t = _gpu_tanh(mgr, X.data)
    out = Tensor(t)
    out._prev = {X}

    def _backward():
        # d/dx tanh(x) = 1 - tanh²(x)
        if X.requires_grad:
            X.grad += out.grad * (1.0 - t ** 2)

    out._backward = _backward
    return out


def gelu(mgr: kp.Manager, X: Tensor) -> Tensor:
    # We need the input for the backward pass — capture it now
    x_data = X.data.copy()
    out = Tensor(_gpu_gelu(mgr, X.data))
    out._prev = {X}

    def _backward():
        # GELU derivative via the tanh approximation
        # Matches what we use in the forward shader
        import numpy as _np
        c     = 0.7978845608  # sqrt(2/pi)
        inner = c * (x_data + 0.044715 * x_data ** 3)
        tanh_inner = _np.tanh(inner)
        dgelu = 0.5 * (1.0 + tanh_inner) + \
                0.5 * x_data * (1.0 - tanh_inner ** 2) * \
                c * (1.0 + 3 * 0.044715 * x_data ** 2)
        if X.requires_grad:
            X.grad += out.grad * dgelu

    out._backward = _backward
    return out


# ── Binary elementwise ────────────────────────────────────────────────────────

def add(mgr: kp.Manager, A: Tensor, B: Tensor) -> Tensor:
    out = Tensor(_gpu_add(mgr, A.data, B.data))
    out._prev = {A, B}

    def _backward():
        # Gradient flows through addition unchanged to both inputs
        if A.requires_grad:
            A.grad += out.grad
        if B.requires_grad:
            B.grad += out.grad

    out._backward = _backward
    return out


def sub(mgr: kp.Manager, A: Tensor, B: Tensor) -> Tensor:
    out = Tensor(_gpu_sub(mgr, A.data, B.data))
    out._prev = {A, B}

    def _backward():
        if A.requires_grad:
            A.grad += out.grad
        if B.requires_grad:
            B.grad -= out.grad   # subtracted, so gradient flips sign

    out._backward = _backward
    return out


def mul(mgr: kp.Manager, A: Tensor, B: Tensor) -> Tensor:
    out = Tensor(_gpu_mul(mgr, A.data, B.data))
    out._prev = {A, B}

    def _backward():
        # Product rule: each input's gradient is the other input's value
        if A.requires_grad:
            A.grad += _gpu_mul(mgr, out.grad, B.data)
        if B.requires_grad:
            B.grad += _gpu_mul(mgr, out.grad, A.data)

    out._backward = _backward
    return out


def div(mgr: kp.Manager, A: Tensor, B: Tensor) -> Tensor:
    out = Tensor(_gpu_div(mgr, A.data, B.data))
    out._prev = {A, B}

    def _backward():
        # dL/dA = dL/dC / B
        # dL/dB = -dL/dC * A / B²
        if A.requires_grad:
            A.grad += _gpu_div(mgr, out.grad, B.data)
        if B.requires_grad:
            B_sq  = _gpu_mul(mgr, B.data, B.data)
            A_div = _gpu_div(mgr, A.data, B_sq)
            B.grad -= _gpu_mul(mgr, out.grad, A_div)

    out._backward = _backward
    return out


# ── Layer norm ────────────────────────────────────────────────────────────────

def layernorm(mgr: kp.Manager,
              X:     Tensor,
              gamma: Tensor,
              beta:  Tensor,
              eps:   float = 1e-5) -> Tensor:

    x_data = X.data
    rows, cols = x_data.shape

    # Compute stats we'll need for backward — on CPU, they're scalars per row
    mean    = x_data.mean(axis=-1, keepdims=True)
    var     = x_data.var(axis=-1,  keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_norm  = (x_data - mean) * inv_std  # normalized, before scale/shift

    out = Tensor(_gpu_layernorm(mgr, x_data, gamma.data, beta.data, eps))
    out._prev = {X, gamma, beta}

    def _backward():
        dy = out.grad  # (rows, cols)

        if gamma.requires_grad:
            # gamma's grad is sum of dy * x_norm over the batch (rows)
            gamma.grad += (dy * x_norm).sum(axis=0)
        if beta.requires_grad:
            beta.grad  += dy.sum(axis=0)

        if X.requires_grad:
            g = gamma.data  # (cols,)
            N = cols

            # Three terms from the layer norm backward derivation:
            # 1. direct term
            # 2. mean correction
            # 3. variance correction
            dy_scaled = dy * g * inv_std
            X.grad += (
                dy_scaled
                - dy_scaled.mean(axis=-1, keepdims=True)
                - x_norm * (dy_scaled * x_norm).mean(axis=-1, keepdims=True)
            )

    out._backward = _backward
    return out


# ── Softmax ───────────────────────────────────────────────────────────────────

def softmax(mgr: kp.Manager, X: Tensor) -> Tensor:
    s = _gpu_softmax(mgr, X.data)   # softmax output, (rows, cols)
    out = Tensor(s)
    out._prev = {X}

    def _backward():
        if X.requires_grad:
            # For each row: dx = s * (dy - sum(dy * s))
            # This is the Jacobian-vector product for softmax,
            # collapsed into a form that doesn't need the full Jacobian matrix
            dy  = out.grad
            dot = (dy * s).sum(axis=-1, keepdims=True)  # sum per row
            X.grad += s * (dy - dot)

    out._backward = _backward
    return out

# ── Transpose ─────────────────────────────────────────────────────────────────

def transpose(mgr: kp.Manager, X: Tensor) -> Tensor:
    from vtrain.ops.transpose import transpose as _gpu_transpose
    out = Tensor(_gpu_transpose(mgr, X.data))
    out._prev = {X}

    def _backward():
        if X.requires_grad:
            X.grad += _gpu_transpose(mgr, out.grad)

    out._backward = _backward
    return out


# ── Differentiable attention ───────────────────────────────────────────────────

def attention(mgr: kp.Manager, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
    """
    Scaled dot-product attention, fully differentiable via composed ops.
    Gradients flow automatically through transpose, matmul, mul, softmax.
    """
    import math
    d_k = Q.data.shape[-1]

    K_T    = transpose(mgr, K)
    scores = matmul(mgr, Q, K_T)

    scale  = Tensor(
        np.full(scores.data.shape, 1.0 / math.sqrt(d_k), dtype=np.float32),
        requires_grad=False
    )
    scores  = mul(mgr, scores, scale)
    weights = softmax(mgr, scores)
    return matmul(mgr, weights, V)
