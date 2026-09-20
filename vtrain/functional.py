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
    from vtrain.gpu_pool import get_pool
    from vtrain.ops.matmul import matmul_gpu as _matmul_resident
    from vtrain.ops.transpose import transpose_gpu as _transpose_resident

    M    = A.shape[0]
    K    = A.shape[1]
    N    = B.shape[1]
    pool = get_pool()

    t_a = A.ensure_on_gpu(mgr)
    t_b = B.ensure_on_gpu(mgr)

    t_c = (pool.acquire(M * N) if pool is not None
           else mgr.tensor(np.zeros(M * N, dtype=np.float32)))

    _matmul_resident(mgr, t_a, t_b, t_c, M, K, N)

    out = Tensor(np.empty((M, N), dtype=np.float32))
    out.mark_gpu_fresh(t_c, (M, N), mgr)
    out._prev = {A, B}

    def _backward():
        # A.grad += out.grad @ B.T
        if A.requires_grad:
            t_b_cur = B.ensure_on_gpu(mgr)
            t_b_T   = (pool.acquire(N * K) if pool is not None
                       else mgr.tensor(np.zeros(N * K, dtype=np.float32)))
            _transpose_resident(mgr, t_b_cur, t_b_T, K, N)

            grad_flat = out.grad.flatten().astype(np.float32)
            t_grad    = (pool.acquire(len(grad_flat)) if pool is not None
                         else mgr.tensor(np.zeros(len(grad_flat), dtype=np.float32)))
            t_grad.data()[:] = grad_flat
            sq = mgr.sequence()
            sq.record(kp.OpSyncDevice([t_grad]))
            sq.eval()

            t_res_a = (pool.acquire(M * K) if pool is not None
                       else mgr.tensor(np.zeros(M * K, dtype=np.float32)))
            _matmul_resident(mgr, t_grad, t_b_T, t_res_a, M, N, K)

            sq2 = mgr.sequence()
            sq2.record(kp.OpSyncLocal([t_res_a]))
            sq2.eval()
            A.grad += t_res_a.data().reshape(M, K)

            if pool is not None:
                pool.release(t_b_T)
                pool.release(t_grad)
                pool.release(t_res_a)

        # B.grad += A.T @ out.grad
        if B.requires_grad:
            t_a_cur = A.ensure_on_gpu(mgr)
            t_a_T   = (pool.acquire(K * M) if pool is not None
                       else mgr.tensor(np.zeros(K * M, dtype=np.float32)))
            _transpose_resident(mgr, t_a_cur, t_a_T, M, K)

            grad_flat = out.grad.flatten().astype(np.float32)
            t_grad    = (pool.acquire(len(grad_flat)) if pool is not None
                         else mgr.tensor(np.zeros(len(grad_flat), dtype=np.float32)))
            t_grad.data()[:] = grad_flat
            sq = mgr.sequence()
            sq.record(kp.OpSyncDevice([t_grad]))
            sq.eval()

            t_res_b = (pool.acquire(K * N) if pool is not None
                       else mgr.tensor(np.zeros(K * N, dtype=np.float32)))
            _matmul_resident(mgr, t_a_T, t_grad, t_res_b, K, M, N)

            sq2 = mgr.sequence()
            sq2.record(kp.OpSyncLocal([t_res_b]))
            sq2.eval()
            B.grad += t_res_b.data().reshape(K, N)

            if pool is not None:
                pool.release(t_a_T)
                pool.release(t_grad)
                pool.release(t_res_b)

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
