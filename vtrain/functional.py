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
    accumulate_gpu,
    binary_gpu,
    unary_backward_gpu,
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
        if A.requires_grad:
            t_b_cur = B.ensure_on_gpu(mgr)
            t_b_T   = (pool.acquire(N * K) if pool is not None
                       else mgr.tensor(np.zeros(N * K, dtype=np.float32)))
            _transpose_resident(mgr, t_b_cur, t_b_T, K, N)

            t_res_a = (pool.acquire(M * K) if pool is not None
                       else mgr.tensor(np.zeros(M * K, dtype=np.float32)))
            _matmul_resident(mgr, out._grad_kp, t_b_T, t_res_a, M, N, K)
            A._ensure_grad_on_gpu(mgr)
            accumulate_gpu(mgr, A._grad_kp, t_res_a, M * K, 1.0)

            if pool is not None:
                pool.release(t_b_T)
                pool.release(t_res_a)

        if B.requires_grad:
            t_a_cur = A.ensure_on_gpu(mgr)
            t_a_T   = (pool.acquire(K * M) if pool is not None
                       else mgr.tensor(np.zeros(K * M, dtype=np.float32)))
            _transpose_resident(mgr, t_a_cur, t_a_T, M, K)

            t_res_b = (pool.acquire(K * N) if pool is not None
                       else mgr.tensor(np.zeros(K * N, dtype=np.float32)))
            _matmul_resident(mgr, t_a_T, out._grad_kp, t_res_b, K, M, N)
            B._ensure_grad_on_gpu(mgr)
            accumulate_gpu(mgr, B._grad_kp, t_res_b, K * N, 1.0)

            if pool is not None:
                pool.release(t_a_T)
                pool.release(t_res_b)

    out._backward = _backward
    return out


# ── Unary elementwise ─────────────────────────────────────────────────────────

def relu(mgr: kp.Manager, X: Tensor) -> Tensor:
    from vtrain.ops.elementwise import unary_gpu
    t_x = X.ensure_on_gpu(mgr)
    n = int(np.prod(X.shape))
    pool = _get_pool()
    t_y = pool.acquire(n) if pool else mgr.tensor(np.zeros(n, dtype=np.float32))
    unary_gpu(mgr, t_x, t_y, n, 'relu')

    out = Tensor(np.empty(X.shape, dtype=np.float32))
    out.mark_gpu_fresh(t_y, X.shape, mgr)
    out._prev = {X}

    def _backward():
        if X.requires_grad:
            X._ensure_grad_on_gpu(mgr)
            unary_backward_gpu(mgr, out._grad_kp, t_x, X._grad_kp, n, 'relu')

    out._backward = _backward
    return out


def sigmoid(mgr: kp.Manager, X: Tensor) -> Tensor:
    from vtrain.ops.elementwise import unary_gpu
    t_x = X.ensure_on_gpu(mgr)
    n = int(np.prod(X.shape))
    pool = _get_pool()
    t_y = pool.acquire(n) if pool else mgr.tensor(np.zeros(n, dtype=np.float32))
    unary_gpu(mgr, t_x, t_y, n, 'sigmoid')

    out = Tensor(np.empty(X.shape, dtype=np.float32))
    out.mark_gpu_fresh(t_y, X.shape, mgr)
    out._prev = {X}

    def _backward():
        if X.requires_grad:
            X._ensure_grad_on_gpu(mgr)
            unary_backward_gpu(mgr, out._grad_kp, t_x, X._grad_kp, n, 'sigmoid')

    out._backward = _backward
    return out


def tanh(mgr: kp.Manager, X: Tensor) -> Tensor:
    from vtrain.ops.elementwise import unary_gpu
    t_x = X.ensure_on_gpu(mgr)
    n = int(np.prod(X.shape))
    pool = _get_pool()
    t_y = pool.acquire(n) if pool else mgr.tensor(np.zeros(n, dtype=np.float32))
    unary_gpu(mgr, t_x, t_y, n, 'tanh')

    out = Tensor(np.empty(X.shape, dtype=np.float32))
    out.mark_gpu_fresh(t_y, X.shape, mgr)
    out._prev = {X}

    def _backward():
        if X.requires_grad:
            X._ensure_grad_on_gpu(mgr)
            unary_backward_gpu(mgr, out._grad_kp, t_x, X._grad_kp, n, 'tanh')

    out._backward = _backward
    return out


def gelu(mgr: kp.Manager, X: Tensor) -> Tensor:
    from vtrain.ops.elementwise import unary_gpu
    t_x = X.ensure_on_gpu(mgr)
    n = int(np.prod(X.shape))
    pool = _get_pool()
    t_y = pool.acquire(n) if pool else mgr.tensor(np.zeros(n, dtype=np.float32))
    unary_gpu(mgr, t_x, t_y, n, 'gelu')

    out = Tensor(np.empty(X.shape, dtype=np.float32))
    out.mark_gpu_fresh(t_y, X.shape, mgr)
    out._prev = {X}

    def _backward():
        if X.requires_grad:
            X._ensure_grad_on_gpu(mgr)
            unary_backward_gpu(mgr, out._grad_kp, t_x, X._grad_kp, n, 'gelu')

    out._backward = _backward
    return out


# ── Binary elementwise ────────────────────────────────────────────────────────

def add(mgr: kp.Manager, A: Tensor, B: Tensor) -> Tensor:
    t_a = A.ensure_on_gpu(mgr)
    t_b = B.ensure_on_gpu(mgr)
    n = int(np.prod(A.shape))
    pool = _get_pool()
    t_c = pool.acquire(n) if pool else mgr.tensor(np.zeros(n, dtype=np.float32))
    binary_gpu(mgr, t_a, t_b, t_c, n, 'add')

    out = Tensor(np.empty(A.shape, dtype=np.float32))
    out.mark_gpu_fresh(t_c, A.shape, mgr)
    out._prev = {A, B}

    def _backward():
        n = int(np.prod(A.shape))
        if A.requires_grad:
            A._ensure_grad_on_gpu(mgr)
            accumulate_gpu(mgr, A._grad_kp, out._grad_kp, n, 1.0)
        if B.requires_grad:
            B._ensure_grad_on_gpu(mgr)
            accumulate_gpu(mgr, B._grad_kp, out._grad_kp, n, 1.0)

    out._backward = _backward
    return out


def sub(mgr: kp.Manager, A: Tensor, B: Tensor) -> Tensor:
    t_a = A.ensure_on_gpu(mgr)
    t_b = B.ensure_on_gpu(mgr)
    n = int(np.prod(A.shape))
    pool = _get_pool()
    t_c = pool.acquire(n) if pool else mgr.tensor(np.zeros(n, dtype=np.float32))
    binary_gpu(mgr, t_a, t_b, t_c, n, 'sub')

    out = Tensor(np.empty(A.shape, dtype=np.float32))
    out.mark_gpu_fresh(t_c, A.shape, mgr)
    out._prev = {A, B}

    def _backward():
        n = int(np.prod(A.shape))
        if A.requires_grad:
            A._ensure_grad_on_gpu(mgr)
            accumulate_gpu(mgr, A._grad_kp, out._grad_kp, n, 1.0)
        if B.requires_grad:
            B._ensure_grad_on_gpu(mgr)
            accumulate_gpu(mgr, B._grad_kp, out._grad_kp, n, -1.0)

    out._backward = _backward
    return out


def mul(mgr: kp.Manager, A: Tensor, B: Tensor) -> Tensor:
    t_a = A.ensure_on_gpu(mgr)
    t_b = B.ensure_on_gpu(mgr)
    n = int(np.prod(A.shape))
    pool = _get_pool()
    t_c = pool.acquire(n) if pool else mgr.tensor(np.zeros(n, dtype=np.float32))
    binary_gpu(mgr, t_a, t_b, t_c, n, 'mul')

    out = Tensor(np.empty(A.shape, dtype=np.float32))
    out.mark_gpu_fresh(t_c, A.shape, mgr)
    out._prev = {A, B}

    def _backward():
        pool = _get_pool()
        n = int(np.prod(A.shape))
        t_ga = pool.acquire(n) if pool else mgr.tensor(np.zeros(n, dtype=np.float32))
        t_gb = pool.acquire(n) if pool else mgr.tensor(np.zeros(n, dtype=np.float32))
        if A.requires_grad:
            t_b = B.ensure_on_gpu(mgr)
            binary_gpu(mgr, out._grad_kp, t_b, t_ga, n, 'mul')
            A._ensure_grad_on_gpu(mgr)
            accumulate_gpu(mgr, A._grad_kp, t_ga, n, 1.0)
        if B.requires_grad:
            t_a = A.ensure_on_gpu(mgr)
            binary_gpu(mgr, out._grad_kp, t_a, t_gb, n, 'mul')
            B._ensure_grad_on_gpu(mgr)
            accumulate_gpu(mgr, B._grad_kp, t_gb, n, 1.0)
        if pool:
            pool.release(t_ga)
            pool.release(t_gb)

    out._backward = _backward
    return out


def div(mgr: kp.Manager, A: Tensor, B: Tensor) -> Tensor:
    t_a = A.ensure_on_gpu(mgr)
    t_b = B.ensure_on_gpu(mgr)
    n = int(np.prod(A.shape))
    pool = _get_pool()
    t_c = pool.acquire(n) if pool else mgr.tensor(np.zeros(n, dtype=np.float32))
    binary_gpu(mgr, t_a, t_b, t_c, n, 'div')

    out = Tensor(np.empty(A.shape, dtype=np.float32))
    out.mark_gpu_fresh(t_c, A.shape, mgr)
    out._prev = {A, B}

    def _backward():
        pool = _get_pool()
        n = int(np.prod(A.shape))
        t_ga = pool.acquire(n) if pool else mgr.tensor(np.zeros(n, dtype=np.float32))
        t_gb = pool.acquire(n) if pool else mgr.tensor(np.zeros(n, dtype=np.float32))
        if A.requires_grad:
            t_b = B.ensure_on_gpu(mgr)
            binary_gpu(mgr, out._grad_kp, t_b, t_ga, n, 'div')
            A._ensure_grad_on_gpu(mgr)
            accumulate_gpu(mgr, A._grad_kp, t_ga, n, 1.0)
            if pool:
                b_sq = pool.acquire(n)
                t_b2 = B.ensure_on_gpu(mgr)
                binary_gpu(mgr, t_b2, t_b2, b_sq, n, 'mul')
                a_dv = pool.acquire(n)
                t_a2 = A.ensure_on_gpu(mgr)
                binary_gpu(mgr, t_a2, b_sq, a_dv, n, 'div')
                binary_gpu(mgr, out._grad_kp, a_dv, t_gb, n, 'mul')
                B._ensure_grad_on_gpu(mgr)
                accumulate_gpu(mgr, B._grad_kp, t_gb, n, -1.0)
                pool.release(b_sq)
                pool.release(a_dv)
        if pool:
            pool.release(t_ga)
            pool.release(t_gb)

    out._backward = _backward
    return out


def _get_pool():
    from vtrain.gpu_pool import get_pool
    return get_pool()


# ── Layer norm ────────────────────────────────────────────────────────────────

def layernorm(mgr: kp.Manager,
              X:     Tensor,
              gamma: Tensor,
              beta:  Tensor,
              eps:   float = 1e-5) -> Tensor:

    x_data = X.data
    rows, cols = x_data.shape

    mean    = x_data.mean(axis=-1, keepdims=True)
    var     = x_data.var(axis=-1,  keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_norm  = (x_data - mean) * inv_std

    out = Tensor(_gpu_layernorm(mgr, x_data, gamma.data, beta.data, eps))
    out._prev = {X, gamma, beta}

    def _backward():
        dy = out.grad

        if gamma.requires_grad:
            gamma._ensure_grad_on_gpu(mgr)
            grad_gamma = (dy * x_norm).sum(axis=0).astype(np.float32)
            gamma._grad_kp.data()[:] = (gamma._grad_kp.data()[:].reshape(gamma.shape) + grad_gamma).flatten()
            sq = gamma._mgr.sequence()
            sq.record(kp.OpSyncDevice([gamma._grad_kp]))
            sq.eval()
        if beta.requires_grad:
            beta._ensure_grad_on_gpu(mgr)
            grad_beta = dy.sum(axis=0).astype(np.float32)
            beta._grad_kp.data()[:] = (beta._grad_kp.data()[:].reshape(beta.shape) + grad_beta).flatten()
            sq = beta._mgr.sequence()
            sq.record(kp.OpSyncDevice([beta._grad_kp]))
            sq.eval()

        if X.requires_grad:
            X._ensure_grad_on_gpu(mgr)
            g = gamma.data
            N = cols

            dy_scaled = dy * g * inv_std
            x_grad = (
                dy_scaled
                - dy_scaled.mean(axis=-1, keepdims=True)
                - x_norm * (dy_scaled * x_norm).mean(axis=-1, keepdims=True)
            ).astype(np.float32)
            X._grad_kp.data()[:] = (X._grad_kp.data()[:].reshape(X.shape) + x_grad).flatten()
            sq = X._mgr.sequence()
            sq.record(kp.OpSyncDevice([X._grad_kp]))
            sq.eval()

    out._backward = _backward
    return out


# ── Softmax ───────────────────────────────────────────────────────────────────

def softmax(mgr: kp.Manager, X: Tensor) -> Tensor:
    from vtrain.ops.softmax import softmax_gpu as _softmax_gpu
    from vtrain.ops.softmax import softmax_backward_gpu as _softmax_backward_gpu
    pool = _get_pool()
    rows, cols = X.shape
    n = rows * cols

    t_x = X.ensure_on_gpu(mgr)
    t_s = pool.acquire(n) if pool else mgr.tensor(np.zeros(n, dtype=np.float32))
    _softmax_gpu(mgr, t_x, t_s, rows, cols)

    out = Tensor(np.empty(X.shape, dtype=np.float32))
    out.mark_gpu_fresh(t_s, X.shape, mgr)
    out._prev = {X}

    def _backward():
        if X.requires_grad:
            X._ensure_grad_on_gpu(mgr)
            _softmax_backward_gpu(mgr, out._grad_kp, t_s, X._grad_kp, rows, cols)

    out._backward = _backward
    return out


# ── Transpose ─────────────────────────────────────────────────────────────────

def transpose(mgr: kp.Manager, X: Tensor) -> Tensor:
    from vtrain.ops.transpose import transpose_gpu as _transpose_resident
    pool = _get_pool()
    rows, cols = X.shape
    n = rows * cols

    t_x = X.ensure_on_gpu(mgr)
    t_y = pool.acquire(n) if pool else mgr.tensor(np.zeros(n, dtype=np.float32))
    _transpose_resident(mgr, t_x, t_y, rows, cols)

    out = Tensor(np.empty(X.shape if rows == cols else (cols, rows), dtype=np.float32))
    out.mark_gpu_fresh(t_y, X.shape if rows == cols else (cols, rows), mgr)
    out._prev = {X}

    def _backward():
        if X.requires_grad:
            pool = _get_pool()
            rows, cols = X.shape
            n = rows * cols
            t_temp = pool.acquire(n) if pool else mgr.tensor(np.zeros(n, dtype=np.float32))
            _transpose_resident(mgr, out._grad_kp, t_temp, cols, rows)
            X._ensure_grad_on_gpu(mgr)
            accumulate_gpu(mgr, X._grad_kp, t_temp, n, 1.0)
            if pool:
                pool.release(t_temp)

    out._backward = _backward
    return out


# ── Differentiable attention ───────────────────────────────────────────────────

def attention(mgr: kp.Manager, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
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