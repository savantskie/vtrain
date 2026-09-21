import numpy as np
import kp
from vtrain.tensor import Tensor
from vtrain.model.linear import Linear, FeedForward
import vtrain.functional as F


class TransformerBlock:
    def __init__(self, d_model: int, n_heads: int, d_ff: int = None):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

        self.W_q = Linear(d_model, d_model, bias=False)
        self.W_k = Linear(d_model, d_model, bias=False)
        self.W_v = Linear(d_model, d_model, bias=False)
        self.W_o = Linear(d_model, d_model, bias=False)

        self.gamma1 = Tensor(np.ones(d_model,  dtype=np.float32), name='ln1_gamma')
        self.beta1  = Tensor(np.zeros(d_model, dtype=np.float32), name='ln1_beta')
        self.gamma2 = Tensor(np.ones(d_model,  dtype=np.float32), name='ln2_gamma')
        self.beta2  = Tensor(np.zeros(d_model, dtype=np.float32), name='ln2_beta')

        self.ff = FeedForward(d_model, d_ff)

    def forward(self, mgr, X: Tensor) -> Tensor:
        from vtrain.gpu_pool import get_pool
        from vtrain.ops.attention import (
            split_heads_gpu, merge_heads_gpu, copy_block_gpu, scatter_head_gpu
        )

        pool       = get_pool()
        B          = X.shape[0]
        head_size  = B * self.d_k
        split_size = self.n_heads * head_size

        normed = F.layernorm(mgr, X, self.gamma1, self.beta1)

        Q = self.W_q.forward(mgr, normed)
        K = self.W_k.forward(mgr, normed)
        V = self.W_v.forward(mgr, normed)

        t_Q_split = pool.acquire(split_size) if pool else mgr.tensor(np.zeros(split_size, dtype=np.float32))
        t_K_split = pool.acquire(split_size) if pool else mgr.tensor(np.zeros(split_size, dtype=np.float32))
        t_V_split = pool.acquire(split_size) if pool else mgr.tensor(np.zeros(split_size, dtype=np.float32))

        split_heads_gpu(mgr, Q.ensure_on_gpu(mgr), t_Q_split, B, self.d_model, self.n_heads, self.d_k)
        split_heads_gpu(mgr, K.ensure_on_gpu(mgr), t_K_split, B, self.d_model, self.n_heads, self.d_k)
        split_heads_gpu(mgr, V.ensure_on_gpu(mgr), t_V_split, B, self.d_model, self.n_heads, self.d_k)

        head_outs = []
        for h in range(self.n_heads):
            offset = h * head_size

            t_Qh = pool.acquire(head_size) if pool else mgr.tensor(np.zeros(head_size, dtype=np.float32))
            t_Kh = pool.acquire(head_size) if pool else mgr.tensor(np.zeros(head_size, dtype=np.float32))
            t_Vh = pool.acquire(head_size) if pool else mgr.tensor(np.zeros(head_size, dtype=np.float32))

            copy_block_gpu(mgr, t_Q_split, t_Qh, head_size, offset, 0, False)
            copy_block_gpu(mgr, t_K_split, t_Kh, head_size, offset, 0, False)
            copy_block_gpu(mgr, t_V_split, t_Vh, head_size, offset, 0, False)

            Q_h = Tensor(np.empty((B, self.d_k), dtype=np.float32), requires_grad=Q.requires_grad)
            K_h = Tensor(np.empty((B, self.d_k), dtype=np.float32), requires_grad=K.requires_grad)
            V_h = Tensor(np.empty((B, self.d_k), dtype=np.float32), requires_grad=V.requires_grad)
            Q_h.mark_gpu_fresh(t_Qh, (B, self.d_k), mgr)
            K_h.mark_gpu_fresh(t_Kh, (B, self.d_k), mgr)
            V_h.mark_gpu_fresh(t_Vh, (B, self.d_k), mgr)

            def _make_qkv_bwd(src, head_t, h_idx,
                               B=B, d_model=self.d_model, d_k=self.d_k, mgr=mgr):
                def _bwd():
                    if not src.requires_grad:
                        return
                    src._ensure_grad_on_gpu(mgr)
                    scatter_head_gpu(mgr, head_t._grad_kp, src._grad_kp,
                                     B, d_model, h_idx, d_k)
                return _bwd

            Q_h._prev = {Q}; Q_h._backward = _make_qkv_bwd(Q, Q_h, h)
            K_h._prev = {K}; K_h._backward = _make_qkv_bwd(K, K_h, h)
            V_h._prev = {V}; V_h._backward = _make_qkv_bwd(V, V_h, h)

            head_outs.append(F.attention(mgr, Q_h, K_h, V_h))

        t_head_split = pool.acquire(split_size) if pool else mgr.tensor(np.zeros(split_size, dtype=np.float32))
        for h, h_out in enumerate(head_outs):
            copy_block_gpu(mgr, h_out._kp_tensor, t_head_split,
                           head_size, 0, h * head_size, False)

        t_concat = pool.acquire(B * self.d_model) if pool else mgr.tensor(np.zeros(B * self.d_model, dtype=np.float32))
        merge_heads_gpu(mgr, t_head_split, t_concat, B, self.d_model, self.n_heads, self.d_k)

        concat = Tensor(np.empty((B, self.d_model), dtype=np.float32))
        concat.mark_gpu_fresh(t_concat, (B, self.d_model), mgr)
        concat._prev = set(head_outs)

        def _concat_backward():
            t_grad_split = pool.acquire(split_size) if pool else mgr.tensor(np.zeros(split_size, dtype=np.float32))
            split_heads_gpu(mgr, concat._grad_kp, t_grad_split,
                            B, self.d_model, self.n_heads, self.d_k)
            for h, h_out in enumerate(head_outs):
                if h_out.requires_grad:
                    h_out._ensure_grad_on_gpu(mgr)
                    copy_block_gpu(mgr, t_grad_split, h_out._grad_kp,
                                   head_size, h * head_size, 0, True)
            if pool:
                pool.release(t_grad_split)
                pool.release(t_Q_split)
                pool.release(t_K_split)
                pool.release(t_V_split)
                pool.release(t_head_split)

        concat._backward = _concat_backward

        attn_out  = self.W_o.forward(mgr, concat)
        residual1 = F.add(mgr, X, attn_out)

        normed2 = F.layernorm(mgr, residual1, self.gamma2, self.beta2)
        ff_out  = self.ff.forward(mgr, normed2)
        output  = F.add(mgr, residual1, ff_out)

        return output

    def parameters(self):
        return (
            self.W_q.parameters() + self.W_k.parameters() +
            self.W_v.parameters() + self.W_o.parameters() +
            [self.gamma1, self.beta1, self.gamma2, self.beta2] +
            self.ff.parameters()
        )

    def __call__(self, mgr, X):
        return self.forward(mgr, X)