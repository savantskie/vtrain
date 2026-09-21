import numpy as np
import kp
from vtrain.tensor import Tensor
from vtrain.model.linear import Linear, FeedForward
import vtrain.functional as F


class TransformerBlock:
    """
    One complete transformer layer (pre-norm style):
      X → LayerNorm → MultiHeadAttention → + residual
        → LayerNorm → FeedForward        → + residual
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int = None):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

        # Attention projections — no bias, standard transformer practice
        self.W_q = Linear(d_model, d_model, bias=False)
        self.W_k = Linear(d_model, d_model, bias=False)
        self.W_v = Linear(d_model, d_model, bias=False)
        self.W_o = Linear(d_model, d_model, bias=False)

        # Layer norm parameters — one pair per sub-layer
        self.gamma1 = Tensor(np.ones(d_model,  dtype=np.float32), name='ln1_gamma')
        self.beta1  = Tensor(np.zeros(d_model, dtype=np.float32), name='ln1_beta')
        self.gamma2 = Tensor(np.ones(d_model,  dtype=np.float32), name='ln2_gamma')
        self.beta2  = Tensor(np.zeros(d_model, dtype=np.float32), name='ln2_beta')

        self.ff = FeedForward(d_model, d_ff)

    def forward(self, mgr, X: Tensor) -> Tensor:
        # ── Attention sub-layer ──────────────────────────────────────
        normed = F.layernorm(mgr, X, self.gamma1, self.beta1)

        Q = self.W_q.forward(mgr, normed)
        K = self.W_k.forward(mgr, normed)
        V = self.W_v.forward(mgr, normed)

        # Split into heads, run attention per head, concat results
        head_outs = []
        for h in range(self.n_heads):
            s, e = h * self.d_k, (h + 1) * self.d_k

            Q_h = Tensor(Q.data[:, s:e].copy(), requires_grad=Q.requires_grad)
            K_h = Tensor(K.data[:, s:e].copy(), requires_grad=K.requires_grad)
            V_h = Tensor(V.data[:, s:e].copy(), requires_grad=V.requires_grad)

            def _make_slice_bwd(src, dst, s, e, mgr=mgr):
                def _bwd():
                    if not src.requires_grad:
                        return
                    src._ensure_grad_on_gpu(mgr)
                    dst._ensure_grad_on_gpu(mgr)
                    sq = mgr.sequence()
                    sq.record(kp.OpSyncLocal([src._grad_kp, dst._grad_kp]))
                    sq.eval()
                    src_np = src._grad_kp.data().reshape(src.shape)
                    dst_np = dst._grad_kp.data().reshape(dst.shape)
                    src_np[:, s:e] += dst_np
                    src._grad_kp.data()[:] = src_np.flatten()
                    sq2 = mgr.sequence()
                    sq2.record(kp.OpSyncDevice([src._grad_kp]))
                    sq2.eval()
                return _bwd

            Q_h._prev = {Q}; Q_h._backward = _make_slice_bwd(Q, Q_h, s, e)
            K_h._prev = {K}; K_h._backward = _make_slice_bwd(K, K_h, s, e)
            V_h._prev = {V}; V_h._backward = _make_slice_bwd(V, V_h, s, e)

            head_outs.append(F.attention(mgr, Q_h, K_h, V_h))

        # Concat heads: list of (seq_len, d_k) → (seq_len, d_model)
        concat      = Tensor(np.concatenate([h.data for h in head_outs], axis=-1))
        concat._prev = set(head_outs)

        def _concat_backward():
            sq = mgr.sequence()
            sq.record(kp.OpSyncLocal([concat._grad_kp]))
            sq.eval()
            concat_np = concat._grad_kp.data().reshape(concat.shape)
            for i, h_out in enumerate(head_outs):
                if h_out.requires_grad:
                    h_out._ensure_grad_on_gpu(mgr)
                    sq2 = mgr.sequence()
                    sq2.record(kp.OpSyncLocal([h_out._grad_kp]))
                    sq2.eval()
                    h_np = h_out._grad_kp.data().reshape(h_out.shape)
                    h_np += concat_np[:, i*self.d_k:(i+1)*self.d_k]
                    h_out._grad_kp.data()[:] = h_np.flatten()
                    sq3 = mgr.sequence()
                    sq3.record(kp.OpSyncDevice([h_out._grad_kp]))
                    sq3.eval()

        concat._backward = _concat_backward

        attn_out  = self.W_o.forward(mgr, concat)
        residual1 = F.add(mgr, X, attn_out)

        # ── Feed-forward sub-layer ───────────────────────────────────
        normed2  = F.layernorm(mgr, residual1, self.gamma2, self.beta2)
        ff_out   = self.ff.forward(mgr, normed2)
        output   = F.add(mgr, residual1, ff_out)

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
