import numpy as np
import kp
import math
from vtrain.tensor import Tensor


class SGD:
    """
    Stochastic Gradient Descent — GPU-resident step.

    weight = weight - lr * gradient
    All operations dispatched as GPU shaders, no CPU numpy.
    """

    def __init__(self, params: list, lr: float = 0.01):
        self.params = params
        self.lr     = lr

    def state_dict(self) -> dict:
        return {"lr": self.lr}

    def load_state_dict(self, state: dict):
        self.lr = state["lr"]

    def step(self):
        from vtrain.shader_utils import compile_shader
        mgr = getattr(self, '_mgr', None)
        if mgr is None:
            for p in self.params:
                if p._mgr is not None:
                    mgr = p._mgr
                    break
        if mgr is None:
            return

        spirv = compile_shader("sgd_step").read_bytes()
        for p in self.params:
            if not p.requires_grad or p._grad_kp is None:
                continue
            flat_size = int(np.prod(p.shape))
            wg_x = math.ceil(flat_size / 256)
            algo = mgr.algorithm(
                [p._kp_tensor, p._grad_kp],
                spirv, (wg_x, 1, 1), [],
                [float(flat_size), float(self.lr)]
            )
            sq = mgr.sequence()
            sq.record(kp.OpAlgoDispatch(algo))
            sq.eval()

    def zero_grad(self):
        for p in self.params:
            if p._grad_kp is not None and p._mgr is not None:
                p._grad_kp.data()[:] = 0.0
                sq = p._mgr.sequence()
                sq.record(kp.OpSyncDevice([p._grad_kp]))
                sq.eval()


class Adam:
    """
    Adam optimizer — Adaptive Moment Estimation.
    Momentum buffers (m, v) are GPU-resident kp.Tensor objects.
    Step dispatches a GPU shader per parameter.
    """

    def __init__(self, params: list, lr: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8):
        self.params = params
        self.lr     = lr
        self.beta1  = beta1
        self.beta2  = beta2
        self.eps    = eps
        self.t      = 0

        mgr = None
        for p in params:
            if p._mgr is not None:
                mgr = p._mgr
                break

        from vtrain.gpu_pool import get_pool
        pool = get_pool()

        self.m = []
        self.v = []
        if mgr is not None and pool is not None:
            for p in params:
                flat_size = int(np.prod(p.shape))
                m_buf = pool.acquire(flat_size)
                v_buf = pool.acquire(flat_size)
                pool.register_persistent(m_buf)
                pool.register_persistent(v_buf)
                m_buf.data()[:] = 0.0
                v_buf.data()[:] = 0.0
                sq = mgr.sequence()
                sq.record(kp.OpSyncDevice([m_buf, v_buf]))
                sq.eval()
                self.m.append(m_buf)
                self.v.append(v_buf)
        else:
            for p in params:
                self.m.append(np.zeros_like(p.data))
                self.v.append(np.zeros_like(p.data))

    def state_dict(self) -> dict:
        m_cpu = []
        v_cpu = []
        mgr = getattr(self, '_mgr', None)
        if mgr is not None:
            for mbuf, vbuf in zip(self.m, self.v):
                sq = mgr.sequence()
                sq.record(kp.OpSyncLocal([mbuf, vbuf]))
                sq.eval()
                m_cpu.append(mbuf.data().copy())
                v_cpu.append(vbuf.data().copy())
        else:
            m_cpu = [arr.copy() for arr in self.m]
            v_cpu = [arr.copy() for arr in self.v]
        return {
            "m":     m_cpu,
            "v":     v_cpu,
            "t":     self.t,
            "lr":    self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps":   self.eps,
        }

    def load_state_dict(self, state: dict):
        mgr = getattr(self, '_mgr', None)
        if mgr is None:
            for p in self.params:
                if p._mgr is not None:
                    mgr = p._mgr
                    self._mgr = mgr
                    break
        if mgr is not None:
            for i in range(len(self.m)):
                self.m[i].data()[:] = state["m"][i].flatten()
                self.v[i].data()[:] = state["v"][i].flatten()
            sq = mgr.sequence()
            sq.record(kp.OpSyncDevice(self.m + self.v))
            sq.eval()
        else:
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
        b1_corr = 1.0 / (1.0 - self.beta1 ** self.t)
        b2_corr = 1.0 / (1.0 - self.beta2 ** self.t)

        from vtrain.shader_utils import compile_shader
        mgr = getattr(self, '_mgr', None)
        if mgr is None:
            for p in self.params:
                if p._mgr is not None:
                    mgr = p._mgr
                    self._mgr = mgr
                    break
        if mgr is None:
            return

        spirv = compile_shader("adam_step").read_bytes()
        for i, p in enumerate(self.params):
            if not p.requires_grad or p._grad_kp is None:
                continue
            flat_size = int(np.prod(p.shape))
            wg_x = math.ceil(flat_size / 256)
            algo = mgr.algorithm(
                [p._kp_tensor, p._grad_kp, self.m[i], self.v[i]],
                spirv, (wg_x, 1, 1), [],
                [float(flat_size), float(self.lr), float(self.beta1),
                 float(self.beta2), float(self.eps), float(self.t),
                 float(b1_corr), float(b2_corr)]
            )
            sq = mgr.sequence()
            sq.record(kp.OpAlgoDispatch(algo))
            sq.eval()

    def zero_grad(self):
        for p in self.params:
            if p._grad_kp is not None and p._mgr is not None:
                p._grad_kp.data()[:] = 0.0
                sq = p._mgr.sequence()
                sq.record(kp.OpSyncDevice([p._grad_kp]))
                sq.eval()