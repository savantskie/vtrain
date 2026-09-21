import numpy as np
import kp
import math
from vtrain.tensor import Tensor


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    diff = pred.data - target.data
    loss = Tensor(np.array(np.mean(diff ** 2), dtype=np.float32),
                  requires_grad=True)
    loss._prev = {pred}

    def _backward():
        if pred.requires_grad:
            N = pred.data.size
            pred.grad += (2.0 * diff / N) * loss.grad

    loss._backward = _backward
    return loss


def cross_entropy_loss_cpu(pred: Tensor, target: Tensor) -> Tensor:
    """CPU cross-entropy — kept for tests and CPU fallback."""
    eps     = 1e-7
    p       = np.clip(pred.data, eps, 1.0)
    batch   = pred.data.shape[0]
    loss_val = -np.mean(np.sum(target.data * np.log(p), axis=-1))
    loss     = Tensor(np.array(loss_val, dtype=np.float32),
                      requires_grad=True)
    loss._prev = {pred}

    def _backward():
        if pred.requires_grad:
            pred.grad += (-target.data / p / batch) * loss.grad

    loss._backward = _backward
    return loss


def cross_entropy_loss_gpu(mgr: kp.Manager, pred: Tensor, target: Tensor) -> Tensor:
    """GPU cross-entropy — uses loss_ce shader. Gradient stays on GPU."""
    from vtrain.gpu_pool import get_pool
    from vtrain.shader_utils import compile_shader
    pool = get_pool()
    batch, n_classes = pred.shape
    n = batch * n_classes

    t_grad = pool.acquire(n) if pool else mgr.tensor(np.zeros(n, dtype=np.float32))
    if pool:
        pool.register_persistent(t_grad)
    t_loss_arr = mgr.tensor(np.zeros(batch, dtype=np.float32))

    t_pred = pred.ensure_on_gpu(mgr)
    t_target = target.ensure_on_gpu(mgr)

    spirv = compile_shader("loss_ce").read_bytes()
    algo = mgr.algorithm(
        [t_pred, t_target, t_grad, t_loss_arr],
        spirv, (batch, 1, 1), [],
        [float(n), float(n_classes), float(1e-7)]
    )
    sq = mgr.sequence()
    sq.record(kp.OpSyncDevice([t_loss_arr]))
    sq.record(kp.OpAlgoDispatch(algo))
    sq.eval()

    sq2 = mgr.sequence()
    sq2.record(kp.OpSyncLocal([t_loss_arr]))
    sq2.eval()
    loss_val = float(np.mean(t_loss_arr.data()[:batch]))

    out = Tensor(np.array(loss_val, dtype=np.float32), requires_grad=True)
    out._mgr = mgr
    out._prev = {pred}

    pred._grad_kp = t_grad
    if pool:
        pool.register_persistent(t_grad)
    pred._grad_data = None
    pred._mgr = mgr

    def _backward():
        pass

    out._backward = _backward
    return out


cross_entropy_loss = cross_entropy_loss_gpu


def one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(labels), n_classes), dtype=np.float32)
    out[np.arange(len(labels)), labels] = 1.0
    return out