import kp
import numpy as np
import math
from vtrain.shader_utils import compile_shader


def softmax(mgr: kp.Manager, X: np.ndarray) -> np.ndarray:
    """
    Softmax over the last dimension.
    X shape: (rows, cols)
    Output: same shape, each row sums to 1.0
    """
    assert X.ndim == 2, "Input must be 2D (rows, cols)"
    rows, cols = X.shape

    x_flat = X.flatten().astype(np.float32)
    y_flat = np.zeros_like(x_flat)

    t_x = mgr.tensor(x_flat)
    t_y = mgr.tensor(y_flat)

    spirv = compile_shader("softmax").read_bytes()

    algo = mgr.algorithm(
        [t_x, t_y],
        spirv,
        (rows, 1, 1),
        [],
        [float(rows), float(cols), float(0), float(0)]
    )

    sq = mgr.sequence()
    sq.record(kp.OpSyncDevice([t_x, t_y]))
    sq.record(kp.OpAlgoDispatch(algo))
    sq.record(kp.OpSyncLocal([t_y]))
    sq.eval()

    return t_y.data().reshape(rows, cols)


# GPU-resident wrappers

def softmax_gpu(mgr, t_x, t_y, rows: int, cols: int) -> None:
    spirv = compile_shader("softmax").read_bytes()
    algo = mgr.algorithm(
        [t_x, t_y], spirv, (rows, 1, 1), [],
        [float(rows), float(cols), float(0), float(0)]
    )
    sq = mgr.sequence()
    sq.record(kp.OpAlgoDispatch(algo))
    sq.eval()


def softmax_backward_gpu(mgr, t_dy, t_s, t_dx, rows: int, cols: int) -> None:
    spirv = compile_shader("softmax_backward").read_bytes()
    algo = mgr.algorithm(
        [t_dy, t_s, t_dx], spirv, (rows, 1, 1), [],
        [float(rows), float(cols)]
    )
    sq = mgr.sequence()
    sq.record(kp.OpAlgoDispatch(algo))
    sq.eval()
