import kp
import numpy as np
import math
from vtrain.shader_utils import compile_shader


def transpose(mgr: kp.Manager, X: np.ndarray) -> np.ndarray:
    """
    Transpose a 2D matrix on the GPU.
    X shape: (rows, cols)
    Returns: (cols, rows)
    """
    assert X.ndim == 2, "Input must be 2D"
    rows, cols = X.shape

    x_flat = X.flatten().astype(np.float32)
    y_flat = np.zeros(cols * rows, dtype=np.float32)

    t_x = mgr.tensor(x_flat)
    t_y = mgr.tensor(y_flat)

    spirv = compile_shader("transpose").read_bytes()

    wg_x = math.ceil(rows / 16)
    wg_y = math.ceil(cols / 16)

    algo = mgr.algorithm(
        [t_x, t_y],
        spirv,
        (wg_x, wg_y, 1),
        [],
        [float(rows), float(cols), float(0), float(0)]
    )

    sq = mgr.sequence()
    sq.record(kp.OpSyncDevice([t_x, t_y]))
    sq.record(kp.OpAlgoDispatch(algo))
    sq.record(kp.OpSyncLocal([t_y]))
    sq.eval()

    return t_y.data().reshape(cols, rows)
