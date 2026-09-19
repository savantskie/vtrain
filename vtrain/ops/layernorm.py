import kp
import numpy as np
import math
from vtrain.shader_utils import compile_shader


def layernorm(mgr: kp.Manager,
              X: np.ndarray,
              gamma: np.ndarray = None,
              beta: np.ndarray  = None,
              eps: float        = 1e-5) -> np.ndarray:
    """
    Layer normalization over the last dimension.
    X shape:     (rows, cols)
    gamma shape: (cols,)  — scale, defaults to all ones
    beta shape:  (cols,)  — shift, defaults to all zeros
    """
    assert X.ndim == 2, "Input must be 2D (rows, cols)"
    rows, cols = X.shape

    # Default gamma=1, beta=0 means "normalize but don't rescale yet"
    if gamma is None:
        gamma = np.ones(cols,  dtype=np.float32)
    if beta is None:
        beta  = np.zeros(cols, dtype=np.float32)

    x_flat     = X.flatten().astype(np.float32)
    y_flat     = np.zeros_like(x_flat)
    gamma_flat = gamma.flatten().astype(np.float32)
    beta_flat  = beta.flatten().astype(np.float32)

    t_x     = mgr.tensor(x_flat)
    t_y     = mgr.tensor(y_flat)
    t_gamma = mgr.tensor(gamma_flat)
    t_beta  = mgr.tensor(beta_flat)

    spirv = compile_shader("layernorm").read_bytes()

    # One workgroup per row — threads within that workgroup
    # cooperate to normalize that row together
    algo = mgr.algorithm(
        [t_x, t_y, t_gamma, t_beta],
        spirv,
        (rows, 1, 1),
        [],
        [float(rows), float(cols), float(eps), float(0)]
    )

    sq = mgr.sequence()
    sq.record(kp.OpSyncDevice([t_x, t_y, t_gamma, t_beta]))
    sq.record(kp.OpAlgoDispatch(algo))
    sq.record(kp.OpSyncLocal([t_y]))
    sq.eval()

    return t_y.data().reshape(rows, cols)
