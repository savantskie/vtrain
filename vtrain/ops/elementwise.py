import kp
import numpy as np
import math
from vtrain.shader_utils import compile_shader

# Match the integers in the .comp shaders
_UNARY_OPS  = {"relu": 0, "sigmoid": 1, "tanh": 2, "gelu": 3}
_BINARY_OPS = {"add":  0, "sub":    1, "mul":  2, "div":  3}


def _unary(mgr: kp.Manager, X: np.ndarray, op: str) -> np.ndarray:
    original_shape = X.shape
    x_flat = X.flatten().astype(np.float32)
    y_flat = np.zeros_like(x_flat)
    n = len(x_flat)

    t_x = mgr.tensor(x_flat)
    t_y = mgr.tensor(y_flat)

    spirv = compile_shader("unary").read_bytes()
    wg_x  = math.ceil(n / 256)

    algo = mgr.algorithm(
        [t_x, t_y],
        spirv,
        (wg_x, 1, 1),
        [],
        [float(n), float(_UNARY_OPS[op])]
    )

    sq = mgr.sequence()
    sq.record(kp.OpSyncDevice([t_x, t_y]))
    sq.record(kp.OpAlgoDispatch(algo))
    sq.record(kp.OpSyncLocal([t_y]))
    sq.eval()

    return t_y.data().reshape(original_shape)


def _binary(mgr: kp.Manager, A: np.ndarray, B: np.ndarray, op: str) -> np.ndarray:
    assert A.shape == B.shape, f"Shape mismatch: {A.shape} vs {B.shape}"
    original_shape = A.shape

    a_flat = A.flatten().astype(np.float32)
    b_flat = B.flatten().astype(np.float32)
    c_flat = np.zeros_like(a_flat)
    n = len(a_flat)

    t_a = mgr.tensor(a_flat)
    t_b = mgr.tensor(b_flat)
    t_c = mgr.tensor(c_flat)

    spirv = compile_shader("binary").read_bytes()
    wg_x  = math.ceil(n / 256)

    algo = mgr.algorithm(
        [t_a, t_b, t_c],
        spirv,
        (wg_x, 1, 1),
        [],
        [float(n), float(_BINARY_OPS[op]), float(0)]
    )

    sq = mgr.sequence()
    sq.record(kp.OpSyncDevice([t_a, t_b, t_c]))
    sq.record(kp.OpAlgoDispatch(algo))
    sq.record(kp.OpSyncLocal([t_c]))
    sq.eval()

    return t_c.data().reshape(original_shape)


# Public API
def relu(mgr, X):     return _unary(mgr, X, "relu")
def sigmoid(mgr, X):  return _unary(mgr, X, "sigmoid")
def tanh_act(mgr, X): return _unary(mgr, X, "tanh")    # named to avoid shadowing math.tanh
def gelu(mgr, X):     return _unary(mgr, X, "gelu")

def add(mgr, A, B): return _binary(mgr, A, B, "add")
def sub(mgr, A, B): return _binary(mgr, A, B, "sub")
def mul(mgr, A, B): return _binary(mgr, A, B, "mul")
def div(mgr, A, B): return _binary(mgr, A, B, "div")
