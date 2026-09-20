import kp
import numpy as np
import math
from vtrain.shader_utils import compile_shader

def matmul(mgr: kp.Manager, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    GPU matrix multiply: A @ B
    A shape: (M, K)
    B shape: (K, N)
    Returns C shape: (M, N)
    """
    assert A.ndim == 2 and B.ndim == 2, "Both inputs must be 2D"
    assert A.shape[1] == B.shape[0], (
        f"Inner dimensions must match: A is {A.shape}, B is {B.shape}"
    )

    M, K = A.shape
    _,  N = B.shape

    # GPU only speaks float32 — cast if needed
    a_flat = A.flatten().astype(np.float32)
    b_flat = B.flatten().astype(np.float32)
    c_flat = np.zeros(M * N, dtype=np.float32)

    # Hand the data to Kompute — it allocates matching buffers on GPU
    t_a = mgr.tensor(a_flat)
    t_b = mgr.tensor(b_flat)
    t_c = mgr.tensor(c_flat)

    # Load SPIR-V (auto-recompiles from source if stale)
    spirv = compile_shader("matmul").read_bytes()

    # Workgroup count: our shader uses 16x16 thread blocks.
    # We need enough blocks to cover every row and column.
    # ceil(M/16) blocks in x covers all rows,
    # ceil(N/16) blocks in y covers all columns.
    # Extra threads that fall outside the matrix bail out via the
    # bounds check in the shader.
    wg_x = math.ceil(M / 16)
    wg_y = math.ceil(N / 16)

    algo = mgr.algorithm(
        [t_a, t_b, t_c],
        spirv,
        (wg_x, wg_y, 1),
        [],                          # specialization constants — none yet
        [float(M), float(K), float(N)]  # push constants — matrix dimensions
    )

    # Sequence = ordered list of GPU commands executed as one batch
    sq = mgr.sequence()
    sq.record(kp.OpSyncDevice([t_a, t_b, t_c]))  # push data to GPU
    sq.record(kp.OpAlgoDispatch(algo))            # run the shader
    sq.record(kp.OpSyncLocal([t_c]))              # pull result back
    sq.eval()                                            # execute everything

    return t_c.data().reshape(M, N)
    
def matmul_gpu(mgr: kp.Manager, t_a: kp.Tensor, t_b: kp.Tensor,
               t_c: kp.Tensor, M: int, K: int, N: int) -> None:
    """
    GPU-resident matmul: runs shader directly on pre-uploaded kp.Tensors.
    No CPU<->GPU transfer — inputs and output all stay on GPU.
    t_c receives the result.
    """
    spirv = compile_shader("matmul").read_bytes()
    wg_x  = math.ceil(M / 16)
    wg_y  = math.ceil(N / 16)

    algo = mgr.algorithm(
        [t_a, t_b, t_c],
        spirv,
        (wg_x, wg_y, 1),
        [],
        [float(M), float(K), float(N)]
    )

    sq = mgr.sequence()
    sq.record(kp.OpAlgoDispatch(algo))
    sq.eval()
    # No OpSyncLocal — result stays on GPU
