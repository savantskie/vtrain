import kp
import numpy as np
import subprocess
import math
from pathlib import Path

SHADER_DIR   = Path(__file__).parent.parent.parent / "shaders"
COMPILED_DIR = Path(__file__).parent.parent.parent / "compiled"


def _compile_shader(shader_name: str) -> Path:
    src = SHADER_DIR   / f"{shader_name}.comp"
    dst = COMPILED_DIR / f"{shader_name}.spv"

    needs_compile = (
        not dst.exists()
        or src.stat().st_mtime > dst.stat().st_mtime
    )

    if needs_compile:
        result = subprocess.run(
            ["glslc", str(src), "-o", str(dst)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Shader compile failed:\n{result.stderr}")

    return dst


def relu(mgr: kp.Manager, X: np.ndarray) -> np.ndarray:
    """
    ReLU activation: max(0, x) applied elementwise.
    Works on any shape — internally flattened, result reshaped back.
    """
    original_shape = X.shape
    x_flat = X.flatten().astype(np.float32)
    y_flat = np.zeros_like(x_flat)

    n = len(x_flat)

    t_x = mgr.tensor(x_flat)
    t_y = mgr.tensor(y_flat)

    spirv = _compile_shader("relu").read_bytes()

    # 1D dispatch: ceil(n / 256) workgroups of 256 threads each
    wg_x = math.ceil(n / 256)

    algo = mgr.algorithm(
        [t_x, t_y],
        spirv,
        (wg_x, 1, 1),
        [],
        [np.uint32(n)]  # push constant — total element count
    )

    sq = mgr.sequence()
    sq.record(kp.OpSyncDevice([t_x, t_y]))
    sq.record(kp.OpAlgoDispatch(algo))
    sq.record(kp.OpSyncLocal([t_y]))
    sq.eval()

    return t_y.data().reshape(original_shape)
