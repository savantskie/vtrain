import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import kp
import numpy as np
from vtrain.ops.elementwise import add

mgr = kp.Manager(0)

A = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
B = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
C = np.zeros(4, dtype=np.float32)

t_a = mgr.tensor(A)
t_b = mgr.tensor(B)
t_c = mgr.tensor(C)

SHADERS_DIR   = Path(__file__).parent.parent / "shaders"
COMPILED_DIR  = Path(__file__).parent.parent / "compiled"

import subprocess
subprocess.run(
    ["glslc",
     str(SHADERS_DIR / "binary_debug.comp"),
     "-o", str(COMPILED_DIR / "binary_debug.spv")],
    check=True
)

spirv = open(COMPILED_DIR / "binary_debug.spv", "rb").read()

algo = mgr.algorithm([t_a, t_b, t_c], spirv, (1, 1, 1), [], [])

sq = mgr.sequence()
sq.record(kp.OpSyncDevice([t_a, t_b, t_c]))
sq.record(kp.OpAlgoDispatch(algo))
sq.record(kp.OpSyncLocal([t_c]))
sq.eval()

print(f"A: {A}")
print(f"B: {B}")
print(f"Got: {t_c.data()}")
