import subprocess
from pathlib import Path

SHADER_DIR   = Path(__file__).parent.parent / "shaders"
COMPILED_DIR = Path(__file__).parent.parent / "compiled"


def compile_shader(shader_name: str) -> Path:
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
