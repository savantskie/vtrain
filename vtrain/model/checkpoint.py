import numpy as np
import json
from pathlib import Path
from vtrain.tensor import Tensor


def save(params: dict, path: str):
    """
    Save model weights to disk.

    params: dict of name → Tensor  e.g. {'W_q': tensor, 'W_k': tensor, ...}
    path:   directory to save into (created if it doesn't exist)

    Saves weights as .npy files, config as config.json.
    """
    save_dir = Path(path)
    save_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for name, tensor in params.items():
        filename = f"{name}.npy"
        np.save(save_dir / filename, tensor.data)
        manifest[name] = {
            "file":  filename,
            "shape": list(tensor.data.shape),
            "dtype": str(tensor.data.dtype),
        }

    with open(save_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved {len(params)} tensors to {save_dir}")


def load(params: dict, path: str):
    """
    Load weights from disk into existing Tensors in-place.

    params: same dict of name → Tensor you passed to save()
    path:   directory previously saved with save()

    Updates tensor.data in place — keeps the same Tensor objects
    so any references elsewhere in the model stay valid.
    """
    save_dir = Path(path)

    with open(save_dir / "manifest.json") as f:
        manifest = json.load(f)

    for name, tensor in params.items():
        if name not in manifest:
            raise KeyError(f"'{name}' not found in checkpoint at {path}")
        data = np.load(save_dir / manifest[name]["file"])
        assert data.shape == tuple(manifest[name]["shape"]), \
            f"Shape mismatch for '{name}': got {data.shape}, expected {manifest[name]['shape']}"
        tensor.data = data.astype(np.float32)

    print(f"Loaded {len(params)} tensors from {save_dir}")


def params_from_block(block, prefix="") -> dict:
    """
    Helper: flatten a TransformerBlock or Linear's parameters into
    a named dict suitable for save()/load().

    Usage:
        p = params_from_block(my_block, prefix="block0")
        save(p, "models/my_model/checkpoints/step_100")
    """
    result = {}
    for i, param in enumerate(block.parameters()):
        name = f"{prefix}_p{i}" if prefix else f"p{i}"
        if param.name:
            name = f"{prefix}_{param.name}" if prefix else param.name
        result[name] = param
    return result
