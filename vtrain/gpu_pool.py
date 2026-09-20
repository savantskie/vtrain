"""
GPU buffer pool — reuse kp.Tensor allocations across op calls instead of
allocating and freeing GPU memory on every single dispatch.
"""

import kp
import numpy as np
from collections import defaultdict

_active_pool = None


def set_pool(pool: "GPUBufferPool") -> None:
    """Register pool as the one all ops will use."""
    global _active_pool
    _active_pool = pool


def get_pool() -> "GPUBufferPool | None":
    return _active_pool


class GPUBufferPool:
    """
    Reusable pool of kp.Tensor GPU buffers, bucketed by flat element count.
    acquire() returns a buffer of the right size (allocates fresh if needed).
    release() returns it for future reuse.
    """

    def __init__(self, mgr: kp.Manager):
        self._mgr  = mgr
        self._pool = defaultdict(list)   # flat_size → [kp.Tensor, ...]

    def acquire(self, size: int) -> kp.Tensor:
        if self._pool[size]:
            return self._pool[size].pop()
        return self._mgr.tensor(np.zeros(size, dtype=np.float32))

    def release(self, buf: kp.Tensor) -> None:
        self._pool[len(buf.data())].append(buf)

    def clear(self) -> None:
        """Drop all pooled buffers. Call at end of training."""
        self._pool.clear()