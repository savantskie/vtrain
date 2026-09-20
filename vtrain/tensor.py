import numpy as np


class Tensor:
    """
    Wraps a numpy array with gradient tracking and autograd support.
    Supports GPU-resident mode: data lives in a kp.Tensor GPU buffer
    and is only synced to CPU when .data is actually accessed (lazy sync).
    """

    def __init__(self, data, requires_grad=True, name=""):
        self._data        = np.array(data, dtype=np.float32)
        self._kp_tensor   = None               # GPU buffer (kp.Tensor), if resident
        self._gpu_fresh   = False              # True = GPU buffer has latest values
        self._mgr         = None               # kp.Manager, set by GPU ops
        self._shape_cache = self._data.shape   # cached — shape never triggers GPU sync
        self.requires_grad = requires_grad
        self.grad          = np.zeros_like(self._data) if requires_grad else None
        self._backward     = lambda: None
        self._prev         = set()
        self.name          = name

    # ── Data property — lazy GPU sync ────────────────────────────────────────

    @property
    def data(self) -> np.ndarray:
        """Return CPU numpy array, syncing from GPU first if GPU has fresh data."""
        if self._gpu_fresh and self._kp_tensor is not None and self._mgr is not None:
            import kp
            sq = self._mgr.sequence()
            sq.record(kp.OpSyncLocal([self._kp_tensor]))
            sq.eval()
            self._data      = (self._kp_tensor.data()
                                .reshape(self._shape_cache)
                                .copy()
                                .astype(np.float32))
            self._gpu_fresh = False
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        """Store new numpy array and mark GPU buffer as stale."""
        self._data        = np.array(value, dtype=np.float32)
        self._shape_cache = self._data.shape
        self._gpu_fresh   = False

    # ── GPU residency helpers ─────────────────────────────────────────────────

    def mark_gpu_fresh(self, kp_tensor, shape: tuple, mgr) -> None:
        """
        Called by GPU ops after writing their result into kp_tensor.
        Marks this Tensor as GPU-resident so the next op can skip the upload.
        """
        self._kp_tensor   = kp_tensor
        self._shape_cache = shape
        self._gpu_fresh   = True
        self._mgr         = mgr

    def ensure_on_gpu(self, mgr):
        """
        Return a kp.Tensor with this Tensor's current data on the GPU.
        Uploads from CPU only if the GPU buffer is stale or missing.
        """
        import kp
        from vtrain.gpu_pool import get_pool
        pool = get_pool()

        if self._gpu_fresh and self._kp_tensor is not None:
            return self._kp_tensor   # already current — skip upload

        flat = self._data.flatten().astype(np.float32)
        size = len(flat)

        # Allocate or reuse a buffer of the right size
        if self._kp_tensor is None or len(self._kp_tensor.data()) != size:
            if self._kp_tensor is not None and pool is not None:
                pool.release(self._kp_tensor)
            self._kp_tensor = (pool.acquire(size) if pool is not None
                               else mgr.tensor(flat))

        # Write into buffer's CPU-side mapping and upload to GPU
        self._kp_tensor.data()[:] = flat
        sq = mgr.sequence()
        sq.record(kp.OpSyncDevice([self._kp_tensor]))
        sq.eval()

        self._gpu_fresh   = True
        self._mgr         = mgr
        return self._kp_tensor

    # ── Autograd ──────────────────────────────────────────────────────────────

    def backward(self):
        topo    = []
        visited = set()

        def build_topo(t):
            if id(t) not in visited:
                visited.add(id(t))
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)

        # Loss is always on CPU — use _data directly, no sync needed
        self.grad = np.ones_like(self._data)

        for t in reversed(topo):
            t._backward()

    def zero_grad(self):
        if self.grad is not None:
            self.grad[:] = 0.0

    def zero_grad_all(self):
        visited = set()

        def _zero(t):
            if id(t) not in visited:
                visited.add(id(t))
                t.zero_grad()
                for child in t._prev:
                    _zero(child)

        _zero(self)

    # ── Shape helpers — never trigger GPU sync ────────────────────────────────

    @property
    def shape(self) -> tuple:
        return self._shape_cache

    @property
    def ndim(self) -> int:
        return len(self._shape_cache)

    def __repr__(self):
        name_str = f" name={self.name!r}" if self.name else ""
        return f"Tensor(shape={self.shape}{name_str})"

    #-- Flush Graphs


def flush_graph(root: 'Tensor', keep: set, pool) -> None:
    """
    After backward(), walk the computation graph and release GPU buffers
    on all intermediate tensors back to the pool, then clear _prev
    references so Python GC can actually collect them.

    keep: set of id(tensor) for model parameters — leave those alone.
    """
    visited = set()

    def _walk(t):
        if id(t) in visited:
            return
        visited.add(id(t))

        for child in t._prev:
            _walk(child)

        if id(t) not in keep:
            if t._kp_tensor is not None and pool is not None:
                pool.release(t._kp_tensor)
                t._kp_tensor = None
                t._gpu_fresh = False
            t.grad  = None
            if t is not root:
                t._data = None
            t._prev = set()       # break reference cycles
            t._backward = None    # release closure-captured numpy arrays

    _walk(root)