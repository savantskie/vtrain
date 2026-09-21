import numpy as np


class Tensor:
    """
    Wraps a numpy array with gradient tracking and autograd support.
    GPU-primary: data lives in a kp.Tensor GPU buffer and is only synced
    to CPU when .data is accessed (lazy sync). Gradients are also GPU-resident.
    """

    def __init__(self, data, requires_grad=True, name=""):
        self._data        = np.array(data, dtype=np.float32)
        self._kp_tensor   = None
        self._gpu_fresh   = False
        self._mgr         = None
        self._shape_cache = self._data.shape
        self.requires_grad = requires_grad
        self._grad_kp     = None    # GPU-resident gradient buffer
        self._grad_data   = None    # CPU copy, lazily synced
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

    # ── Gradient property — GPU-resident ─────────────────────────────────────

    @property
    def grad(self) -> np.ndarray:
        """Return CPU numpy gradient, syncing from GPU if needed."""
        if self._grad_kp is not None and self._mgr is not None:
            if self._grad_data is None:
                import kp
                sq = self._mgr.sequence()
                sq.record(kp.OpSyncLocal([self._grad_kp]))
                sq.eval()
                self._grad_data = (self._grad_kp.data()
                                   .reshape(self._shape_cache)
                                   .copy()
                                   .astype(np.float32))
            return self._grad_data
        return None

    @grad.setter
    def grad(self, value):
        """Set gradient from numpy (for compatibility). Marks GPU stale."""
        self._grad_data = np.array(value, dtype=np.float32)
        self._grad_kp = None

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

    def _ensure_grad_on_gpu(self, mgr):
        """Return kp.Tensor for gradient, uploading from CPU copy if needed."""
        from vtrain.gpu_pool import get_pool
        pool = get_pool()
        flat_size = int(np.prod(self._shape_cache))

        if self._grad_kp is not None and self._mgr is mgr:
            return self._grad_kp

        if self._grad_kp is None:
            self._grad_kp = (pool.acquire(flat_size) if pool is not None
                             else mgr.tensor(np.zeros(flat_size, dtype=np.float32)))
            if pool is not None:
                pool.register_persistent(self._grad_kp)

        if self._grad_data is not None:
            self._grad_kp.data()[:] = self._grad_data.flatten()
            import kp
            sq = mgr.sequence()
            sq.record(kp.OpSyncDevice([self._grad_kp]))
            sq.eval()
        else:
            self._grad_kp.data()[:] = 0.0
            import kp
            sq = mgr.sequence()
            sq.record(kp.OpSyncDevice([self._grad_kp]))
            sq.eval()

        self._mgr = mgr
        return self._grad_kp

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

        from vtrain.gpu_pool import get_pool
        pool = get_pool()
        import kp

        # Initialize loss gradient
        flat_size = int(np.prod(self._shape_cache))
        self._grad_kp = (pool.acquire(flat_size) if pool is not None
                         else self._mgr.tensor(np.ones(flat_size, dtype=np.float32)))
        if pool is not None:
            pool.register_persistent(self._grad_kp)
        self._grad_kp.data()[:] = np.ones(flat_size, dtype=np.float32)
        sq = self._mgr.sequence()
        sq.record(kp.OpSyncDevice([self._grad_kp]))
        sq.eval()
        self._grad_data = None

        # Pre-initialize grad_kp for every node so closures accumulate safely
        for t in topo:
            if t is not self and t._grad_kp is None:
                fs = int(np.prod(t._shape_cache))
                t._grad_kp = (pool.acquire(fs) if pool is not None
                              else self._mgr.tensor(np.zeros(fs, dtype=np.float32)))
                if pool is not None:
                    pool.register_persistent(t._grad_kp)
                t._grad_kp.data()[:] = 0.0
                sq2 = self._mgr.sequence()
                sq2.record(kp.OpSyncDevice([t._grad_kp]))
                sq2.eval()
                t._grad_data = None
                if t._mgr is None:
                    t._mgr = self._mgr

        for t in reversed(topo):
            t._backward()

    def zero_grad(self):
        if self._grad_kp is not None and self._mgr is not None:
            import kp
            self._grad_kp.data()[:] = 0.0
            sq = self._mgr.sequence()
            sq.record(kp.OpSyncDevice([self._grad_kp]))
            sq.eval()
        self._grad_data = None

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
            if t._grad_kp is not None and pool is not None:
                pool.release(t._grad_kp)
                t._grad_kp = None
            t._grad_data = None
            if t is not root:
                t._data = None
            t._prev = set()
            t._backward = None

    _walk(root)