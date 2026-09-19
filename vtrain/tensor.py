import numpy as np


class Tensor:
    """
    Wraps a numpy array with gradient tracking and autograd support.
    The GPU ops in functional.py create these and wire up the backward
    closures — this class just manages the graph and the backward pass.
    """

    def __init__(self, data, requires_grad=True, name=""):
        self.data          = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad          = np.zeros_like(self.data) if requires_grad else None
        self._backward     = lambda: None  # no-op until an op registers one
        self._prev         = set()         # Tensors this one was computed from
        self.name          = name          # optional label for debugging

    def backward(self):
        """
        Run backprop from this tensor back through the whole graph.
        Assumes this is a scalar output (loss), or will sum gradients
        across the output if not.
        """
        # Step 1: topological sort — depth-first, children before parents
        # This guarantees that when we backprop through a node, all the
        # gradients flowing INTO it from later nodes are already accumulated
        topo    = []
        visited = set()

        def build_topo(t):
            if id(t) not in visited:
                visited.add(id(t))
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)

        # Step 2: seed — the gradient of the output w.r.t. itself is 1
        # (if this is a scalar loss, that's literally just 1.0)
        self.grad = np.ones_like(self.data)

        # Step 3: unwind the graph in reverse order, firing each
        # backward closure. Each closure deposits gradients into
        # its input tensors' .grad fields.
        for t in reversed(topo):
            t._backward()

    def zero_grad(self):
        """Reset gradient accumulator to zero. Call before each training step."""
        if self.grad is not None:
            self.grad = np.zeros_like(self.data)

    def zero_grad_all(self):
        """Zero gradients for this tensor and all tensors in its graph."""
        visited = set()

        def _zero(t):
            if id(t) not in visited:
                visited.add(id(t))
                t.zero_grad()
                for child in t._prev:
                    _zero(child)

        _zero(self)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        name_str = f" name={self.name!r}" if self.name else ""
        return f"Tensor(shape={self.shape}{name_str})"
