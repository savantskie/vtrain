import numpy as np
from vtrain.tensor import Tensor


def grad_check(fn, inputs, eps=1e-3, atol=1e-3, name=""):
    """
    Verify backward pass correctness by comparing analytical gradients
    (from .backward()) against numerical gradients (finite difference).

    fn:     callable(inputs) -> Tensor scalar (or will be summed to scalar)
    inputs: list of Tensors with requires_grad=True
    eps:    finite difference step size
    atol:   tolerance for pass/fail

    How it works:
    - Numerical gradient for element [i] of input X:
        (f(X with x[i]+eps) - f(X with x[i]-eps)) / (2*eps)
    - This approximates the true gradient without any math — just observation
    - If our backward shader is correct, it should match to within atol
    """
    prefix = f"[{name}] " if name else ""

    # Step 1: get analytical gradients from our backward pass
    for inp in inputs:
        inp.zero_grad()

    out = fn(inputs)
    out.backward()
    analytical = [inp.grad.copy() for inp in inputs]

    # Step 2: numerical gradients via finite difference
    numerical = []
    for inp in inputs:
        num_grad    = np.zeros_like(inp.data)
        flat_data   = inp.data.flat

        for i in range(inp.data.size):
            orig = inp.data.flat[i]

            # f(x + eps)
            inp.data.flat[i] = orig + eps
            for t in inputs:
                t.zero_grad()
            out_plus = fn(inputs)

            # f(x - eps)
            inp.data.flat[i] = orig - eps
            for t in inputs:
                t.zero_grad()
            out_minus = fn(inputs)

            # Central difference
            num_grad.flat[i] = (
                out_plus.data.sum() - out_minus.data.sum()
            ) / (2.0 * eps)

            # Restore
            inp.data.flat[i] = orig

        numerical.append(num_grad)

    # Restore analytical grads (numerical pass zeroed them)
    for inp, ana in zip(inputs, analytical):
        inp.grad = ana

    # Step 3: compare
    all_passed = True
    for i, (ana, num) in enumerate(zip(analytical, numerical)):
        max_err     = np.abs(ana - num).max()
        rel_err     = max_err / (np.abs(num).max() + 1e-8)
        passed      = max_err < atol
        all_passed  = all_passed and passed
        status      = "✓" if passed else "✗"
        print(f"{status} {prefix}input[{i}]  "
              f"max_err={max_err:.2e}  rel_err={rel_err:.2e}")

    return all_passed
