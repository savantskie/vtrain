import numpy as np
from vtrain.tensor import Tensor


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    Mean Squared Error — average of (pred - target)²
    Standard loss for regression tasks.

    pred/target shape: any, but must match.
    Returns: scalar Tensor
    """
    diff = pred.data - target.data
    loss = Tensor(np.array(np.mean(diff ** 2), dtype=np.float32),
                  requires_grad=True)
    loss._prev = {pred}

    def _backward():
        if pred.requires_grad:
            N = pred.data.size
            # Gradient: 2*(pred - target)/N
            pred.grad += (2.0 * diff / N) * loss.grad

    loss._backward = _backward
    return loss


def cross_entropy_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    Cross-entropy loss — expects pred to already be softmax probabilities.

    pred shape:   (batch, n_classes) — softmax output
    target shape: (batch, n_classes) — one-hot encoded labels
    Returns: scalar Tensor

    Why one-hot: keeps it consistent with the rest of our Tensor pipeline.
    A helper below converts integer class labels to one-hot if needed.
    """
    eps     = 1e-7   # prevents log(0) which is -inf
    p       = np.clip(pred.data, eps, 1.0)
    batch   = pred.data.shape[0]

    loss_val = -np.mean(np.sum(target.data * np.log(p), axis=-1))
    loss     = Tensor(np.array(loss_val, dtype=np.float32),
                      requires_grad=True)
    loss._prev = {pred}

    def _backward():
        if pred.requires_grad:
            # Gradient: -(target / pred) / batch
            pred.grad += (-target.data / p / batch) * loss.grad

    loss._backward = _backward
    return loss


def one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Convert integer class labels to one-hot encoding.
    labels shape: (batch,) of integers in [0, n_classes)
    Returns:      (batch, n_classes) float32
    """
    out = np.zeros((len(labels), n_classes), dtype=np.float32)
    out[np.arange(len(labels)), labels] = 1.0
    return out
