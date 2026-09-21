import kp
import numpy as np
from vtrain.ops.matmul import matmul
from vtrain.ops.softmax import softmax
from vtrain.ops.elementwise import mul
from vtrain.shader_utils import compile_shader

def attention(mgr: kp.Manager,
              Q: np.ndarray,
              K: np.ndarray,
              V: np.ndarray,
              mask: np.ndarray = None) -> np.ndarray:
    d_k = Q.shape[-1]
    K_T = np.ascontiguousarray(K.T)
    scores = matmul(mgr, Q, K_T)
    scale_factor = np.full(scores.shape,
                           1.0 / np.sqrt(d_k),
                           dtype=np.float32)
    scores = mul(mgr, scores, scale_factor)
    if mask is not None:
        scores = scores + mask
    weights = softmax(mgr, scores)
    output = matmul(mgr, weights, V)
    return output


def multihead_attention(mgr: kp.Manager,
                        X: np.ndarray,
                        W_q: np.ndarray,
                        W_k: np.ndarray,
                        W_v: np.ndarray,
                        W_o: np.ndarray,
                        n_heads: int) -> np.ndarray:
    seq_len, d_model = X.shape
    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    d_k = d_model // n_heads

    Q = matmul(mgr, X, W_q)
    K = matmul(mgr, X, W_k)
    V = matmul(mgr, X, W_v)

    Q = Q.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    K = K.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    V = V.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)

    head_outputs = []
    for h in range(n_heads):
        head_out = attention(mgr, Q[h], K[h], V[h])
        head_outputs.append(head_out)

    concat = np.concatenate(head_outputs, axis=-1)
    output = matmul(mgr, concat, W_o)
    return output



import math

def split_heads_gpu(mgr, t_src, t_dst, B: int, d_model: int, n_heads: int, d_k: int) -> None:
    """GPU-resident: reshape (B, d_model) -> (n_heads, B, d_k) flat"""
    spirv = compile_shader("split_heads").read_bytes()
    total = n_heads * B * d_k
    wg_x  = math.ceil(total / 256)
    algo  = mgr.algorithm(
        [t_src, t_dst], spirv, (wg_x, 1, 1), [],
        [float(B), float(d_model), float(n_heads), float(d_k)]
    )
    sq = mgr.sequence()
    sq.record(kp.OpAlgoDispatch(algo))
    sq.eval()


def merge_heads_gpu(mgr, t_src, t_dst, B: int, d_model: int, n_heads: int, d_k: int) -> None:
    """GPU-resident: reshape (n_heads, B, d_k) flat -> (B, d_model)"""
    spirv = compile_shader("merge_heads").read_bytes()
    total = n_heads * B * d_k
    wg_x  = math.ceil(total / 256)
    algo  = mgr.algorithm(
        [t_src, t_dst], spirv, (wg_x, 1, 1), [],
        [float(B), float(d_model), float(n_heads), float(d_k)]
    )
    sq = mgr.sequence()
    sq.record(kp.OpAlgoDispatch(algo))
    sq.eval()


def copy_block_gpu(mgr, t_src, t_dst, n: int,
                   src_offset: int, dst_offset: int, accumulate: bool) -> None:
    """Copy or accumulate a contiguous block between GPU buffers with offsets."""
    spirv = compile_shader("copy_block").read_bytes()
    wg_x  = math.ceil(n / 256)
    algo  = mgr.algorithm(
        [t_src, t_dst], spirv, (wg_x, 1, 1), [],
        [float(n), float(src_offset), float(dst_offset),
         float(1.0 if accumulate else 0.0)]
    )
    sq = mgr.sequence()
    sq.record(kp.OpAlgoDispatch(algo))
    sq.eval()


def scatter_head_gpu(mgr, t_src, t_dst,
                     B: int, d_model: int, head: int, d_k: int) -> None:
    """Scatter-accumulate head gradient (B*d_k) into full gradient (B*d_model)."""
    spirv = compile_shader("scatter_head").read_bytes()
    total = B * d_k
    wg_x  = math.ceil(total / 256)
    algo  = mgr.algorithm(
        [t_src, t_dst], spirv, (wg_x, 1, 1), [],
        [float(B), float(d_model), float(head), float(d_k)]
    )
    sq = mgr.sequence()
    sq.record(kp.OpAlgoDispatch(algo))
    sq.eval()