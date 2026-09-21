# VTrain System Map

## Overview

VTrain is a from-scratch, Vulkan-accelerated transformer language model training
framework built entirely in Python with GLSL compute shaders. Zero dependency on
PyTorch, JAX, TensorFlow, or CUDA. Every GPU operation — matmul, softmax, layer
norm, activation functions — runs through Vulkan compute shaders compiled from
GLSL to SPIR-V, dispatched via the Kompute library.

## Architecture

```
train_wiki.py  ───┐
generate.py    ───┤
                   v
            ┌─ vtrain/ ──────────────────────┐
            │  tensor.py      (autograd)     │
            │  functional.py  (diff ops)     │
            │  optim.py       (SGD, Adam)    │
            │  loss.py        (MSE, CE)      │
            │  train.py       (Trainer)      │
            │  grad_check.py  (verify grads) │
            │  shader_utils.py(compile .comp)│
            │  gpu_pool.py    (buffer reuse) │
             │  gpu_detect.py  (device info)  │
            │                                 │
            │  ops/ ──────────────────────┐  │
            │  │  matmul.py   (GPU matmul)│  │
            │  │  elementwise.py(unary/   │  │
            │  │    binary ops)          │  │
            │  │  softmax.py              │  │
            │  │  layernorm.py            │  │
            │  │  transpose.py            │  │
            │  │  attention.py (composed) │  │
            │  └──────────────────────────┘  │
            │                                 │
            │  model/ ───────────────────┐   │
            │  │  linear.py   (Linear,   │   │
            │  │    FeedForward)         │   │
            │  │  transformer.py(Block)  │   │
            │  │  lm.py  (SmallLM)      │   │
            │  │  checkpoint.py(save/    │   │
            │  │    load)               │   │
            │  └──────────────────────────┘  │
            │                                 │
            │  data/ ───────────────────┐   │
            │     dataset.py(CharDataset)│   │
            │     wiki_extract.py       │   │
            │     (XML dump -> text)    │   │
            └─────────────────────────────┘
                    │
                    v
            ┌─ shaders/ ───┐
            │ .comp GLSL   │── glslc ──> compiled/.spv
            └──────────────┘
                    │
                    v
            ┌─ Kompute (kp) ──────────────┐
            │  Vulkan compute abstraction │
            │  Buffer mgmt, shader        │
            │  dispatch, sync ops         │
            └─────────────────────────────┘
                    │
                    v
              Vulkan GPU
```

## Component Details

### Tensor (`vtrain/tensor.py`)

Wraps numpy arrays with autograd support. Key design:

- **GPU-primary storage**: Data lives in a `kp.Tensor` GPU buffer and is only synced
  to CPU when `.data` is accessed (lazy sync)
- **GPU-resident gradients**: `_grad_kp` replaces `self.grad` as a numpy array.
  `grad` is now a property with lazy GPU→CPU sync. `_ensure_grad_on_gpu()` ensures
  the gradient buffer exists on GPU before any backward closure uses it.
- **Pre-initialized backward**: `backward()` allocates `_grad_kp` for every node in
  the computation graph before running any closure, so closures can safely
  accumulate into GPU buffers without checking for null.
- **Buffer pool**: `ensure_on_gpu()` uses `get_pool()` for GPU buffer allocation
- **Autograd**: topological sort + backward closures in reverse order
- **flush_graph()**: After backward, releases GPU buffers (`_kp_tensor` and
  `_grad_kp`) for non-parameter tensors and breaks reference cycles

### GPU Operations (`vtrain/ops/`)

Every op follows the same dual pattern:

**CPU-round-trip form** (e.g., `matmul(mgr, A_np, B_np)`):
1. Flatten inputs to float32
2. Create kp.Tensors via `mgr.tensor()`
3. Compile SPIR-V from .comp via `compile_shader()`
4. Create kp.Algorithm with push constants
5. Sequence: OpSyncDevice → OpAlgoDispatch → OpSyncLocal
6. Return reshaped result

**GPU-resident form** (e.g., `matmul_gpu(mgr, t_a, t_b, t_c, M, K, N)`):
- Operates on pre-uploaded kp.Tensors
- No sync operations — result stays on GPU
- Used by functional.py's backward passes via buffer pool

### Shader Details

| Shader | Workgroup | Buffers | Operation |
|--------|-----------|---------|-----------|
| matmul.comp | 16×16 | A, B, C | C[i][j] = sum_k A[i][k] * B[k][j] |
| unary.comp | 256×1 | X, Y | Switch: 0=ReLU, 1=sigmoid, 2=tanh, 3=GELU |
| binary.comp | 256×1 | A, B, C | Switch: 0=add, 1=sub, 2=mul, 3=div |
| softmax.comp | 256×1 | X, Y | 3-pass: max, exp+sum, normalize |
| layernorm.comp | 256×1 | X, Y, γ, β | 2-pass: mean/var, normalize+scale |
| transpose.comp | 16×16 | X, Y | Tiled with bank-conflict avoidance |
| adam_step.comp | 256×1 | p, g, m, v | Per-element Adam update, 8 push constants |
| sgd_step.comp | 256×1 | p, g | Per-element SGD update |
| accumulate.comp | 256×1 | dst, src | dst += sign * src (sign push constant) |
| unary_backward.comp | 256×1 | go, x, gi | gi += go * derivative(x) for relu/sigmoid/tanh/gelu |
| softmax_backward.comp | 256×1 | dy, s, dx | dx += s * (dy - sum(dy*s)), shared-memory reduction |
| loss_ce.comp | 256×1 | p, t, g, loss | Cross-entropy loss + gradient, 1 workgroup/batch element |
| split_heads.comp | 256×1 | src, dst | Q(B, d_model) → Q_heads(n_heads, B, d_k) |
| merge_heads.comp | 256×1 | src, dst | Inverse of split_heads |

### Autograd (`vtrain/functional.py`)

Every op function takes `kp.Manager` + Tensor inputs and:
1. Calls `ensure_on_gpu()` on inputs to ensure GPU buffers exist
2. Acquires output GPU buffer from pool
3. Dispatches GPU-resident op (no CPU transfer)
4. Wraps result in Tensor with `_prev` set to inputs and `mark_gpu_fresh()`
5. Defines `_backward` closure that accumulates gradients via GPU ops

**All backward closures use GPU ops.** No numpy math in any backward path:
- matmul: GPU-resident matmul + transpose
- unary ops (relu, sigmoid, tanh, gelu): `unary_backward_gpu` shader
- binary ops (add, sub, mul, div): `accumulate_gpu` + `binary_gpu`
- softmax: `softmax_backward_gpu` shader (shared-memory reduction)
- layer norm: CPU math but gradients synced to GPU via `_ensure_grad_on_gpu()`

### Model Architecture

**SmallLM**: Embedding → N×TransformerBlock → Linear head → logits

**TransformerBlock**: Pre-norm design:
- LayerNorm → Multi-Head Attention → residual
- LayerNorm → FeedForward (Linear→GELU→Linear) → residual

Head slicing is done on CPU via closure factories to avoid Python's
loop-variable capture bug.

### Training Loop

Configurable via CLI args in `train_wiki.py`:
- Data loading through CharDataset (character-level tokenizer)
- Model construction from config
- Checkpoint resume from last saved step (weights + optimizer state)
- Parameter upload to GPU and persistent registration in buffer pool
- Per step: zero_grad (GPU) → get_batch → forward (GPU-resident) →
  softmax → CE loss (GPU shader) → backward (GPU-resident) →
  flush_graph → optimizer step (GPU shader, per-parameter dispatch)

No CPU round-trips during training. CPU only involved for:
- Loss scalar read (one float per log interval)
- Checkpoint save/load (one GPU→CPU sync per parameter)
- Console logging

## Known Quirks

1. **Buffer qualifiers**: `readonly` and `writeonly` on GLSL buffer declarations
   cause silent zero output in Kompute's descriptor layout. All buffers are
   declared without qualifiers.
2. **Push constant sizing**: Ops with 3 buffers require a float padding constant
   in the push constant block or Kompute miscalculates buffer size.
3. **Kompute API**: The built-from-source version uses `kp.OpSyncDevice` and
   `kp.OpSyncLocal`. The PyPI package uses different names — and the PyPI
   package itself is broken on CMake 4.x, which is why this project builds
   Kompute from source.
4. **GPU-resident gradients**: Gradients are stored as GPU `kp.Tensor` buffers,
   not numpy arrays. Accessing `tensor.grad` triggers a GPU→CPU sync. Backward
   closures accumulate into `_grad_kp` directly via GPU shader dispatches.
5. **Pre-initialized backward**: `backward()` allocates gradient buffers for every
   node in the autograd graph before running any closure. This prevents null
   pointer crashes but uses more GPU memory during the backward pass. Buffers
   are freed by `flush_graph()` after the optimizer step.
6. **Indices in get_batch()**: `CharDataset.get_batch()` returns float32 arrays
   for both input and target. Targets are implicitly cast — no separate integer
   target path exists yet.

## Test Coverage

Tests cover all GPU ops, autograd correctness, loss functions, optimizers,
model forward/backward shapes, and checkpoint save/load. Run with:
```bash
python3 -m pytest tests/ -v
```