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

- **Lazy GPU sync**: `.data` property syncs GPU→CPU only when accessed
- **GPU residency**: `mark_gpu_fresh()`/`ensure_on_gpu()` keep data on GPU
  between chained operations, avoiding uploads
- **Buffer pool**: `ensure_on_gpu()` checks `get_pool()` before allocating
- **Autograd**: topological sort + backward closures in reverse order
- **flush_graph()**: After backward, releases GPU buffers for non-parameter
  tensors and breaks reference cycles to prevent memory leaks

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

### Autograd (`vtrain/functional.py`)

Every op function takes `kp.Manager` + Tensor inputs and:
1. Calls raw GPU op on `.data`
2. Wraps result in Tensor with `_prev` set to inputs
3. Defines `_backward` closure that accumulates gradients

Matmul backward uses GPU-resident ops (transpose + matmul with buffer pool).
Other ops compute gradients on CPU via numpy arithmetic.

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
- Checkpoint resume from last saved step
- Per step: zero_grad → get_batch → forward → softmax → CE loss →
  backward → flush_graph → gc.collect → malloc_trim → optimizer step

The flush_graph + gc.collect + malloc_trim sequence prevents system RAM
growth over long training runs.

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
4. **zero_grad()**: Uses in-place `p.grad[:] = 0.0` rather than allocating
   new arrays. Prevents system RAM growth over many training steps.
5. **Indices in get_batch()**: `CharDataset.get_batch()` returns float32 arrays
   for both input and target. Targets are implicitly cast — no separate
   integer target path exists yet.

## Test Coverage

Tests cover all GPU ops, autograd correctness, loss functions, optimizers,
model forward/backward shapes, and checkpoint save/load. Run with:
```bash
python3 -m pytest tests/ -v
```