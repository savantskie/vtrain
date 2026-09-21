# VTrain - Vulkan ML Training Framework

A from-scratch machine learning training framework built on Vulkan compute shaders.
No CUDA. No ROCm. No vendor lock-in. If your GPU speaks Vulkan, it can train.

Built and tested on dual AMD MI50 32GB cards running Ubuntu 22.04 with Vulkan 1.4.x.
Intended for anyone with Vulkan-capable hardware that the major training frameworks
have left behind - older AMD cards, workstation GPUs, anything that falls outside
the official ROCm or CUDA support matrix but still has real compute capability.

## Why this exists

Every major ML training framework assumes CUDA or ROCm. If you have hardware that
doesn't fit neatly into either of those - say, a pair of AMD MI50s that AMD dropped
from ROCm support - your options are basically "upgrade your hardware" or "give up."

VTrain is the third option. Vulkan runs on virtually every modern GPU regardless of
vendor or generation. This framework proves that a complete training stack - forward
pass, backward pass, autograd, optimizers, the whole thing - can be built on Vulkan
compute shaders with no dependency on vendor-specific ML runtimes.

## What's in the box

A complete training framework built from the ground up:

- GLSL compute shaders for every operation, compiled to SPIR-V via glslc
- Full forward pass: matmul, elementwise ops (ReLU, GELU, sigmoid, tanh, add, sub,
  mul, div), layer normalization, softmax, transpose, attention
- Full backward pass with GPU shaders for every op — no CPU fallback
- GPU-resident gradients: every backward closure dispatches GPU shaders
  (matmul, unary_backward, softmax_backward, accumulate, etc.)
- GPU-resident optimizer: Adam/SGD step dispatched as GPU shader per parameter,
  momentum buffers stored as GPU kp.Tensor
- GPU-resident loss: cross-entropy computed entirely on GPU
- GPU-primary tensors: data and gradients live in kp.Tensor buffers on GPU.
  CPU sync only happens for loss scalar reads, checkpoint saves, and logging.
- GPU buffer reuse pool to avoid allocating GPU memory every dispatch
- Autograd system with topological sort and numerical gradient verification
- Adam optimizer with GPU-resident momentum/velocity buffers
- Transformer architecture: embeddings, multi-head attention, feed-forward layers,
  residual connections
- Character-level language model (SmallLM) ready to train on any text corpus
- Wikipedia data pipeline: download, extract, clean, and train
- Training loop with checkpointing, crash recovery, and resume support
  — optimizer state saved alongside weights, resumed training continues
  with full momentum/velocity intact
- Text generation from trained checkpoints
- GPU device detection utility (vtrain/gpu_detect.py)

## Hardware requirements

- Any Vulkan 1.1+ capable GPU (AMD, NVIDIA, Intel - anything with a Vulkan driver)
- Tested on: AMD MI50 32GB (gfx906, RADV driver, Vulkan 1.4.335)
- **Single-GPU training is implemented and operational. Multi-GPU Vulkan training is planned and currently under development.**
- Linux recommended. Other platforms untested but should work if Vulkan is available.
- No ROCm required. No CUDA required. No specific driver version required beyond
  basic Vulkan support.

## System dependencies

Ubuntu/Debian:
```bash
sudo apt install git cmake build-essential python3 python3-venv glslang-tools vulkan-tools libvulkan-dev
```

Other distros: install the equivalent packages. You need git, cmake, a C++ compiler,
Python 3.8+, glslc (from glslang-tools or shaderc), and the Vulkan loader.

## Setup

```bash
git clone https://github.com/savantskie/vtrain.git
cd vtrain
bash setup.sh
source .venv/bin/activate
```

The setup script handles everything - checks dependencies, creates a virtual
environment, builds Kompute from source, and verifies the installation against
your GPU.

## Training on Wikipedia

Download and extract Simple English Wikipedia (recommended for a first run - small,
clean, and fast to iterate on):

```bash
wget https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2
python3 vtrain/data/wiki_extract.py --dump simplewiki-latest-pages-articles.xml.bz2 --output simplewiki.txt
```

Train:

```bash
python3 train_wiki.py --data simplewiki.txt --run-dir models/run1
```

Full option list:

```
--data              Path to training text file (required)
--run-dir           Directory for checkpoints and logs (default: models/run1)
--max-chars         Characters to load from data file (default: 10000000)
--seq-len           Sequence length (default: 64)
--batch-size        Batch size (default: 16)
--d-model           Model dimension (default: 128)
--n-heads           Attention heads (default: 4)
--n-layers          Transformer layers (default: 2)
--lr                Learning rate (default: 0.0003)
--max-steps         Training steps (default: 10000)
--checkpoint-every  Steps between checkpoints (default: 200)
--log-every         Steps between log lines (default: 50)
--device            Vulkan device index (default: 0)
```

## Training on your own data

Point it at any plain text file:

```bash
python3 train_wiki.py --data /path/to/your/text.txt --run-dir models/myrun
```

The character-level tokenizer builds its vocabulary automatically from whatever
text you give it. No preprocessing required beyond having a plain text file.

## Text generation

Generate text from a trained checkpoint:

```bash
python3 generate.py --vocab models/run1/vocab.json --checkpoint models/run1/checkpoints/step_010000_final
```

Full option list:

```
--vocab         Path to vocab.json (required)
--checkpoint    Path to checkpoint directory (required)
--d-model       Model dimension (default: 128)
--n-heads       Attention heads (default: 4)
--n-layers      Transformer layers (default: 2)
--seed          Seed text (default: "The history of")
--n-chars       Characters to generate (default: 500)
--temperature   Sampling temperature (default: 0.2)
--device        Vulkan device index (default: 0)
```

## Running the tests

```bash
source .venv/bin/activate
python3 -m pytest tests/ -v
```

Tests cover every shader, the full autograd system, loss functions, optimizers,
model forward and backward passes, and checkpoint save/load.

## Project structure

```
vtrain/
  ops/          - GPU op wrappers (matmul, elementwise, layernorm, softmax, transpose)
  model/        - Model architecture (linear, transformer, language model, checkpoint)
  data/         - Data pipeline (character dataset, Wikipedia extractor)
  tensor.py     - Tensor class with autograd (GPU-resident lazy sync)
  functional.py - Differentiable GPU op wrappers
  gpu_pool.py   - GPU buffer reuse pool
  grad_check.py - Numerical gradient verification
  loss.py       - Loss functions (CPU and GPU)
  optim.py      - SGD and Adam optimizers (GPU shader dispatch)
  gpu_detect.py - Vulkan device enumeration
  train.py      - Reusable training loop
shaders/        - GLSL compute shader source
compiled/       - SPIR-V compiled shaders (generated at runtime)
models/         - Training runs and checkpoints
tests/          - Test suite
train_wiki.py   - Wikipedia training script
  generate.py     - Text generation from trained checkpoints
  VTRAIN_SYSTEM_MAP.md - Complete system architecture reference
  setup.sh        - One-shot environment setup
build_kompute.sh - Kompute build script
```

## Known quirks

If you're building on top of this or debugging issues, a few things that aren't
obvious from the Kompute documentation:

1. Buffer qualifiers: readonly and writeonly on buffer declarations cause silent
   zero output in Kompute's descriptor layout. All buffers are declared without
   qualifiers.
2. Push constant sizing: ops with 3 buffers require a float padding constant in
   the push constant block or Kompute miscalculates the buffer size. Silent failure.
3. Kompute API: the built-from-source version uses kp.OpSyncDevice and
   kp.OpSyncLocal. The PyPI package docs reference kp.OpTensorSyncDevice which
   does not exist in the actual build - and the PyPI package itself is broken on
   CMake 4.x anyway, which is why this repo builds Kompute from source.
4. GPU-resident gradients: gradients are stored as GPU kp.Tensor buffers,
   not numpy arrays. Accessing tensor.grad triggers a GPU→CPU sync.
   backward() pre-initializes gradient buffers for every node in the
   autograd graph before running closures, preventing null-pointer crashes.
   Buffers are freed by flush_graph() after the optimizer step.
5. Optimizer state (Adam momentum/velocity) is checkpointed alongside
   weights. Resume preserves the optimizer's step counter and velocity
   estimates, so loss converges as if training was never interrupted.
6. Cross-entropy loss now requires a kp.Manager argument for GPU dispatch.
   The old CPU-only signature cross_entropy_loss(pred, target) is kept
   for tests. Use cross_entropy_loss_gpu(mgr, pred, target) for training.

## License

MIT
