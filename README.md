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
- Full backward pass with GPU shaders for every op - no CPU fallback
- GPU buffer reuse pool to avoid allocating GPU memory every dispatch
- Memory leak prevention: in-place gradient zeroing, computation graph flushing,
  and periodic glibc arena trimming keep system RAM stable over long training runs
- Autograd system with topological sort and numerical gradient verification
- Loss functions: MSE and cross-entropy
- Optimizers: SGD and Adam
- Transformer architecture: embeddings, multi-head attention, feed-forward layers,
  residual connections
- Character-level language model (SmallLM) ready to train on any text corpus
- Wikipedia data pipeline: download, extract, clean, and train
- Training loop with checkpointing, crash recovery, and resume support
- Text generation from trained checkpoints

## Hardware requirements

- Any Vulkan 1.1+ capable GPU (AMD, NVIDIA, Intel - anything with a Vulkan driver)
- Tested on: AMD MI50 32GB (gfx906, RADV driver, Vulkan 1.4.335)
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
git clone https://github.com/yourusername/vtrain.git
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
  tensor.py     - Tensor class with autograd
  functional.py - Differentiable GPU op wrappers
  grad_check.py - Numerical gradient verification
  loss.py       - Loss functions
  optim.py      - SGD and Adam optimizers
  train.py      - Reusable training loop
shaders/        - GLSL compute shader source
compiled/       - SPIR-V compiled shaders (generated at runtime)
models/         - Training runs and checkpoints
tests/          - Test suite
train_wiki.py   - Wikipedia training script
  generate.py     - Text generation from trained checkpoints
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
4. System RAM growth over long training runs is prevented by in-place gradient
   zeroing (allocating new arrays every step slowly fragments glibc's malloc
   arena), computation graph flushing after every backward pass, and periodic
   gc.collect() + malloc_trim() calls. If you see RAM growing unboundedly,
   check that your training loop includes all three.

## License

MIT
