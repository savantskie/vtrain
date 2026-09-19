#!/bin/bash
set -e

echo "Building Kompute from source..."

# Check dependencies
for cmd in git cmake c++ python3; do
    if ! command -v "$cmd" &> /dev/null; then
        echo "Error: $cmd is required but not installed."
        exit 1
    fi
done

# Check for Vulkan
if ! python3 -c "import ctypes; ctypes.CDLL('libvulkan.so.1')" &> /dev/null; then
    echo "Error: libvulkan not found. Install vulkan-tools and libvulkan-dev."
    exit 1
fi

# Find Python being used
PYTHON=$(which python3)
VENV_PYTHON="$(pwd)/.venv/bin/python3"
if [ -f "$VENV_PYTHON" ]; then
    PYTHON="$VENV_PYTHON"
fi

echo "Using Python: $PYTHON"

# Clone if not already present
if [ ! -d "kompute-src" ]; then
    git clone https://github.com/KomputeProject/kompute.git kompute-src
    cd kompute-src
    git submodule update --init --recursive
    cd ..
else
    echo "kompute-src already exists, skipping clone."
fi

# Build
cd kompute-src
mkdir -p build && cd build

cmake .. \
    -DKOMPUTE_OPT_BUILD_PYTHON=ON \
    -DKOMPUTE_OPT_LOG_LEVEL=Off \
    -DKOMPUTE_OPT_USE_SPDLOG=Off \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE="$PYTHON"

make -j$(nproc)

cd ../..

# Find the .so and copy it to the venv
SO=$(find kompute-src/build -name "kp.cpython-*.so" | head -1)
if [ -z "$SO" ]; then
    echo "Error: could not find compiled kp .so file."
    exit 1
fi

SITE_PACKAGES=$("$PYTHON" -c "import site; print(site.getsitepackages()[0])")
cp "$SO" "$SITE_PACKAGES/"

echo "Kompute installed to $SITE_PACKAGES/$(basename $SO)"
echo "Build complete."
