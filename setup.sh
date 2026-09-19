#!/bin/bash
set -e

echo "================================================"
echo " VTrain - Vulkan ML Training Framework"
echo " Setup Script"
echo "================================================"
echo ""

# Check OS
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Warning: This framework was built and tested on Linux."
    echo "Other platforms are untested. Proceeding anyway..."
    echo ""
fi

# Check for required system packages
echo "Checking system dependencies..."
MISSING=()

for cmd in git cmake c++ python3 glslc; do
    if ! command -v "$cmd" &> /dev/null; then
        MISSING+=("$cmd")
    fi
done

if ! python3 -c "import ctypes; ctypes.CDLL('libvulkan.so.1')" &> /dev/null; then
    MISSING+=("libvulkan (install libvulkan-dev)")
fi

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "Error: the following required dependencies are missing:"
    for m in "${MISSING[@]}"; do
        echo "  - $m"
    done
    echo ""
    echo "On Ubuntu/Debian:"
    echo "  sudo apt install git cmake build-essential python3 python3-venv glslang-tools vulkan-tools libvulkan-dev"
    exit 1
fi

echo "All system dependencies found."
echo ""

# Check for Vulkan devices
echo "Checking for Vulkan devices..."
if ! vulkaninfo --summary &> /dev/null; then
    echo "Error: no Vulkan devices found. Check your GPU drivers."
    exit 1
fi

DEVICE_COUNT=$(vulkaninfo --summary 2>/dev/null | grep -c "deviceName" || true)
echo "Found $DEVICE_COUNT Vulkan device(s)."
echo ""

# Create virtual environment
echo "Creating Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Virtual environment created at .venv/"
else
    echo ".venv already exists, skipping."
fi

source .venv/bin/activate

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "Python dependencies installed."

# Build Kompute
echo ""
echo "Building Kompute (this takes a few minutes)..."
bash build_kompute.sh

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "
import kp
import numpy as np
mgr = kp.Manager(0)
print('Kompute OK - device 0 initialized')
" && echo "Installation verified." || echo "Warning: verification failed. Check output above."

echo ""
echo "================================================"
echo " Setup complete."
echo ""
echo " Activate your environment:"
echo "   source .venv/bin/activate"
echo ""
echo " Download Wikipedia data (optional):"
echo "   wget https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
echo "   python3 vtrain/data/wiki_extract.py --dump simplewiki-latest-pages-articles.xml.bz2 --output simplewiki.txt"
echo ""
echo " Train a model:"
echo "   python3 train_wiki.py --data simplewiki.txt --run-dir models/run1"
echo ""
echo " Run tests:"
echo "   python3 -m pytest tests/"
echo "================================================"
