#!/bin/bash
# set -euo pipefail

# ENV_PREFIX="$PWD/.venv"

module load conda

if [ -d "$ENV_PREFIX" ]; then
    echo "Environment already exists at $ENV_PREFIX, activating..."
else
    echo "Creating conda environment at $ENV_PREFIX ..."
    conda create --prefix "$ENV_PREFIX" python=3.12 -y
fi

conda activate "$ENV_PREFIX"

# PyTorch with CUDA (Perlmutter A100s, CUDA 12.x); bundles triton automatically
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install project in editable mode (pulls in all deps from pyproject.toml)
pip install -e .

# Extra deps for benchmarking
pip install triton

# Init submodules (needed for HGCalGeo geometry files)
git submodule update --init --recursive

echo ""
echo "=== Verification ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import triton; print(f'Triton {triton.__version__}')"
python -c "from calodiffusion.utils.triton_sparse import HAS_TRITON; print(f'triton_sparse HAS_TRITON={HAS_TRITON}')"
echo "Done. Activate with: module load conda && conda activate $ENV_PREFIX"
