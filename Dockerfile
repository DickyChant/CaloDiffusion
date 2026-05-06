# CaloDiffusion — Triton-accelerated sparse decoding container
#
# Base: NVIDIA PyTorch image (CUDA 12.4, cuDNN, NCCL, Python 3.12). PyTorch
# wheels here ship Triton bundled, so the triton_sparse kernels run out of the
# box on any Volta+ GPU.
#
# Build:
#   docker build -t calodiffusion:triton .
#
# Run (interactive, with a host data dir mounted):
#   docker run --rm -it --gpus all \
#       -v $PWD/data:/workspace/data \
#       calodiffusion:triton bash
#
# The two SSH-only submodules (CaloChallenge, CMSHGCaloChallenge) are NOT
# initialized in the image — clone them on the host and bind-mount, or run
# `git submodule update --init` inside the container with a forwarded SSH
# agent. The public HGCalShowers submodule is initialized at build time.

ARG PYTORCH_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/CaloDiffusion

# Install Python deps first to maximize layer cache hits.
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && \
    pip install \
        "dask>=2023.3.0" "gpyopt>=1.2.6" "mlflow>=1.26.1" "pydot>=1.2.4" \
        "tables>=3.9.2" "h5py>=3.11.0" "einops>=0.8.0" "scikit-learn>=1.5.2" \
        "torchinfo>=1.8.0" "optuna>=4.0.0" "fvcore>=0.1.5" "torchsde>=0.2.6" \
        "click>=8.0.1" "mplhep>=0.3.57" "pytest" "pytest-dependency"

# Pull only the public submodule via HTTPS so the build needs no SSH key.
COPY .gitmodules ./
RUN git init -q && \
    git submodule add -f https://github.com/OzAmram/HGCalShowers HGCalShowers || true

COPY . .

RUN pip install --no-deps -e .

# Quick sanity check at build time.
RUN python -c "import torch; print('torch', torch.__version__); \
import triton; print('triton', triton.__version__); \
from calodiffusion.utils.triton_sparse import HAS_TRITON; \
assert HAS_TRITON, 'triton import failed'; print('triton_sparse OK')"

CMD ["bash"]
