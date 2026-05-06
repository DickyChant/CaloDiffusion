# CaloDiffusion — Triton-accelerated sparse decoding container
#
# Base: PyTorch 2.5.1 + CUDA 12.4 + cuDNN9 (Triton 3.1 ships in the wheel).
# Out of the box on any Volta+ GPU.
#
# Build (run from repo root, with HGCalShowers/ and CMSHGCaloChallenge/
# submodules populated):
#   git submodule update --init --recursive
#   docker build -t calodiffusion:triton .
#
# Sample interactively, with host data + a trained model mounted:
#   docker run --rm -it --gpus all \
#       -v $PWD/data:/workspace/data \
#       -v $PWD/trained_models:/workspace/models \
#       calodiffusion:triton bash
#
# Generate a CMSHGCaloChallenge submission (5/50/500 GeV) in one shot:
#   docker run --rm --gpus all \
#       -v $PWD/trained_models:/workspace/models \
#       -v $PWD/submission:/workspace/CaloDiffusion/submission \
#       -e CONFIG=calodiffusion/configs/config_HGCal_pions.json \
#       -e MODEL_DIR=/workspace/models/my_pion_run \
#       -e N_EVENTS=50000 -e BATCH_SIZE=128 \
#       calodiffusion:triton submit
#
# The two SSH-only submodules (CaloChallenge, CMSHGCaloChallenge) need to be
# populated on the host before `docker build`, since the image cannot use the
# host's SSH keys. HGCalShowers is public and is also pulled in via the host
# checkout. If a submodule directory is empty, the build will fail loudly at
# the verification step rather than producing a broken image.

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

# Python deps first for layer-cache friendliness.
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && \
    pip install \
        "dask>=2023.3.0" "gpyopt>=1.2.6" "mlflow>=1.26.1" "pydot>=1.2.4" \
        "tables>=3.9.2" "h5py>=3.11.0" "einops>=0.8.0" "scikit-learn>=1.5.2" \
        "torchinfo>=1.8.0" "optuna>=4.0.0" "fvcore>=0.1.5" "torchsde>=0.2.6" \
        "click>=8.0.1" "mplhep>=0.3.57" "pytest" "pytest-dependency"

# Project + populated submodules from the host.
COPY . .

RUN pip install --no-deps -e .

# Verify submodules are populated and Triton wires up.
RUN test -f HGCalShowers/HGCalGeo.py \
    || (echo "HGCalShowers submodule is empty — run 'git submodule update --init --recursive' on the host before building" && exit 1)
RUN test -f CMSHGCaloChallenge/hgcal_metrics.py \
    || (echo "CMSHGCaloChallenge submodule is empty — run 'git submodule update --init --recursive' on the host before building" && exit 1)
RUN python -c "import torch; print('torch', torch.__version__); \
import triton; print('triton', triton.__version__); \
from calodiffusion.utils.triton_sparse import HAS_TRITON; \
assert HAS_TRITON, 'triton import failed'; print('triton_sparse OK')"

# Convenience entrypoint: 'submit' runs the 3-energy submission, anything else
# is exec'd directly so the image still behaves like a generic shell container.
COPY scripts/run_submission.sh /usr/local/bin/calodif-submit
RUN chmod +x /usr/local/bin/calodif-submit

ENTRYPOINT ["/bin/bash", "-c", "if [ \"$1\" = submit ]; then exec calodif-submit; else exec \"$@\"; fi", "--"]
CMD ["bash"]
