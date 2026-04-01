# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CaloDiffusion — a PyTorch 2.0+ implementation of diffusion models for calorimeter shower simulation in particle physics. Generates shower events for CaloChallenge and HGCal datasets. The repo name on disk is `triton_sparse_decode` but the package is `calodiffusion`.

## Build & Install

```bash
pip install -e .
```

Requires Python >=3.9. Key deps: torch>=2.0.0, h5py, mlflow, optuna, einops, torchsde.

## CLI Entry Points

```bash
# Training
calodif-train -d DATA-DIR -c CONFIG.json --checkpoint SAVE-DIR diffusion

# Inference / Sampling
calodif-inference --n-events N -c CONFIG.json --model-loc MODEL-PATH sample

# Plotting
calodif-inference -c CONFIG.json plot --generated RESULTS.h5f
```

## Testing

```bash
pip install pytest pytest-dependency
# All tests (excluding HGCal)
python3 -m pytest tests/test_execution.py -m "not hgcal" --data-dir ./test_data/
# HGCal tests only (uses mock random data, no real data needed)
python3 -m pytest tests/test_execution.py -m "hgcal"
# Single test
python3 -m pytest tests/test_execution.py::test_name -v
```

CI runs two separate workflows: `test-calochallenge.yml` (downloads real Zenodo data) and `test-hgcal.yml` (mock data).

## Architecture

### Core Abstractions (all in `calodiffusion/models/`)

- **`Diffusion`** (`diffusion.py`) — Abstract base class. Defines `init_model()`, `forward()`, `__call__()`, `noise_generation()`, `sample()`. Handles noise scheduling, loss computation, and sampling orchestration.
- **`CaloDiffusion`** (`calodiffusion.py`) — Concrete implementation with calorimeter-specific logic (cylindrical coordinates, energy conditioning, shower embeddings).
- **`Loss`** (`loss.py`) — Pluggable loss base class. Loaded dynamically from config (`TRAINING_OBJ`, `LOSS_TYPE`).
- **`Sample`** / **`DDim`** (`sample.py`) — Pluggable sampler base class. Selected via config `SAMPLER` key.

### Neural Network Architectures (`models/models.py`)

- **`ResNet`** — Fully-connected conditional model
- **`CondUnet`** — Convolutional U-Net with time/energy conditioning
- **`CylindricalConv`** / **`CylindricalConvTrans`** — Convolutions respecting phi-circular boundary conditions

### Training (`calodiffusion/train/`)

- **`Train`** (`train.py`) — Abstract base with data loading, checkpoint management
- **`TrainDiffusion`** (`train_diffusion.py`) — Concrete training loop for diffusion models
- **`TrainLayerModel`** (`train_layer_model.py`) — Layer-specific training variant

### Data Pipeline (`calodiffusion/utils/`)

- **`Dataset`** (`dataset.py`) — `IterableDataset` for streaming HDF5 files, multi-worker support
- **`utils.py`** — Device management (`get_device()`), data splitting, model loading
- **`sampling.py`** — Beta schedules (cosine, linear, log), Karras steps, ancestral sampling

### Configuration

Everything is config-driven via JSON files in `calodiffusion/configs/`. Key fields:
- `FILES`/`EVAL`: dataset HDF5 filenames
- `SHAPE_ORIG`/`SHAPE_PAD`/`SHAPE_FINAL`: data reshaping dimensions
- `SAMPLER`, `TRAINING_OBJ`, `LOSS_TYPE`: pluggable component selection
- `NSTEPS`, `BATCH`, `LR`, `MAXEPOCH`: training hyperparameters
- `CYLINDRICAL`, `NOISE_SCHED`, `TIME_EMBED`, `SHOWER_EMBED`: model configuration
- `EMIN`/`EMAX`: energy range for conditioning

### Data Flow

**Training:** JSON config → `TrainDiffusion` → creates `CaloDiffusion` model → training loop (batch → add noise → predict → loss → backprop) → checkpoint

**Inference:** Load checkpoint + config → generate noise → iterative denoising via sampler over `num_steps` → final shower output (HDF5)

### Submodules

- `CaloChallenge/` — Challenge evaluation code
- `HGCalShowers/` — HGCal dataset utilities
- `CMSHGCaloChallenge/` — CMS HGCal challenge code
