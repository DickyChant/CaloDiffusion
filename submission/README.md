# CMSHGCaloChallenge — CaloDiffusion submission (Triton sparse decoding)

Triton-accelerated sparse-decoding variant of the CaloDiffusion sampler. The
HGCal `Decoder` automatically routes through the fused Triton CSC kernel
(`calodiffusion/utils/triton_sparse.py:sparse_decode_csc`) when CUDA + Triton
are available; the kernel only touches the ~5 nonzero entries per column
(out of N=22268 for pion) and amortizes its precompute via a per-Decoder
cache keyed on the underlying tensor's `data_ptr`. Trainable Decoders fall
back to the tiled fused kernel.

## Tarball layout

The submitted `.tar.gz` contains:

```
.
├── container.def
├── container.sif                ← built by you
├── run-photon-sample.sh
├── run-pion-sample.sh
├── README.md
├── configs/
│   ├── HGCal_photons.json       ← submission-friendly (relative BIN_FILE)
│   └── HGCal_pions.json
└── CaloDiffusion/               ← the runtime code (this repo)
    ├── calodiffusion/
    ├── HGCalShowers/            ← submodule, populated
    ├── pyproject.toml
    └── checkpoints/
        ├── HGCal_photon_april14_Diffusion/checkpoint.pth
        ├── HGCal_photon_april14_LayerModel/checkpoint.pth
        ├── HGCal_pion_oct17_Diffusion/checkpoint.pth
        └── HGCal_pion_oct17_LayerModel/checkpoint.pth
```

## End-to-end build

From the CaloDiffusion repo root, with checkpoints under `./checkpoints/`:

```bash
# 1. Build the apptainer image (one-off; ~3.4 GB)
cd submission/
apptainer build --fakeroot container.sif container.def

# 2. Stage runtime code + checkpoints and bundle everything into a tarball
./package.sh
```

`package.sh` rsyncs the repo into `submission/CaloDiffusion/`, copies the
four checkpoint dirs, and produces `calodiffusion-triton-submission.tar.gz`.

## Run the sample scripts

```bash
./run-pion-sample.sh   <batch_size> <n_samples> <energy>     # energy ∈ {5,50,500}
./run-photon-sample.sh <batch_size> <n_samples> <energy>
```

Recommended batch sizes (matching the training configs):
- photon: `--batch-size 80`
- pion:   `--batch-size 100` (uses `--sparse-per-batch` since that's how the model was trained)

Outputs land alongside the script as
`test_generation_calodif_{pion,photon}_E<energy>.h5`.

The runners do `pip install --no-deps -e .` inside the container before
sampling — a no-op on subsequent invocations. `--nv` exposes the GPU; `--pwd`
sets the apptainer working directory to `CaloDiffusion/` so the
relative `BIN_FILE` paths in the configs resolve.
