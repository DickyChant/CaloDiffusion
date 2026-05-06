# CMSHGCaloChallenge — CaloDiffusion submission (Triton sparse decoding)

Triton-accelerated sparse decoding variant of the CaloDiffusion sampler. The
HGCal `Decoder` automatically routes through a fused Triton CSC kernel
(`calodiffusion/utils/triton_sparse.py:sparse_decode_csc`) when CUDA + Triton
are available; the kernel is ~6× faster than the dense sparse-mat path on
pion (large `N`) and amortizes the per-shape precompute via a cache on the
`Decoder` instance.

## Tarball layout

The submitted `.tar.gz` must contain:

```
.
├── container.def
├── container.sif        ← built by you (apptainer build --fakeroot)
├── run-photon-sample.sh
├── run-pion-sample.sh
├── README.md
└── CaloDiffusion/       ← the runtime code (this repo, with submodules init'd)
    ├── calodiffusion/
    ├── HGCalShowers/    ← submodule
    ├── pyproject.toml
    └── checkpoints/
        ├── checkpoint_HGCal_pions.pth
        └── checkpoint_HGCal_photons.pth
```

## Build the image

```
cd submission/
apptainer build --fakeroot container.sif container.def
```

## Run the sample scripts

```
./run-pion-sample.sh   <batch_size> <n_samples> <energy>     # energy ∈ {5,50,500}
./run-photon-sample.sh <batch_size> <n_samples> <energy>
```

Outputs land alongside the script as `test_generation_calodif_{pion,photon}_E<energy>.h5`.

The runners do `pip install --no-deps -e CaloDiffusion/` inside the container
before sampling — this is a no-op on second invocation. Triton + CUDA are
required at runtime; `--nv` is passed to apptainer to expose the GPU.

## Packaging

```
tar czf calodiffusion-triton-submission.tar.gz \
    container.def container.sif \
    run-pion-sample.sh run-photon-sample.sh README.md CaloDiffusion/
```
