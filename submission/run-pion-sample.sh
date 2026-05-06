#!/bin/bash
# CMSHGCaloChallenge submission runner — pion shower generation.
#
# Usage: $0 <batch_size> <n_samples> <energy>
#   batch_size : sampling batch size (e.g. 100)
#   n_samples  : total number of showers to generate
#   energy     : 5, 50, or 500 (GeV)
#
# Output: ./test_generation_calodif_pion_E<energy>.h5
#
# Tarball layout assumed:
#   ./container.sif
#   ./run-pion-sample.sh   (this file)
#   ./configs/HGCal_pions.json
#   ./CaloDiffusion/             ← package source
#   ./CaloDiffusion/HGCalShowers/geoms/HGCal_geo_2024_large.pkl
#   ./CaloDiffusion/checkpoints/HGCal_pion_oct17_Diffusion/checkpoint.pth
#   ./CaloDiffusion/checkpoints/HGCal_pion_oct17_LayerModel/checkpoint.pth

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Error: missing required arguments" >&2
    echo "Usage: $0 <batch_size> <n_samples> <energy>" >&2
    exit 1
fi

BATCH="$1"
N_SAMPLES="$2"
ENERGY="$3"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF="$HERE/container.sif"
SRC="$HERE/CaloDiffusion"
CONFIG="$HERE/configs/HGCal_pions.json"
DIFFU_CKPT="$SRC/checkpoints/HGCal_pion_oct17_Diffusion/checkpoint.pth"
LAYER_CKPT="$SRC/checkpoints/HGCal_pion_oct17_LayerModel/checkpoint.pth"
OUT="$HERE/test_generation_calodif_pion_E${ENERGY}.h5"

for f in "$SIF" "$CONFIG" "$DIFFU_CKPT" "$LAYER_CKPT"; do
    test -f "$f" || { echo "Missing file: $f" >&2; exit 2; }
done

echo "[pion] batch=$BATCH n_samples=$N_SAMPLES energy=${ENERGY} GeV → $OUT"

# Editable install + sample. Sparse decoding routes through the Triton CSC kernel
# automatically (see CaloDiffusion/calodiffusion/utils/HGCal_utils.py:Decoder.forward).
# --sparse-per-batch matches how the pion model was trained.
apptainer exec --nv --pwd "$SRC" -B "$HERE" "$SIF" pip install --no-deps -e .
apptainer exec --nv --pwd "$SRC" -B "$HERE" "$SIF" \
    calodif-inference \
        --n-events "$N_SAMPLES" \
        -c "$CONFIG" \
        --hgcal \
        sample \
            --model-loc "$DIFFU_CKPT" \
            --batch-size "$BATCH" \
            --energy "$ENERGY" \
            --sample-algo DDim \
            --sample-steps 200 \
            --sparse-decoding \
            --sparse-per-batch \
            --generated "$OUT" \
            layer \
                --layer-model "$LAYER_CKPT"
