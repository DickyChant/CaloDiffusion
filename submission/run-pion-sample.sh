#!/bin/bash
# CMSHGCaloChallenge submission runner — pion shower generation.
#
# Usage: $0 <batch_size> <n_samples> <energy>
#   batch_size : sampling batch size (e.g. 128)
#   n_samples  : total number of showers to generate
#   energy     : 5, 50, or 500 (GeV)
#
# Output: ./test_generation_calodif_pion_E<energy>.h5
#
# Layout assumed inside the submission tarball:
#   ./container.sif
#   ./run-pion-sample.sh   (this file)
#   ./CaloDiffusion/       (the package source tree)
#   ./CaloDiffusion/checkpoints/checkpoint_HGCal_pions.pth

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

CONFIG="$SRC/calodiffusion/configs/config_HGCal_pions.json"
CHECKPOINT="$SRC/checkpoints/checkpoint_HGCal_pions.pth"
OUT="$HERE/test_generation_calodif_pion_E${ENERGY}.h5"

echo "[pion] batch=$BATCH n_samples=$N_SAMPLES energy=${ENERGY} GeV → $OUT"

# Editable install + sample. Sparse decoding routes through the Triton CSC kernel
# automatically (see calodiffusion/utils/HGCal_utils.py:Decoder.forward).
apptainer exec --nv -B "$HERE" "$SIF" pip install --no-deps -e "$SRC"
apptainer exec --nv -B "$HERE" "$SIF" \
    calodif-inference \
        --n-events "$N_SAMPLES" \
        -c "$CONFIG" \
        --hgcal \
        sample \
            --model-loc "$CHECKPOINT" \
            --batch-size "$BATCH" \
            --energy "$ENERGY" \
            --sparse-decoding \
            --generated "$OUT"
