#!/usr/bin/env bash
# Generate CMSHGCaloChallenge submission showers at the three fixed energies
# (5, 50, 500 GeV) using the Triton-accelerated sparse decoder.
#
# Required env:
#   CONFIG       path to a HGCal CaloDiffusion config json (pion or photon)
#   MODEL_DIR    directory containing the trained checkpoint
#
# Optional env (with defaults):
#   N_EVENTS     events per energy bucket           [50000]
#   BATCH_SIZE   sampling batch size                [128]
#   SAMPLE_STEPS denoising steps                    [200]
#   SAMPLE_ALGO  sampler                            [DDim]
#   OUT_DIR      where to drop generated h5 files   [./submission]
#   ENERGIES     space-separated energy buckets     [5 50 500]
#   EXTRA        extra flags passed through to `calodif-inference sample`
#
# Output layout:
#   $OUT_DIR/E5/generated_*.h5
#   $OUT_DIR/E50/generated_*.h5
#   $OUT_DIR/E500/generated_*.h5

set -euo pipefail

: "${CONFIG:?CONFIG path is required (e.g. calodiffusion/configs/config_HGCal_pions.json)}"
: "${MODEL_DIR:?MODEL_DIR path is required}"

N_EVENTS="${N_EVENTS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SAMPLE_STEPS="${SAMPLE_STEPS:-200}"
SAMPLE_ALGO="${SAMPLE_ALGO:-DDim}"
OUT_DIR="${OUT_DIR:-./submission}"
ENERGIES="${ENERGIES:-5 50 500}"
EXTRA="${EXTRA:-}"

mkdir -p "$OUT_DIR"

for E in $ENERGIES; do
  sub="$OUT_DIR/E${E}"
  mkdir -p "$sub"
  out="$sub/generated_E${E}.h5"
  echo "=== Sampling E=${E} GeV → $out (n=$N_EVENTS, batch=$BATCH_SIZE, steps=$SAMPLE_STEPS) ==="
  calodif-inference \
    -c "$CONFIG" \
    --n-events "$N_EVENTS" \
    sample \
      --model-loc "$MODEL_DIR" \
      --sample-algo "$SAMPLE_ALGO" \
      --sample-steps "$SAMPLE_STEPS" \
      --batch-size "$BATCH_SIZE" \
      --energy "$E" \
      --sparse-decoding \
      --generated "$out" \
      $EXTRA
done

echo "=== Done. Submission files under $OUT_DIR ==="
ls -lh "$OUT_DIR"/*/*.h5
