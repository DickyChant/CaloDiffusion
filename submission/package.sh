#!/bin/bash
# Build a CMSHGCaloChallenge submission tarball from the current repo state.
#
# Produces:
#   submission/CaloDiffusion/             ← copy of this repo (no .git, no submission/)
#   submission/CaloDiffusion/checkpoints/ ← copied/symlinked from $CHECKPOINTS_SRC
#   submission/calodiffusion-triton-submission.tar.gz
#
# Required env (overridable):
#   REPO_ROOT       absolute path to the CaloDiffusion repo  [auto-detected]
#   CHECKPOINTS_SRC absolute path to host checkpoints dir    [REPO_ROOT/checkpoints]
#   SIF             absolute path to container.sif           [submission/container.sif]
#   OUT             tarball name                             [calodiffusion-triton-submission.tar.gz]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$HERE/.." && pwd)}"
CHECKPOINTS_SRC="${CHECKPOINTS_SRC:-$REPO_ROOT/checkpoints}"
SIF="${SIF:-$HERE/container.sif}"
OUT="${OUT:-calodiffusion-triton-submission.tar.gz}"

test -f "$SIF" || { echo "container.sif missing — build it first: 'apptainer build --fakeroot $SIF $HERE/container.def'" >&2; exit 1; }
test -d "$CHECKPOINTS_SRC" || { echo "Checkpoints dir not found: $CHECKPOINTS_SRC" >&2; exit 1; }

STAGE="$HERE/CaloDiffusion"
echo "Staging CaloDiffusion source → $STAGE"
rm -rf "$STAGE"
rsync -a \
    --exclude='.git' \
    --exclude='submission' \
    --exclude='__pycache__' \
    --exclude='.venv' \
    --exclude='.triton*' \
    --exclude='*.pyc' \
    --exclude='checkpoints' \
    --exclude='trained_models' \
    --exclude='data' \
    "$REPO_ROOT/" "$STAGE/"

echo "Linking checkpoints → $STAGE/checkpoints"
mkdir -p "$STAGE/checkpoints"
for sub in HGCal_photon_april14_Diffusion HGCal_photon_april14_LayerModel \
           HGCal_pion_oct17_Diffusion HGCal_pion_oct17_LayerModel; do
    src="$CHECKPOINTS_SRC/$sub"
    test -d "$src" || { echo "Missing checkpoint subdir: $src" >&2; exit 1; }
    test -f "$src/checkpoint.pth" || { echo "Missing checkpoint.pth in: $src" >&2; exit 1; }
    cp -r "$src" "$STAGE/checkpoints/"
done

echo "Building tarball → $HERE/$OUT"
cd "$HERE"
tar czf "$OUT" \
    container.def \
    container.sif \
    run-pion-sample.sh run-photon-sample.sh \
    README.md \
    configs \
    CaloDiffusion

echo "Done: $HERE/$OUT ($(du -h "$OUT" | cut -f1))"
