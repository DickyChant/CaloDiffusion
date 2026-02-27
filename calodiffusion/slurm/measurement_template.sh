#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH --output=__JOB_OUT__/slurm_%j.out
#SBATCH --error=__JOB_OUT__/slurm_%j.err
#SBATCH --partition=__PARTITION__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=__MEMORY__
#SBATCH --constraint=(__CONSTRAINT__)
#SBATCH --time=__TIME_LIMIT__
__ACCOUNT_LINE__

set -euo pipefail
unset LD_PRELOAD || true

REPO_DIR=__REPO_DIR__
DATA_DIR=__DATA_DIR__
N_EVENTS=__N_EVENTS__
SAMPLE_STEPS=__SAMPLE_STEPS__
BATCH_SIZES_VALUE=__BATCH_SIZES__
AUTO_PREPARE_DUMMY_VALUE=__AUTO_PREPARE_DUMMY__
PHOTON_CONFIG_VALUE=__PHOTON_CONFIG__
PION_CONFIG_VALUE=__PION_CONFIG__
PHOTON_MODEL_VALUE=__PHOTON_MODEL__
PION_MODEL_VALUE=__PION_MODEL__
CONDA_ENV_VALUE=__CONDA_ENV__
OUTPUT_ROOT_VALUE=__OUTPUT_ROOT__

OUT_ROOT="${OUTPUT_ROOT_VALUE}/__JOB_NAME__/${SLURM_JOB_ID}"
mkdir -p "$OUT_ROOT"

if [[ -n "$CONDA_ENV_VALUE" && "$CONDA_ENV_VALUE" != "none" ]]; then
    if command -v module >/dev/null 2>&1; then
        module load conda || module load python || true
    fi
    if [[ -f "$HOME/.bashrc" ]]; then
        source "$HOME/.bashrc"
    fi
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base 2>/dev/null || true)"
        if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
            source "$CONDA_BASE/etc/profile.d/conda.sh"
        fi
        conda activate "$CONDA_ENV_VALUE"
    else
        echo "WARNING: conda not found; continuing without conda env activation."
    fi
fi

cd "$REPO_DIR"

export WORK_DIR="$OUT_ROOT"
export LOGFILE="$OUT_ROOT/inference_measurement.log"
export BATCH_SIZES="$BATCH_SIZES_VALUE"
export AUTO_PREPARE_DUMMY="$AUTO_PREPARE_DUMMY_VALUE"

if [[ -n "$PHOTON_CONFIG_VALUE" ]]; then
    export PHOTON_CONFIG="$PHOTON_CONFIG_VALUE"
fi
if [[ -n "$PION_CONFIG_VALUE" ]]; then
    export PION_CONFIG="$PION_CONFIG_VALUE"
fi
if [[ -n "$PHOTON_MODEL_VALUE" ]]; then
    export PHOTON_MODEL="$PHOTON_MODEL_VALUE"
fi
if [[ -n "$PION_MODEL_VALUE" ]]; then
    export PION_MODEL="$PION_MODEL_VALUE"
fi

echo "Running measurement at $(date)"
echo "WORK_DIR=$WORK_DIR"
echo "DATA_DIR=$DATA_DIR"
echo "N_EVENTS=$N_EVENTS SAMPLE_STEPS=$SAMPLE_STEPS BATCH_SIZES=$BATCH_SIZES"

bash "$REPO_DIR/measurement.sh" "$DATA_DIR" "$N_EVENTS" "$SAMPLE_STEPS"

echo "Measurement completed at $(date)"
echo "Artifacts: $WORK_DIR"
