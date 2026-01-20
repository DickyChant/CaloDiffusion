#!/bin/bash
#SBATCH --job-name=calodiff_1gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --account=m4707

# ============================================================================
# NERSC Perlmutter Single-GPU Training Script for CaloDiffusion
# ============================================================================
# 
# Usage: sbatch perlmutter_single_gpu.sh
# ============================================================================

# Configuration - UPDATE THESE FOR YOUR RUN
CONFIG="/global/homes/s/sqian/calodiff/CaloDiffusion/calodiffusion/configs/config_HGCal_photons_nersc.json"
DATA_FOLDER="/pscratch/sd/s/sqian/HGCal_sim_samples/SinglePhoton"
CHECKPOINT_FOLDER="/pscratch/sd/s/sqian/CaloDiffu_HGCAL/models"
MODEL_TYPE="diffusion"

# ============================================================================
# Environment Setup
# ============================================================================

module load conda
conda activate calo-diff

mkdir -p logs

echo "=============================================="
echo "SLURM Job Information"
echo "=============================================="
echo "Job ID:           $SLURM_JOB_ID"
echo "Node:             $SLURM_NODELIST"
echo "GPUs:             $SLURM_GPUS"
echo "=============================================="

# ============================================================================
# Training Execution (Single GPU - No DDP)
# ============================================================================

cd /global/homes/s/sqian/calodiff/CaloDiffusion

echo "Starting single-GPU training at $(date)"

python -m calodiffusion.training \
    --config "$CONFIG" \
    --data-folder "$DATA_FOLDER" \
    --checkpoint "$CHECKPOINT_FOLDER" \
    $MODEL_TYPE

echo "Training completed at $(date)"
