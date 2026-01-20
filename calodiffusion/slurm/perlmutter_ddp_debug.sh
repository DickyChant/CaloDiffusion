#!/bin/bash
#SBATCH --job-name=calodiff_ddp_debug
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --account=m2612

# ============================================================================
# NERSC Perlmutter 2-GPU DDP Debug Script for CaloDiffusion
# ============================================================================
# 
# Uses debug queue (30 min max) for quick DDP testing with 2 GPUs
# Usage: sbatch perlmutter_ddp_debug.sh
# ============================================================================

# Configuration
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

# DDP Configuration
MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_ADDR
export MASTER_PORT=29500

echo "=============================================="
echo "DDP Debug Session (2 GPUs)"
echo "=============================================="
echo "Job ID:           $SLURM_JOB_ID"
echo "Node:             $SLURM_NODELIST"
echo "Tasks per Node:   $SLURM_NTASKS_PER_NODE"
echo "Master Address:   $MASTER_ADDR"
echo "Master Port:      $MASTER_PORT"
echo "World Size:       $SLURM_NTASKS"
echo "=============================================="

# ============================================================================
# Training Execution
# ============================================================================

cd /global/homes/s/sqian/calodiff/CaloDiffusion

echo "Starting 2-GPU DDP debug training at $(date)"

srun --export=ALL \
    python -m calodiffusion.training \
    --config "$CONFIG" \
    --data-folder "$DATA_FOLDER" \
    --checkpoint "$CHECKPOINT_FOLDER" \
    $MODEL_TYPE

echo "Training completed at $(date) with exit code: $?"
