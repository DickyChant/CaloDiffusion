#!/bin/bash
#SBATCH --job-name=calodiff_ddp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --account=m2612

# ============================================================================
# NERSC Perlmutter DDP Training Script for CaloDiffusion
# ============================================================================
# 
# Usage:
#   1. Update SBATCH directives above as needed (account, time, nodes, etc.)
#   2. Set CONFIG, DATA_FOLDER, and CHECKPOINT_FOLDER variables below
#   3. Submit with: sbatch perlmutter_ddp.sh
#
# For multi-node training, increase --nodes and adjust MASTER_ADDR logic
# ============================================================================

# Configuration - UPDATE THESE FOR YOUR RUN
CONFIG="/global/homes/s/sqian/calodiff/CaloDiffusion/calodiffusion/configs/config_HGCal_photons_nersc.json"
DATA_FOLDER="/pscratch/sd/s/sqian/HGCal_sim_samples/SinglePhoton"
CHECKPOINT_FOLDER="/pscratch/sd/s/sqian/CaloDiffu_HGCAL/models"
MODEL_TYPE="diffusion"  # Options: diffusion, layer

# Extra arguments (optional)
EXTRA_ARGS=""

# ============================================================================
# Environment Setup
# ============================================================================

# Load conda and activate environment
module load conda
conda activate calo-diff

# Create log directory if it doesn't exist
mkdir -p logs

# Print job information
echo "=============================================="
echo "SLURM Job Information"
echo "=============================================="
echo "Job ID:           $SLURM_JOB_ID"
echo "Job Name:         $SLURM_JOB_NAME"
echo "Node List:        $SLURM_NODELIST"
echo "Num Nodes:        $SLURM_NNODES"
echo "Tasks per Node:   $SLURM_NTASKS_PER_NODE"
echo "GPUs per Node:    $SLURM_GPUS_PER_NODE"
echo "CPUs per Task:    $SLURM_CPUS_PER_TASK"
echo "=============================================="

# ============================================================================
# DDP Configuration
# ============================================================================

# Get master node address (first node in allocation)
MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_ADDR
export MASTER_PORT=29500

echo "Master Address:   $MASTER_ADDR"
echo "Master Port:      $MASTER_PORT"
echo "World Size:       $(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))"
echo "=============================================="

# ============================================================================
# Training Execution
# ============================================================================

cd /global/homes/s/sqian/calodiff/CaloDiffusion

echo "Starting training at $(date)"
echo "Config:           $CONFIG"
echo "Data Folder:      $DATA_FOLDER"
echo "Checkpoint:       $CHECKPOINT_FOLDER"
echo "Model Type:       $MODEL_TYPE"
echo "=============================================="

# Use srun to launch distributed training
# Each task will be assigned to one GPU automatically via SLURM
srun --export=ALL \
    python -m calodiffusion.training \
    --config "$CONFIG" \
    --data-folder "$DATA_FOLDER" \
    --checkpoint "$CHECKPOINT_FOLDER" \
    $EXTRA_ARGS \
    $MODEL_TYPE

exit_code=$?

echo "=============================================="
echo "Training completed at $(date)"
echo "Exit code: $exit_code"
echo "=============================================="

exit $exit_code
