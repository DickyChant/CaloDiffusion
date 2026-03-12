#!/bin/bash
#SBATCH --job-name=tune-glam
#SBATCH --nodes=1
#SBATCH --account=m2612
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --ntasks=1
#SBATCH -G 4
#SBATCH --cpus-per-task=64
#SBATCH --time=03:00:00

# Tune GLaM grid dimensions for photon or pion showers.
# Usage: sbatch slurm/tune_glam.sh photon
#        sbatch slurm/tune_glam.sh pion

PARTICLE=${1:-photon}
TUNE_GLAM_DIR=/pscratch/sd/s/sqian/tune_glam
RESULTS_DIR=${HOME}/glam_tune_results/${PARTICLE}

module load conda
conda activate ${TUNE_GLAM_DIR}/.venv

cd ${TUNE_GLAM_DIR}

python tune_glam.py \
    --config configs/tune_glam_${PARTICLE}.json \
    --results-dir ${RESULTS_DIR} \
    --n-trials 100

echo "Done. Results saved to ${RESULTS_DIR}"
