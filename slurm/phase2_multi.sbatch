#!/bin/bash
#SBATCH --job-name=ieeg_p2m
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=00:45:00
#SBATCH --array=1-35
#SBATCH --output=logs/p2m_%A_%a.out
#SBATCH --error=logs/p2m_%A_%a.out
# Tasks 1-30: leave-one-out + scaling pretraining for participant P01..P30.
# Tasks 31-35: pooled pretraining on all 30 participants (seeds 1-5).
# Resubmitting is safe: finished pretraining is skipped.
set -eo pipefail
source slurm/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python scripts/11_run_phase2_multi.py --array-index "$SLURM_ARRAY_TASK_ID" --config configs/phase2.json
