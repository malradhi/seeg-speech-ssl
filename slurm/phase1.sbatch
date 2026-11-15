#!/bin/bash
#SBATCH --job-name=ieeg_p1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=01:00:00
#SBATCH --array=1-30
#SBATCH --output=logs/p1_%A_%a.out
#SBATCH --error=logs/p1_%A_%a.out
# One array task per participant; 8 worker processes share one A100.
# Resubmitting is safe: finished runs are skipped.
set -eo pipefail
source slurm/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python scripts/10_run_phase1.py --array-index "$SLURM_ARRAY_TASK_ID" \
    --config configs/phase1.json --phase phase1 --workers 8
