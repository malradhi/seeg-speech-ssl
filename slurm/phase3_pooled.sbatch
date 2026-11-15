#!/bin/bash
#SBATCH --job-name=ieeg_p3pool
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=00:30:00
#SBATCH --array=1-3
#SBATCH --output=logs/p3pool_%A_%a.out
#SBATCH --error=logs/p3pool_%A_%a.out
# Pooled pretraining of the large Transformer on all 30 participants, one seed per task.
# Resubmitting is safe: a finished seed is skipped.
set -eo pipefail
source slurm/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python scripts/13_run_phase3_pooled.py --array-index "$SLURM_ARRAY_TASK_ID" --config configs/phase3.json
