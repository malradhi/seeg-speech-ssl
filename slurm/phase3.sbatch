#!/bin/bash
#SBATCH --job-name=ieeg_p3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=01:30:00
#SBATCH --array=1-30
#SBATCH --output=logs/p3_%A_%a.out
#SBATCH --error=logs/p3_%A_%a.out
# One task per participant; needs all three phase3_pooled tasks finished (submit with
# --dependency=afterok:<pooled job id>, see KOMONDOR.md, step 7). Resubmitting is safe.
set -eo pipefail
source slurm/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python scripts/14_run_phase3.py --array-index "$SLURM_ARRAY_TASK_ID" --config configs/phase3.json --workers 8
