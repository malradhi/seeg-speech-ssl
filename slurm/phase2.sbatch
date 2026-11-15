#!/bin/bash
#SBATCH --job-name=ieeg_p2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=02:00:00
#SBATCH --array=1-30
#SBATCH --output=logs/p2_%A_%a.out
#SBATCH --error=logs/p2_%A_%a.out
# One task per participant; needs ALL phase2_multi tasks finished (submit with
# --dependency=afterok:<multi job id>, see docs/RUNNING_THE_STUDY.md, step 6). Resubmitting is safe.
set -eo pipefail
source slurm/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python scripts/12_run_phase2.py --array-index "$SLURM_ARRAY_TASK_ID" --config configs/phase2.json --workers 8
