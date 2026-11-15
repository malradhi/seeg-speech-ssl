#!/bin/bash
#SBATCH --job-name=ieeg_p4b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6000M
#SBATCH --time=02:00:00
#SBATCH --array=1-30
#SBATCH --output=logs/p4b_%A_%a.out
#SBATCH --error=logs/p4b_%A_%a.out
# One task per participant: self-pretraining, the 1e-2 grid edge, then BrainBERT with 64
# components per contact (re-extracted; ~0.6 GB per participant in derived_brainbert_k64).
# Expected 45-75 min per task. Resubmitting is safe: finished work is skipped.
set -eo pipefail
source slurm/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python scripts/17_run_phase4b.py --array-index "$SLURM_ARRAY_TASK_ID" --config configs/phase4b.json --workers 8
