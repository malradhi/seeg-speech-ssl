#!/bin/bash
#SBATCH --job-name=ieeg_p4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=01:30:00
#SBATCH --array=1-30
#SBATCH --output=logs/p4_%A_%a.out
#SBATCH --error=logs/p4_%A_%a.out
# One task per participant: BrainBERT features, then the Phase 4 runs. Needs the BrainBERT
# checkpoint (scripts/04_download_brainbert.py). Resubmitting is safe: finished work is skipped.
set -eo pipefail
source slurm/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python scripts/16_run_phase4.py --array-index "$SLURM_ARRAY_TASK_ID" --config configs/phase4.json --workers 8
