#!/bin/bash
#SBATCH --job-name=ieeg_prep
#SBATCH --partition=cpu
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2000M
#SBATCH --array=1-30
#SBATCH --output=logs/prep_%A_%a.out
#SBATCH --error=logs/prep_%A_%a.out
# One array task per participant (P01..P30). Submit from the repository root.
set -eo pipefail
source slurm/env.sh
python scripts/02_preprocess.py --array-index "$SLURM_ARRAY_TASK_ID"
