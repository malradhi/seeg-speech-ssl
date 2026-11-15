#!/bin/bash
#SBATCH --job-name=ieeg_prep_fast
#SBATCH --partition=test
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000M
#SBATCH --output=logs/prep_fast_%j.out
#SBATCH --error=logs/prep_fast_%j.out
# All 30 participants in one job on the (usually idle) test partition, 6 at a time.
set -eo pipefail
source slurm/env.sh
printf "P%02d\n" $(seq 1 30) | xargs -P 6 -I{} python scripts/02_preprocess.py --participants {}
echo "PREPROCESSING FINISHED"
