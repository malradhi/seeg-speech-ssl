#!/bin/bash
#SBATCH --job-name=ieeg_contam
#SBATCH --partition=test
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000M
#SBATCH --output=logs/contam_%j.out
#SBATCH --error=logs/contam_%j.out
# Acoustic-contamination audit for all 30 participants, 4 at a time (CPU only).
set -eo pipefail
source slurm/env.sh
printf "P%02d\n" $(seq 1 30) | xargs -P 4 -I{} python scripts/05_contamination.py --participants {}
echo "CONTAMINATION AUDIT FINISHED"
