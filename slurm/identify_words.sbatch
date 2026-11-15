#!/bin/bash
#SBATCH --job-name=ieeg_ident5w
#SBATCH --partition=test
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000M
#SBATCH --output=logs/ident5w_%j.out
#SBATCH --error=logs/ident5w_%j.out
# Phase 5 rerun with word labels (CPU): identification plus accuracy over pairs of different words,
# 8 participants at a time; measured 4 min for all 30. Finished participants are skipped, so if
# the job stops early simply submit it again.
set -eo pipefail
source slurm/env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
printf "P%02d\n" $(seq 1 30) | xargs -P 8 -I{} python scripts/23_identify.py --participants {} --config configs/phase5_words.json
echo "participants identified: $(ls $IEEG_ROOT/results/ident5w/P*.json 2>/dev/null | wc -l) / 30"
echo "PHASE5 WORDS FINISHED"
