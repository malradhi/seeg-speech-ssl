#!/bin/bash
#SBATCH --job-name=ieeg_eval4
#SBATCH --partition=test
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000M
#SBATCH --output=logs/eval4_%j.out
#SBATCH --error=logs/eval4_%j.out
# Evaluation suite for the Phase 4 conditions (CPU), 8 participants at a time. Finished
# participants are skipped, so if the job stops early simply submit it again.
set -eo pipefail
source slurm/env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
printf "P%02d\n" $(seq 1 30) | xargs -P 8 -I{} python scripts/15_evaluate.py --participants {} --config configs/phase4_eval.json
echo "participants evaluated: $(ls $IEEG_ROOT/results/eval4/P*.json 2>/dev/null | wc -l) / 30"
echo "EVALUATION FINISHED"
