#!/bin/bash
#SBATCH --job-name=ieeg_eval
#SBATCH --partition=test
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000M
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.out
# Evaluation suite (CPU) for all 30 participants, 8 at a time. Needs every Phase 3
# participant task finished. Finished participants are skipped, so if the job hits the
# time limit simply submit it again.
set -eo pipefail
source slurm/env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
printf "P%02d\n" $(seq 1 30) | xargs -P 8 -I{} python scripts/15_evaluate.py --participants {} --config configs/phase3.json
echo "participants evaluated: $(ls $IEEG_ROOT/results/eval/P*.json 2>/dev/null | wc -l) / 30"
echo "EVALUATION FINISHED"
