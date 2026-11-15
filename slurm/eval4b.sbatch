#!/bin/bash
#SBATCH --job-name=ieeg_eval4b
#SBATCH --partition=test
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000M
#SBATCH --array=1-2
#SBATCH --output=logs/eval4b_%A_%a.out
#SBATCH --error=logs/eval4b_%A_%a.out
# Evaluation suite (CPU) on the seven-rate selections and the Phase 4b conditions. Two tasks
# (P01-P15, P16-P30), 8 participants at a time; expected 20-35 min each. Finished participants
# are skipped, so if a task stops early simply submit the job again.
set -eo pipefail
source slurm/env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
first=$(( (SLURM_ARRAY_TASK_ID - 1) * 15 + 1 ))
printf "P%02d\n" $(seq $first $((first + 14))) | xargs -P 8 -I{} python scripts/15_evaluate.py --participants {} --config configs/phase4b_eval.json
echo "participants evaluated: $(ls $IEEG_ROOT/results/eval4b/P*.json 2>/dev/null | wc -l) / 30"
echo "EVALUATION FINISHED"
