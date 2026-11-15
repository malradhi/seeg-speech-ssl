#!/bin/bash
#SBATCH --job-name=ieeg_smoke3
#SBATCH --partition=test
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000M
#SBATCH --output=logs/smoke3_%j.out
#SBATCH --error=logs/smoke3_%j.out
# Phase 3 GPU smoke test (~10 min): every Phase 3 code path on 3 participants with tiny
# budgets, written to results/smoke3 and results/smoke3_eval (never results/phase3).
set -eo pipefail
source slurm/env.sh
python scripts/13_run_phase3_pooled.py --array-index 1 --config configs/smoke3.json
python scripts/14_run_phase3.py --participants P01 --config configs/smoke3.json --workers 4
python scripts/15_evaluate.py --participants P01 --config configs/smoke3.json --overwrite
python scripts/21_quicklook.py --phase smoke3
python scripts/22_eval_summary.py --dir smoke3_eval
echo "SMOKE3 FINISHED"
