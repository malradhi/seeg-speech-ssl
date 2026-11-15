#!/bin/bash
#SBATCH --job-name=ieeg_smoke4
#SBATCH --partition=test
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000M
#SBATCH --output=logs/smoke4_%j.out
#SBATCH --error=logs/smoke4_%j.out
# Phase 4 GPU smoke test: BrainBERT features for P01 (real checkpoint) and every Phase 4 code
# path with tiny budgets; writes only results/smoke4, results/smoke4_eval, derived_brainbert_smoke.
set -eo pipefail
source slurm/env.sh
python scripts/16_run_phase4.py --participants P01 --config configs/smoke4.json --workers 4
python scripts/15_evaluate.py --participants P01 --config configs/smoke4.json --overwrite
python scripts/21_quicklook.py --phase smoke4
python scripts/22_eval_summary.py --dir smoke4_eval
echo "SMOKE4 FINISHED"
