#!/bin/bash
#SBATCH --job-name=ieeg_smoke2
#SBATCH --partition=test
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000M
#SBATCH --output=logs/smoke2_%j.out
#SBATCH --error=logs/smoke2_%j.out
# Phase 2 GPU smoke test (~10 min): every Phase 2 code path on 3 participants with tiny
# budgets, written to results/smoke2. Needs the Phase 1 encoders of P01-P03.
set -eo pipefail
source slurm/env.sh
for i in 1 2 3 4; do python scripts/11_run_phase2_multi.py --array-index $i --config configs/smoke2.json; done
python scripts/12_run_phase2.py --participants P01 --config configs/smoke2.json --workers 4
python scripts/05_contamination.py --participants P01 --config configs/smoke2.json
python scripts/21_quicklook.py --phase smoke2
python scripts/20_collect.py --phase smoke2
echo "SMOKE2 FINISHED"
