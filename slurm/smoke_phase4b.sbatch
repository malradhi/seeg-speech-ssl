#!/bin/bash
#SBATCH --job-name=ieeg_smoke4b
#SBATCH --partition=test
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6000M
#SBATCH --output=logs/smoke4b_%j.out
#SBATCH --error=logs/smoke4b_%j.out
# Phase 4b GPU smoke test on P01 with tiny budgets: every Phase 4b code path, BrainBERT k=64
# extraction with the real checkpoint. Reads the real Phase 1/3/4 encoders and Phase 4 BrainBERT
# features; writes only results/smoke4b, results/smoke4b_eval, derived_brainbert_k64_smoke.
set -eo pipefail
source slurm/env.sh
python scripts/17_run_phase4b.py --participants P01 --config configs/smoke4b.json --workers 4
python scripts/15_evaluate.py --participants P01 --config configs/smoke4b.json --overwrite
python scripts/21_quicklook.py --phase smoke4b
python scripts/22_eval_summary.py --dir smoke4b_eval
echo "SMOKE4B FINISHED"
