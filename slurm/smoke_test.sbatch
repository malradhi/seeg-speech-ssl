#!/bin/bash
#SBATCH --job-name=ieeg_smoke
#SBATCH --partition=test
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000M
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.out
# Submit from the repository root:  sbatch slurm/smoke_test.sbatch
# Needs P15 downloaded (python scripts/01_download_osf.py --only P15).
set -eo pipefail
source slurm/env.sh
nvidia-smi || true
python scripts/00_check_env.py
python scripts/02_preprocess.py --participants P15
python scripts/03_audit.py --participants P15
python scripts/10_run_phase1.py --participants P15 --config configs/smoke.json --phase smoke --workers 4
python scripts/21_quicklook.py --phase smoke
python scripts/20_collect.py --phase smoke
echo "SMOKE TEST FINISHED"
