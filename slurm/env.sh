# Source from the repository root before running anything:
#     source slurm/env.sh
#
# >>> EDIT: where the data and results go (about 35 GB at the peak; see docs/RUNNING_THE_STUDY.md) <<<
export IEEG_ROOT="${IEEG_ROOT:-$HOME/seeg_data}"

# >>> EDIT: your SLURM project account, if your cluster needs one (read by sbatch) <<<
# export SBATCH_ACCOUNT=your_project_account

# The module names below are those of the Komondor cluster; elsewhere, activate an
# environment with Python >= 3.8, PyTorch, NumPy and SciPy instead (see README.md).

module load pytorch/2.4.1 >/dev/null 2>&1 || echo "WARNING: could not load pytorch/2.4.1 (try: module avail pytorch)"

# Never fall back to the system python3 (3.6): the code needs Python >= 3.8.
_ieeg_py_ok() {
    command -v python >/dev/null 2>&1 &&
    python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' >/dev/null 2>&1
}
if ! _ieeg_py_ok; then
    # The PyTorch conda environment may be mounted only on compute nodes.
    # On the login node cray-python is enough for download / audit / collect.
    module load cray-python >/dev/null 2>&1 || true
    if ! command -v python >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
        python() { python3 "$@"; }
    fi
fi
if _ieeg_py_ok; then
    echo "Python OK: $(python --version 2>&1)  ($(python -c 'import sys; print(sys.executable)'))"
else
    echo "ERROR: no Python >= 3.8 available. Check: module avail python; module avail pytorch"
fi

export REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
# One BLAS thread per worker process (the runner starts several workers per GPU).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$IEEG_ROOT" "$REPO/logs"
echo "IEEG_ROOT=$IEEG_ROOT  REPO=$REPO"
