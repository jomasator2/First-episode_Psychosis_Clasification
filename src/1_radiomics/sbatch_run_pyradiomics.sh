#!/bin/bash
#SBATCH --job-name=radiomics
#SBATCH --partition=gpuceib           # options: gpuceib, gpu, bigmem, fast, long
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
##SBATCH --gres=gpu:0                 # uncomment and set GPUs if needed

set -euo pipefail

# -----------------------------
# 1) Move to the repo root (directory where this script lives)
# -----------------------------
cd "$(dirname "$0")"

# -----------------------------
# 2) Load/activate a Python environment
#    Priority: conda env -> module+venv -> user local python
# -----------------------------
ENV_NAME="radiomics"
PYTHON_VERSION="3.10"

if command -v conda >/dev/null 2>&1; then
  # Use conda if available
  eval "$(conda shell.bash hook)"
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda activate "$ENV_NAME"
  else
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
    conda activate "$ENV_NAME"
  fi
  # Install dependencies if present
  if [ -f requirements.txt ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
  fi

elif module avail python >/dev/null 2>&1; then
  # Fallback: system module + venv
  module load python
  if [ ! -d .venv ]; then
    python -m venv .venv
  fi
  source .venv/bin/activate
  if [ -f requirements.txt ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
  fi

else
  # Last resort: user local bin path
  export PYTHONPATH="$HOME/.local/bin:${PYTHONPATH:-}"
  export PATH="$HOME/.local/bin:$PATH"
fi

# -----------------------------
# 3) Show versions for reproducibility
# -----------------------------
python -V
pip --version || true

# -----------------------------
# 4) Define inputs/outputs and run the pipeline
# -----------------------------
INPUT_DIR="PATH_TO_YOUR_FREESURFER_DIRECTORY"
SUBJECTS_CSV="data/freesurfer_subjects.csv"
RESULTS_CSV="data/freesurfer_subjects_results.csv"
PARAMS_FILE="Params.yaml"
LOG_FILE="data/logfile.txt"

# Generate subject list from FreeSurfer outputs
srun python generate_csv_freesurfer.py \
  "$INPUT_DIR" \
  -o "$SUBJECTS_CSV"

# Compute radiomics with parameters and logging
srun python calculate_orig.py \
  "./$SUBJECTS_CSV" \
  "./$RESULTS_CSV" \
  --param "$PARAMS_FILE" \
  --logfile "$LOG_FILE"

# -----------------------------
# 5) Clean up environment (best-effort)
# -----------------------------
conda deactivate 2>/dev/null || deactivate 2>/dev/null || true
