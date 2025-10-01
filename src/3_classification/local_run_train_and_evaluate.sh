#!/bin/bash

# set -euo pipefail
set -euo pipefail

# -----------------------------
# 1) Activate environment
# -----------------------------
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate pyradiomics
elif [ -d ".venv" ]; then
  source .venv/bin/activate
else
  echo "⚠️ No conda or .venv found, using system Python."
fi

#python -m pip install scienceplots lime scikit-optimize
mkdir -p log
mkdir -p ../4_results_viewer/data

python -u 1_train_and_evaluate.py \
    --csv features_outcome_df_processed_true.tsv \
    --calculate_differences \
    --fine_tune_best_model \
    --results_base ../4_results_viewer/data \
    --ratios 40 \
    -v