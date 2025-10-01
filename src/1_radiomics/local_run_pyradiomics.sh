#!/bin/bash
# --------------------------------------------
# Run Radiomics extraction locally (no SLURM)
# --------------------------------------------

set -euo pipefail

# 1) Activate environment
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate radiomics
elif [ -d ".venv" ]; then
  source .venv/bin/activate
else
  echo "⚠️ No conda or .venv found, using system Python."
fi

# 2) Define inputs and outputs
INPUT_DIR="/home/jmsaborit/remoto/FISABIO_datalake/p0042024/derivatives/simplified_freesurfer_3_cases/"
SUBJECTS_CSV="./data/freesurfer_subjects.tsv"
OUTPUT_CSV="../2_preprocess_data/data/df_processed_example.tsv"
PARAMS_FILE="Params.yaml"
LOG_FILE="logs/logfile.txt"

# Generate folders data and logs
mkdir -p data logs

# 3) Generate CSV with subjects and masks
python generate_csv_freesurfer.py \
  "$INPUT_DIR" \
  -o "$SUBJECTS_CSV"

# 4) Run extractor
python calculate.py \
  "$SUBJECTS_CSV" \
  "$OUTPUT_CSV" \
  --param "$PARAMS_FILE" \
  --logfile "$LOG_FILE"
