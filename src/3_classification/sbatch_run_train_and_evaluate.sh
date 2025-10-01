#!/bin/bash
#SBATCH --job-name=radiomics

#SBATCH --partition=gpuceib 
## gpuceib, gpu, bigmem, fast, long

#SBATCH --cpus-per-task=1
#SBATCH --mem=30G

#SBATCH --output=log/radiomics.out   # salida estándar
#SBATCH --error=log/radiomics.err    # salida de errores

##SBATCH --gres=gpu:0

module load PyTorch/2.1.2-Miniconda3-CUDA-11.8.0
# export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH"
# export PATH="/usr/local/cuda-11.7/bin:$PATH"
export PYTHONPATH="/home/jmsaborit/.local/bin:$PYTHONPATH"
export PATH="/home/jmsaborit/.local/bin:$PATH"

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