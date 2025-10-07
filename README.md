# First-Episode Psychosis Classification - Complete Workflow

Machine learning models for classification of First-Episode Psychosis (FEP) vs Healthy Controls using structural MRI radiomic features. This document describes the complete execution flow from data preprocessing to results visualization.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Step 1: Radiomics Feature Extraction](#step-1-radiomics-feature-extraction)
4. [Step 2: Data Preprocessing](#step-2-data-preprocessing)
5. [Step 3: Classification and Model Training](#step-3-classification-and-model-training)
6. [Step 4: Results Visualization](#step-4-results-visualization)
7. [Complete Pipeline Execution](#complete-pipeline-execution)

---

## Overview

The pipeline consists of four main steps:

1. **Radiomics Feature Extraction**: Extract radiomic features from MRI images using PyRadiomics
2. **Data Preprocessing**: Clean, normalize, and prepare features for machine learning
3. **Classification**: Train and evaluate machine learning models with explainability techniques
4. **Results Visualization**: Analyze and visualize model performance and interpretability

---

## Prerequisites

### System Requirements
- Python 3.11.9
- Conda or Miniconda
- Access to MRI data (FreeSurfer processed images)

### Directory Structure
```
First-episode_Psychosis_Clasification/
├── src/
│   ├── 1_radiomics/          # Feature extraction
│   ├── 2_preprocess_data/     # Data preprocessing
│   ├── 3_classification/      # Model training
│   └── 4_results_viewer/      # Results visualization
├── README.md
└── README_all.md
```

---

## Step 1: Radiomics Feature Extraction

This step extracts radiomic features from structural MRI images using PyRadiomics.

### 1.1 Setup Environment

Navigate to the radiomics directory:
```bash
cd src/1_radiomics
```

Create and activate the conda environment:
```bash
conda env create -f enviroment.yml
conda activate radiomics
```

### 1.2 Input Requirements

- **Input Directory**: FreeSurfer processed MRI images with segmentation masks
- **Configuration**: `Params.yaml` file with PyRadiomics parameters
- **Output**: CSV/TSV file with extracted radiomic features

### 1.3 Run Feature Extraction

#### Local Execution
```bash
./local_run_pyradiomics.sh
```

This script will:
1. Generate a CSV with subject IDs and mask paths from FreeSurfer data
2. Extract radiomic features for each subject and ROI
3. Save features to `../2_preprocess_data/data/df_processed_example.tsv`
4. Generate logs in `logs/logfile.txt`

#### HPC/SLURM Execution
```bash
sbatch run_radiomics.sh
```

### 1.4 Key Scripts

- **`generate_csv_freesurfer.py`**: Creates subject list from FreeSurfer directory
- **`calculate.py`**: Main feature extraction script using PyRadiomics
- **`Params.yaml`**: PyRadiomics configuration parameters

### 1.5 Output

The output TSV file contains:
- Subject identifiers
- ROI labels
- First-order statistics (mean, median, variance, etc.)
- Texture features (GLCM, GLRLM, GLSZM, etc.)
- Shape features (volume, surface area, sphericity, etc.)

---

## Step 2: Data Preprocessing

This step cleans and prepares the radiomic features for machine learning.

### 2.1 Setup Environment

Navigate to the preprocessing directory:
```bash
cd ../2_preprocess_data
```

Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate jupyter_venv
```

### 2.2 Run Preprocessing

Open the Jupyter notebook:
```bash
jupyter notebook preprocess.ipynb
```

Or use JupyterLab:
```bash
jupyter lab preprocess.ipynb
```

### 2.3 Preprocessing Steps

The notebook performs:
1. **Data Loading**: Load radiomic features from Step 1, or use the original dataset from [Zenodo](https://zenodo.org/records/17285665) (automatically downloaded by `preprocess.ipynb`)
2. **Quality Control**: Remove features with missing values or low variance
3. **Feature Selection**: Select relevant features based on statistical tests
4. **Normalization**: Standardize or normalize feature values
5. **Data Splitting**: Create train/test splits
6. **Export**: Save processed data for classification

### 2.4 Output

Processed features are saved to:
- `data/features_outcome_df_processed_true.tsv`

This file is used as input for the classification step.

---

## Step 3: Classification and Model Training

This step trains machine learning models and evaluates their performance with explainability techniques.

### 3.1 Setup Environment

Navigate to the classification directory:
```bash
cd ../3_classification
```

Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate pyclassification
```

### 3.2 Run Classification Pipeline

#### Local Execution
```bash
./local_run_train_and_evaluate.sh
```

This executes the complete classification pipeline with:
- Multiple ML algorithms (Random Forest, SVM, Logistic Regression, etc.)
- Cross-validation
- Hyperparameter tuning
- Model evaluation
- Explainability analysis (SHAP, LIME)

#### HPC/SLURM Execution
```bash
sbatch sbatch_run_train_and_evaluate.sh
```

### 3.3 Pipeline Components

The classification pipeline consists of three main scripts:

#### 3.3.1 Model Training and Evaluation (`1_train_and_evaluate.py`)
```bash
python 1_train_and_evaluate.py \
    --csv features_outcome_df_processed_true.tsv \
    --calculate_differences \
    --fine_tune_best_model \
    --results_base ../4_results_viewer/data \
    --ratios 40 \
    -v
```

**Parameters:**
- `--csv`: Input features file
- `--calculate_differences`: Compute statistical differences between groups
- `--fine_tune_best_model`: Perform hyperparameter optimization
- `--results_base`: Output directory for results
- `--ratios`: Train/test split ratio (e.g., 40 = 40% test)
- `-v`: Verbose output

**Output:**
- Model performance metrics (accuracy, precision, recall, F1-score, AUC)
- Cross-validation results
- Feature importance rankings
- Confusion matrices

#### 3.3.2 Model Comparison (`2_model_differences.py`)
Analyzes statistical differences between model performances.

#### 3.3.3 Best Model Retraining (`3_retrain_best_model_and_evaluate.py`)
Retrains the best performing model with optimized hyperparameters and generates:
- Final model predictions
- SHAP values for feature importance
- LIME explanations for individual predictions
- Calibration plots
- ROC curves

### 3.4 Supported Models

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting
- XGBoost
- Neural Networks

### 3.5 Explainability Techniques

- **SHAP (SHapley Additive exPlanations)**: Global and local feature importance
- **LIME (Local Interpretable Model-agnostic Explanations)**: Instance-level explanations
- **Feature Importance**: Model-specific feature rankings
- **Partial Dependence Plots**: Feature effect visualization

### 3.6 Output Files

Results are saved to `../4_results_viewer/data/`:
- `model_results.pkl`: Trained models and predictions
- `feature_importance.csv`: Feature importance scores
- `shap_values.pkl`: SHAP analysis results
- `performance_metrics.csv`: Model evaluation metrics
- `confusion_matrix.png`: Confusion matrix visualization

---

## Step 4: Results Visualization

This step provides interactive visualization and analysis of model results.

### 4.1 Setup Environment

Navigate to the results viewer directory:
```bash
cd ../4_results_viewer
```

Create and activate the conda environment:
```bash
conda env create -f enviroment.yml
conda activate jupyter_venv
```

### 4.2 View Results

Open the results notebook:
```bash
jupyter notebook results.ipynb
```

Or use JupyterLab:
```bash
jupyter lab results.ipynb
```

### 4.3 Visualization Components

The notebook includes:

1. **Performance Metrics**
   - Accuracy, precision, recall, F1-score
   - ROC curves and AUC scores
   - Precision-recall curves
   - Calibration plots

2. **Model Comparison**
   - Side-by-side model performance
   - Statistical significance tests
   - Cross-validation stability

3. **Feature Analysis**
   - Feature importance rankings
   - SHAP summary plots
   - SHAP dependence plots
   - Feature correlation heatmaps

4. **Explainability Visualizations**
   - SHAP waterfall plots for individual predictions
   - LIME explanations
   - Decision boundary visualizations
   - Feature contribution plots

5. **Clinical Interpretability**
   - Top discriminative features
   - ROI-specific analysis
   - Biomarker identification

### 4.4 Interactive Exploration

The notebook provides interactive widgets for:
- Selecting different models
- Filtering features by importance
- Exploring individual predictions
- Comparing feature effects across groups

---

## Complete Pipeline Execution

### Sequential Execution (Recommended for First Run)

Execute each step in order:

```bash
# Step 1: Feature Extraction
cd src/1_radiomics
conda env create -f enviroment.yml
conda activate radiomics
./local_run_pyradiomics.sh

# Step 2: Preprocessing
cd ../2_preprocess_data
conda env create -f environment.yml
conda activate jupyter_venv
jupyter notebook preprocess.ipynb
# Run all cells in the notebook, then close

# Step 3: Classification
cd ../3_classification
conda env create -f environment.yml
conda activate pyclassification
./local_run_train_and_evaluate.sh

# Step 4: Results Visualization
cd ../4_results_viewer
conda env create -f enviroment.yml
conda activate jupyter_venv
jupyter notebook results.ipynb
```

### Automated Pipeline (After Initial Setup)

Once environments are created, you can run the pipeline with:

```bash
# From project root
cd src/1_radiomics && conda activate radiomics && ./local_run_pyradiomics.sh && \
cd ../3_classification && conda activate pyclassification && ./local_run_train_and_evaluate.sh
```

Then manually run the preprocessing and results notebooks.

### HPC/SLURM Pipeline

For high-performance computing:

```bash
# Step 1: Submit radiomics job
cd src/1_radiomics
sbatch run_radiomics.sh

# Wait for completion, then run preprocessing notebook

# Step 3: Submit classification job
cd ../3_classification
sbatch sbatch_run_train_and_evaluate.sh

# Wait for completion, then run results notebook
```

---

## Environment Management

### List All Environments
```bash
conda env list
```

### Activate Specific Environment
```bash
conda activate radiomics              # For feature extraction
conda activate jupyter_venv           # For preprocessing and visualization
conda activate pyclassification       # For classification
```

### Remove Environment (if needed)
```bash
conda env remove -n radiomics
conda env remove -n jupyter_venv
conda env remove -n pyclassification
```

### Update Environment
```bash
conda env update -f environment.yml --prune
```

---

## Troubleshooting

### Common Issues

1. **Environment Activation Fails**
   ```bash
   eval "$(conda shell.bash hook)"
   conda activate <env_name>
   ```

2. **Missing Dependencies**
   ```bash
   conda env update -f environment.yml
   ```

3. **Memory Issues During Classification**
   - Reduce the number of features in preprocessing
   - Use feature selection techniques
   - Increase available RAM or use HPC

4. **SHAP/LIME Computation Slow**
   - Reduce the number of samples for explanation
   - Use approximate SHAP methods
   - Run on HPC with more resources

### Log Files

Check log files for debugging:
- `src/1_radiomics/logs/logfile.txt`: Feature extraction logs
- `src/3_classification/log/`: Classification logs

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{fep_classification,
  title={First-Episode Psychosis Classification using MRI Radiomics},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/First-episode_Psychosis_Clasification}
}
```

---

## License

See LICENSE file for details.

---

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---
