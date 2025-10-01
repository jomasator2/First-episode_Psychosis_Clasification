#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classifier evaluation on a radiomics table.

Input:
- A CSV/TSV where the first column is 'Label' (0 = control, 1 = disease)
  and each remaining column is a radiomic feature (one subject per row).

What it does:
- (Optional) univariate feature screening (normality test → t-test or Mann–Whitney)
- Stratified repeated k-fold CV (no grouping)
- Trains several classifiers (SVM, Logistic Regression, RF, NB, KNN, GB)
- Saves per-fold metrics, predictions, selected features, and ROC curves
- (Optional) runs model comparisons and fine-tuning helper scripts

Outputs:
- results CSV, predictions CSV, selected-features TXT, and ROC plots
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, shapiro, ttest_ind

from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold

# Plotting (headless-safe)
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
import scienceplots  # noqa: F401, E402

plt.style.use(["science", "grid"])
mpl.rcParams["text.usetex"] = False
DPI = 300

# -----------------------------
# Logging
# -----------------------------
LOGGER = logging.getLogger("radiomics.eval")

def setup_logging(verbose: int = 0) -> None:
    level = logging.WARNING if verbose <= 0 else logging.INFO if verbose == 1 else logging.DEBUG
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    LOGGER.setLevel(level)
    LOGGER.handlers[:] = [handler]

def log_info(msg: str) -> None:
    LOGGER.info(f"✅ {msg}")

def log_warning(msg: str) -> None:
    LOGGER.warning(f"⚠️ {msg}")

def log_error(msg: str) -> None:
    LOGGER.error(f"❌ {msg}")

# -----------------------------
# Utilities
# -----------------------------

# --- add near the top (after imports) ---
def default_ratios() -> list[float | int]:
    """Default sweep of ratios."""
    return (
        [round(x, 3) for x in np.arange(0.001, 0.01, 0.001)]
        + [round(x, 2) for x in np.arange(0.01, 0.1, 0.01)]
        + [round(x, 1) for x in np.arange(0.1, 1.0, 0.1)]
        + list(range(1, 22))
        + [23, 25]
    )


def parse_alpha_from_ratio(ratio_str: str | None, default: float = 0.05) -> float:
    """
    Extract an alpha value in (0, 1) from strings like:
      'p-value', 'p05', 'p-value-05', 'p-0.1', 'p=0.001', 'alpha=5e-2'
    Accepts commas as decimals ('p-0,05').
    Falls back to `default` if not found/invalid.
    """
    if ratio_str is None:
        return float(default)

    s = str(ratio_str).strip().lower().replace("pvalue", "p-value").replace(",", ".")
    m = re.search(r"([0-9]+(?:\.[0-9]+)?(?:e-?\d+)?)\s*$", s)
    if not m:
        return float(default)

    tok = m.group(1)
    try:
        if "." in tok or "e" in tok:
            val = float(tok)
            return val if 0.0 < val < 1.0 else float(default)
        # digits only → interpret "05"→0.05, "001"→0.001, "5"→0.5
        val = float(f"0.{tok}")
        return val if 0.0 < val < 1.0 else float(default)
    except Exception:
        return float(default)


def get_models(random_state: int = 42) -> List[Tuple[str, object]]:
    """
    Define classifier pipelines (standardize → variance filter → model).
    Returns:
        List of (model_name, sklearn_pipeline)
    """
    pipe_svc = make_pipeline(
        StandardScaler(),
        VarianceThreshold(),
        SVC(random_state=random_state, class_weight="balanced", probability=True),
    )
    pipe_lr = make_pipeline(
        StandardScaler(),
        VarianceThreshold(),
        LogisticRegression(
            penalty="elasticnet",
            l1_ratio=0.5,
            class_weight="balanced",
            random_state=random_state,
            solver="saga",
            max_iter=10_000,
        ),
    )
    pipe_rf = make_pipeline(
        StandardScaler(),
        VarianceThreshold(),
        RandomForestClassifier(
            n_jobs=-1, class_weight="balanced_subsample", random_state=random_state
        ),
    )
    pipe_nb = make_pipeline(StandardScaler(), VarianceThreshold(), GaussianNB())
    pipe_knn = make_pipeline(
        StandardScaler(),
        VarianceThreshold(),
        KNeighborsClassifier(n_jobs=-1),
    )
    pipe_gb = make_pipeline(
        StandardScaler(),
        VarianceThreshold(),
        GradientBoostingClassifier(random_state=random_state),
    )

    return [
        ("SVM", pipe_svc),
        ("Logistic Regression", pipe_lr),
        ("Random Forest", pipe_rf),
        ("Naive Bayes", pipe_nb),
        ("KNN", pipe_knn),
        ("Gradient Boosting", pipe_gb),
    ]


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 1,
    base_random_state: int = 42,
):
    """
    Repeated stratified k-fold CV (no groups).
    Returns:
        fold_results: list of per-fold metric dicts
        pred_vals: dict with raw predictions per fold
    """
    fold_results: List[Dict] = []
    folds_data: List[Dict] = []
    global_fold_index = 0

    for rep in range(n_repeats):
        current_rs = base_random_state + rep
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=current_rs)

        for train_idx, val_idx in splitter.split(X, y):
            global_fold_index += 1
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model.fit(X_train, y_train)

            # Train preds/probs
            y_train_pred = model.predict(X_train)
            if hasattr(model, "predict_proba"):
                y_train_prob = model.predict_proba(X_train)[:, 1]
            elif hasattr(model, "decision_function"):
                y_train_prob = model.decision_function(X_train)
            else:
                y_train_prob = None

            try:
                train_auc = roc_auc_score(y_train, y_train_prob) if y_train_prob is not None else np.nan
            except Exception:
                train_auc = np.nan
            train_f1 = f1_score(y_train, y_train_pred, average="binary")

            # Val preds/probs
            y_val_pred = model.predict(X_val)
            if hasattr(model, "predict_proba"):
                y_val_prob = model.predict_proba(X_val)[:, 1]
            elif hasattr(model, "decision_function"):
                y_val_prob = model.decision_function(X_val)
            else:
                y_val_prob = None

            try:
                val_auc = roc_auc_score(y_val, y_val_prob) if y_val_prob is not None else np.nan
            except Exception:
                val_auc = np.nan

            val_mcc = matthews_corrcoef(y_val, y_val_pred)
            val_kappa = cohen_kappa_score(y_val, y_val_pred)
            val_f1_binary = f1_score(y_val, y_val_pred, average="binary")
            val_f1_macro = f1_score(y_val, y_val_pred, average="macro")
            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_balanced_accuracy = balanced_accuracy_score(y_val, y_val_pred)
            val_sensitivity = recall_score(y_val, y_val_pred, pos_label=1)
            val_specificity = recall_score(y_val, y_val_pred, pos_label=0)
            val_ppv = precision_score(y_val, y_val_pred, pos_label=1)

            cm = confusion_matrix(y_val, y_val_pred)
            val_npv = cm[0, 0] / (cm[0, 0] + cm[1, 0]) if (cm[0, 0] + cm[1, 0]) > 0 else np.nan

            per_class_precision = precision_score(y_val, y_val_pred, average=None)
            per_class_recall = recall_score(y_val, y_val_pred, average=None)
            per_class_f1 = f1_score(y_val, y_val_pred, average=None)

            per_class_accuracy: List[float] = []
            for i in range(len(cm)):
                row_sum = np.sum(cm[i, :])
                per_class_accuracy.append(cm[i, i] / row_sum if row_sum > 0 else np.nan)

            fold_metrics = {
                "Fold": global_fold_index,
                "Repeat": rep + 1,
                "train_auc": train_auc,
                "train_f1": train_f1,
                "val_auc": val_auc,
                "val_mcc": val_mcc,
                "val_kappa": val_kappa,
                "val_f1_binary": val_f1_binary,
                "val_f1_macro": val_f1_macro,
                "val_accuracy": val_accuracy,
                "val_sensitivity": val_sensitivity,
                "val_specificity": val_specificity,
                "val_ppv": val_ppv,
                "val_npv": val_npv,
                "val_balanced_accuracy": val_balanced_accuracy,
                "per_class_precision": per_class_precision.tolist(),
                "per_class_recall": per_class_recall.tolist(),
                "per_class_f1": per_class_f1.tolist(),
                "per_class_accuracy": per_class_accuracy,
            }
            fold_results.append(fold_metrics)

            folds_data.append(
                {
                    "fold_index": global_fold_index,
                    "Repeat": rep + 1,
                    "y_val": y_val,
                    "y_val_pred": y_val_pred,
                    "y_val_prob": y_val_prob,
                }
            )

    return fold_results, {"folds": folds_data}


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    """
    1) Parse CLI
    2) Load table (first column 'Label')
    3) Optional univariate feature selection
    4) CV training/evaluation
    5) Save results, predictions, features, and ROC curves
    """
    parser = argparse.ArgumentParser(
        description="Model evaluation with stratified cross-validation (radiomics table)."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the input table where the first column is 'Label' and the rest are features.",
    )
    parser.add_argument(
        "--results_base",
        type=Path,
        default="./results",
        help="Base directory where results will be stored.",
    )
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds for StratifiedKFold.")
    parser.add_argument("--n_repeats", type=int, default=10, help="Number of repeated CV runs.")
    parser.add_argument(
        "--feature_strategy",
        type=str,
        choices=["all", "most_discriminant"],
        default="most_discriminant",
        help="Feature selection strategy.",
    )
    parser.add_argument(
        "--calculate_differences",
        action="store_true",
        default=True,
        help="If enabled, run 2_model_differences.py after evaluation.",
    )
    parser.add_argument(
        "--fine_tune_best_model",
        action="store_true",
        default=True,
        help="If enabled, run 3_retrain_best_model_and_evaluate.py on the best model.",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        default=None,
        help=(
            "List of ratios to evaluate. Accepts numbers (e.g. 0.04 5) "
            "and/or strings like 'p-0.05' or 'alpha=5e-2'. "
            "Default is a comprehensive sweep."
        ),
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase logging (-v=INFO, -vv=DEBUG)")
    args = parser.parse_args()

    # Inicializar logging
    setup_logging(args.verbose)

    # --- Load data ---
    df = pd.read_csv(args.csv, sep="\t")
    if "Label" not in df.columns:
        raise ValueError("Input must contain a column named 'Label' as the first column.")
    y = df["Label"].values.astype(int)

    # Encode labels (in case they aren’t exactly 0/1)
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Example feature-ratio sweep 
    ratios = args.ratios if args.ratios is not None else default_ratios()
    for ratio in ratios:
        X = df.drop(columns=["Label"])
        log_info(f"Evaluating ratio={ratio}")

        # Results directories
        base_dir = args.results_base / f"results_{ratio}_ratio"
        strat_dir = base_dir / args.feature_strategy
        strat_dir.mkdir(parents=True, exist_ok=True)
        experiment_dir = strat_dir / "radiomics_experiment"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        log_info(f"Results will be saved in: {experiment_dir}")

        # --- Univariate feature selection ---
        selected_features = X.columns
        if args.feature_strategy == "most_discriminant":
            log_info(">> Performing univariate feature screening...")
            fs_dir = experiment_dir / "feature_selection"
            images_dir = fs_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            feature_names: List[str] = []
            sensitivity_list: List[float] = []
            specificity_list: List[float] = []
            auc_list: List[float] = []
            threshold_list: List[float] = []
            test_type_list: List[str] = []
            pvalue_list: List[float] = []
            pos_vs_neg_list: List[str] = []

            for column in X.columns:
                # Normality (Shapiro) to choose parametric vs non-parametric test
                stat, p_norm = shapiro(X[column])
                a_dist = X[column][y == 0]
                b_dist = X[column][y == 1]

                feature_names.append(column)
                alpha = 0.05
                if p_norm > alpha:
                    test_type_list.append("t-test")
                    _, pval = ttest_ind(a_dist, b_dist)
                else:
                    test_type_list.append("mann-whitney U-test")
                    _, pval = mannwhitneyu(a_dist, b_dist)
                pvalue_list.append(float(pval))

                # Single-feature ROC
                fpr, tpr, thresholds = metrics.roc_curve(y, X[column], pos_label=1)
                auc_val = metrics.auc(fpr, tpr)
                pos_vs_neg = ">"
                if auc_val < 0.5:
                    fpr, tpr, thresholds = metrics.roc_curve(y, X[column], pos_label=0)
                    auc_val = metrics.auc(fpr, tpr)
                    pos_vs_neg = "<"
                auc_list.append(float(auc_val))
                pos_vs_neg_list.append(pos_vs_neg)

                roc_df = pd.DataFrame(
                    {"fpr": fpr, "tpr": tpr, "1-fpr": 1 - fpr, "tf": tpr - (1 - fpr), "thresholds": thresholds}
                )
                cutoff_df = roc_df.iloc[(roc_df.tf - 0).abs().argsort()[:1]]
                sensitivity_list.append(float(cutoff_df["tpr"].values[0]))
                specificity_list.append(float(cutoff_df["1-fpr"].values[0]))
                threshold_list.append(float(cutoff_df["thresholds"].values[0]))

            train_auc_pvals_df = pd.DataFrame(
                list(
                    zip(
                        auc_list,
                        pos_vs_neg_list,
                        threshold_list,
                        sensitivity_list,
                        specificity_list,
                        test_type_list,
                        pvalue_list,
                    )
                ),
                index=feature_names,
                columns=[
                    "AUC",
                    "Pos.vs.Neg.",
                    "Cutoff-Threshold",
                    "Sensitivity",
                    "Specificity",
                    "Test",
                    "p-value",
                ],
            ).sort_values(by="p-value", ascending=True)

            # Selection by p-threshold or by top-k ratio
            train_auc_pvals_df_aux = train_auc_pvals_df.copy(deep=True)
            if isinstance(ratio, str) and "p" in ratio.lower():
                alpha = parse_alpha_from_ratio(ratio)
                pvals = train_auc_pvals_df_aux["p-value"].astype(float)
                mask = pvals.notna() & (pvals <= alpha)
                selected_features = train_auc_pvals_df_aux.index[mask]
                num_features_model = int(mask.sum())
                log_info(f"--> Threshold p ≤ {alpha:.4g}. Selected {num_features_model} features.")
            else:
                ratio = int(ratio)
                n_subjects = X.shape[0]
                log_info(f"nsubjects = {n_subjects}")
                log_info(f"ratio = {ratio}")
                num_features_model = int(n_subjects // ratio)
                selected_features = train_auc_pvals_df_aux.index[0:num_features_model]
                log_info(f"--> Selected {num_features_model} features (top by p-value).")

            X = X[selected_features]

            fs_dir_path = experiment_dir / "feature_selection"
            fs_dir_path.mkdir(parents=True, exist_ok=True)
            df_path_1 = fs_dir_path / "train_auc_pvals_df.csv"
            train_auc_pvals_df.to_csv(df_path_1)
            log_info(f"  --> Saved CSV: {df_path_1}\n")

            # Save quick violin & ROC for top-20
            top_20 = train_auc_pvals_df.index[:20]
            for rank, feature_name in enumerate(top_20, start=1):
                safe_feat_name = feature_name.replace("/", "_")
                feat_folder = images_dir / f"{rank}_{safe_feat_name}"
                feat_folder.mkdir(parents=True, exist_ok=True)

                # Violin
                plt.figure(figsize=(9, 9))
                sns.violinplot(x=y, y=df[feature_name], color="grey")
                plt.title(f"Distribution of {feature_name}", fontsize=14)
                plt.xlabel("Classes")
                plt.xticks([0, 1], ["control", "disease"], fontsize=12)
                plt.savefig(feat_folder / f"{safe_feat_name}_violinplot.png", dpi=DPI)
                plt.close()

                # ROC
                fpr, tpr, _ = metrics.roc_curve(y, df[feature_name], pos_label=1)
                auc_val = metrics.auc(fpr, tpr)
                plt.figure(figsize=(6, 6))
                plt.plot(fpr, tpr, marker=".", label=f"{feature_name} (AUC={auc_val:.3f})")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.legend()
                plt.title(f"ROC: {feature_name}")
                plt.savefig(feat_folder / f"{safe_feat_name}_ROC.png", dpi=DPI)
                plt.close()

        else:
            log_info(">> Using ALL features (no selection).")

        # --- Training & evaluation ---
        models = get_models(random_state=42)
        all_results: List[Dict] = []
        preds_data: List[Dict] = []

        for model_name, model in models:
            log_info(f"Evaluating {model_name}...")
            fold_metrics_list, pred_vals = evaluate_model(
                model,
                X,
                y,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
                base_random_state=42,
            )
            for fold_metrics in fold_metrics_list:
                fold_metrics["Classifier"] = model_name
                all_results.append(fold_metrics)
            preds_data.append({"Classifier": model_name, "folds": pred_vals["folds"]})

        df_results = pd.DataFrame(all_results)
        fixed_cols = ["Classifier", "Fold", "Repeat"]
        other_cols = [c for c in df_results.columns if c not in fixed_cols]
        df_results = df_results[fixed_cols + other_cols].sort_values(by=["Classifier", "Fold"])

        csv_basename = Path(args.csv).stem
        feat_str = args.feature_strategy
        results_filename = f"results_{csv_basename}_{feat_str}.csv"
        results_filepath = experiment_dir / results_filename
        df_results.to_csv(results_filepath, index=False)
        log_info(f"Results saved to '{results_filepath}'")

        # Save raw per-fold predictions
        records_for_csv: List[Dict] = []
        for item in preds_data:
            clf_name = item["Classifier"]
            for fold_info in item["folds"]:
                records_for_csv.append(
                    {
                        "Classifier": clf_name,
                        "Fold": fold_info["fold_index"],
                        "Repeat": fold_info["Repeat"],
                        "y_val": fold_info["y_val"].tolist(),
                        "y_pred": fold_info["y_val_pred"].tolist(),
                        "y_prob": [] if fold_info["y_val_prob"] is None else fold_info["y_val_prob"].tolist(),
                    }
                )
        df_preds = pd.DataFrame(records_for_csv)
        preds_filename = f"preds_{csv_basename}_{feat_str}.csv"
        preds_filepath = experiment_dir / preds_filename
        df_preds.to_csv(preds_filepath, index=False)
        log_info(f"Predictions saved to '{preds_filepath}'")

        # Save selected features
        variables_txt_path = experiment_dir / "selected_features.txt"
        with open(variables_txt_path, "w") as f:
            for feat in selected_features:
                f.write(str(feat) + "\n")
            log_info(f"Selected features written to: {variables_txt_path}")

        # --- ROC curves (best and median fold per classifier) ---
        log_info("\nGenerating ROC curves: best and median folds per classifier...")

        roc_dir = experiment_dir / "ROC_curves"
        roc_dir.mkdir(parents=True, exist_ok=True)
        curves_info_optimal: List[Dict] = []
        curves_info_median: List[Dict] = []

        classifiers = df_results["Classifier"].unique()
        for clf_name in classifiers:
            df_clf = df_results[df_results["Classifier"] == clf_name]
            best_fold_idx = df_clf["val_auc"].idxmax()
            best_fold_num = df_clf.loc[best_fold_idx, "Fold"]
            median_auc = df_clf["val_auc"].median()
            median_fold_idx = (df_clf["val_auc"] - median_auc).abs().idxmin()
            median_fold_num = df_clf.loc[median_fold_idx, "Fold"]

            df_best = df_preds[(df_preds["Classifier"] == clf_name) & (df_preds["Fold"] == best_fold_num)]
            if len(df_best) > 0:
                y_true = df_best.iloc[0]["y_val"]
                y_prob = df_best.iloc[0]["y_prob"]
                if y_prob:
                    fpr_b, tpr_b, _ = metrics.roc_curve(y_true, y_prob, pos_label=1)
                    auc_b = metrics.auc(fpr_b, tpr_b)
                    curves_info_optimal.append(
                        {"classifier": clf_name, "fold": best_fold_num, "fpr": fpr_b, "tpr": tpr_b, "auc": auc_b}
                    )

            df_med = df_preds[(df_preds["Classifier"] == clf_name) & (df_preds["Fold"] == median_fold_num)]
            if len(df_med) > 0:
                y_true = df_med.iloc[0]["y_val"]
                y_prob = df_med.iloc[0]["y_prob"]
                if y_prob:
                    fpr_m, tpr_m, _ = metrics.roc_curve(y_true, y_prob, pos_label=1)
                    auc_m = metrics.auc(fpr_m, tpr_m)
                    curves_info_median.append(
                        {"classifier": clf_name, "fold": median_fold_num, "fpr": fpr_m, "tpr": tpr_m, "auc": auc_m}
                    )

        curves_info_optimal.sort(key=lambda x: x["auc"], reverse=True)
        curves_info_median.sort(key=lambda x: x["auc"], reverse=True)

        # Color palette and mapping
        my_colors = ["#0072B2", "#009E73", "#D55E00", "#CC78BC", "#DE8F05", "#56B4E9"]
        my_palette = sns.color_palette(my_colors)
        fixed_classifiers = ["SVM", "Logistic Regression", "Random Forest", "Naive Bayes", "KNN", "Gradient Boosting"]
        color_mapping: Dict[str, tuple] = {clf: my_palette[i] for i, clf in enumerate(fixed_classifiers)}

        # Plot best-fold ROCs
        fig_opt, ax_opt = plt.subplots(figsize=(8, 6))
        for info in curves_info_optimal:
            clf_name = info["classifier"]
            ax_opt.plot(
                info["fpr"],
                info["tpr"],
                label=f"{clf_name} (Fold={info['fold']}, AUC={info['auc']:.3f})",
                color=color_mapping.get(clf_name, "black"),
            )
        ax_opt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="_nolegend_")
        ax_opt.set_xlabel("False Positive Rate", fontsize=12, labelpad=10)
        ax_opt.set_ylabel("True Positive Rate", fontsize=12, labelpad=10)
        ax_opt.set_title("ROC curves: best folds per model", fontsize=14)
        ax_opt.legend(fontsize=10)
        for line in ax_opt.get_legend().get_lines():
            line.set_linewidth(2.5)
        fig_opt.tight_layout()
        roc_plot_path_opt = roc_dir / "roc_optimal_folds.png"
        plt.savefig(roc_plot_path_opt, dpi=DPI, bbox_inches="tight")
        plt.close(fig_opt)
        log_info(f"ROC (best folds) saved to: {roc_plot_path_opt}")

        # Plot median-fold ROCs
        fig_med, ax_med = plt.subplots(figsize=(8, 6))
        for info in curves_info_median:
            clf_name = info["classifier"]
            ax_med.plot(
                info["fpr"],
                info["tpr"],
                label=f"{clf_name} (Fold={info['fold']}, AUC={info['auc']:.3f})",
                color=color_mapping.get(clf_name, "black"),
            )
        ax_med.plot([0, 1], [0, 1], linestyle="--", color="gray", label="_nolegend_")
        ax_med.set_xlabel("False Positive Rate", fontsize=12, labelpad=10)
        ax_med.set_ylabel("True Positive Rate", fontsize=12, labelpad=10)
        ax_med.set_title("ROC curves: median folds per model", fontsize=14)
        ax_med.legend(fontsize=10)
        for line in ax_med.get_legend().get_lines():
            line.set_linewidth(2.5)
        fig_med.tight_layout()
        roc_plot_path_med = roc_dir / "roc_median_folds.png"
        plt.savefig(roc_plot_path_med, dpi=DPI, bbox_inches="tight")
        plt.close(fig_med)
        log_info(f"ROC (median folds) saved to: {roc_plot_path_med}")

        # --- Optional: model comparisons & fine-tuning ---
        if args.calculate_differences:
            log_info("\nRunning model comparisons (2_model_differences.py)...")
            model_diff_dir = experiment_dir / "model_differences"
            model_diff_dir.mkdir(parents=True, exist_ok=True)
            postprocess_cmd = [
                "python3",
                "2_model_differences.py",
                "--csv_preds",
                preds_filepath,
                "--csv_results",
                results_filepath,
                "--metric",
                "val_auc",
                "--alpha",
                "0.05",
                "--outdir",
                model_diff_dir,
                "-v",
            ]
            subprocess.call(postprocess_cmd)
        else:
            log_info("\nSkipping model comparisons.")

        if args.fine_tune_best_model:
            if curves_info_optimal:
                best_model = curves_info_optimal[0]["classifier"]
                model_mapping = {
                    "SVM": "SVM",
                    "Logistic Regression": "LogisticRegression",
                    "Random Forest": "RandomForest",
                    "Naive Bayes": "NaiveBayes",
                    "KNN": "KNN",
                    "Gradient Boosting": "GradientBoosting",
                    
                }
                best_model_finetune = model_mapping.get(best_model, best_model)
                log_info(f"Fine-tuning best model: {best_model_finetune}")
                fine_tune_cmd = [
                    "python3",
                    "3_retrain_best_model_and_evaluate.py",
                    "--csv",
                    args.csv,
                    "--model",
                    best_model_finetune,
                    "--variables",
                    variables_txt_path,
                    "-v",
                ]
                subprocess.call(fine_tune_cmd)
            else:
                log_warn("No information available to determine the best model.")
        else:
            log_warn("Skipping best-model fine-tuning.")


if __name__ == "__main__":
    main()


