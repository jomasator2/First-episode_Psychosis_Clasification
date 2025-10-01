#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advanced optimization and evaluation of the best model.

This script performs:
1) Bayesian hyperparameter tuning (BayesSearchCV)
2) Evaluation on a hold-out test set
3) Probability calibration (Platt scaling)
4) Explainability analysis with SHAP and LIME
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import joblib
import matplotlib as mpl
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns
import shap
from lime.lime_tabular import LimeTabularExplainer
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import mannwhitneyu
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from statsmodels.stats.multitest import multipletests
from sklearn.naive_bayes import GaussianNB

# -----------------------------
# Plot setup (headless-safe)
# -----------------------------
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.style.use(["science", "grid"])
mpl.rcParams["text.usetex"] = False
DPI = 300

# -----------------------------
# Logging
# -----------------------------
LOGGER = logging.getLogger("radiomics.best_model")

def setup_logging(verbose: int = 0) -> None:
    """
    Configure root logger.
    - verbose = 0 → WARNING
    - verbose = 1 → INFO
    - verbose >=2 → DEBUG
    """
    level = logging.WARNING if verbose <= 0 else logging.INFO if verbose == 1 else logging.DEBUG
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    LOGGER.setLevel(level)
    LOGGER.handlers[:] = [handler]

def log_info(msg: str) -> None:
    LOGGER.info(f"✅ {msg}")

def log_warn(msg: str) -> None:
    LOGGER.warning(f"⚠️ {msg}")

def log_error(msg: str) -> None:
    LOGGER.error(f"❌ {msg}")


# =========================
# -------  SHAP  ----------
# =========================
def perform_shap_analysis(
    X_data: pd.DataFrame,
    y_data: np.ndarray,
    model_clf,
    preprocessor,
    shap_dir: str | Path,
    report_path: str | Path,
    dataset_name: str = "dataset",
    dpi: int = DPI,
    log_beeswarm: bool = True,
) -> Tuple[bool, pd.Index | None, "shap._explanation.Explanation | None", List[str] | None]:
    """
    Run SHAP analysis and generate:
      - Feature-wise statistical test (Mann–Whitney U + Holm correction)
      - Class-ordered heatmap
      - Beeswarm (standard, optional log scale)
      - Beeswarm ordered by max |SHAP|
      - Scatter plots for top features
    """
    shap_dir = Path(shap_dir)
    shap_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_path)

    log_info(f"Running SHAP analysis for {dataset_name}...")
    try:
        # ----- Preprocessing (replay the pipeline: StandardScaler -> VarianceThreshold) -----
        scaler = preprocessor.steps[0][1]
        X_scaled = pd.DataFrame(scaler.transform(X_data), index=X_data.index, columns=X_data.columns)

        vt = preprocessor.steps[1][1]
        mask = vt.get_support()
        selected_features = X_data.columns[mask]
        X_transformed = pd.DataFrame(vt.transform(X_scaled.values), index=X_data.index, columns=selected_features)

        # ----- Choose SHAP explainer -----
        if isinstance(model_clf, (RandomForestClassifier, GradientBoostingClassifier)):
            explainer = shap.TreeExplainer(model_clf)
        elif isinstance(model_clf, LogisticRegression):
            try:
                explainer = shap.LinearExplainer(model_clf, X_transformed)
            except Exception:
                background = shap.kmeans(X_transformed, 50)
                explainer = shap.KernelExplainer(model_clf.predict_proba, background)
        else:
            background = shap.kmeans(X_transformed, 50)
            explainer = shap.KernelExplainer(model_clf.predict_proba, background)

        shap_values = explainer(X_transformed)
        joblib.dump(shap_values, shap_dir / "shap_values.pkl")

        # Binary case → keep positive class
        if getattr(shap_values, "values", None) is not None and shap_values.values.ndim > 2:
            shap_values = shap_values[:, :, 1]

        # ----- Part 1: Feature-wise statistical test (U-test + Holm) -----
        log_info(f"Running Mann–Whitney U tests with Holm correction on {dataset_name}...")
        shap_matrix = pd.DataFrame(shap_values.values, index=X_transformed.index, columns=X_transformed.columns)

        features_test: List[str] = []
        pvalues_raw: List[float] = []

        for feat in shap_matrix.columns:
            shap_c0 = shap_matrix.loc[y_data == 0, feat]
            shap_c1 = shap_matrix.loc[y_data == 1, feat]
            _, pval = mannwhitneyu(shap_c0, shap_c1, alternative="two-sided")
            features_test.append(feat)
            pvalues_raw.append(float(pval))

        alpha = 0.05
        reject, pvals_corr, _, _ = multipletests(pvalues_raw, alpha=alpha, method="holm")

        lines: List[str] = []
        lines.append("=================================")
        lines.append(f"MANN–WHITNEY U (SHAP by feature) + Holm correction for {dataset_name}")
        lines.append("Comparison: Class 0 vs Class 1")
        lines.append(f"alpha = {alpha}")
        lines.append(f"Total features: {len(features_test)}")
        lines.append("=================================\n")
        lines.append("Per-feature results (raw & corrected p-values):")

        significant: List[Tuple[str, float, float]] = []
        for feat, p_raw, p_cor, rej in zip(features_test, pvalues_raw, pvals_corr, reject):
            tag = "=> SIGNIFICANT" if rej else "=> not significant"
            if rej:
                significant.append((feat, p_raw, p_cor))
            lines.append(f"    {feat}: raw={p_raw:.4e}, corrected={p_cor:.4e} {tag}")

        lines.append("")
        lines.append(f" Total significant comparisons: {len(significant)}")
        if not significant:
            lines.append("    None.")
        else:
            for feat, p_raw, p_cor in significant:
                lines.append(f"    {feat}: raw={p_raw:.4e}, corrected={p_cor:.4e} => SIGNIFICANT")
        lines.append("")

        test_txt = shap_dir / "shap_statistical_test.txt"
        test_txt.write_text("\n".join(lines), encoding="utf-8")
        log_info(f"Statistical test saved: {test_txt}")

        # ----- Part 2: Heatmap ordered by class -----
        log_info(f"Generating class-ordered SHAP heatmap for {dataset_name}...")
        idx_c0 = np.where(y_data == 0)[0]
        idx_c1 = np.where(y_data == 1)[0]
        idx_order = np.concatenate([idx_c0, idx_c1])

        heatmap_path = shap_dir / "shap_heatmap.png"
        shap.plots.heatmap(shap_values, show=False, instance_order=idx_order)
        fig = plt.gcf()
        ax = plt.gca()
        split = len(idx_c0)
        ax.axvline(split - 0.5, color="black", linewidth=1, zorder=10)
        n_total = len(idx_order)
        ax.text((split / 2) / n_total, 1.01, "Class 0", ha="center", va="bottom", transform=ax.transAxes)
        ax.text((split + len(idx_c1) / 2) / n_total, 1.01, "Class 1", ha="center", va="bottom", transform=ax.transAxes)
        fig.set_size_inches(10, 6)
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=DPI, bbox_inches="tight")
        plt.close()
        log_info(f"Heatmap saved: {heatmap_path}")

        # ----- Part 3: Beeswarm (standard, log scale) -----
        beeswarm_std = shap_dir / "shap_beeswarm.png"
        shap.plots.beeswarm(shap_values, max_display=20, log_scale=True, show=False)
        fig = plt.gcf()
        plt.tight_layout()
        fig.set_size_inches(14, 8)
        plt.subplots_adjust(left=0.4, right=0.95)
        plt.savefig(beeswarm_std, dpi=dpi, bbox_inches="tight")
        plt.close()
        log_info(f"Beeswarm (standard) saved: {beeswarm_std}")

        # ----- Part 4: Beeswarm ordered by max |SHAP| -----
        beeswarm_ord = shap_dir / "shap_beeswarm_ordered_by_max_abs.png"
        shap.plots.beeswarm(shap_values, order=shap_values.abs.max(0), max_display=20, log_scale=True, show=False)
        fig = plt.gcf()
        plt.tight_layout()
        fig.set_size_inches(14, 8)
        plt.subplots_adjust(left=0.4, right=0.95)
        plt.savefig(beeswarm_ord, dpi=dpi, bbox_inches="tight")
        plt.close()
        log_info(f"Beeswarm (ordered by max |SHAP|) saved: {beeswarm_ord}")

        # ----- Part 5: Scatter plots for top features -----
        scatter_dir = shap_dir / "scatter_plots"
        scatter_dir.mkdir(parents=True, exist_ok=True)

        mean_abs = np.abs(shap_values.values).mean(axis=0)
        top_idx = np.argsort(mean_abs)[-15:]
        top_idx = top_idx[np.argsort(mean_abs[top_idx])[::-1]]
        top_features = list(X_transformed.columns[top_idx])

        for i, feat in enumerate(top_features, start=1):
            outpath = scatter_dir / f"{i:02d}_{feat}.png"
            shap.plots.scatter(shap_values[:, feat], color=shap_values, show=False)
            fig = plt.gcf()
            ax = plt.gca()
            if log_beeswarm:
                ax.set_xscale("log")
            fig.set_size_inches(10, 6)
            plt.tight_layout()
            plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
            plt.close()

        log_info(f"Scatter plots saved under: {scatter_dir}")
        return True, selected_features, shap_values, top_features

    except Exception as e:
        with open(report_path, "a", encoding="utf-8") as f_out:
            f_out.write(f"=== SHAP Analysis ({dataset_name}) ===\n")
            f_out.write(" SHAP not generated (unsupported model or error):\n")
            f_out.write(f"  {repr(e)}\n\n")
        log_error(f"SHAP analysis error for {dataset_name}: {e}")
        return False, None, None, None


# =========================
# -------  LIME  ----------
# =========================
def _extract_feature_name_from_lime(feat_str: str) -> str:
    """Extract original feature name from LIME token (removes ranges/extra text)."""
    tokens = re.findall(r"[A-Za-z0-9_\.\-]+", feat_str)
    valid = [t for t in tokens if re.search("[A-Za-z]", t)]
    if not valid:
        return feat_str.strip()
    return max(valid, key=len)


def explain_lime_instance(
    X_data: np.ndarray,
    index: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_clf,
    explainer: LimeTabularExplainer,
    lime_dir: str | Path,
    instance_label: str = "instance",
    dpi: int = DPI,
) -> None:
    """Generate & save LIME explanation for a single instance."""
    lime_dir = Path(lime_dir)
    lime_dir.mkdir(parents=True, exist_ok=True)

    exp = explainer.explain_instance(
        data_row=X_data[index], predict_fn=model_clf.predict_proba, num_features=10
    )

    txt_path = lime_dir / f"lime_explanation_{instance_label}_{index}.txt"
    fig_path = lime_dir / f"lime_explanation_{instance_label}_{index}.png"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"=== LIME Explanation for {instance_label} (index: {index}) ===\n\n")
        f.write(f"True label: {y_true[index]}\n")
        f.write(f"Predicted label: {y_pred[index]}\n")
        f.write(f"Probabilities: {model_clf.predict_proba([X_data[index]])}\n\n")
        f.write("Local feature importances:\n")
        for feat_info in exp.as_list():
            f.write(f"  {feat_info[0]}: {feat_info[1]:.4f}\n")

    with plt.style.context("default"):
        fig = exp.as_pyplot_figure()
        ax = plt.gca()

        pos_color = "#0072B2"
        neg_color = "#E69F00"

        # Recolor default red/green bars for consistency
        for rect in ax.patches:
            if rect.get_facecolor() == (0.0, 1.0, 0.0, 1.0):
                rect.set_facecolor(pos_color)
            elif rect.get_facecolor() == (1.0, 0.0, 0.0, 1.0):
                rect.set_facecolor(neg_color)

        plt.title(f"LIME Explanation - {instance_label} (index={index})")
        plt.savefig(fig_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    log_info(f"LIME saved for {instance_label} #{index}:\n  {txt_path}\n  {fig_path}")


def generate_lime_explanations_for_misclassifications(
    X_test_lime: np.ndarray,
    y_test: np.ndarray,
    model_clf,
    explainer: LimeTabularExplainer,
    lime_dir: str | Path,
) -> None:
    """
    Produce LIME explanations for representative examples of:
    TN (0→0), TP (1→1), FP (0→1), FN (1→0) on the test set.
    """
    y_pred = model_clf.predict(X_test_lime)

    tn_idx = np.where((y_test == 0) & (y_pred == 0))[0]
    tp_idx = np.where((y_test == 1) & (y_pred == 1))[0]
    fp_idx = np.where((y_test == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y_test == 1) & (y_pred == 0))[0]

    def _explain_first(indices: np.ndarray, label: str):
        if len(indices) > 0:
            idx = int(indices[0])
            explain_lime_instance(
                X_data=X_test_lime,
                index=idx,
                y_true=y_test,
                y_pred=y_pred,
                model_clf=model_clf,
                explainer=explainer,
                lime_dir=lime_dir,
                instance_label=label,
            )
        else:
            log_warn(f"No instances for {label}.")

    _explain_first(tn_idx, "TN")
    _explain_first(tp_idx, "TP")
    _explain_first(fp_idx, "FP")
    _explain_first(fn_idx, "FN")


def perform_lime_analysis(
    X_data: pd.DataFrame,
    y_data: np.ndarray,
    model_clf,
    preprocessor,
    lime_dir: str | Path,
    selected_features: pd.Index,
    report_path: str | Path,
    shap_top_features: List[str] | None = None,
    dataset_name: str = "dataset",
) -> bool:
    """
    Global LIME analysis:
      - Fit a tabular explainer on preprocessed features
      - Aggregate local explanations
      - Visualize top-15 features (min-max & log-normalized color scales)
      - Save representative instance explanations (TN/TP/FP/FN)
    """
    lime_dir = Path(lime_dir)
    lime_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_path)

    log_info(f"Running LIME analysis for {dataset_name}...")
    try:
        # Preprocess (replay pipeline)
        X_lime = preprocessor.transform(X_data)

        # Configure explainer
        explainer = LimeTabularExplainer(
            training_data=X_lime,
            feature_names=list(selected_features),
            class_names=["0", "1"],
            discretize_continuous=True,
            random_state=42,
        )

        # Collect local explanations across instances (class=1)
        records: List[dict] = []
        for i in range(len(X_lime)):
            exp = explainer.explain_instance(data_row=X_lime[i], predict_fn=model_clf.predict_proba)
            for (feat_str, weight) in exp.as_list(label=1):
                feat_name = _extract_feature_name_from_lime(feat_str)
                col_idx = selected_features.get_loc(feat_name) if feat_name in selected_features else None
                feat_value = X_lime[i, col_idx] if col_idx is not None else np.nan
                records.append(
                    {"instance": i, "feature": feat_name, "weight": weight, "feature_value": float(feat_value)}
                )

        df_lime = pd.DataFrame(records)
        df_lime["abs_weight"] = df_lime["weight"].abs()

        # Top-15 features by mean |weight|
        top_features = (
            df_lime.groupby("feature")["abs_weight"].mean().sort_values(ascending=False).head(15).index.tolist()
        )
        df_lime_top = df_lime[df_lime["feature"].isin(top_features)].copy()

        # Min-max normalization per feature
        df_lime_top["val_min"] = df_lime_top.groupby("feature")["feature_value"].transform("min")
        df_lime_top["val_max"] = df_lime_top.groupby("feature")["feature_value"].transform("max")
        df_lime_top["feature_value_norm"] = (df_lime_top["feature_value"] - df_lime_top["val_min"]) / (
            df_lime_top["val_max"] - df_lime_top["val_min"]
        )

        # Log-normalized variant
        def _shift_pos(x):
            return x - x.min() + 1e-10 if x.min() <= 0 else x

        df_lime_top["feature_value_shift"] = df_lime_top.groupby("feature")["feature_value"].transform(_shift_pos)
        df_lime_top["feature_value_log"] = np.log1p(df_lime_top["feature_value_shift"])
        df_lime_top["log_min"] = df_lime_top.groupby("feature")["feature_value_log"].transform("min")
        df_lime_top["log_max"] = df_lime_top.groupby("feature")["feature_value_log"].transform("max")
        df_lime_top["feature_value_log_norm"] = (df_lime_top["feature_value_log"] - df_lime_top["log_min"]) / (
            df_lime_top["log_max"] - df_lime_top["log_min"]
        )

        # Feature display order (prefer SHAP order if provided)
        lime_feats = df_lime_top["feature"].unique().tolist()
        if shap_top_features is not None:
            shap_order = list(shap_top_features)
            final_order = [f for f in shap_order if f in lime_feats] + [f for f in lime_feats if f not in shap_order]
        else:
            final_order = lime_feats

        # Custom colormap (shared)
        colors = [(0.0, "#008afb"), (0.2, "#008afb"), (0.7, "#ff0052"), (1.0, "#ff0052")]
        cmap = LinearSegmentedColormap.from_list("lime_cmap", colors)

        # Plot A: min-max normalized pseudo-beeswarm
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.stripplot(
            data=df_lime_top,
            x="weight",
            y="feature",
            hue="feature_value_norm",
            palette=cmap,
            hue_norm=(0, 1),
            orient="h",
            size=5,
            dodge=False,
            legend=False,
            order=final_order,
            ax=ax,
        )
        ax.axvline(0, color="black", linestyle="--")
        ax.set_title(f"LIME weights — Top 15 features ({dataset_name}, Min–Max normalization)")
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Feature value (normalized [0..1])")
        plt.tight_layout()
        fig_path = Path(lime_dir) / "lime_pseudo_beeswarm.png"
        plt.savefig(fig_path, dpi=DPI, bbox_inches="tight")
        plt.close()
        log_info(f"LIME visualization (min–max) saved: {fig_path}")

        # Plot B: log-normalized pseudo-beeswarm
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.stripplot(
            data=df_lime_top,
            x="weight",
            y="feature",
            hue="feature_value_log_norm",
            palette=cmap,
            hue_norm=(0, 1),
            orient="h",
            size=5,
            dodge=False,
            legend=False,
            order=final_order,
            ax=ax,
        )
        ax.axvline(0, color="black", linestyle="--")
        ax.set_title(f"LIME weights — Top 15 features ({dataset_name}, Log normalization)")
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Feature value (log-normalized [0..1])")
        plt.tight_layout()
        fig_path_log = Path(lime_dir) / "lime_pseudo_beeswarm_log.png"
        plt.savefig(fig_path_log, dpi=DPI, bbox_inches="tight")
        plt.close()
        log_info(f"LIME visualization (log) saved: {fig_path_log}")

        # Individual explanations: TN/TP/FP/FN
        indiv_dir = Path(lime_dir) / "individual_analysis"
        indiv_dir.mkdir(parents=True, exist_ok=True)
        generate_lime_explanations_for_misclassifications(
            X_test_lime=X_lime, y_test=y_data, model_clf=model_clf, explainer=explainer, lime_dir=indiv_dir
        )
        return True

    except Exception as e:
        with open(report_path, "a", encoding="utf-8") as f_out:
            f_out.write(f"\n=== LIME Analysis ({dataset_name}) ===\n")
            f_out.write(" LIME not generated (unsupported model or error):\n")
            f_out.write(f"  {repr(e)}\n\n")
        log_error(f"LIME analysis error for {dataset_name}: {e}")
        return False


# =========================
# -------- MAIN  ----------
# =========================
def main() -> None:
    """Fine-tune, evaluate, calibrate, and explain the selected model."""
    parser = argparse.ArgumentParser(
        description=(
            "Train and tune a model using CV on the training set and hold-out evaluation on a final test set. "
            "Then calibrate probabilities and run SHAP/LIME if possible."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="features_all_gland.csv",
        help="Path to the features CSV (default: features_all_gland.csv).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["SVM", "LogisticRegression", "RandomForest", "NaiveBayes", "KNN", "GradientBoosting"],
        help="Model to fine-tune/evaluate.",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of CV folds for BayesSearchCV (default: 5).",
    )
    parser.add_argument(
        "--variables",
        type=str,
        required=True,
        help="Path to selected features file (e.g., selected_features.txt).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v → INFO, -vv → DEBUG).",
    )
    args = parser.parse_args()
    setup_logging(args.verbose)

    log_info("Starting model fine-tuning.")
    log_info(f"Selected model: {args.model}")
    log_info(f"CSV used: {args.csv}")
    log_info(f"Variables file: {args.variables}")

    # Output dirs
    variables_path = Path(args.variables).resolve()
    base_dir = variables_path.parent
    out_dir = base_dir / "best_results"
    calib_dir = out_dir / "calibration"
    explain_dir = out_dir / "explicability"
    train_exp_dir = explain_dir / "train"
    test_exp_dir = explain_dir / "test"
    train_shap_dir = train_exp_dir / "SHAP"
    train_lime_dir = train_exp_dir / "LIME"
    test_shap_dir = test_exp_dir / "SHAP"
    test_lime_dir = test_exp_dir / "LIME"

    for d in [out_dir, calib_dir, explain_dir, train_exp_dir, test_exp_dir, train_shap_dir, train_lime_dir, test_shap_dir, test_lime_dir]:
        d.mkdir(parents=True, exist_ok=True)
    log_info(f"Output folder: {out_dir if out_dir.is_absolute() else out_dir}")

    # -------------------------------
    # 1) Load data and split
    # -------------------------------
    data_path = Path(args.csv)
    log_info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path, sep="\t")
    log_info(f"Loaded data: shape={df.shape}")

    if "Label" not in df.columns:
        log_error("Input CSV must contain a 'Label' column.")
        raise SystemExit(1)

    y = df["Label"].values
    X = df.drop(columns=["Label"])

    # 1.1) Filter to selected features
    log_info(f"Filtering features using: {args.variables}")
    used_vars = [line.strip() for line in Path(args.variables).read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = [c for c in used_vars if c not in X.columns]
    if missing:
        log_warn(f"{len(missing)} features listed but missing in CSV (first 5): {missing[:5]}")
    X = X[[c for c in used_vars if c in X.columns]]

    # Hold-out split (stratified)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -------------------------------
    # 2) Pipeline & search space
    # -------------------------------
    n_folds = int(args.n_folds)
    scoring = {"roc_auc": "roc_auc", "f1": "f1", "balanced_accuracy": "balanced_accuracy"}
    refit_metric = "roc_auc"
    rs = 42

    if args.model == "SVM":
        pipe = make_pipeline(StandardScaler(), VarianceThreshold(), SVC(random_state=rs, probability=True))
        search_space = {
            "svc__C": Real(1e-4, 1e3, prior="log-uniform"),
            "svc__kernel": Categorical(["linear", "rbf", "poly"]),
            "svc__gamma": Real(1e-4, 1e3, prior="log-uniform"),
            "svc__coef0": Real(0, 1),
        }

    elif args.model == "LogisticRegression":
        pipe = make_pipeline(
            StandardScaler(),
            VarianceThreshold(),
            LogisticRegression(class_weight="balanced", random_state=rs, solver="saga", max_iter=10_000),
        )
        search_space = {
            "logisticregression__C": Real(1e-4, 1e3, prior="log-uniform"),
            "logisticregression__penalty": Categorical(["l1", "l2", "elasticnet"]),
            "logisticregression__l1_ratio": Real(0.1, 0.9),
        }

    elif args.model == "RandomForest":
        pipe = make_pipeline(
            StandardScaler(),
            VarianceThreshold(),
            RandomForestClassifier(n_jobs=-1, class_weight="balanced_subsample", random_state=rs),
        )
        search_space = {
            "randomforestclassifier__n_estimators": Integer(50, 1024),
            "randomforestclassifier__max_depth": Integer(1, 10),
            "randomforestclassifier__max_features": Categorical(["sqrt", "log2", None]),
            "randomforestclassifier__min_samples_split": Integer(2, 20),
        }

    elif args.model == "NaiveBayes":
        pipe = make_pipeline(StandardScaler(), VarianceThreshold(), GaussianNB())
        search_space = {}  # no hyperparameters to tune

    elif args.model == "KNN":
        pipe = make_pipeline(StandardScaler(), VarianceThreshold(), KNeighborsClassifier(n_jobs=-1))
        search_space = {
            "kneighborsclassifier__n_neighbors": Integer(2, 8),
            "kneighborsclassifier__weights": Categorical(["uniform", "distance"]),
        }

    elif args.model == "GradientBoosting":
        pipe = make_pipeline(StandardScaler(), VarianceThreshold(), GradientBoostingClassifier(random_state=rs))
        search_space = {
            "gradientboostingclassifier__n_estimators": Integer(50, 1024),
            "gradientboostingclassifier__learning_rate": Real(1e-4, 0.1, prior="log-uniform"),
            "gradientboostingclassifier__max_depth": Integer(1, 10),
            "gradientboostingclassifier__subsample": Real(0.5, 1.0),
            "gradientboostingclassifier__max_features": Categorical(["sqrt", "log2", None]),
        }

    else:
        log_error(f"Unknown model '{args.model}'.")
        raise SystemExit(2)

    # -------------------------------
    # 3) Bayesian optimization (CV)
    # -------------------------------
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rs)
    log_info("Starting Bayesian optimization with BayesSearchCV...")

    search = BayesSearchCV(
        estimator=pipe,
        search_spaces=search_space,
        scoring=scoring,
        refit=refit_metric,
        cv=cv,
        n_jobs=-1,
        random_state=rs,
    )
    search.fit(X_train_full, y_train_full)
    best_estimator = search.best_estimator_
    log_info("Optimization complete.")
    log_info(f"Best params: {search.best_params_}")

    est_path = out_dir / "best_estimator.pkl"
    joblib.dump(best_estimator, est_path)
    log_info(f"Best estimator saved: {est_path}")

    # -------------------------------
    # 4) Report CV results
    # -------------------------------
    report_path = out_dir / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"=== Fine-tuning: {args.model} ===\n\n")
        f.write(f"Best params (by {refit_metric}): {search.best_params_}\n\n")
        f.write("=== CV Results (BayesSearch) ===\n")
        idx_best = search.best_index_
        for key in scoring:
            mean_test = search.cv_results_[f"mean_test_{key}"][idx_best]
            std_test = search.cv_results_[f"std_test_{key}"][idx_best]
            f.write(f"  CV {key}: {mean_test:.3f} +/- {std_test:.3f}\n")
        f.write("\n")
    log_info(f"Report initialized at: {report_path}")

    # -------------------------------
    # 5) Test evaluation (uncalibrated)
    # -------------------------------
    log_info("Evaluating on test set (uncalibrated)...")
    y_test_pred = best_estimator.predict(X_test)

    # Confusion matrix (uncalibrated)
    cm_fig = out_dir / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.grid(False)
    disp = ConfusionMatrixDisplay.from_estimator(best_estimator, X_test, y_test, ax=ax, cmap="cividis")
    ax.set_title(f"{args.model} (UNcalibrated)", fontsize=12)
    n_classes = len(disp.display_labels)
    ax.set_xticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="--", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.tight_layout()
    plt.savefig(cm_fig, dpi=DPI, bbox_inches="tight")
    plt.close()
    log_info(f"Confusion matrix saved: {cm_fig}")

    # Test metrics
    if hasattr(best_estimator, "predict_proba"):
        auc_ = roc_auc_score(y_test, best_estimator.predict_proba(X_test)[:, 1])
    elif hasattr(best_estimator, "decision_function"):
        auc_ = roc_auc_score(y_test, best_estimator.decision_function(X_test))
    else:
        auc_ = np.nan

    mcc = matthews_corrcoef(y_test, y_test_pred)
    kappa = cohen_kappa_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    acc = accuracy_score(y_test, y_test_pred)
    sens = recall_score(y_test, y_test_pred, pos_label=1)
    spec = recall_score(y_test, y_test_pred, pos_label=0)
    ppv = precision_score(y_test, y_test_pred, pos_label=1)
    npv = precision_score(y_test, y_test_pred, pos_label=0)
    balacc = balanced_accuracy_score(y_test, y_test_pred)
    report_cr = classification_report(y_test, y_test_pred)

    with open(report_path, "a", encoding="utf-8") as f:
        f.write("=== Test Evaluation (UNcalibrated) ===\n")
        f.write(f"  Confusion Matrix figure: {cm_fig}\n")
        f.write(f"  AUC: {auc_:.3f}\n")
        f.write(f"  MCC: {mcc:.3f}\n")
        f.write(f"  Kappa: {kappa:.3f}\n")
        f.write(f"  F1: {f1:.3f}\n")
        f.write(f"  Accuracy: {acc:.3f}\n")
        f.write(f"  Sensitivity: {sens:.3f}\n")
        f.write(f"  Specificity: {spec:.3f}\n")
        f.write(f"  PPV: {ppv:.3f}\n")
        f.write(f"  NPV: {npv:.3f}\n")
        f.write(f"  Balanced Accuracy: {balacc:.3f}\n\n")
        f.write("=== Classification Report ===\n")
        f.write(report_cr)
        f.write("\n\n")
    log_info("Uncalibrated test metrics appended to report.")

    # -------------------------------
    # 6) Calibration (Platt, cv=5) + threshold tuning
    # -------------------------------
    log_info("Calibrating with Platt scaling (sigmoid, cv=5)...")
    cal_clf = CalibratedClassifierCV(best_estimator, method="sigmoid", cv=5)
    cal_clf.fit(X_train_full, y_train_full)

    # Calibration plots
    cal_pre = calib_dir / "calibration_pre.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    CalibrationDisplay.from_estimator(best_estimator, X_test, y_test, n_bins=10, name=f"{args.model}_pre", ax=ax)
    for line in ax.get_lines():
        line.set_color("black")
    leg = ax.get_legend()
    if leg:
        for t in leg.get_texts():
            t.set_color("black")
        for line in leg.get_lines():
            line.set_color("black")
        for patch in leg.get_patches():
            patch.set_edgecolor("black")
            patch.set_facecolor("black")
    ax.set_title(f"Calibration Curve (pre), {args.model}", fontsize=14)
    plt.savefig(cal_pre, dpi=DPI, bbox_inches="tight")
    plt.close()
    log_info(f"Calibration (pre) saved: {cal_pre}")

    cal_post = calib_dir / "calibration_post.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    CalibrationDisplay.from_estimator(cal_clf, X_test, y_test, n_bins=10, name=f"{args.model}_post", ax=ax)
    ax.set_title(f"Calibration Curve (post), {args.model}", fontsize=14)
    for line in ax.get_lines():
        line.set_color("black")
    leg = ax.get_legend()
    if leg:
        for t in leg.get_texts():
            t.set_color("black")
        for line in leg.get_lines():
            line.set_color("black")
        for patch in leg.get_patches():
            patch.set_edgecolor("black")
            patch.set_facecolor("black")
    plt.savefig(cal_post, dpi=DPI, bbox_inches="tight")
    plt.close()
    log_info(f"Calibration (post) saved: {cal_post}")

    # Threshold sweep (optimize F1)
    thresholds = np.linspace(0.1, 0.9, 9)
    best_thresh, best_f1 = None, -np.inf
    sweep_rows: List[dict] = []
    probs = cal_clf.predict_proba(X_test)[:, 1]
    for th in thresholds:
        pred_th = (probs >= th).astype(int)
        f1_val = f1_score(y_test, pred_th)
        sweep_rows.append({"threshold": float(th), "f1": float(f1_val)})
        if f1_val > best_f1:
            best_f1, best_thresh = f1_val, float(th)

    y_best = (probs >= best_thresh).astype(int)
    auc_b = roc_auc_score(y_test, probs)
    mcc_b = matthews_corrcoef(y_test, y_best)
    kappa_b = cohen_kappa_score(y_test, y_best)
    f1_b = f1_score(y_test, y_best)
    acc_b = accuracy_score(y_test, y_best)
    sens_b = recall_score(y_test, y_best, pos_label=1)
    spec_b = recall_score(y_test, y_best, pos_label=0)
    ppv_b = precision_score(y_test, y_best, pos_label=1)
    npv_b = precision_score(y_test, y_best, pos_label=0)
    balacc_b = balanced_accuracy_score(y_test, y_best)
    report_cr_b = classification_report(y_test, y_best)

    with open(report_path, "a", encoding="utf-8") as f:
        f.write("=== Threshold Tuning (best threshold results) ===\n")
        f.write("Per-threshold results:\n")
        for r in sweep_rows:
            f.write(f"Threshold: {r['threshold']:.2f} - F1: {r['f1']:.3f}\n")
        f.write(f"\nBest threshold (by F1): {best_thresh:.2f}\n")
        f.write(f"\nClassification Report (threshold {best_thresh:.2f}):\n")
        f.write(report_cr_b)
        f.write("\n")
        f.write(f"AUC: {auc_b:.3f}\n")
        f.write(f"MCC: {mcc_b:.3f}\n")
        f.write(f"Kappa: {kappa_b:.3f}\n")
        f.write(f"F1: {f1_b:.3f}\n")
        f.write(f"Accuracy: {acc_b:.3f}\n")
        f.write(f"Sensitivity: {sens_b:.3f}\n")
        f.write(f"Specificity: {spec_b:.3f}\n")
        f.write(f"PPV: {ppv_b:.3f}\n")
        f.write(f"NPV: {npv_b:.3f}\n")
        f.write(f"Balanced Accuracy: {balacc_b:.3f}\n\n")
    log_info(f"Threshold tuning complete (best={best_thresh:.2f}, F1={best_f1:.3f}).")

    cm_best = confusion_matrix(y_test, y_best)
    cm_best_fig = calib_dir / "confusion_matrix_best_threshold.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.grid(False)
    ConfusionMatrixDisplay(confusion_matrix=cm_best).plot(ax=ax, cmap="cividis")
    ax.set_title(f"{args.model} (Calibrated, threshold={best_thresh:.2f})", fontsize=12)
    n_classes = cm_best.shape[0]
    ax.set_xticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="--", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.tight_layout()
    plt.savefig(cm_best_fig, dpi=DPI, bbox_inches="tight")
    plt.close()
    log_info(f"Confusion matrix (calibrated) saved: {cm_best_fig}")

    # -------------------------------
    # 7) Explainability: SHAP & LIME
    # -------------------------------
    preprocessor = deepcopy(best_estimator)
    preprocessor.steps.pop(-1)  # remove final classifier

    model_clf = best_estimator.steps[-1][1]

    # SHAP: train set
    train_ok, selected_feats, train_shap_values, train_top = perform_shap_analysis(
        X_data=X_train_full,
        y_data=y_train_full,
        model_clf=model_clf,
        preprocessor=preprocessor,
        shap_dir=train_shap_dir,
        report_path=report_path,
        dataset_name="train",
    )

    # SHAP: test set
    test_ok, _, test_shap_values, test_top = perform_shap_analysis(
        X_data=X_test,
        y_data=y_test,
        model_clf=model_clf,
        preprocessor=preprocessor,
        shap_dir=test_shap_dir,
        report_path=report_path,
        dataset_name="test",
    )

    # LIME: train/test (only if SHAP preprocessing succeeded)
    if train_ok and selected_feats is not None:
        _ = perform_lime_analysis(
            X_data=X_train_full,
            y_data=y_train_full,
            model_clf=model_clf,
            preprocessor=preprocessor,
            lime_dir=train_lime_dir,
            selected_features=selected_feats,
            report_path=report_path,
            shap_top_features=train_top,
            dataset_name="train",
        )

    if test_ok and selected_feats is not None:
        _ = perform_lime_analysis(
            X_data=X_test,
            y_data=y_test,
            model_clf=model_clf,
            preprocessor=preprocessor,
            lime_dir=test_lime_dir,
            selected_features=selected_feats,
            report_path=report_path,
            shap_top_features=test_top,
            dataset_name="test",
        )

    log_info(f"Done. Report saved to: {report_path}")


if __name__ == "__main__":
    main()
