#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Statistical comparison of classification models.

This script:
- Performs a global Friedman test to check if there are significant differences
  in performance across multiple classifiers.
- If significant, runs pairwise Wilcoxon signed-rank tests with Holm correction
  for multiple comparisons.
- Saves a textual summary and generates visualizations:
  (1) Boxplot of validation metric distributions.
  (2) Heatmap of corrected pairwise p-values.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

# Plot backend (headless-safe) & style
mpl.use("Agg")
plt.style.use(["science", "grid"])
mpl.rcParams["text.usetex"] = False
DPI = 300

# -----------------------------
# Logging helpers
# -----------------------------
LOGGER = logging.getLogger("radiomics.stats")

def setup_logging(verbose: int = 0) -> None:
    """
    Configure logger level:
      0 → WARNING, 1 → INFO, 2+ → DEBUG
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

def log_err(msg: str) -> None:
    LOGGER.error(f"❌ {msg}")


def main() -> None:
    """
    Entry point:
    - Load model results (CSV).
    - Run Friedman test (global).
    - If significant, run pairwise Wilcoxon with Holm correction.
    - Save textual summary and plots.
    """
    parser = argparse.ArgumentParser(
        description="Statistical comparison of classifiers (Friedman + Wilcoxon post-hoc)."
    )
    parser.add_argument("--csv_preds", type=str, required=True, help="Path to predictions CSV.")
    parser.add_argument("--csv_results", type=str, required=True, help="Path to results-per-fold CSV.")
    parser.add_argument("--metric", type=str, default="val_auc", help="Metric column to compare.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (default=0.05).")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory for results.")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase logging (-v=INFO, -vv=DEBUG)")
    args = parser.parse_args()
    setup_logging(args.verbose)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Load results ---
    log_info(f"Loading results CSV: {args.csv_results}")
    try:
        df_results = pd.read_csv(args.csv_results)
    except Exception as e:
        log_err(f"Failed to read results CSV: {e}")
        raise SystemExit(1)

    metric_col = args.metric
    alpha = args.alpha

    if "Classifier" not in df_results.columns or "Fold" not in df_results.columns or metric_col not in df_results.columns:
        log_err("CSV must contain 'Classifier', 'Fold', and the selected metric column.")
        raise SystemExit(1)

    classifiers = df_results["Classifier"].unique()
    log_info(f"Found {len(classifiers)} classifiers: {list(classifiers)}")

    df_results = df_results.sort_values(by=["Classifier", "Fold"])

    # Pivot: rows = folds, cols = classifiers, values = metric
    pivot_df = df_results.pivot_table(index="Fold", columns="Classifier", values=metric_col)

    # Drop folds with NaNs
    if pivot_df.isnull().any().any():
        n_rows_before = pivot_df.shape[0]
        pivot_df = pivot_df.dropna(axis=0)
        n_dropped = n_rows_before - pivot_df.shape[0]
        log_warn(f"NaN values found; dropped {n_dropped} fold(s) with missing metrics.")

    if pivot_df.shape[0] < 2 or pivot_df.shape[1] < 2:
        log_err("Not enough data for statistical tests (need ≥2 folds and ≥2 classifiers).")
        raise SystemExit(1)

    # Order classifiers by median performance
    median_per_clf = df_results.groupby("Classifier")[metric_col].median().sort_values(ascending=False)
    ordered_classifiers = median_per_clf.index.tolist()
    pivot_df = pivot_df[ordered_classifiers]
    log_info("Classifiers ordered by median performance (desc).")

    # --- Friedman test ---
    log_info("Running Friedman test (global)...")
    data_for_friedman = [pivot_df[clf].values for clf in pivot_df.columns]
    stat, p_value = friedmanchisquare(*data_for_friedman)

    summary_lines: list[str] = []
    summary_lines.append("=" * 33)
    summary_lines.append(f"FRIEDMAN TEST for metric: {metric_col}")
    summary_lines.append(f"Statistic: {stat:.4f}, p-value: {p_value:.4e}")
    summary_lines.append(f"alpha = {alpha}")
    if p_value < alpha:
        summary_lines.append("=> Significant differences detected (reject H0).")
        log_info(f"Friedman significant (p={p_value:.3e}) → proceeding with pairwise tests.")
    else:
        summary_lines.append("=> No significant differences (fail to reject H0).")
        log_warn(f"Friedman not significant (p={p_value:.3e}).")

    summary_lines.append("=" * 33 + "\n")

    # --- Pairwise post-hoc Wilcoxon ---
    pairwise_matrix = None
    if p_value < alpha:
        clfs = pivot_df.columns.tolist()
        n_clfs = len(clfs)
        pairwise_matrix = np.ones((n_clfs, n_clfs))

        pvals, pairs = [], []
        log_info("Running pairwise Wilcoxon signed-rank tests with Holm correction...")
        for i in range(n_clfs):
            for j in range(i + 1, n_clfs):
                scores_i = pivot_df.iloc[:, i].values
                scores_j = pivot_df.iloc[:, j].values
                try:
                    w_stat, p_val_pair = wilcoxon(scores_i, scores_j, alternative="two-sided")
                except ValueError as e:
                    # Happens if differences are zero for all pairs
                    log_warn(f"Wilcoxon skipped for {clfs[i]} vs {clfs[j]}: {e}")
                    p_val_pair = 1.0
                pvals.append(p_val_pair)
                pairs.append((i, j))

        # Holm correction
        reject, pvals_corr, _, _ = multipletests(pvals, alpha=alpha, method="holm")

        # Fill matrix
        for (i, j), p_corr in zip(pairs, pvals_corr):
            pairwise_matrix[i, j] = pairwise_matrix[j, i] = p_corr

        summary_lines.append("Pairwise comparisons (Wilcoxon + Holm correction):")
        sig_lines = []
        for (i, j), p_corr, rej in zip(pairs, pvals_corr, reject):
            c1, c2 = clfs[i], clfs[j]
            res = f"    {c1} vs {c2}: corrected p={p_corr:.4e}"
            res += " => SIGNIFICANT" if rej else " => not significant"
            summary_lines.append(res)
            if rej:
                sig_lines.append(res)

        summary_lines.append("\nSignificant comparisons:")
        if sig_lines:
            summary_lines.extend(sig_lines)
        else:
            summary_lines.append("    None detected.")
    else:
        summary_lines.append("Skipping pairwise comparisons (Friedman not significant).")

    # Save summary
    summary_path = outdir / "model_differences_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    log_info(f"Statistical summary saved: {summary_path}")

    # --- Plot 1: Boxplot ---
    log_info("Creating boxplot of metric distributions...")
    plt.figure(figsize=(8, 5))
    pivot_df.boxplot(
        boxprops=dict(color="black"),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(color="black"),
    )
    plt.title(f"Distribution of {metric_col} by classifier")
    plt.ylabel(metric_col)
    plt.xticks(rotation=45, ha="right")
    boxplot_path = outdir / "boxplot_metric.png"
    plt.savefig(boxplot_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    log_info(f"Boxplot saved: {boxplot_path}")

    # --- Plot 2: Heatmap of corrected p-values ---
    if pairwise_matrix is not None:
        log_info("Creating heatmap of corrected pairwise p-values...")
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.imshow(pairwise_matrix, interpolation="nearest", cmap="cividis", aspect="auto")
        ax.set_title("Corrected pairwise p-values (Wilcoxon)")

        clfs = pivot_df.columns.tolist()
        ax.set_xticks(np.arange(len(clfs)))
        ax.set_yticks(np.arange(len(clfs)))
        ax.set_xticklabels(clfs, rotation=45, ha="right")
        ax.set_yticklabels(clfs)

        ax.set_xticks(np.arange(-0.5, len(clfs), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(clfs), 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="--", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

        for i in range(len(clfs)):
            for j in range(len(clfs)):
                pval_ij = pairwise_matrix[i, j]
                color = "white" if pval_ij < 0.05 else "black"
                ax.text(j, i, f"{pval_ij:.3f}", ha="center", va="center", color=color, fontsize=8)

        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        heatmap_path = outdir / "heatmap_pvalues.png"
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=DPI)
        plt.close()
        log_info(f"Heatmap saved: {heatmap_path}")


if __name__ == "__main__":
    main()
