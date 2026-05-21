# src/visualisation.py
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EVAL_DIR = "results/evaluation"


def plot_conformal_coverage(coverage_records, save_dir=EVAL_DIR):
    """
    Two-panel bar chart: per-dataset overall coverage and per-cluster coverage.
    Draws a horizontal dashed line at the nominal 90% coverage target.

    coverage_records: list of dicts with keys
        {dataset, cluster, coverage, n_samples}
        where cluster is "overall" for per-dataset bars or "cluster_X" for
        per-cluster bars.
    """
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(coverage_records)

    per_ds  = df[df["cluster"] == "overall"].reset_index(drop=True)
    per_cls = df[df["cluster"] != "overall"].reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, data, title in [
        (axes[0], per_ds,  "Coverage per Dataset"),
        (axes[1], per_cls, "Coverage per Operating-Condition Cluster"),
    ]:
        x = np.arange(len(data))
        bars = ax.bar(x, data["coverage"], color="steelblue", alpha=0.8)
        ax.axhline(0.9, color="red", linestyle="--", linewidth=1.5, label="Target 90%")

        labels = [f"{r['dataset']}\n{r['cluster']}" for _, r in data.iterrows()]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Empirical Coverage")
        ax.set_title(title)
        ax.legend(fontsize=8)

        for bar, (_, row) in zip(bars, data.iterrows()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{row['coverage']:.2f}\n(n={row['n_samples']})",
                ha="center", va="bottom", fontsize=7,
            )

    fig.suptitle("Conformal Prediction Coverage Validation", fontsize=12)
    path = os.path.join(save_dir, "conformal_coverage.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Coverage plot saved to %s", path)


def plot_rul_predictions(unit_dfs, save_path, lead_time_threshold=30):
    """
    Per-unit subplot: true RUL trajectory, predicted RUL, and shaded 90%
    conformal interval.  Annotates the lead-time threshold.

    unit_dfs: list of dicts with keys
        {unit_id, cycles, rul_true, rul_pred, lower, upper}
    save_path: output .png path
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    n = len(unit_dfs)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, unit in enumerate(unit_dfs):
        ax = axes[idx // ncols][idx % ncols]
        cycles = np.asarray(unit["cycles"])
        ax.plot(cycles, unit["rul_true"],  color="black",  linewidth=2,   label="True RUL")
        ax.plot(cycles, unit["rul_pred"],  color="#1f77b4", linewidth=1.5, label="Predicted")
        ax.fill_between(
            cycles, unit["lower"], unit["upper"],
            color="#1f77b4", alpha=0.2, label="90% interval",
        )
        ax.axhline(
            lead_time_threshold, color="darkorange",
            linestyle="--", linewidth=1, label=f"Lead time ({lead_time_threshold} cyc)",
        )
        ax.set_title(f"Unit {unit['unit_id']}", fontsize=10)
        ax.set_xlabel("Cycle")
        ax.set_ylabel("RUL")
        ax.legend(fontsize=7)
        ax.set_ylim(bottom=0)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("RUL prediction plot saved to %s", save_path)


def plot_feature_importance(model, feat_cols, save_dir=EVAL_DIR, top_n=30):
    """
    Horizontal bar chart of XGBoost gain importance, colour-coded by feature
    group: raw_sensor, rolling_mean, rolling_std, cycle_norm, dataset_id.
    """
    os.makedirs(save_dir, exist_ok=True)

    importances = model.model.feature_importances_
    df = pd.DataFrame({"feature": feat_cols, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)

    def _group(name):
        if "_roll_mean_" in name:
            return "rolling_mean"
        if "_roll_std_" in name:
            return "rolling_std"
        if name == "dataset_id":
            return "dataset_id"
        if name == "cycle_norm":
            return "cycle_norm"
        return "raw_sensor"

    group_colors = {
        "raw_sensor":   "#4C72B0",
        "rolling_mean": "#DD8452",
        "rolling_std":  "#55A868",
        "dataset_id":   "#C44E52",
        "cycle_norm":   "#8172B2",
    }
    df["group"] = df["feature"].map(_group)
    colors = df["group"].map(group_colors)

    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.32)))
    ax.barh(df["feature"], df["importance"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title(f"Top {len(df)} Feature Importances — Joint Model")

    handles = [
        mpatches.Patch(color=c, label=g.replace("_", " "))
        for g, c in group_colors.items()
        if g in df["group"].values
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9)

    path = os.path.join(save_dir, "feature_importance.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance plot saved to %s", path)
