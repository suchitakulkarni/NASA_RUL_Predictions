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


def plot_lead_time_distribution(per_unit_df, threshold, save_dir=EVAL_DIR):
    """
    Histogram of alarm lead times, split by TP (near-failure) and FP (healthy).

    per_unit_df : DataFrame returned by evaluation.lead_time_full_trajectory,
                  must contain 'needs_maintenance' and 'lead_time_cycles' columns.
    threshold   : maintenance alarm threshold in cycles.
    """
    os.makedirs(save_dir, exist_ok=True)

    detected = per_unit_df[
        per_unit_df["alarm_fired"] & (per_unit_df["lead_time_cycles"].fillna(-1) > 0)
    ]
    tp_df = detected[detected["needs_maintenance"]]    # near-failure, correctly caught
    fp_df = detected[~detected["needs_maintenance"]]   # healthy, falsely alarmed
    n_fn  = int(per_unit_df["needs_maintenance"].sum()) - len(tp_df)  # missed near-failure

    fig, ax = plt.subplots(figsize=(10, 5))

    bins = np.linspace(0, detected["lead_time_cycles"].max() + 10, 30)

    if len(tp_df) > 0:
        ax.hist(tp_df["lead_time_cycles"].values, bins=bins, color="green", alpha=0.75,
                label=f"TP — near-failure, correctly detected (n={len(tp_df)})")
    if len(fp_df) > 0:
        ax.hist(fp_df["lead_time_cycles"].values, bins=bins, color="orange", alpha=0.75,
                label=f"FP — healthy engines, false alarm (n={len(fp_df)})")
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2,
               label=f"Maintenance threshold ({threshold} cycles)")

    ax.set_xlabel("Lead Time: Cycles Before Engine Failure", fontsize=14)
    ax.set_ylabel("Number of Engines", fontsize=14)
    ax.set_title("Alarm Lead Time Distribution — Full Trajectory Analysis", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)

    all_lt = detected["lead_time_cycles"].values
    stats_text = (
        f"FN (missed near-failure): {n_fn}\n"
        f"Mean lead time (TP+FP): {np.mean(all_lt):.0f} cycles\n"
        f"Median lead time (TP+FP): {np.median(all_lt):.0f} cycles"
    )
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    fig.tight_layout()
    path = os.path.join(save_dir, "lead_time_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Lead-time distribution plot saved to %s", path)


def plot_business_summary(lead_time_metrics, improvement_metrics, cost_metrics,
                          save_dir=EVAL_DIR):
    """
    Three-panel business summary.

    Panel 1 — Detection: detected vs missed engines at the maintenance threshold,
               annotated with mean lead time.
    Panel 2 — RMSE improvement over naive mean predictor per dataset.
    Panel 3 — Cost savings vs reactive (wait-for-failure) baseline.

    lead_time_metrics : dict from evaluation.lead_time_full_trajectory
    improvement_metrics : list of {dataset, improvement_pct}
    cost_metrics : dict from evaluation.cost_savings_from_detections
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Panel 1: Confusion matrix as annotated 2×2 grid ---
    ax = axes[0]
    # Extra room: left for row labels, top for column headers
    ax.set_xlim(-0.65, 2.1)
    ax.set_ylim(-0.2, 2.7)
    ax.axis("off")

    lm = lead_time_metrics
    cells = [
        # (col, row, value, label, bg_color)
        (0, 1, lm["tp"], "TP\n(caught near-failure)", "green"),
        (1, 1, lm["fn"], "FN\n(missed near-failure)", "red"),
        (0, 0, lm["fp"], "FP\n(false alarm)",          "orange"),
        (1, 0, lm["tn"], "TN\n(correctly quiet)",      "blue"),
    ]
    for col, row, val, label, color in cells:
        ax.add_patch(plt.Rectangle(
            (col, row), 1, 1,
            color=color, alpha=0.25, linewidth=2, edgecolor=color,
        ))
        ax.text(col + 0.5, row + 0.65, str(val),
                ha="center", va="center", fontsize=16, fontweight="bold")
        ax.text(col + 0.5, row + 0.25, label,
                ha="center", va="center", fontsize=10)

    # Column headers — pure data coordinates
    ax.text(1.0, 2.55, "Alarm Fired", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(0.5, 2.25, "Yes", ha="center", va="center", fontsize=11)
    ax.text(1.5, 2.25, "No",  ha="center", va="center", fontsize=11)
    # Row headers — pure data coordinates
    ax.text(-0.1, 1.5, "Near\nfailure", ha="right", va="center", fontsize=11)
    ax.text(-0.1, 0.5, "Healthy",       ha="right", va="center", fontsize=11)

    thr = lm["threshold_cycles"]
    ax.set_title(
        f"Detection Confusion Matrix (threshold={thr} cycles)\n"
        f"Recall={lm['recall_pct']:.1f}%  Precision={lm['precision_pct']:.1f}%  "
        f"Specificity={lm['specificity_pct']:.1f}%",
        fontsize=12,
    )

    # --- Panel 2: RMSE improvement per dataset ---
    ax = axes[1]
    datasets     = [r["dataset"] for r in improvement_metrics]
    improvements = [r["improvement_pct"] for r in improvement_metrics]
    x = np.arange(len(datasets))
    bars = ax.bar(x, improvements, color="blue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.set_ylabel("RMSE Improvement (%)", fontsize=14)
    ax.set_title("Improvement vs\nNaive Mean Predictor", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.axhline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=12)

    # --- Panel 3: Cost breakdown for the at-risk fleet ---
    ax = axes[2]
    # Stacked bar: model cost = TP savings + FP waste + FN unplanned
    tp_cost   = cost_metrics["tp"] * cost_metrics["cost_planned"]   if "tp" in cost_metrics else 0
    fp_cost   = cost_metrics["false_alarm_cost"]
    fn_cost   = cost_metrics["fn"] * cost_metrics["cost_unplanned"] if "fn" in cost_metrics else 0
    baseline  = cost_metrics["baseline_cost"]

    x = np.array([0, 1])
    ax.bar([0], [baseline], color="red",    alpha=0.8, width=0.4, label="Unplanned failures")
    # Stacked model bar
    ax.bar([1], [tp_cost],          color="green",  alpha=0.8, width=0.4, label="Planned (TP)")
    ax.bar([1], [fp_cost],          color="orange", alpha=0.8, width=0.4,
           bottom=[tp_cost],              label="Wasted alarm (FP)")
    ax.bar([1], [fn_cost],          color="red",    alpha=0.5, width=0.4,
           bottom=[tp_cost + fp_cost],    label="Unplanned missed (FN)")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Reactive\nBaseline", "With Model"], fontsize=12)
    ax.set_ylabel("Cost for At-Risk Fleet ($)", fontsize=14)
    ax.set_title(
        f"Cost on {cost_metrics['n_near_failure']} Near-Failure Engines\n"
        f"(unplanned=${cost_metrics['cost_unplanned']:,.0f} / "
        f"planned=${cost_metrics['cost_planned']:,.0f})",
        fontsize=14,
    )
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.text(0.5, 0.97,
            f"Net savings: ${cost_metrics['net_savings']:,.0f} "
            f"({cost_metrics['savings_pct']:.1f}%)",
            transform=ax.transAxes, ha="center", va="top", fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("Business Value Summary — XGBoost + Conformal Prediction", fontsize=14)
    fig.tight_layout()
    path = os.path.join(save_dir, "business_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Business summary plot saved to %s", path)


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
        bars = ax.bar(x, data["coverage"], color="blue", alpha=0.8)
        ax.axhline(0.9, color="red", linestyle="--", linewidth=1.5, label="Target 90%")

        labels = [f"{r['dataset']}\n{r['cluster']}" for _, r in data.iterrows()]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Empirical Coverage", fontsize=14)
        ax.set_title(title, fontsize=14)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=12)

        for bar, (_, row) in zip(bars, data.iterrows()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{row['coverage']:.2f}\n(n={row['n_samples']})",
                ha="center", va="bottom", fontsize=12,
            )

    fig.suptitle("Conformal Prediction Coverage Validation", fontsize=14)
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
        ax.plot(cycles, unit["rul_pred"],  color="blue", linewidth=1.5, label="Predicted")
        ax.fill_between(
            cycles, unit["lower"], unit["upper"],
            color="blue", alpha=0.2, label="90% interval",
        )
        ax.axhline(
            lead_time_threshold, color="red",
            linestyle="--", linewidth=1, label=f"Lead time ({lead_time_threshold} cyc)",
        )
        ax.set_title(f"Unit {unit['unit_id']}", fontsize=14)
        ax.set_xlabel("Cycle", fontsize=14)
        ax.set_ylabel("RUL", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=12)
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
        "raw_sensor":   "blue",
        "rolling_mean": "red",
        "rolling_std":  "green",
        "dataset_id":   "magenta",
        "cycle_norm":   "cyan",
    }
    df["group"] = df["feature"].map(_group)
    colors = df["group"].map(group_colors)

    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.32)))
    ax.barh(df["feature"], df["importance"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Gain)", fontsize=14)
    ax.set_title(f"Top {len(df)} Feature Importances — Joint Model", fontsize=14)
    ax.tick_params(labelsize=12)

    handles = [
        mpatches.Patch(color=c, label=g.replace("_", " "))
        for g, c in group_colors.items()
        if g in df["group"].values
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=12)

    path = os.path.join(save_dir, "feature_importance.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance plot saved to %s", path)


def plot_per_dataset_comparison(ds_name, separate_data, joint_data, title, save_dir=EVAL_DIR):
    """
    Scatter of true vs predicted RUL for one dataset, overlaying separate and joint model.

    separate_data / joint_data: dict with keys
        {y_true, y_pred, rmse, lower (or None), upper (or None)}
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    all_vals = []

    for data, color, label_prefix in [
        (separate_data, "blue",  "Separate"),
        (joint_data,    "red",   "Joint"),
    ]:
        y_true = np.asarray(data["y_true"])
        y_pred = np.asarray(data["y_pred"])
        all_vals.extend(y_true.tolist())
        all_vals.extend(y_pred.tolist())

        lower = data.get("lower")
        upper = data.get("upper")
        label = f"{label_prefix}  RMSE={data['rmse']:.1f}"

        if lower is not None and upper is not None:
            err_lo = np.clip(y_pred - np.asarray(lower), 0, None)
            err_hi = np.clip(np.asarray(upper) - y_pred, 0, None)
            ax.errorbar(
                y_true, y_pred,
                yerr=[err_lo, err_hi],
                fmt="o", color=color, alpha=0.45, markersize=3,
                elinewidth=0.6, capsize=0,
                label=label,
            )
        else:
            ax.scatter(y_true, y_pred, alpha=0.5, s=20, color=color, label=label)

    lim = max(all_vals) * 1.05 if all_vals else 1.0
    ax.plot([0, lim], [0, lim], "k--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("True RUL", fontsize=14)
    ax.set_ylabel("Predicted RUL", fontsize=14)
    ax.set_title(f"{ds_name} — Separate vs Joint Model", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)

    plot_path = os.path.join(save_dir, f"{title}.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Comparison plot saved to %s", plot_path)
    return plot_path


def plot_pred_vs_true(dataset_results, title="pred_vs_true_combined", save_dir=EVAL_DIR):
    """
    Scatter plot of true vs predicted RUL, one series per dataset.

    When per-unit conformal intervals are available (keys "lower" and "upper" in
    each dataset dict), vertical error bars are drawn to show prediction uncertainty.

    dataset_results : dict {dataset_name: {
        "y_true":  ndarray,
        "y_pred":  ndarray,
        "rmse":    float,
        "lower":   ndarray or None,   # conformal lower bound per sample
        "upper":   ndarray or None,   # conformal upper bound per sample
    }}
    """
    os.makedirs(save_dir, exist_ok=True)
    colors = ["blue", "red", "green", "magenta"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    all_vals = []

    for (ds_name, data), color in zip(dataset_results.items(), colors):
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        all_vals.extend(y_true.tolist())
        all_vals.extend(y_pred.tolist())

        lower = data.get("lower")
        upper = data.get("upper")

        if lower is not None and upper is not None:
            err_lo = np.clip(y_pred - lower, 0, None)
            err_hi = np.clip(upper - y_pred, 0, None)
            ax.errorbar(
                y_true, y_pred,
                yerr=[err_lo, err_hi],
                fmt="o", color=color, alpha=0.45, markersize=3,
                elinewidth=0.6, capsize=0,
                label=f"{ds_name}  RMSE={data['rmse']:.1f}",
            )
        else:
            ax.scatter(
                y_true, y_pred,
                alpha=0.5, s=20, color=color,
                label=f"{ds_name}  RMSE={data['rmse']:.1f}",
            )

    lim = max(all_vals) * 1.05 if all_vals else 1.0
    ax.plot([0, lim], [0, lim], "k--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("True RUL", fontsize=14)
    ax.set_ylabel("Predicted RUL", fontsize=14)
    ax.set_title("Predicted vs True RUL", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)

    plot_path = os.path.join(save_dir, f"{title}.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved to %s", plot_path)
    return plot_path
