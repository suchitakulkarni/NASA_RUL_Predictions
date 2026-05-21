#!/usr/bin/env python3
"""
Phase 1 EDA — six diagnostic plots that motivate every engineering decision.
Run from project root: python eda.py
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers the 3d projection

from src.utils import setup_logging
from src.data import load_raw
from src.condition_normaliser import ConditionNormaliser
from src.feature_engineering import DEFAULT_SENSORS_TO_DROP

RESULTS_EDA = os.path.join("results", "eda")
DATA_DIR = "CMAPSSData"
DATASETS = ["FD001", "FD002", "FD003", "FD004"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
OP_COLS = ["op1", "op2", "op3"]
SENSORS_TO_PLOT = [f"s{i}" for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
N_CLUSTERS = 6
N_UNITS_SAMPLE = 10
SEED = 42


def _cluster_colors():
    cmap = plt.get_cmap("tab10")
    return [cmap(i) for i in range(N_CLUSTERS)]


def _save(fig, path):
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved %s", path)


def _load_all_train():
    frames = []
    for ds_id in DATASETS:
        df, _, _ = load_raw(DATA_DIR, ds_id)
        df["dataset_id"] = ds_id
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    logging.info(
        "Combined train: %d rows, %d total units across %s",
        len(combined),
        combined.groupby(["dataset_id", "unit_no"]).ngroups,
        DATASETS,
    )
    return combined


def _fit_normaliser(all_train):
    norm = ConditionNormaliser(n_clusters=N_CLUSTERS, random_state=SEED)
    norm.fit(all_train, SENSOR_COLS)
    return norm


# ---------------------------------------------------------------------------
# 1.1  Raw sensor trajectories
# ---------------------------------------------------------------------------

def plot_1_1_raw_sensor_trajectories(norm, rng):
    logging.info("=== 1.1  Raw sensor trajectories ===")
    colors = _cluster_colors()
    n_cols = 4
    n_rows = (len(SENSORS_TO_PLOT) + n_cols - 1) // n_cols

    for ds_id in DATASETS:
        train_df, _, _ = load_raw(DATA_DIR, ds_id)
        all_units = train_df["unit_no"].unique()
        sampled = rng.choice(all_units, size=min(N_UNITS_SAMPLE, len(all_units)), replace=False)
        df_s = train_df[train_df["unit_no"].isin(sampled)].copy()
        df_s["cluster"] = norm.kmeans.predict(df_s[OP_COLS])

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows), squeeze=False)
        flat_axes = axes.flatten()

        for idx, sensor in enumerate(SENSORS_TO_PLOT):
            ax = flat_axes[idx]
            for unit in sampled:
                du = df_s[df_s["unit_no"] == unit].sort_values("time")
                ax.scatter(
                    du["time"].values,
                    du[sensor].values,
                    c=[colors[c] for c in du["cluster"].values],
                    s=3, alpha=0.5, linewidths=0,
                )
            ax.set_title(sensor, fontsize=9)
            ax.set_xlabel("Cycle", fontsize=7)
            ax.tick_params(labelsize=7)

        for idx in range(len(SENSORS_TO_PLOT), len(flat_axes)):
            flat_axes[idx].set_visible(False)

        handles = [
            plt.Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=colors[c], markersize=7,
                label=f"Cluster {c}",
            )
            for c in range(N_CLUSTERS)
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            fontsize=9, ncol=N_CLUSTERS,
            title="Op. condition cluster",
        )
        fig.suptitle(
            f"Raw sensor trajectories — {ds_id}  (n={len(sampled)} units, points coloured by cluster)",
            fontsize=12,
        )
        plt.tight_layout(rect=[0, 0.07, 1, 0.96])
        _save(fig, os.path.join(RESULTS_EDA, f"raw_sensor_trajectories_{ds_id.lower()}.png"))


# ---------------------------------------------------------------------------
# 1.2  Near-constant sensor identification
# ---------------------------------------------------------------------------

def plot_1_2_sensor_variance(all_train, norm):
    logging.info("=== 1.2  Sensor variance ===")
    # Stds must be computed on condition-normalised data: sensors that are constant
    # within each operating condition have near-zero std after normalisation regardless
    # of their raw scale (e.g. s1 has high raw std because it differs across conditions,
    # but zero within-condition variance and hence std ≈ 0 after normalisation).
    logging.info("Applying condition normalisation before computing per-sensor std")
    all_norm = norm.transform(all_train)
    stds = all_norm[SENSOR_COLS].std().sort_values()

    # Sensors constant within every cluster transform to std ≈ 0; s16 (near-constant) lands at 0.464.
    # s6/s10 normalise to std = 1 by construction despite low degradation signal — dropped by convention.
    threshold = 0.5

    fig, ax = plt.subplots(figsize=(14, 5))
    bar_colors = ["#d62728" if v <= threshold else "#1f77b4" for v in stds.values]
    # Use log scale so near-zero bars (std ≈ 0) are visible alongside std ≈ 1 sensors.
    # Clip to 1e-4 so zero-std sensors have a visible bar rather than disappearing entirely.
    plot_vals = stds.values.clip(min=1e-4)
    ax.bar(range(len(stds)), plot_vals, color=bar_colors, tick_label=stds.index)
    ax.set_yscale("log")
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1.2,
               label=f"Drop threshold = {threshold}  (below = near-constant within clusters)")
    ax.set_xlabel("Sensor")
    ax.set_ylabel("Standard deviation after condition normalisation (log scale)")
    ax.set_title(
        "Per-sensor std after condition normalisation\n"
        "Red bars = near-constant within every operating condition = no degradation signal"
    )
    ax.legend()
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()

    dropped = stds[stds <= threshold].index.tolist()
    logging.info("Sensors below threshold (std <= %.2f after condition normalisation): %s", threshold, dropped)
    known_drops = set(DEFAULT_SENSORS_TO_DROP)
    identified = set(dropped)
    missed = known_drops - identified
    if missed:
        logging.info(
            "Sensors in DEFAULT_SENSORS_TO_DROP not captured by threshold (%s) — "
            "excluded on domain knowledge / low degradation signal rather than near-zero variance",
            sorted(missed),
        )
    logging.info(
        "Agreement with DEFAULT_SENSORS_TO_DROP: matched=%s, extra=%s",
        sorted(identified & known_drops),
        sorted(identified - known_drops),
    )
    _save(fig, os.path.join(RESULTS_EDA, "sensor_variance.png"))


# ---------------------------------------------------------------------------
# 1.3  Operating condition clustering
# ---------------------------------------------------------------------------

def plot_1_3_operating_conditions(all_train, norm):
    logging.info("=== 1.3  Operating condition clusters ===")
    df_plot = all_train.sample(n=min(8000, len(all_train)), random_state=SEED)
    clusters = norm.kmeans.predict(df_plot[OP_COLS])
    colors = _cluster_colors()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for c in range(N_CLUSTERS):
        mask = clusters == c
        ax.scatter(
            df_plot.loc[mask, "op1"].values,
            df_plot.loc[mask, "op2"].values,
            df_plot.loc[mask, "op3"].values,
            color=colors[c], s=8, alpha=0.4, label=f"Cluster {c}",
        )

    ax.set_xlabel("op_setting_1", labelpad=8)
    ax.set_ylabel("op_setting_2", labelpad=8)
    ax.set_zlabel("op_setting_3", labelpad=8)
    ax.set_title(
        "Operating condition clusters (k=6)\nFD001/FD003 collapse to a single cluster",
        fontsize=11,
    )
    ax.legend(fontsize=9)

    centroids = pd.DataFrame(norm.kmeans.cluster_centers_, columns=OP_COLS)
    logging.info("Cluster centroids:\n%s", centroids.round(4).to_string())
    _save(fig, os.path.join(RESULTS_EDA, "operating_condition_clusters.png"))

    centroid_path = os.path.join(RESULTS_EDA, "cluster_centroids.csv")
    centroids.to_csv(centroid_path, index_label="cluster")
    logging.info("Cluster centroids saved to %s", centroid_path)


# ---------------------------------------------------------------------------
# 1.4  Before vs after condition normalisation
# ---------------------------------------------------------------------------

def plot_1_4_normalisation_effect(norm, rng):
    logging.info("=== 1.4  Normalisation effect ===")
    train_df, _, _ = load_raw(DATA_DIR, "FD002")
    all_units = train_df["unit_no"].unique()
    sampled = rng.choice(all_units, size=min(N_UNITS_SAMPLE, len(all_units)), replace=False)
    df_raw = train_df[train_df["unit_no"].isin(sampled)].copy()
    df_norm = norm.transform(df_raw)

    sensor = "s11"
    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for unit in sampled:
        du_raw  = df_raw[df_raw["unit_no"]   == unit].sort_values("time")
        du_norm = df_norm[df_norm["unit_no"] == unit].sort_values("time")
        ax_raw.plot(du_raw["time"].values,  du_raw[sensor].values,  alpha=0.7, linewidth=0.9)
        ax_norm.plot(du_norm["time"].values, du_norm[sensor].values, alpha=0.7, linewidth=0.9)

    ax_raw.set_title(f"FD002 — {sensor}  raw", fontsize=11)
    ax_raw.set_xlabel("Cycle")
    ax_raw.set_ylabel("Sensor value (raw)")

    ax_norm.set_title(f"FD002 — {sensor}  condition-normalised", fontsize=11)
    ax_norm.set_xlabel("Cycle")
    ax_norm.set_ylabel("Sensor value (normalised)")

    fig.suptitle(
        "Condition normalisation removes operating-condition variance, revealing degradation trend",
        fontsize=12,
    )
    plt.tight_layout()
    _save(fig, os.path.join(RESULTS_EDA, "normalisation_effect_sensor_11.png"))


# ---------------------------------------------------------------------------
# 1.5  Rolling statistics justification
# ---------------------------------------------------------------------------

def plot_1_5_rolling_stats(norm, rng):
    logging.info("=== 1.5  Rolling statistics justification ===")
    train_df, _, _ = load_raw(DATA_DIR, "FD004")
    max_cycles = train_df.groupby("unit_no")["time"].max()
    long_units = max_cycles[max_cycles >= 80].index
    unit_id = int(rng.choice(long_units))
    logging.info("Selected FD004 unit %d for rolling stats plot", unit_id)

    df_unit = train_df[train_df["unit_no"] == unit_id].copy().sort_values("time")
    df_norm = norm.transform(df_unit)

    sensor = "s11"
    windows = [5, 10, 20]
    sig = df_norm[sensor].values
    cycles = df_norm["time"].values

    fig, (ax_mean, ax_std) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax_mean.plot(cycles, sig, color="lightgray", linewidth=1.0, alpha=0.9, label="Normalised signal")
    for w in windows:
        roll_mean = pd.Series(sig).rolling(w, min_periods=1).mean().values
        ax_mean.plot(cycles, roll_mean, linewidth=1.8, label=f"Window = {w}")
    ax_mean.set_ylabel(f"{sensor} (normalised)")
    ax_mean.set_title(f"Rolling mean smooths cycle-to-cycle noise — FD004 unit {unit_id}")
    ax_mean.legend(fontsize=9)

    for w in windows:
        roll_std = pd.Series(sig).rolling(w, min_periods=2).std().fillna(0).values
        ax_std.plot(cycles, roll_std, linewidth=1.8, label=f"Window = {w}")
    ax_std.set_xlabel("Cycle")
    ax_std.set_ylabel(f"{sensor} rolling std")
    ax_std.set_title("Rolling std captures local noise level")
    ax_std.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, os.path.join(RESULTS_EDA, "rolling_stats_justification.png"))


# ---------------------------------------------------------------------------
# 1.6  RUL target design
# ---------------------------------------------------------------------------

def plot_1_6_rul_target():
    logging.info("=== 1.6  RUL target design ===")
    train_df, _, _ = load_raw(DATA_DIR, "FD001")
    max_cycles = train_df.groupby("unit_no")["time"].max()
    # Pick a unit long enough for capping to be clearly visible
    long_units = max_cycles[max_cycles > 150].index
    unit_id = int(long_units[0])
    logging.info("Selected FD001 unit %d for RUL target plot (max_cycle=%d)", unit_id, max_cycles[unit_id])

    df_unit = train_df[train_df["unit_no"] == unit_id].sort_values("time")
    max_cycle = df_unit["time"].max()
    cycles = df_unit["time"].values
    rul_linear = max_cycle - cycles
    cap = 125
    rul_capped = np.minimum(rul_linear, cap)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cycles, rul_linear, linewidth=2, label="Linear RUL (uncapped)", color="#1f77b4")
    ax.plot(cycles, rul_capped, linewidth=2, linestyle="--",
            label=f"Piecewise RUL (cap = {cap})", color="#d62728")
    ax.axhline(cap, color="black", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL (cycles remaining)")
    ax.set_title(
        f"RUL target design — FD001 unit {unit_id}\n"
        f"Capping at {cap} cycles removes uninformative early-life behaviour"
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save(fig, os.path.join(RESULTS_EDA, "rul_target_comparison.png"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    setup_logging()
    os.makedirs(RESULTS_EDA, exist_ok=True)
    rng = np.random.default_rng(SEED)

    logging.info("Phase 1 EDA starting — output dir: %s", RESULTS_EDA)

    all_train = _load_all_train()
    norm = _fit_normaliser(all_train)

    plot_1_1_raw_sensor_trajectories(norm, rng)
    plot_1_2_sensor_variance(all_train, norm)
    plot_1_3_operating_conditions(all_train, norm)
    plot_1_4_normalisation_effect(norm, rng)
    plot_1_5_rolling_stats(norm, rng)
    plot_1_6_rul_target()

    logging.info("Phase 1 EDA complete — 6 plot groups saved to %s", RESULTS_EDA)


if __name__ == "__main__":
    main()
