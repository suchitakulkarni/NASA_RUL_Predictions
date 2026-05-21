#!/usr/bin/env python3
"""
ablation.py — Phase 2 ablation study for the CMAPSS RUL pipeline.

Produces:
  results/ablation/cross_dataset_rmse_raw.png
  results/ablation/cross_dataset_rmse_normalised.png
  results/ablation/cross_dataset_rmse_comparison.png
  results/ablation/separate_vs_joint_rmse.png

Usage:
    python ablation.py --trials 50
    python ablation.py --trials 50 --cap 125 --window 5 10
"""
import argparse
import concurrent.futures
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.utils import setup_logging
from src.data import load_raw
from src.condition_normaliser import ConditionNormaliser
from src.feature_engineering import FeatureEngineer
from src.rul_target import compute_rul
from src.model import RULModel

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
DATA_DIR = "./CMAPSSData"
SENSOR_COLS = ["s" + str(i) for i in range(1, 22)]
ABLATION_DIR = "results/ablation"
ABLATION_MODELS_DIR = os.path.join(ABLATION_DIR, "models")

logger = logging.getLogger(__name__)


# ── CPU parallelism helpers ───────────────────────────────────────────────────

def _n_workers():
    """Number of outer parallel training jobs — one per dataset."""
    return min(len(DATASETS), os.cpu_count() or 4)


def _per_model_jobs():
    """XGBoost threads per model when running _n_workers() models in parallel."""
    return max(1, (os.cpu_count() or 4) // _n_workers())


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_all(cap):
    out = {}
    for ds in DATASETS:
        train_df, test_df, rul_labels = load_raw(DATA_DIR, ds)
        train_df = compute_rul(train_df, cap=cap)
        out[ds] = (train_df, test_df, rul_labels)
    return out


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _last_cycle(df):
    return df.sort_values("time").groupby("unit_no").last().reset_index()


# ── 2.1  Raw cross-dataset heatmap ───────────────────────────────────────────
# Features: op1/2/3 + s1-s21 normalised by StandardScaler fit on train_i.
# No rolling statistics. No condition normalisation.

def _cross_dataset_rmse_raw(dataset_map, cap):
    raw_feat_cols = ["op1", "op2", "op3"] + SENSOR_COLS

    # Pre-compute: scaler+train matrix per training dataset,
    # and raw (unscaled) last-cycle test arrays per test dataset.
    train_data = {}
    test_raw = {}
    for ds, (train_df, test_df, rul_labels) in dataset_map.items():
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train_df[raw_feat_cols].values)
        y_tr = train_df["RUL"].values
        last = _last_cycle(test_df).merge(rul_labels, on="unit_no")
        train_data[ds] = (X_tr, y_tr, scaler)
        test_raw[ds] = (last[raw_feat_cols].values, last["RUL"].clip(0, cap).values)

    n_jobs = _per_model_jobs()

    def _train_row(train_ds):
        X_tr, y_tr, scaler = train_data[train_ds]
        model = XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            verbosity=0, random_state=42, n_jobs=n_jobs,
        )
        model.fit(X_tr, y_tr, verbose=False)
        logger.info("2.1 | trained on %s (%d samples)", train_ds, len(X_tr))
        row = {}
        for test_ds in DATASETS:
            X_test_raw, y_test = test_raw[test_ds]
            X_test = scaler.transform(X_test_raw)
            row[test_ds] = _rmse(y_test, model.predict(X_test))
            logger.info("2.1 | train=%-6s test=%-6s RMSE=%.2f",
                        train_ds, test_ds, row[test_ds])
        return train_ds, row

    n = len(DATASETS)
    matrix = np.zeros((n, n))
    with concurrent.futures.ThreadPoolExecutor(max_workers=_n_workers()) as pool:
        for train_ds, row in pool.map(_train_row, DATASETS):
            i = DATASETS.index(train_ds)
            for j, test_ds in enumerate(DATASETS):
                matrix[i, j] = row[test_ds]

    return matrix


# ── 2.2  Condition-normalised cross-dataset heatmap ──────────────────────────
# Features: global ConditionNormaliser + FeatureEngineer (no dataset_id).

def _cross_dataset_rmse_normalised(dataset_map, normaliser, fe, cap):
    # Pre-compute normalised+engineered data for all datasets
    norm_data = {}
    for ds, (train_df, test_df, rul_labels) in dataset_map.items():
        eng_train, feat_cols = fe.transform(
            normaliser.transform(train_df), dataset_id=None
        )
        eng_test, _ = fe.transform(
            normaliser.transform(test_df), dataset_id=None
        )
        last = _last_cycle(eng_test).merge(rul_labels, on="unit_no")
        norm_data[ds] = (
            eng_train[feat_cols].values,
            eng_train["RUL"].values,
            feat_cols,
            last[feat_cols].values,
            last["RUL"].clip(0, cap).values,
        )

    n_jobs = _per_model_jobs()

    def _train_row(train_ds):
        X_tr, y_tr, feat_cols, _, _ = norm_data[train_ds]
        model = XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            verbosity=0, random_state=42, n_jobs=n_jobs,
        )
        model.fit(X_tr, y_tr, verbose=False)
        logger.info("2.2 | trained on %s (%d samples)", train_ds, len(X_tr))
        row = {}
        for test_ds in DATASETS:
            _, _, _, X_test, y_test = norm_data[test_ds]
            row[test_ds] = _rmse(y_test, model.predict(X_test))
            logger.info("2.2 | train=%-6s test=%-6s RMSE=%.2f",
                        train_ds, test_ds, row[test_ds])
        return train_ds, row

    n = len(DATASETS)
    matrix = np.zeros((n, n))
    with concurrent.futures.ThreadPoolExecutor(max_workers=_n_workers()) as pool:
        for train_ds, row in pool.map(_train_row, DATASETS):
            i = DATASETS.index(train_ds)
            for j, test_ds in enumerate(DATASETS):
                matrix[i, j] = row[test_ds]

    return matrix


# ── 2.3  Separate vs Joint model comparison ───────────────────────────────────

def _eval_model(model, last_df, rul_labels, feat_cols, cap):
    last = last_df.merge(rul_labels, on="unit_no")
    y_test = last["RUL"].clip(0, cap).values
    return _rmse(y_test, model.predict(last[feat_cols].values))


def _separate_rmse(dataset_map, normaliser, fe, cap, trials):
    # Pre-compute normalised features outside the threads (normaliser.transform
    # is read-only, safe to call from multiple threads, but doing it here keeps
    # each worker function self-contained and avoids redundant work).
    prepared = {}
    for i, ds in enumerate(DATASETS, start=1):
        train_df, test_df, rul_labels = dataset_map[ds]
        norm_train = normaliser.transform(train_df)
        norm_test = normaliser.transform(test_df)
        eng_train, feat_cols = fe.transform(norm_train, dataset_id=None)
        eng_test, _ = fe.transform(norm_test, dataset_id=None)
        prepared[ds] = (eng_train, feat_cols, _last_cycle(eng_test), rul_labels)

    n_jobs = _per_model_jobs()

    def _train_one(ds):
        eng_train, feat_cols, last_test, rul_labels = prepared[ds]
        model_path = os.path.join(ABLATION_MODELS_DIR, f"separate_{ds}.pkl")
        if os.path.exists(model_path):
            logger.info("2.3 | loading %s", model_path)
            model = RULModel.load(model_path)
        else:
            logger.info("2.3 | training separate model for %s (%d samples, %d trials)",
                        ds, len(eng_train), trials)
            model = RULModel()
            model.train(
                eng_train[feat_cols].values, eng_train["RUL"].values,
                optuna_trials=trials, n_jobs=n_jobs,
            )
            model.save(model_path)
        rmse = _eval_model(model, last_test, rul_labels, feat_cols, cap)
        logger.info("2.3 | separate %-6s RMSE=%.2f", ds, rmse)
        return ds, rmse

    result = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=_n_workers()) as pool:
        for ds, rmse in pool.map(_train_one, DATASETS):
            result[ds] = rmse

    return result


def _joint_rmse(dataset_map, normaliser, fe, cap, trials):
    X_parts, y_parts, strat_parts, test_data = [], [], [], {}

    for i, ds in enumerate(DATASETS, start=1):
        train_df, test_df, rul_labels = dataset_map[ds]
        norm_train = normaliser.transform(train_df)
        norm_test = normaliser.transform(test_df)
        eng_train, feat_cols = fe.transform(norm_train, dataset_id=i)
        eng_test, _ = fe.transform(norm_test, dataset_id=i)
        X_parts.append(eng_train[feat_cols].values)
        y_parts.append(eng_train["RUL"].values)
        strat_parts.append(np.full(len(eng_train), i, dtype=int))
        test_data[ds] = (_last_cycle(eng_test), rul_labels, feat_cols)

    model_path = os.path.join(ABLATION_MODELS_DIR, "joint.pkl")
    if os.path.exists(model_path):
        logger.info("2.3 | loading existing joint model")
        model = RULModel.load(model_path)
    else:
        X_all = np.concatenate(X_parts)
        y_all = np.concatenate(y_parts)
        strat = np.concatenate(strat_parts)
        logger.info("2.3 | training joint model (%d samples, %d trials)", len(X_all), trials)
        model = RULModel()
        # Joint model trains on all data sequentially — use all cores.
        model.train(X_all, y_all, optuna_trials=trials, stratify=strat, n_jobs=-1)
        model.save(model_path)

    result = {}
    for ds, (last_test, rul_labels, feat_cols) in test_data.items():
        rmse = _eval_model(model, last_test, rul_labels, feat_cols, cap)
        result[ds] = rmse
        logger.info("2.3 | joint    %-6s RMSE=%.2f", ds, rmse)

    return result


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_heatmap(matrix, title, path, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True, fmt=".1f",
        xticklabels=DATASETS, yticklabels=DATASETS,
        cmap="YlOrRd", vmin=vmin, vmax=vmax,
        ax=ax,
        cbar_kws={"label": "RMSE (cycles)"},
    )
    ax.set_xlabel("Test Dataset")
    ax.set_ylabel("Train Dataset")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved: %s", path)


def _plot_heatmap_comparison(raw_matrix, norm_matrix, path):
    vmin = min(raw_matrix.min(), norm_matrix.min())
    vmax = max(raw_matrix.max(), norm_matrix.max())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, matrix, subtitle in zip(
        axes,
        [raw_matrix, norm_matrix],
        ["Raw Features", "Condition-Normalised Features"],
    ):
        sns.heatmap(
            matrix,
            annot=True, fmt=".1f",
            xticklabels=DATASETS, yticklabels=DATASETS,
            cmap="YlOrRd", vmin=vmin, vmax=vmax,
            ax=ax,
            cbar_kws={"label": "RMSE (cycles)"},
        )
        ax.set_xlabel("Test Dataset")
        ax.set_ylabel("Train Dataset")
        ax.set_title(f"Cross-Dataset RMSE\n{subtitle}")

    plt.suptitle(
        "Effect of Condition Normalisation on Cross-Dataset Generalisation",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", path)


def _plot_bar_chart(separate_rmse, joint_rmse, path):
    x = np.arange(len(DATASETS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(
        x - width / 2, [separate_rmse[ds] for ds in DATASETS],
        width, label="Separate model", color="#4C72B0",
    )
    bars2 = ax.bar(
        x + width / 2, [joint_rmse[ds] for ds in DATASETS],
        width, label="Joint model", color="#DD8452",
    )
    ax.set_xlabel("Dataset")
    ax.set_ylabel("RMSE (cycles)")
    ax.set_title("Separate vs Joint Model — Test RMSE per Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.legend()
    ax.bar_label(bars1, fmt="%.1f", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.1f", padding=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved: %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2 ablation study")
    parser.add_argument("--cap",    type=int, default=125,
                        help="RUL cap in cycles (default 125)")
    parser.add_argument("--window", type=int, nargs="+", default=[5, 10],
                        help="Rolling window sizes for normalised features (default 5 10)")
    parser.add_argument("--trials", type=int, default=50,
                        help="Optuna trials for 2.3 separate/joint models (default 50)")
    args = parser.parse_args()

    setup_logging()
    os.makedirs(ABLATION_DIR, exist_ok=True)
    os.makedirs(ABLATION_MODELS_DIR, exist_ok=True)

    n_cpu = os.cpu_count() or 4
    logger.info(
        "Phase 2 ablation | cap=%d, windows=%s, optuna_trials=%d, "
        "workers=%d, xgb_jobs_per_model=%d, total_cpus=%d",
        args.cap, args.window, args.trials,
        _n_workers(), _per_model_jobs(), n_cpu,
    )

    dataset_map = _load_all(args.cap)

    # Fit global normaliser and feature engineer on all training data combined
    all_train = pd.concat(
        [train_df for train_df, _, _ in dataset_map.values()], ignore_index=True
    )
    sensor_cols_present = [c for c in SENSOR_COLS if c in all_train.columns]
    normaliser = ConditionNormaliser(n_clusters=6)
    normaliser.fit(all_train, sensor_cols_present)

    fe = FeatureEngineer(window_sizes=args.window)
    fe.fit(all_train)

    # 2.1 — Raw cross-dataset heatmap
    logger.info("=== 2.1: Raw cross-dataset RMSE heatmap ===")
    raw_matrix = _cross_dataset_rmse_raw(dataset_map, args.cap)
    _plot_heatmap(
        raw_matrix,
        "Cross-Dataset RMSE (Raw Features)",
        os.path.join(ABLATION_DIR, "cross_dataset_rmse_raw.png"),
    )

    # 2.2 — Normalised cross-dataset heatmap
    logger.info("=== 2.2: Normalised cross-dataset RMSE heatmap ===")
    norm_matrix = _cross_dataset_rmse_normalised(dataset_map, normaliser, fe, args.cap)
    _plot_heatmap(
        norm_matrix,
        "Cross-Dataset RMSE (Condition-Normalised Features)",
        os.path.join(ABLATION_DIR, "cross_dataset_rmse_normalised.png"),
    )
    _plot_heatmap_comparison(
        raw_matrix, norm_matrix,
        os.path.join(ABLATION_DIR, "cross_dataset_rmse_comparison.png"),
    )

    # 2.3 — Separate vs Joint with Optuna
    logger.info("=== 2.3: Separate vs Joint model RMSE ===")
    sep_rmse = _separate_rmse(dataset_map, normaliser, fe, args.cap, args.trials)
    jnt_rmse = _joint_rmse(dataset_map, normaliser, fe, args.cap, args.trials)
    _plot_bar_chart(
        sep_rmse, jnt_rmse,
        os.path.join(ABLATION_DIR, "separate_vs_joint_rmse.png"),
    )

    logger.info("Phase 2 ablation complete.")


if __name__ == "__main__":
    main()
