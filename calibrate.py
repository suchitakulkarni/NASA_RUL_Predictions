#!/usr/bin/env python3
"""
calibrate.py — Phase 5: conformal calibration for the joint RUL model.

Loads results/models/joint_model.pkl and results/models/condition_normaliser.pkl,
calibrates a ConformalPredictor on a held-out fraction of training units, saves
the calibrated predictor to results/models/conformal_predictor.pkl, and writes
the coverage validation plot.

Usage:
    python calibrate.py [--alpha 0.1] [--cal-frac 0.2] [--cap 125] [--window 5 10]
"""
import argparse
import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import setup_logging
from src.data import load_raw
from src.condition_normaliser import ConditionNormaliser
from src.feature_engineering import FeatureEngineer
from src.rul_target import compute_rul
from src.model import RULModel
from src.conformal import ConformalPredictor
from src import visualisation

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
DATA_DIR = "./CMAPSSData"
MODELS_DIR = "results/models"

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Conformal calibration for joint RUL model")
    parser.add_argument("--alpha",    type=float, default=0.1,
                        help="Miscoverage level (default 0.1 = 90% coverage target)")
    parser.add_argument("--cal-frac", type=float, default=0.2,
                        help="Fraction of training units to hold out for calibration")
    parser.add_argument("--cap",      type=int,   default=125,
                        help="RUL cap in cycles — must match train.py --cap")
    parser.add_argument("--window",   type=int, nargs="+", default=[5, 10],
                        help="Rolling window sizes — must match train.py --window")
    args = parser.parse_args()

    setup_logging()
    os.makedirs("results/evaluation", exist_ok=True)

    logger.info(
        "Conformal calibration: alpha=%.2f, cal_frac=%.2f, cap=%d, window=%s",
        args.alpha, args.cal_frac, args.cap, args.window,
    )

    model     = RULModel.load(os.path.join(MODELS_DIR, "joint_model.pkl"))
    normaliser = ConditionNormaliser.load(os.path.join(MODELS_DIR, "condition_normaliser.pkl"))

    # Rebuild FeatureEngineer with the same global max_cycle used during training
    all_train_parts = []
    for ds in DATASETS:
        train_df, _, _ = load_raw(DATA_DIR, ds)
        train_df = compute_rul(train_df, cap=args.cap)
        all_train_parts.append(train_df)
    all_train = pd.concat(all_train_parts, ignore_index=True)

    fe = FeatureEngineer(window_sizes=args.window)
    fe.fit(all_train)

    X_cal_parts, y_cal_parts, cluster_parts = [], [], []
    n_cal_per_ds = {}

    for i, ds in enumerate(DATASETS, start=1):
        train_df, _, _ = load_raw(DATA_DIR, ds)
        train_df = compute_rul(train_df, cap=args.cap)

        unique_units = train_df["unit_no"].unique()
        # random_state=99 gives a different split from the Optuna holdout (seed 42)
        _, cal_units = train_test_split(
            unique_units, test_size=args.cal_frac, random_state=99
        )

        cal_df = train_df[train_df["unit_no"].isin(cal_units)].copy()
        n_units_cal = len(cal_units)

        # Cluster assignment on raw op settings before normalisation
        clusters = normaliser.predict_clusters(cal_df)
        norm_cal = normaliser.transform(cal_df)
        eng_cal, feat_cols = fe.transform(norm_cal, dataset_id=i)

        X_cal_parts.append(eng_cal[feat_cols].values)
        y_cal_parts.append(eng_cal["RUL"].values)
        cluster_parts.append(clusters)
        n_cal_per_ds[ds] = len(eng_cal)

        logger.info(
            "%s calibration split: %d/%d units, %d rows",
            ds, n_units_cal, len(unique_units), len(eng_cal),
        )

    X_cal    = np.concatenate(X_cal_parts)
    y_cal    = np.concatenate(y_cal_parts)
    clusters = np.concatenate(cluster_parts)

    conformal = ConformalPredictor()
    conformal.calibrate(model, X_cal, y_cal, clusters, alpha=args.alpha)

    # Coverage validation records
    y_pred_cal, lower, upper = conformal.predict_with_interval(model, X_cal, clusters)
    covered = (y_cal >= lower) & (y_cal <= upper)

    coverage_records = []

    # Per-dataset overall coverage
    offset = 0
    for ds in DATASETS:
        n = n_cal_per_ds[ds]
        cov = float(covered[offset:offset + n].mean())
        coverage_records.append({
            "dataset": ds, "cluster": "overall",
            "coverage": cov, "n_samples": n,
        })
        logger.info("%s overall coverage: %.4f (n=%d)", ds, cov, n)
        offset += n

    # Per operating-condition cluster across all datasets
    for c in np.unique(clusters):
        mask = clusters == c
        cov  = float(covered[mask].mean())
        coverage_records.append({
            "dataset": "all",
            "cluster": f"cluster_{c}",
            "coverage": cov,
            "n_samples": int(mask.sum()),
        })

    visualisation.plot_conformal_coverage(coverage_records)

    conformal.save(os.path.join(MODELS_DIR, "conformal_predictor.pkl"))
    logger.info("Conformal calibration complete.")


if __name__ == "__main__":
    main()
