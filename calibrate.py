#!/usr/bin/env python3
"""
calibrate.py — Phase 5: conformal calibration for the joint RUL model.

Loads the joint model and condition normaliser produced by train.py --mode joint,
calibrates a ConformalPredictor on a held-out fraction of training units, and
saves the result alongside the coverage validation plot.

The --run-tag and --features arguments must match what was passed to train.py so
the correct artifact files are found.

Usage:
    python calibrate.py [--alpha 0.1] [--cal-frac 0.2] [--cap 125]
    python calibrate.py --features engineered --run-tag my_exp
"""
import argparse
import logging
import os

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

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
    parser.add_argument("--features", choices=["raw", "engineered"], default="engineered",
                        help="Feature set used when training — determines which "
                             "condition_normaliser artifact to load (default: engineered)")
    parser.add_argument("--run-tag",  type=str, default=None,
                        help="Artifact tag passed to train.py; defaults to "
                             "'joint_{features}'. Must match the tag used during training.")
    args = parser.parse_args()

    setup_logging()
    os.makedirs("results/evaluation", exist_ok=True)

    artifact_tag = args.run_tag if args.run_tag else f"joint_{args.features}"
    logger.info(
        "Conformal calibration: alpha=%.2f, cal_frac=%.2f, cap=%d, artifact_tag=%s",
        args.alpha, args.cal_frac, args.cap, artifact_tag,
    )

    model = RULModel.load(os.path.join(MODELS_DIR, f"{artifact_tag}_model.pkl"))
    normaliser = (
        ConditionNormaliser.load(os.path.join(MODELS_DIR, f"condition_normaliser_{artifact_tag}.pkl"))
        if args.features == "engineered" else None
    )

    # Load the saved FeatureEngineer — window sizes and sensor drops must match training exactly.
    fe_path = os.path.join(MODELS_DIR, f"feature_engineer_{artifact_tag}.pkl")
    if not os.path.exists(fe_path):
        raise FileNotFoundError(
            f"{fe_path} not found — run train.py --mode joint --features {args.features} first."
        )
    fe = FeatureEngineer.load(fe_path)
    logger.info("FeatureEngineer loaded from %s (window_sizes=%s)", fe_path, fe.window_sizes)

    X_cal_parts, y_cal_parts, cluster_parts = [], [], []
    X_fit_parts, y_fit_parts = [], []
    n_cal_per_ds = {}
    feat_cols = None

    for i, ds in enumerate(DATASETS, start=1):
        train_df, _, _ = load_raw(DATA_DIR, ds)
        train_df = compute_rul(train_df, cap=args.cap)

        unique_units = train_df["unit_no"].unique()
        # random_state=99 gives a different split from the Optuna holdout (seed 42)
        fit_units, cal_units = train_test_split(
            unique_units, test_size=args.cal_frac, random_state=99
        )

        cal_df = train_df[train_df["unit_no"].isin(cal_units)].copy()
        fit_df = train_df[train_df["unit_no"].isin(fit_units)].copy()

        if args.features == "engineered":
            clusters = normaliser.predict_clusters(cal_df)
            norm_cal = normaliser.transform(cal_df)
            norm_fit = normaliser.transform(fit_df)
            eng_cal, feat_cols = fe.transform(norm_cal, dataset_id=i)
            eng_fit, _         = fe.transform(norm_fit, dataset_id=i)
        else:
            clusters = np.zeros(len(cal_df), dtype=int)
            feat_cols = ["time"] + list(fe.sensor_cols) + ["dataset_id"]
            cal_df["dataset_id"] = i
            fit_df["dataset_id"] = i
            eng_cal = cal_df
            eng_fit = fit_df

        X_cal_parts.append(eng_cal[feat_cols].values)
        y_cal_parts.append(eng_cal["RUL"].values)
        cluster_parts.append(clusters)
        X_fit_parts.append(eng_fit[feat_cols].values)
        y_fit_parts.append(eng_fit["RUL"].values)
        n_cal_per_ds[ds] = len(eng_cal)

        logger.info(
            "%s: %d fit units / %d cal units — %d fit rows, %d cal rows",
            ds, len(fit_units), len(cal_units), len(eng_fit), len(eng_cal),
        )

    X_cal    = np.concatenate(X_cal_parts)
    y_cal    = np.concatenate(y_cal_parts)
    clusters = np.concatenate(cluster_parts)
    X_fit    = np.concatenate(X_fit_parts)
    y_fit    = np.concatenate(y_fit_parts)

    # Refit a temporary model on fit_units (cal_units excluded) using the same
    # hyperparameters found during training. Calibrating on the full training
    # model's residuals would be invalid — the model has already seen those rows
    # and its residuals are too small, giving intervals that are too narrow.
    mono = None
    if feat_cols is not None and "cycle_norm" in feat_cols:
        mono = tuple(-1 if c == "cycle_norm" else 0 for c in feat_cols)

    cal_params = {**model.best_params, "verbosity": 0, "random_state": 42}
    if mono is not None:
        cal_params["monotone_constraints"] = mono

    logger.info(
        "Refitting temporary calibration model on %d rows (%.0f%% of training data)",
        len(X_fit), 100.0 * len(X_fit) / (len(X_fit) + len(X_cal)),
    )
    cal_xgb = XGBRegressor(**cal_params)
    cal_xgb.fit(X_fit, y_fit)

    class _TempModel:
        """Thin wrapper so ConformalPredictor.calibrate() can call .predict()."""
        def predict(self, X):
            return cal_xgb.predict(np.asarray(X, dtype=float))

    conformal = ConformalPredictor()
    conformal.calibrate(_TempModel(), X_cal, y_cal, clusters, alpha=args.alpha)

    # Coverage validation — use the deployment model (fit on all data) to
    # check how well the calibrated intervals cover the cal set.
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

    conformal.save(os.path.join(MODELS_DIR, f"conformal_predictor_{artifact_tag}.pkl"))
    logger.info("Conformal calibration complete.")


if __name__ == "__main__":
    main()
