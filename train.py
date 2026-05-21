#!/usr/bin/env python3
"""
train.py — Train RUL model on CMAPSS datasets.

Usage:
    python train.py --mode joint --trials 100 --cap 125 --window 5 10
    python train.py --mode separate --trials 50 --cap 125
"""
import argparse
import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.utils import setup_logging
from src.data import load_raw
from src.condition_normaliser import ConditionNormaliser
from src.feature_engineering import FeatureEngineer
from src.rul_target import compute_rul
from src.model import RULModel

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
DATA_DIR = "./CMAPSSData"
SENSOR_COLS = ["s" + str(i) for i in range(1, 22)]
MODELS_DIR = "results/models"

logger = logging.getLogger(__name__)


def _test_rmse(model, test_df, rul_labels, feat_cols, cap):
    """Evaluate model on official test set (last observed cycle per unit)."""
    test_last = test_df.groupby("unit_no").last().reset_index()
    test_last = test_last.merge(rul_labels, on="unit_no")
    y_test = test_last["RUL"].clip(0, cap).values
    X_test = test_last[feat_cols].values
    y_pred = model.predict(X_test)
    return float(np.sqrt(mean_squared_error(y_test, y_pred)))


def main():
    parser = argparse.ArgumentParser(description="Train RUL model on CMAPSS datasets")
    parser.add_argument("--mode",    choices=["separate", "joint"], default="joint",
                        help="separate: one model per dataset; joint: one model on all four")
    parser.add_argument("--trials",  type=int, default=100,
                        help="Number of Optuna trials (default 100)")
    parser.add_argument("--cap",     type=int, default=125,
                        help="RUL cap in cycles (default 125)")
    parser.add_argument("--window",  type=int, nargs="+", default=[5, 10],
                        help="Rolling window sizes (default 5 10)")
    args = parser.parse_args()

    setup_logging()
    os.makedirs(MODELS_DIR, exist_ok=True)

    logger.info(
        "Training: mode=%s, trials=%d, cap=%d, windows=%s",
        args.mode, args.trials, args.cap, args.window,
    )

    # Load all four datasets (raw, time column preserved)
    dataset_map = {}
    for ds in DATASETS:
        train_df, test_df, rul_labels = load_raw(DATA_DIR, ds)
        train_df = compute_rul(train_df, cap=args.cap)
        dataset_map[ds] = (train_df, test_df, rul_labels)

    # Fit ConditionNormaliser on all training data combined
    all_train = pd.concat(
        [df for df, _, _ in dataset_map.values()], ignore_index=True
    )
    sensor_cols = [c for c in SENSOR_COLS if c in all_train.columns]
    normaliser = ConditionNormaliser(n_clusters=6)
    normaliser.fit(all_train, sensor_cols)
    normaliser.save(os.path.join(MODELS_DIR, "condition_normaliser.pkl"))

    # Fit FeatureEngineer on all training data (global max_cycle)
    fe = FeatureEngineer(window_sizes=args.window)
    fe.fit(all_train)

    if args.mode == "joint":
        _train_joint(dataset_map, normaliser, fe, args)
    else:
        _train_separate(dataset_map, normaliser, fe, args)


def _train_joint(dataset_map, normaliser, fe, args):
    X_parts, y_parts, strat_parts, test_data = [], [], [], {}

    for i, ds in enumerate(DATASETS, start=1):
        train_df, test_df, rul_labels = dataset_map[ds]

        norm_train = normaliser.transform(train_df)
        norm_test  = normaliser.transform(test_df)

        eng_train, feat_cols = fe.transform(norm_train, dataset_id=i)
        eng_test,  _         = fe.transform(norm_test,  dataset_id=i)

        X_parts.append(eng_train[feat_cols].values)
        y_parts.append(eng_train["RUL"].values)
        strat_parts.append(np.full(len(eng_train), i, dtype=int))
        test_data[ds] = (eng_test, rul_labels, feat_cols)

    X_all    = np.concatenate(X_parts)
    y_all    = np.concatenate(y_parts)
    strat    = np.concatenate(strat_parts)

    logger.info("Joint training matrix: X=%s y=%s", X_all.shape, y_all.shape)

    model = RULModel()
    model.train(X_all, y_all, optuna_trials=args.trials, stratify=strat)
    model.save(os.path.join(MODELS_DIR, "joint_model.pkl"))

    for ds, (eng_test, rul_labels, feat_cols) in test_data.items():
        rmse = _test_rmse(model, eng_test, rul_labels, feat_cols, args.cap)
        logger.info("Joint model test RMSE on %s: %.4f", ds, rmse)


def _train_separate(dataset_map, normaliser, fe, args):
    for i, ds in enumerate(DATASETS, start=1):
        train_df, test_df, rul_labels = dataset_map[ds]

        norm_train = normaliser.transform(train_df)
        norm_test  = normaliser.transform(test_df)

        # No dataset_id for separate models — constant feature adds no signal
        eng_train, feat_cols = fe.transform(norm_train, dataset_id=None)
        eng_test,  _         = fe.transform(norm_test,  dataset_id=None)

        X_train = eng_train[feat_cols].values
        y_train = eng_train["RUL"].values

        logger.info("Separate model for %s: X=%s", ds, X_train.shape)

        model = RULModel()
        model.train(X_train, y_train, optuna_trials=args.trials)
        model.save(os.path.join(MODELS_DIR, f"separate_model_{ds}.pkl"))

        rmse = _test_rmse(model, eng_test, rul_labels, feat_cols, args.cap)
        logger.info("Separate model test RMSE on %s: %.4f", ds, rmse)


if __name__ == "__main__":
    main()
