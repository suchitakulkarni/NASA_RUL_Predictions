#!/usr/bin/env python3
"""
train.py — Train RUL model on CMAPSS datasets.

Config is loaded from configs/default.yaml. CLI flags override individual values
for quick experiments without editing the file.

Usage:
    python train.py --mode joint    --features engineered
    python train.py --mode separate --features raw
    python train.py --mode joint    --features engineered --trials 50 --cap 100 --window 5 10
"""
import argparse
import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.utils import setup_logging
from src.config import Config
from src.data import load_raw
from src.condition_normaliser import ConditionNormaliser
from src.feature_engineering import FeatureEngineer
from src.rul_target import compute_rul
from src.model import RULModel

SENSOR_COLS = ["s" + str(i) for i in range(1, 22)]

logger = logging.getLogger(__name__)


def _test_rmse(model, test_df, rul_labels, feat_cols, cap):
    """Evaluate model on official test set (last observed cycle per unit)."""
    test_last = test_df.groupby("unit_no").last().reset_index()
    test_last = test_last.merge(rul_labels, on="unit_no")
    y_test = test_last["RUL"].clip(0, cap).values
    X_test = test_last[feat_cols].values
    y_pred = model.predict(X_test)
    return float(np.sqrt(mean_squared_error(y_test, y_pred)))


def _make_features(train_df, test_df, ds_id, normaliser, fe, mode, features):
    """
    Prepare (X_train, y_train, processed_test_df, feat_cols) for one dataset.

    raw:        same kept sensor columns + raw cycle time, no rolling stats or
                condition normalisation.
    engineered: condition-normalised sensors + rolling stats + cycle_norm.

    ds_id is appended as dataset_id only in joint mode.
    """
    include_ds = (mode == "joint")

    if features == "engineered":
        norm_train = normaliser.transform(train_df)
        norm_test  = normaliser.transform(test_df)
        eng_train, feat_cols = fe.transform(norm_train, dataset_id=ds_id if include_ds else None)
        eng_test,  _         = fe.transform(norm_test,  dataset_id=ds_id if include_ds else None)
    else:
        feat_cols = ["time"] + list(fe.sensor_cols)
        eng_train = train_df.copy()
        eng_test  = test_df.copy()
        if include_ds:
            eng_train["dataset_id"] = ds_id
            eng_test["dataset_id"]  = ds_id
            feat_cols = feat_cols + ["dataset_id"]

    X_train = eng_train[feat_cols].values
    y_train = eng_train["RUL"].values
    unit_ids = eng_train["unit_no"].values
    return X_train, y_train, unit_ids, eng_test, feat_cols


def main():
    parser = argparse.ArgumentParser(description="Train RUL model on CMAPSS datasets")
    parser.add_argument("--config",   default="configs/default.yaml",
                        help="Path to YAML config (default: configs/default.yaml)")
    parser.add_argument("--mode",     choices=["separate", "joint"], default="joint",
                        help="separate: one model per dataset; joint: one model on all four")
    parser.add_argument("--features", choices=["raw", "engineered"], default="engineered",
                        help="raw: sensor values + time only; engineered: rolling stats + cycle_norm")
    # CLI overrides — when provided, take precedence over the config file
    parser.add_argument("--cap",     type=int,       default=None,
                        help="Override data.rul_cap in config")
    parser.add_argument("--trials",  type=int,       default=None,
                        help="Override model.optuna_trials in config")
    parser.add_argument("--window",  type=int, nargs="+", default=None,
                        help="Override features.window_sizes in config")
    parser.add_argument("--n-jobs",  type=int, default=1,
                        help="XGBoost threads per model; set to 1 when running "
                             "multiple train.py instances in parallel (default: 1)")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="Optional suffix for all saved artifact filenames, "
                             "e.g. 'exp1'. Defaults to '{mode}_{features}'. "
                             "Use distinct tags when running parallel experiments.")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    # Apply CLI overrides
    if args.cap is not None:
        cfg.data.rul_cap = args.cap
    if args.trials is not None:
        cfg.model.optuna_trials = args.trials
    if args.window is not None:
        cfg.features.window_sizes = args.window

    setup_logging()
    os.makedirs(cfg.paths.models_dir, exist_ok=True)

    artifact_tag = args.run_tag if args.run_tag else f"{args.mode}_{args.features}"
    logger.info(
        "Training: mode=%s, features=%s, trials=%d, cap=%d, windows=%s, artifact_tag=%s",
        args.mode, args.features, cfg.model.optuna_trials,
        cfg.data.rul_cap, cfg.features.window_sizes, artifact_tag,
    )
    logger.info("Full config: %s", cfg.to_flat_dict())

    dataset_map = {}
    for ds in cfg.data.datasets:
        train_df, test_df, rul_labels = load_raw(cfg.data.dir, ds)
        train_df = compute_rul(train_df, cap=cfg.data.rul_cap)
        dataset_map[ds] = (train_df, test_df, rul_labels)

    all_train = pd.concat(
        [df for df, _, _ in dataset_map.values()], ignore_index=True
    )
    sensor_cols = [c for c in SENSOR_COLS if c in all_train.columns]

    # Always fit both; fe.sensor_cols is needed even in raw mode
    normaliser = ConditionNormaliser(
        n_clusters=cfg.condition_normaliser.n_clusters,
        n_init=cfg.condition_normaliser.n_init,
        random_state=cfg.condition_normaliser.random_state,
    )
    normaliser.fit(all_train, sensor_cols)

    fe = FeatureEngineer(
        drop_sensors=cfg.features.sensors_to_drop,
        window_sizes=cfg.features.window_sizes,
    )
    fe.fit(all_train)

    fe.save(os.path.join(cfg.paths.models_dir, f"feature_engineer_{artifact_tag}.pkl"))
    normaliser.save(os.path.join(cfg.paths.models_dir, f"condition_normaliser_{artifact_tag}.pkl"))

    if args.mode == "joint":
        _train_joint(dataset_map, normaliser, fe, cfg, args, artifact_tag)
    else:
        _train_separate(dataset_map, normaliser, fe, cfg, args, artifact_tag)


def _train_joint(dataset_map, normaliser, fe, cfg, args, artifact_tag):
    X_parts, y_parts, strat_parts, uid_parts, test_data = [], [], [], [], {}

    for i, ds in enumerate(cfg.data.datasets, start=1):
        train_df, test_df, rul_labels = dataset_map[ds]
        X_train, y_train, unit_ids, eng_test, feat_cols = _make_features(
            train_df, test_df, i, normaliser, fe, "joint", args.features
        )
        X_parts.append(X_train)
        y_parts.append(y_train)
        uid_parts.append(unit_ids)
        strat_parts.append(np.full(len(X_train), i, dtype=int))
        test_data[ds] = (eng_test, rul_labels, feat_cols)

    X_all = np.concatenate(X_parts)
    y_all = np.concatenate(y_parts)
    uid_all = np.concatenate(uid_parts)
    strat = np.concatenate(strat_parts)

    logger.info("Joint training matrix: X=%s y=%s", X_all.shape, y_all.shape)

    model = RULModel()
    model.train(
        X_all, y_all, cfg=cfg.model, stratify=strat,
        feat_cols=feat_cols, groups=uid_all, n_jobs=args.n_jobs,
    )
    model.save(os.path.join(cfg.paths.models_dir, f"{artifact_tag}_model.pkl"))

    for ds, (eng_test, rul_labels, feat_cols) in test_data.items():
        rmse = _test_rmse(model, eng_test, rul_labels, feat_cols, cfg.data.rul_cap)
        logger.info("Joint %s model RMSE on %s: %.4f", args.features, ds, rmse)


def _train_separate(dataset_map, normaliser, fe, cfg, args, artifact_tag):
    for i, ds in enumerate(cfg.data.datasets, start=1):
        train_df, test_df, rul_labels = dataset_map[ds]
        X_train, y_train, unit_ids, eng_test, feat_cols = _make_features(
            train_df, test_df, i, normaliser, fe, "separate", args.features
        )

        logger.info("Separate %s model for %s: X=%s", args.features, ds, X_train.shape)

        model = RULModel()
        model.train(
            X_train, y_train, cfg=cfg.model,
            feat_cols=feat_cols, groups=unit_ids, n_jobs=args.n_jobs,
        )
        model.save(os.path.join(cfg.paths.models_dir, f"{artifact_tag}_{ds}.pkl"))

        rmse = _test_rmse(model, eng_test, rul_labels, feat_cols, cfg.data.rul_cap)
        logger.info("Separate %s model RMSE on %s: %.4f", args.features, ds, rmse)


if __name__ == "__main__":
    main()
