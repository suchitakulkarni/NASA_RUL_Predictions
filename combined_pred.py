#!/usr/bin/env python3
"""
combined_pred.py — Evaluate separately-trained models across all four CMAPSS datasets.

Loads the four separate models (one per dataset) produced by train.py --mode separate,
applies the correct preprocessing pipeline, and reports per-dataset and overall RMSE.

Usage:
    python combined_pred.py --features engineered
    python combined_pred.py --features raw
    python combined_pred.py --features engineered --config configs/default.yaml
"""
import argparse
import logging
import os

import numpy as np
from sklearn.metrics import mean_squared_error

from src.utils import setup_logging
from src.config import Config
from src.data import load_raw
from src.condition_normaliser import ConditionNormaliser
from src.feature_engineering import FeatureEngineer
from src.model import RULModel

logger = logging.getLogger(__name__)


def _preprocess_test(test_df, normaliser, fe, features):
    """Mirror the separate-mode preprocessing from train.py."""
    if features == "engineered":
        norm_test = normaliser.transform(test_df)
        eng_test, feat_cols = fe.transform(norm_test, dataset_id=None)
    else:
        feat_cols = ["time"] + list(fe.sensor_cols)
        eng_test = test_df.copy()
    return eng_test, feat_cols


def _predict_rmse(model, eng_test, rul_labels, feat_cols, cap):
    test_last = eng_test.groupby("unit_no").last().reset_index()
    test_last = test_last.merge(rul_labels, on="unit_no")
    y_true = test_last["RUL"].clip(0, cap).values
    y_pred = model.predict(test_last[feat_cols].values)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return y_true, y_pred, rmse


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate separate-mode models across all CMAPSS datasets"
    )
    parser.add_argument("--features", choices=["raw", "engineered"], required=True,
                        help="Which set of models to evaluate (raw or engineered)")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to YAML config (default: configs/default.yaml)")
    parser.add_argument("--cap", type=int, default=None,
                        help="Override data.rul_cap from config")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.cap is not None:
        cfg.data.rul_cap = args.cap

    setup_logging()

    fe_path = os.path.join(cfg.paths.models_dir, "feature_engineer.pkl")
    if not os.path.exists(fe_path):
        raise FileNotFoundError(
            f"{fe_path} not found — run train.py first to generate preprocessing artifacts."
        )
    fe = FeatureEngineer.load(fe_path)

    normaliser = None
    if args.features == "engineered":
        norm_path = os.path.join(cfg.paths.models_dir, "condition_normaliser.pkl")
        if not os.path.exists(norm_path):
            raise FileNotFoundError(
                f"{norm_path} not found — run train.py --features engineered first."
            )
        normaliser = ConditionNormaliser.load(norm_path)

    all_y_true, all_y_pred = [], []
    results = {}

    for ds in cfg.data.datasets:
        model_path = os.path.join(cfg.paths.models_dir, f"separate_{args.features}_{ds}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"{model_path} not found — run "
                f"train.py --mode separate --features {args.features} first."
            )
        model = RULModel.load(model_path)

        _, test_df, rul_labels = load_raw(cfg.data.dir, ds)
        eng_test, feat_cols = _preprocess_test(test_df, normaliser, fe, args.features)

        y_true, y_pred, rmse = _predict_rmse(model, eng_test, rul_labels, feat_cols, cfg.data.rul_cap)
        results[ds] = rmse
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)
        logger.info("%s RMSE: %.4f", ds, rmse)

    overall = float(np.sqrt(mean_squared_error(
        np.concatenate(all_y_true), np.concatenate(all_y_pred)
    )))

    print(f"\n{'='*45}")
    print(f"  Separate models — features={args.features}, cap={cfg.data.rul_cap}")
    print(f"{'='*45}")
    print(f"  {'Dataset':<10} {'RMSE':>10}")
    print(f"  {'-'*22}")
    for ds, rmse in results.items():
        print(f"  {ds:<10} {rmse:>10.4f}")
    print(f"  {'-'*22}")
    print(f"  {'Overall':<10} {overall:>10.4f}")
    print(f"{'='*45}\n")


if __name__ == "__main__":
    main()
