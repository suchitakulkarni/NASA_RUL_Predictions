#!/usr/bin/env python3
"""
combined_pred.py — Evaluate separately-trained models across all four CMAPSS datasets.

Loads the four separate models (one per dataset) produced by train.py --mode separate,
applies the correct preprocessing pipeline, and reports per-dataset and overall RMSE.
Conformal prediction intervals are overlaid on the scatter plot when a calibrated
conformal predictor is found for the corresponding joint model artifact tag.

Usage:
    python combined_pred.py --features engineered
    python combined_pred.py --features raw
    python combined_pred.py --features engineered --run-tag my_exp
    python combined_pred.py --features engineered --config configs/default.yaml
"""
import argparse
import logging
import os

import numpy as np
from sklearn.metrics import mean_squared_error

from src.utils import setup_logging, RESULTS_DIR
from src.config import Config
from src.data import load_raw
from src.condition_normaliser import ConditionNormaliser
from src.conformal import ConformalPredictor
from src.feature_engineering import FeatureEngineer
from src.model import RULModel
from src import visualisation

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


def _predict(model, eng_test, rul_labels, feat_cols, cap, normaliser=None, conformal=None):
    """
    Return (y_true, y_pred, rmse, lower, upper).
    lower/upper are conformal intervals when conformal is provided.
    When normaliser is None (raw mode) all samples map to cluster 0 (global quantile).
    """
    test_last = eng_test.groupby("unit_no").last().reset_index()
    test_last = test_last.merge(rul_labels, on="unit_no")
    y_true = test_last["RUL"].clip(0, cap).values
    X_test = test_last[feat_cols].values

    lower = upper = None
    if conformal is not None:
        clusters = (
            normaliser.predict_clusters(test_last)
            if normaliser is not None
            else np.zeros(len(X_test), dtype=int)
        )
        y_pred, lower, upper = conformal.predict_with_interval(model, X_test, clusters)
    else:
        y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return y_true, y_pred, rmse, lower, upper


def _get_joint_data(joint_model, joint_fe, joint_normaliser, conformal,
                    test_df, rul_labels, cap, dataset_id, features):
    """Predict with the joint model on one dataset's test set."""
    if features == "engineered":
        # Must transform the full test_df before taking the last row — rolling statistics
        # computed on a single row per unit (after groupby.last) are meaningless.
        norm_test = joint_normaliser.transform(test_df)
        eng_test, feat_cols = joint_fe.transform(norm_test, dataset_id=dataset_id)
        test_last = eng_test.groupby("unit_no").last().reset_index()
        test_last = test_last.merge(rul_labels, on="unit_no")
        y_true = test_last["RUL"].clip(0, cap).values
        X_test = test_last[feat_cols].values
        if conformal is not None:
            raw_last = test_df.groupby("unit_no").last().reset_index()
            clusters = joint_normaliser.predict_clusters(raw_last)
        else:
            clusters = None
    else:
        test_last = test_df.groupby("unit_no").last().reset_index()
        test_last = test_last.merge(rul_labels, on="unit_no")
        y_true = test_last["RUL"].clip(0, cap).values
        test_last = test_last.copy()
        test_last["dataset_id"] = dataset_id
        feat_cols = ["time"] + list(joint_fe.sensor_cols) + ["dataset_id"]
        X_test = test_last[feat_cols].values
        clusters = np.zeros(len(X_test), dtype=int) if conformal is not None else None

    lower = upper = None
    if conformal is not None:
        y_pred, lower, upper = conformal.predict_with_interval(joint_model, X_test, clusters)
    else:
        y_pred = joint_model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"y_true": y_true, "y_pred": y_pred, "rmse": rmse, "lower": lower, "upper": upper}


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
    parser.add_argument("--run-tag", type=str, default=None,
                        help="Artifact tag used when running train.py; defaults to "
                             "'separate_{features}'. Must match the --run-tag passed "
                             "to train.py.")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.cap is not None:
        cfg.data.rul_cap = args.cap

    setup_logging()

    artifact_tag = args.run_tag if args.run_tag else f"separate_{args.features}"
    logger.info("Using artifact_tag=%s", artifact_tag)

    fe_path = os.path.join(cfg.paths.models_dir, f"feature_engineer_{artifact_tag}.pkl")
    if not os.path.exists(fe_path):
        raise FileNotFoundError(
            f"{fe_path} not found — run train.py --mode separate --features {args.features} first."
        )
    fe = FeatureEngineer.load(fe_path)

    normaliser = None
    if args.features == "engineered":
        norm_path = os.path.join(cfg.paths.models_dir, f"condition_normaliser_{artifact_tag}.pkl")
        if not os.path.exists(norm_path):
            raise FileNotFoundError(
                f"{norm_path} not found — run train.py --mode separate --features engineered first."
            )
        normaliser = ConditionNormaliser.load(norm_path)

    # Conformal predictor is calibrated on the joint model; load it if present.
    # Raw mode uses cluster=0 (global quantile) since there is no condition normaliser.
    joint_tag = artifact_tag.replace("separate_", "joint_", 1)
    conformal = None
    conf_path = os.path.join(cfg.paths.models_dir, f"conformal_predictor_{joint_tag}.pkl")
    if os.path.exists(conf_path):
        conformal = ConformalPredictor.load(conf_path)
        logger.info("Conformal predictor loaded from %s — uncertainty will be shown", conf_path)
    else:
        logger.info("No conformal predictor found at %s — plotting point predictions only", conf_path)

    # Joint model artifacts for comparison plots
    joint_model = None
    joint_fe = None
    joint_normaliser = None
    joint_model_path = os.path.join(cfg.paths.models_dir, f"{joint_tag}_model.pkl")
    joint_fe_path    = os.path.join(cfg.paths.models_dir, f"feature_engineer_{joint_tag}.pkl")
    joint_norm_path  = os.path.join(cfg.paths.models_dir, f"condition_normaliser_{joint_tag}.pkl")
    if os.path.exists(joint_model_path) and os.path.exists(joint_fe_path):
        joint_model = RULModel.load(joint_model_path)
        joint_fe    = FeatureEngineer.load(joint_fe_path)
        if args.features == "engineered" and os.path.exists(joint_norm_path):
            joint_normaliser = ConditionNormaliser.load(joint_norm_path)
        logger.info("Joint %s model loaded for per-dataset comparison plots", args.features)
    else:
        logger.warning(
            "Joint model artifacts not found at %s — comparison plots will be skipped",
            joint_model_path,
        )

    all_y_true, all_y_pred = [], []
    results = {}
    dataset_results = {}

    for i, ds in enumerate(cfg.data.datasets, start=1):
        model_path = os.path.join(cfg.paths.models_dir, f"{artifact_tag}_{ds}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"{model_path} not found — run "
                f"train.py --mode separate --features {args.features} first."
            )
        model = RULModel.load(model_path)

        _, test_df, rul_labels = load_raw(cfg.data.dir, ds)
        eng_test, feat_cols = _preprocess_test(test_df, normaliser, fe, args.features)

        y_true, y_pred, rmse, lower, upper = _predict(
            model, eng_test, rul_labels, feat_cols, cfg.data.rul_cap, normaliser, conformal
        )
        results[ds] = rmse
        dataset_results[ds] = {"y_true": y_true, "y_pred": y_pred, "rmse": rmse,
                                "lower": lower, "upper": upper}
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)
        logger.info("%s RMSE: %.4f", ds, rmse)

        # Per-dataset comparison plot: separate model vs joint model
        if joint_model is not None:
            joint_data = _get_joint_data(
                joint_model, joint_fe, joint_normaliser, conformal,
                test_df, rul_labels, cfg.data.rul_cap, i, args.features,
            )
            comp_title = f"pred_vs_true_{artifact_tag}_{ds}"
            visualisation.plot_per_dataset_comparison(
                ds,
                {"y_true": y_true, "y_pred": y_pred, "rmse": rmse, "lower": lower, "upper": upper},
                joint_data,
                comp_title,
                save_dir=RESULTS_DIR,
            )
            logger.info("%s joint RMSE: %.4f", ds, joint_data["rmse"])

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

    plot_title = f"pred_vs_true_{artifact_tag}"
    plot_path = visualisation.plot_pred_vs_true(
        dataset_results, title=plot_title, save_dir=RESULTS_DIR
    )
    print(f"  Plot saved to {plot_path}\n")


if __name__ == "__main__":
    main()
