#!/usr/bin/env python3
"""
run_evaluation.py — Phase 6: final metrics, prediction plots, feature importance.

Generates:
    results/evaluation/metrics_summary.csv
    results/evaluation/rul_predictions_fd001.png  (one per dataset)
    results/evaluation/rul_predictions_fd002.png
    results/evaluation/rul_predictions_fd003.png
    results/evaluation/rul_predictions_fd004.png
    results/evaluation/feature_importance.png
    results/evaluation/cost_summary.csv

Requires that train.py (joint) and calibrate.py have been run first.

Usage:
    python run_evaluation.py [--cap 125] [--n-units 5] [--lead-time 30]
                             [--cost-unplanned 100000] [--cost-planned 20000]
"""
import argparse
import logging
import os
import random

import numpy as np
import pandas as pd

from src.utils import setup_logging
from src.data import load_raw
from src.condition_normaliser import ConditionNormaliser
from src.feature_engineering import FeatureEngineer
from src.model import RULModel
from src.conformal import ConformalPredictor
from src.evaluation import compute_metrics, save_metrics_table, cost_savings
from src import visualisation

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
DATA_DIR = "./CMAPSSData"
MODELS_DIR = "results/models"
EVAL_DIR  = "results/evaluation"

logger = logging.getLogger(__name__)


def _build_unit_trajectories(
    test_df, rul_labels_dict, normaliser, fe, model, conformal,
    dataset_id, cap, feat_cols, n_units,
):
    """
    For a random sample of test units, build cycle-by-cycle prediction dicts
    suitable for visualisation.plot_rul_predictions.

    True RUL at cycle t:  true_rul_at_last_cycle + (last_cycle - t)
    Capped at `cap`.
    """
    all_units = test_df["unit_no"].unique().tolist()
    sampled = random.sample(all_units, min(n_units, len(all_units)))

    unit_dfs = []
    for uid in sampled:
        unit_df = test_df[test_df["unit_no"] == uid].copy()
        cycles = unit_df["time"].values
        last_cycle = int(cycles[-1])
        true_rul_at_end = min(float(rul_labels_dict[uid]), float(cap))

        rul_true = np.clip(true_rul_at_end + (last_cycle - cycles), 0.0, float(cap))

        clusters = normaliser.predict_clusters(unit_df)
        norm_df  = normaliser.transform(unit_df)
        eng_df, _ = fe.transform(norm_df, dataset_id=dataset_id)
        X = eng_df[feat_cols].values

        y_pred, lower, upper = conformal.predict_with_interval(model, X, clusters)

        unit_dfs.append({
            "unit_id":  uid,
            "cycles":   cycles,
            "rul_true": rul_true,
            "rul_pred": y_pred,
            "lower":    lower,
            "upper":    upper,
        })

    return unit_dfs


def main():
    parser = argparse.ArgumentParser(description="Phase 6: final evaluation and plots")
    parser.add_argument("--cap",            type=int,   default=125)
    parser.add_argument("--n-units",        type=int,   default=5,
                        help="Test units to plot per dataset")
    parser.add_argument("--lead-time",      type=int,   default=30,
                        help="Lead-time threshold annotated on RUL plots (cycles)")
    parser.add_argument("--cost-unplanned", type=float, default=100_000.0,
                        help="Cost of unplanned downtime per unit")
    parser.add_argument("--cost-planned",   type=float, default=20_000.0,
                        help="Cost of scheduled maintenance per unit")
    parser.add_argument("--features",       choices=["raw", "engineered"], default="engineered",
                        help="Feature set the joint model was trained with (default: engineered)")
    parser.add_argument("--run-tag",        type=str, default=None,
                        help="Artifact tag passed to train.py and calibrate.py; "
                             "defaults to 'joint_{features}'. Must match training.")
    args = parser.parse_args()

    setup_logging()
    os.makedirs(EVAL_DIR, exist_ok=True)

    random.seed(42)
    np.random.seed(42)

    artifact_tag = args.run_tag if args.run_tag else f"joint_{args.features}"
    logger.info("Loading artifacts with tag=%s", artifact_tag)

    model      = RULModel.load(os.path.join(MODELS_DIR, f"{artifact_tag}_model.pkl"))
    normaliser = ConditionNormaliser.load(os.path.join(MODELS_DIR, f"condition_normaliser_{artifact_tag}.pkl"))
    conformal  = ConformalPredictor.load(os.path.join(MODELS_DIR, f"conformal_predictor_{artifact_tag}.pkl"))

    fe_path = os.path.join(MODELS_DIR, f"feature_engineer_{artifact_tag}.pkl")
    if not os.path.exists(fe_path):
        raise FileNotFoundError(
            f"{fe_path} not found — run train.py --mode joint --features {args.features} first."
        )
    fe = FeatureEngineer.load(fe_path)
    logger.info("FeatureEngineer loaded from %s (window_sizes=%s)", fe_path, fe.window_sizes)

    metrics_rows = []
    all_lower, all_upper = [], []
    feat_cols_final = None

    for i, ds in enumerate(DATASETS, start=1):
        _, test_df, rul_labels = load_raw(DATA_DIR, ds)

        # Process full test_df so rolling statistics are computed over all cycles,
        # then take the last observed cycle per unit as the official evaluation point.
        norm_test = normaliser.transform(test_df)
        eng_test, feat_cols = fe.transform(norm_test, dataset_id=i)
        feat_cols_final = feat_cols

        test_last = eng_test.groupby("unit_no").last().reset_index()
        test_last = test_last.merge(rul_labels, on="unit_no")
        y_true = test_last["RUL"].clip(0, args.cap).values
        X_test = test_last[feat_cols].values

        # Clusters from raw (pre-normalisation) last row
        raw_last = test_df.groupby("unit_no").last().reset_index()
        clusters = normaliser.predict_clusters(raw_last)

        y_pred, lower, upper = conformal.predict_with_interval(model, X_test, clusters)
        all_lower.extend(lower.tolist())
        all_upper.extend(upper.tolist())

        test_coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
        logger.info(
            "%s test: n=%d, coverage=%.4f", ds, len(y_true), test_coverage
        )

        metrics_rows.append(compute_metrics(y_true, y_pred, dataset_name=ds))

        # Cycle-by-cycle trajectory plots for sampled test units
        rul_labels_dict = rul_labels.set_index("unit_no")["RUL"].to_dict()
        unit_dfs = _build_unit_trajectories(
            test_df, rul_labels_dict, normaliser, fe, model, conformal,
            dataset_id=i, cap=args.cap, feat_cols=feat_cols, n_units=args.n_units,
        )
        save_path = os.path.join(EVAL_DIR, f"rul_predictions_{ds.lower()}.png")
        visualisation.plot_rul_predictions(
            unit_dfs, save_path, lead_time_threshold=args.lead_time
        )

    save_metrics_table(metrics_rows, os.path.join(EVAL_DIR, "metrics_summary.csv"))

    # Cost savings summary across all test units
    savings = cost_savings(
        np.array(all_lower), np.array(all_upper),
        cost_unplanned=args.cost_unplanned,
        cost_planned=args.cost_planned,
    )
    pd.DataFrame([savings]).to_csv(
        os.path.join(EVAL_DIR, "cost_summary.csv"), index=False
    )
    logger.info("Cost summary saved to %s/cost_summary.csv", EVAL_DIR)

    # Feature importance (joint model, last feat_cols which include dataset_id)
    if feat_cols_final is not None:
        visualisation.plot_feature_importance(model, feat_cols_final)

    logger.info("Phase 6 evaluation complete. Results in %s/", EVAL_DIR)


if __name__ == "__main__":
    main()
