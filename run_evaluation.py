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
from src.evaluation import (compute_metrics, save_metrics_table,
                            percentage_improvement,
                            lead_time_full_trajectory, cost_savings_from_detections)
from src import visualisation

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
DATA_DIR = "./CMAPSSData"
MODELS_DIR = "results/models"
EVAL_DIR  = "results/evaluation"

logger = logging.getLogger(__name__)


def _build_all_unit_trajectories(
    test_df, rul_labels_dict, normaliser, fe, model, conformal,
    dataset_id, cap, feat_cols,
):
    """
    Build cycle-by-cycle prediction dicts for ALL units in test_df.

    Processes the entire dataset in one pass (efficient) then splits by unit.
    Returns a list of dicts with keys:
        unit_id, cycles, rul_true, rul_pred, lower, upper, true_rul_at_end
    """
    clusters_all = normaliser.predict_clusters(test_df)
    norm_df      = normaliser.transform(test_df)
    eng_df, _    = fe.transform(norm_df, dataset_id=dataset_id)
    X_all        = eng_df[feat_cols].values

    y_pred_all, lower_all, upper_all = conformal.predict_with_interval(
        model, X_all, clusters_all
    )

    eng_df = eng_df.copy()
    eng_df["_pred"]    = y_pred_all
    eng_df["_lower"]   = lower_all
    eng_df["_upper"]   = upper_all
    eng_df["_cluster"] = clusters_all

    unit_dfs = []
    for uid, group in eng_df.groupby("unit_no"):
        cycles          = group["time"].values
        last_cycle      = int(cycles[-1])
        true_rul_at_end = min(float(rul_labels_dict.get(uid, 0)), float(cap))
        rul_true        = np.clip(
            true_rul_at_end + (last_cycle - cycles), 0.0, float(cap)
        )
        unit_dfs.append({
            "unit_id":         uid,
            "cycles":          cycles,
            "rul_true":        rul_true,
            "rul_pred":        group["_pred"].values,
            "lower":           group["_lower"].values,
            "upper":           group["_upper"].values,
            "true_rul_at_end": true_rul_at_end,
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

    metrics_rows          = []
    per_ds_arrays         = {}   # ds -> (y_true, y_pred, lower, upper)
    all_unit_trajectories = []   # full cycle-by-cycle data for all units
    feat_cols_final       = None

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

        coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
        logger.info("%s test: n=%d, coverage=%.4f", ds, len(y_true), coverage)

        per_ds_arrays[ds] = (y_true, y_pred, lower, upper)
        metrics_rows.append(compute_metrics(y_true, y_pred, dataset_name=ds))

        # Full cycle-by-cycle trajectories for all units (used for lead-time analysis)
        rul_labels_dict = rul_labels.set_index("unit_no")["RUL"].to_dict()
        ds_trajectories = _build_all_unit_trajectories(
            test_df, rul_labels_dict, normaliser, fe, model, conformal,
            dataset_id=i, cap=args.cap, feat_cols=feat_cols,
        )
        all_unit_trajectories.extend(ds_trajectories)

        # Sample a subset for the RUL prediction plots
        sampled_units = random.sample(
            [t["unit_id"] for t in ds_trajectories],
            min(args.n_units, len(ds_trajectories)),
        )
        plot_trajs = [t for t in ds_trajectories if t["unit_id"] in sampled_units]
        save_path = os.path.join(EVAL_DIR, f"rul_predictions_{ds.lower()}.png")
        visualisation.plot_rul_predictions(
            plot_trajs, save_path, lead_time_threshold=args.lead_time
        )

    save_metrics_table(metrics_rows, os.path.join(EVAL_DIR, "metrics_summary.csv"))

    # RMSE improvement vs naive mean predictor
    y_true_all = np.concatenate([v[0] for v in per_ds_arrays.values()])
    y_pred_all = np.concatenate([v[1] for v in per_ds_arrays.values()])
    per_ds_improvement = [
        {"dataset": ds, "improvement_pct": percentage_improvement(yt, yp)["improvement_pct"]}
        for ds, (yt, yp, _, _) in per_ds_arrays.items()
    ]
    overall_improvement = percentage_improvement(y_true_all, y_pred_all)

    # Lead-time analysis using full cycle-by-cycle trajectories across all units
    per_unit_df, lead_metrics = lead_time_full_trajectory(
        all_unit_trajectories, threshold=args.lead_time
    )
    per_unit_df.to_csv(os.path.join(EVAL_DIR, "lead_time_per_unit.csv"), index=False)
    logger.info("Per-unit lead-time saved to %s/lead_time_per_unit.csv", EVAL_DIR)

    # Cost savings: use full confusion matrix so false alarms are properly penalised
    cost_metrics = cost_savings_from_detections(
        tp=lead_metrics["tp"],
        fp=lead_metrics["fp"],
        fn=lead_metrics["fn"],
        tn=lead_metrics["tn"],
        cost_unplanned=args.cost_unplanned,
        cost_planned=args.cost_planned,
    )

    pd.DataFrame([{**overall_improvement, **lead_metrics, **cost_metrics}]).to_csv(
        os.path.join(EVAL_DIR, "business_metrics.csv"), index=False
    )
    logger.info("Business metrics saved to %s/business_metrics.csv", EVAL_DIR)

    visualisation.plot_lead_time_distribution(per_unit_df, threshold=args.lead_time)
    visualisation.plot_business_summary(lead_metrics, per_ds_improvement, cost_metrics)

    # Feature importance (joint model, last feat_cols which include dataset_id)
    if feat_cols_final is not None:
        visualisation.plot_feature_importance(model, feat_cols_final)

    logger.info("Phase 6 evaluation complete. Results in %s/", EVAL_DIR)


if __name__ == "__main__":
    main()
