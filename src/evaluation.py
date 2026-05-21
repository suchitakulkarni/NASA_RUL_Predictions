# src/evaluation.py
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def cmapss_score(y_true, y_pred):
    """
    Standard CMAPSS asymmetric scoring function (sum over test units).

    d = y_pred - y_true (positive means late prediction).
    Late predictions (d >= 0) are penalised with exp(d/10) - 1.
    Early predictions (d < 0) are penalised with exp(-d/13) - 1.
    Penalises late predictions more heavily, reflecting the cost of missing
    a failure vs. scheduling unnecessary maintenance.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(scores.sum())


def compute_metrics(y_true, y_pred, dataset_name=""):
    """
    Compute RMSE, MAE, and CMAPSS score for a single dataset split.

    Returns a dict with keys: dataset, rmse, mae, cmapss_score.
    """
    r = rmse(y_true, y_pred)
    m = mae(y_true, y_pred)
    s = cmapss_score(y_true, y_pred)
    logger.info(
        "%s | RMSE=%.4f  MAE=%.4f  CMAPSS_score=%.2f  n=%d",
        dataset_name, r, m, s, len(y_true),
    )
    return {"dataset": dataset_name, "rmse": r, "mae": m, "cmapss_score": s}


def save_metrics_table(rows, path):
    """
    Save a list of metric dicts (from compute_metrics) to CSV and log summary.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info("Metrics table saved to %s\n%s", path, df.to_string(index=False))
    return df


def cost_savings(rul_lower, rul_upper, cost_unplanned, cost_planned):
    """
    Estimate expected cost savings from using the conformal prediction interval
    to schedule maintenance proactively vs. reacting after failure.

    Model: a unit whose true failure falls inside the prediction interval is
    assumed to be caught and scheduled (incurring cost_planned).  The
    probability of catching is approximated by min(interval_width / 125, 1.0)
    where 125 is the RUL cap.  Savings per unit = probability * (cost_unplanned
    - cost_planned).  This is a linear illustration for the portfolio; a
    production model would use survival analysis to integrate over the failure
    distribution.

    Returns a summary dict.
    """
    rul_lower = np.asarray(rul_lower, dtype=float)
    rul_upper = np.asarray(rul_upper, dtype=float)

    widths = rul_upper - rul_lower
    mean_width = float(np.mean(widths))
    n_units = len(rul_lower)

    prob_catch = np.minimum(widths / 125.0, 1.0)
    saving_per_unit = prob_catch * (cost_unplanned - cost_planned)
    total_saving = float(saving_per_unit.sum())

    summary = {
        "n_units": n_units,
        "mean_interval_width_cycles": round(mean_width, 2),
        "cost_unplanned": cost_unplanned,
        "cost_planned": cost_planned,
        "total_expected_savings": round(total_saving, 2),
        "mean_savings_per_unit": round(float(saving_per_unit.mean()), 2),
    }
    logger.info("Cost savings summary: %s", summary)
    return summary
