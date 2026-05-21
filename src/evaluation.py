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


def percentage_improvement(y_true, y_pred):
    """
    RMSE improvement over a naive mean predictor (always predicts the training
    mean RUL).  Returns a dict with naive RMSE, model RMSE, and % improvement.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    naive_pred = np.full_like(y_true, y_true.mean())
    naive_rmse = float(np.sqrt(np.mean((y_true - naive_pred) ** 2)))
    model_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    improvement_pct = 100.0 * (naive_rmse - model_rmse) / naive_rmse

    result = {
        "naive_rmse": round(naive_rmse, 2),
        "model_rmse": round(model_rmse, 2),
        "improvement_pct": round(improvement_pct, 1),
    }
    logger.info(
        "RMSE improvement vs naive mean: %.1f%% (naive=%.2f, model=%.2f)",
        improvement_pct, naive_rmse, model_rmse,
    )
    return result


def lead_time_full_trajectory(unit_trajectories, threshold=30):
    """
    Compute true lead-time metrics from full cycle-by-cycle predictions.

    For each unit, finds the FIRST cycle where the conformal lower bound drops
    below `threshold` — this is when the maintenance alarm fires.

        failure_cycle = last_test_cycle + true_rul_at_last_cycle
        lead_time     = failure_cycle − alarm_cycle

    A positive lead time means the alarm fired before the engine actually fails,
    giving the maintenance team time to act.  A missed unit is one where the
    lower bound never crosses the threshold during the entire test window.

    Returns (per_unit_df, aggregate_dict).
    """
    rows = []
    for unit in unit_trajectories:
        cycles          = np.asarray(unit["cycles"])
        lower           = np.asarray(unit["lower"])
        true_rul_at_end = float(unit["true_rul_at_end"])
        last_cycle      = int(cycles[-1])
        failure_cycle   = last_cycle + true_rul_at_end

        alarm_mask = lower < threshold
        if alarm_mask.any():
            alarm_idx   = int(np.argmax(alarm_mask))   # first True
            alarm_cycle = int(cycles[alarm_idx])
            lead_time   = failure_cycle - alarm_cycle
        else:
            alarm_cycle = None
            lead_time   = None

        rows.append({
            "unit_id":          unit["unit_id"],
            "true_rul_at_end":  true_rul_at_end,
            "failure_cycle":    failure_cycle,
            "alarm_cycle":      alarm_cycle,
            "lead_time_cycles": lead_time,
            "alarm_fired":      bool(alarm_mask.any()),
        })

    df = pd.DataFrame(rows)

    # Alarm fired before the engine actually fails
    detected = df["alarm_fired"] & (df["lead_time_cycles"].fillna(-1) > 0)
    df["detected_early"] = detected

    # Ground truth: was the engine genuinely near failure during the test window?
    # true_rul_at_end is the remaining life at the LAST observed cycle.
    # If it is <= threshold, the engine was within the danger zone at end of monitoring.
    df["needs_maintenance"] = df["true_rul_at_end"] <= threshold

    tp = int(( detected &  df["needs_maintenance"]).sum())   # caught real near-failure
    fp = int(( detected & ~df["needs_maintenance"]).sum())   # alarmed healthy engine
    fn = int((~detected &  df["needs_maintenance"]).sum())   # missed real near-failure
    tn = int((~detected & ~df["needs_maintenance"]).sum())   # correctly quiet, healthy

    precision   = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall      = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    f1          = (2 * precision * recall / (precision + recall)
                   if (precision + recall) > 0 else float("nan"))

    lead_times_arr = df.loc[detected, "lead_time_cycles"].values

    def _s(arr, fn):
        return round(float(fn(arr)), 1) if len(arr) > 0 else float("nan")

    metrics = {
        "threshold_cycles":        threshold,
        "n_units":                 len(df),
        # Confusion matrix
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_near_failure":          tp + fn,   # engines genuinely in danger zone
        "n_healthy":               fp + tn,   # engines with ample remaining life
        # Classification metrics
        "recall_pct":              round(100.0 * recall,      1),   # of near-failure, % caught
        "precision_pct":           round(100.0 * precision,   1),   # of alarms, % genuine
        "specificity_pct":         round(100.0 * specificity, 1),   # of healthy, % correctly quiet
        "f1":                      round(f1, 3),
        # Lead time (for correctly detected near-failure engines only)
        "n_detected":              int(detected.sum()),
        "n_missed":                int((~detected).sum()),
        "mean_lead_time_cycles":   _s(lead_times_arr, np.mean),
        "median_lead_time_cycles": _s(lead_times_arr, np.median),
        "min_lead_time_cycles":    _s(lead_times_arr, np.min),
    }
    logger.info(
        "Full-trajectory lead-time (threshold=%d): "
        "TP=%d FP=%d FN=%d TN=%d | recall=%.1f%% precision=%.1f%% | "
        "mean_lead=%.1f cycles median_lead=%.1f cycles",
        threshold, tp, fp, fn, tn,
        100.0 * recall, 100.0 * precision,
        metrics["mean_lead_time_cycles"], metrics["median_lead_time_cycles"],
    )
    return df, metrics


def cost_savings_from_detections(tp, fp, fn, tn, cost_unplanned, cost_planned):
    """
    Compute maintenance cost under the model vs a reactive (wait-for-failure) baseline.

    Uses the full confusion matrix so false alarms are properly costed.

    Populations
    -----------
    TP : near-failure engines correctly alarmed → pay cost_planned (saved from unplanned)
    FP : healthy engines falsely alarmed       → pay cost_planned (wasted spend)
    FN : near-failure engines missed           → pay cost_unplanned (failure happens)
    TN : healthy engines correctly quiet       → $0 in this window (not yet at risk)

    Baseline : every near-failure engine fails unplanned → (TP+FN) × cost_unplanned.
    Model    : TP × cost_planned + FP × cost_planned + FN × cost_unplanned.

    Net savings = baseline − model
               = TP × (cost_unplanned − cost_planned) − FP × cost_planned
    """
    n_near_failure  = tp + fn
    baseline_cost   = n_near_failure * cost_unplanned
    model_cost      = tp * cost_planned + fp * cost_planned + fn * cost_unplanned
    net_savings     = baseline_cost - model_cost
    gross_savings   = tp * (cost_unplanned - cost_planned)
    false_alarm_cost = fp * cost_planned

    result = {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "cost_unplanned":   cost_unplanned,
        "cost_planned":     cost_planned,
        "n_near_failure":   n_near_failure,
        "baseline_cost":    round(baseline_cost,    2),
        "model_cost":       round(model_cost,        2),
        "gross_savings":    round(gross_savings,     2),
        "false_alarm_cost": round(false_alarm_cost,  2),
        "net_savings":      round(net_savings,        2),
        "savings_pct":      round(
            100.0 * net_savings / baseline_cost, 1
        ) if baseline_cost > 0 else 0.0,
    }
    logger.info(
        "Cost savings (at-risk fleet of %d engines): "
        "baseline=$%.0f  model=$%.0f  net_savings=$%.0f (%.1f%%)\n"
        "  Gross savings from TP: $%.0f | False alarm cost (FP): $%.0f",
        n_near_failure, baseline_cost, model_cost, net_savings, result["savings_pct"],
        gross_savings, false_alarm_cost,
    )
    return result
