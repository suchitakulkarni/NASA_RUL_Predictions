"""
Compute lead-time statistics for individual and ensemble XGBoost RUL models.

Lead time = (step index where true RUL first crosses alarm_threshold)
          - (step index where predicted RUL first crosses alarm_threshold)

  > 0  → early warning  (model alarmed before true failure zone — good)
  < 0  → late warning   (model alarmed after the engine entered the critical zone)

Two exclusion filters are applied per unit and reported:
  1. Sub-threshold / flat trajectories: true RUL never falls to alarm_threshold
     within the test window (no meaningful crossing time can be defined).
  2. High-error units: per-unit MAE of the median (q=0.50) prediction exceeds
     mae_threshold (prediction is too far off for lead time to be informative).

Results are printed per dataset and saved to results/lead_time_results.csv.
"""

import numpy as np
import pandas as pd
import os

from src.data import load_datasets, normalize_data
from src.model import create_sliding_window
from test_performance import load_individual_model, load_ensemble_model

# ── Configuration ─────────────────────────────────────────────────────────────
ALARM_THRESHOLD = 30   # cycles: critical-zone boundary
MAE_THRESHOLD   = 50   # cycles: exclude unit if per-unit median MAE exceeds this
DATASET_IDS     = ["FD001", "FD002", "FD003", "FD004"]
OUTPUT_PATH     = "results/lead_time_results.csv"
# ──────────────────────────────────────────────────────────────────────────────


def load_test_data_with_uid(dataset_id):
    """Return (X_flat, y, uid) for the validation/test split of dataset_id."""
    data_train, data_valid, RUL_labels, feature_cols = load_datasets(dataset_id=dataset_id)
    data_train, data_valid, _ = normalize_data(data_train, data_valid, feature_cols)
    x_valid, y_valid, uid_valid = create_sliding_window(data_valid, feature_cols)
    X_valid = x_valid.reshape(x_valid.shape[0], -1)
    return X_valid, y_valid, uid_valid


def first_crossing_index(values, threshold):
    """Index of first entry <= threshold, or None if never crossed."""
    idx = np.where(values <= threshold)[0]
    return int(idx[0]) if len(idx) > 0 else None


def compute_unit_lead_times(y_true, y_pred, uid, alarm_threshold, mae_threshold):
    """
    Per-unit lead time computation with exclusion filtering.

    Returns
    -------
    results        : list of dicts for included units
    excluded_flat  : list of unit IDs excluded as sub-threshold
    excluded_error : list of unit IDs excluded as high-error
    """
    df = pd.DataFrame({"unit_no": uid, "true_RUL": y_true, "pred_RUL": y_pred})
    units = df["unit_no"].unique()

    results, excluded_flat, excluded_error = [], [], []

    for unit in units:
        mask = df["unit_no"] == unit
        true_rul = df.loc[mask, "true_RUL"].values
        pred_rul = df.loc[mask, "pred_RUL"].values

        # --- Exclusion 1: true RUL never reaches alarm threshold ---
        t_true = first_crossing_index(true_rul, alarm_threshold)
        if t_true is None:
            excluded_flat.append(unit)
            continue

        # --- Exclusion 2: prediction too far from truth ---
        unit_mae = float(np.mean(np.abs(true_rul - pred_rul)))
        if unit_mae > mae_threshold:
            excluded_error.append(unit)
            continue

        t_pred = first_crossing_index(pred_rul, alarm_threshold)
        # If model never crosses alarm: treat as a maximally late warning
        # (it is included but flagged via t_pred=None → lead_time negative)
        if t_pred is None:
            lead_time = t_true - len(true_rul)   # large negative
        else:
            lead_time = t_true - t_pred

        results.append({
            "unit_no": unit,
            "lead_time": lead_time,
            "t_true_crossing": t_true,
            "t_pred_crossing": t_pred,
            "unit_mae": unit_mae,
            "pred_never_crosses": t_pred is None,
        })

    return results, excluded_flat, excluded_error


def print_lead_time_stats(label, results_df, n_total, mae_threshold):
    n_incl = len(results_df)
    if n_incl == 0:
        print(f"    {label}: no units remaining after exclusions.")
        return

    lt = results_df["lead_time"]
    print(f"    {label}:")
    print(f"      included units : {n_incl} / {n_total}")
    print(f"      mean ± std     : {lt.mean():.1f} ± {lt.std():.1f} cycles")
    print(f"      median         : {lt.median():.1f} cycles")
    print(f"      early (> 0)    : {(lt > 0).sum()}  ({100*(lt>0).mean():.1f}%)")
    print(f"      late  (≤ 0)    : {(lt <= 0).sum()}  ({100*(lt<=0).mean():.1f}%)")
    never = results_df["pred_never_crosses"].sum()
    if never:
        print(f"      pred never crossed alarm: {never}")


def run_dataset(test_ds, alarm_threshold, mae_threshold):
    """
    Compute lead times for one test dataset using:
      - the "home" individual model (trained on the same dataset)
      - the ensemble model
    Returns a DataFrame of per-unit results with a 'model' column.
    """
    print(f"\n{'─'*60}")
    print(f"  Test dataset: {test_ds}")
    print(f"{'─'*60}")

    X_test, y_test, uid_test = load_test_data_with_uid(test_ds)
    n_total = len(np.unique(uid_test))
    print(f"  Total units in test set: {n_total}")

    all_rows = []

    # ── Individual model (home: trained on same dataset) ──────────────────────
    indiv_model = load_individual_model(test_ds, 0.50)
    y_pred_indiv = indiv_model.predict(X_test)
    res_indiv, excl_flat_i, excl_err_i = compute_unit_lead_times(
        y_test, y_pred_indiv, uid_test, alarm_threshold, mae_threshold
    )
    print(f"\n  Individual (train={test_ds}, q=0.50):")
    print(f"    excluded — flat / sub-threshold : {len(excl_flat_i)}"
          f"  ({100*len(excl_flat_i)/n_total:.1f}%)")
    print(f"    excluded — MAE > {mae_threshold} cycles   : {len(excl_err_i)}"
          f"  ({100*len(excl_err_i)/n_total:.1f}%)")
    df_indiv = pd.DataFrame(res_indiv)
    if len(df_indiv):
        df_indiv["model"] = f"individual_{test_ds}"
        df_indiv["test_ds"] = test_ds
        print_lead_time_stats("Lead time", df_indiv, n_total, mae_threshold)
        all_rows.append(df_indiv)

    # ── Ensemble model ─────────────────────────────────────────────────────────
    ensemble_model = load_ensemble_model(0.50)
    y_pred_ens = ensemble_model.predict(X_test)
    res_ens, excl_flat_e, excl_err_e = compute_unit_lead_times(
        y_test, y_pred_ens, uid_test, alarm_threshold, mae_threshold
    )
    print(f"\n  Ensemble (q=0.50):")
    print(f"    excluded — flat / sub-threshold : {len(excl_flat_e)}"
          f"  ({100*len(excl_flat_e)/n_total:.1f}%)")
    print(f"    excluded — MAE > {mae_threshold} cycles   : {len(excl_err_e)}"
          f"  ({100*len(excl_err_e)/n_total:.1f}%)")
    df_ens = pd.DataFrame(res_ens)
    if len(df_ens):
        df_ens["model"] = "ensemble"
        df_ens["test_ds"] = test_ds
        print_lead_time_stats("Lead time", df_ens, n_total, mae_threshold)
        all_rows.append(df_ens)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


# ── Main ───────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Lead Time Analysis")
print(f"  alarm_threshold = {ALARM_THRESHOLD} cycles")
print(f"  mae_threshold   = {MAE_THRESHOLD} cycles")
print("=" * 60)

all_datasets_results = []
for ds in DATASET_IDS:
    df_ds = run_dataset(ds, ALARM_THRESHOLD, MAE_THRESHOLD)
    all_datasets_results.append(df_ds)

combined = pd.concat(all_datasets_results, ignore_index=True)

# ── Cross-dataset summary ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  Overall Summary (all datasets combined)")
print(f"{'='*60}")
for model_label in combined["model"].unique():
    sub = combined[combined["model"] == model_label]
    lt = sub["lead_time"]
    print(f"\n  {model_label}:")
    print(f"    included units : {len(sub)}")
    print(f"    mean ± std     : {lt.mean():.1f} ± {lt.std():.1f} cycles")
    print(f"    median         : {lt.median():.1f} cycles")
    print(f"    early (> 0)    : {(lt > 0).sum()}  ({100*(lt>0).mean():.1f}%)")
    print(f"    late  (≤ 0)    : {(lt <= 0).sum()}  ({100*(lt<=0).mean():.1f}%)")

# ── Save results ───────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
combined.to_csv(OUTPUT_PATH, index=False)
print(f"\nPer-unit results saved to {OUTPUT_PATH}")
