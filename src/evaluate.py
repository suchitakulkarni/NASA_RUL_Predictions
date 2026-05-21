# src/evaluate.py
import logging
import os
import matplotlib.pyplot as plt
import pandas as pd
from src.utils import RESULTS_DIR

logger = logging.getLogger(__name__)


def write_results_to_csv(uid_test, ypreds, y_test, quantiles=None):
    """
    Write per-sample predictions for all quantiles alongside true RUL to CSV.

    Parameters
    ----------
    uid_test  : array-like, unit IDs for each test sample
    ypreds    : dict {quantile: predictions array} as returned by predict()
    y_test    : array-like, true RUL values
    quantiles : list of quantiles; must match keys in ypreds
    """
    if quantiles is None:
        quantiles = [0.05, 0.5, 0.95]

    logger.info("Writing predictions to CSV: %d samples, quantiles=%s", len(y_test), quantiles)

    for q in quantiles:
        if q not in ypreds:
            logger.error("Quantile q=%.2f missing from ypreds dict", q)
            raise KeyError(f"Missing predictions for quantile {q}")

    df_preds = pd.DataFrame({"unit_no": uid_test, "RUL_true": y_test})
    for q in quantiles:
        col = f"RUL_pred_{str(q).replace('.', '')}"
        df_preds[col] = ypreds[q]

    out_path = os.path.join(RESULTS_DIR, "final_results.csv")
    df_preds.to_csv(out_path, index=False)
    logger.info("Predictions saved to %s", out_path)
    logger.debug("Prediction dataframe shape: %s", df_preds.shape)

    return df_preds


def plot_results(df_preds, RUL_labels, title="pred_vs_true"):
    """
    Plot median prediction vs true RUL with 90% quantile interval.

    Parameters
    ----------
    df_preds   : DataFrame as returned by write_results_to_csv()
    RUL_labels : DataFrame with columns [unit_no, RUL] -- true end-of-life RUL per unit
    title      : filename stem for the saved plot
    """
    logger.info("Plotting prediction vs true RUL for %d units", df_preds["unit_no"].nunique())

    RUL_per_unit = df_preds.groupby("unit_no").last().reset_index()
    logger.debug("RUL_per_unit shape before merge: %s", RUL_per_unit.shape)

    RUL_per_unit = RUL_per_unit.merge(RUL_labels, on="unit_no")
    RUL_per_unit = RUL_per_unit.sort_values(by="RUL").reset_index(drop=True)

    logger.debug("RUL_per_unit shape after merge: %s", RUL_per_unit.shape)
    logger.info("True RUL range: min=%.1f, max=%.1f", RUL_per_unit["RUL"].min(), RUL_per_unit["RUL"].max())

    # Check for expected columns
    required_cols = ["RUL", "RUL_pred_005", "RUL_pred_05", "RUL_pred_095"]
    missing = [c for c in required_cols if c not in RUL_per_unit.columns]
    if missing:
        logger.error("Missing columns for plotting: %s", missing)
        raise ValueError(f"Missing columns in RUL_per_unit: {missing}")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")

    ax.plot(RUL_per_unit["RUL"], RUL_per_unit["RUL"],
            label="Perfect prediction", color="black", linestyle="--")
    ax.plot(RUL_per_unit["RUL"], RUL_per_unit["RUL_pred_05"],
            label="Median (0.5)", color="blue")
    ax.fill_between(
        RUL_per_unit["RUL"],
        RUL_per_unit["RUL_pred_005"],
        RUL_per_unit["RUL_pred_095"],
        color="gray", alpha=0.3, label="90% Interval"
    )

    ax.set_xlabel("True RUL", fontsize=14)
    ax.set_ylabel("Predicted RUL", fontsize=14)
    ax.set_title("Quantile Prediction vs True RUL", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)

    csv_path = os.path.join(RESULTS_DIR, "RUL_per_unit.csv")
    RUL_per_unit.to_csv(csv_path, index=False)
    logger.info("Per-unit RUL summary saved to %s", csv_path)

    plot_path = os.path.join(RESULTS_DIR, f"{title}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    logger.info("Plot saved to %s", plot_path)

    return fig


