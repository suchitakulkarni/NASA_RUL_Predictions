# src/rul_target.py
# Expects a DataFrame that still has the 'time' (cycle) column.
# In the new pipeline, call before dropping 'time' from the raw data.
import logging
import numpy as np

logger = logging.getLogger(__name__)


def compute_rul(df, cap=125):
    """
    Piecewise-linear RUL per unit, capped at `cap` cycles.
    Capping prevents the model wasting capacity on unpredictable
    early-life cycles where RUL is effectively uninformative.
    """
    df = df.copy()
    max_cycles = df.groupby("unit_no")["time"].transform("max")
    df["RUL"] = (max_cycles - df["time"]).clip(upper=cap)
    logger.info(
        "RUL computed: cap=%d, min=%.1f, max=%.1f, mean=%.1f",
        cap, df["RUL"].min(), df["RUL"].max(), df["RUL"].mean(),
    )
    return df


def compute_lead_time(rul_predictions, threshold):
    """
    Cycles of advance warning before predicted RUL crosses `threshold`.
    Returns 0 for units already at or below the threshold.
    """
    rul_predictions = np.asarray(rul_predictions)
    lead_times = np.maximum(rul_predictions - threshold, 0)
    logger.info(
        "Lead time (threshold=%d): mean=%.1f, median=%.1f, already_below=%d",
        threshold,
        lead_times.mean(),
        float(np.median(lead_times)),
        int((rul_predictions <= threshold).sum()),
    )
    return lead_times
