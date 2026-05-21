# src/feature_engineering.py
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Sensors with near-zero variance across all operating conditions — confirmed
# empirically across FD001-FD004 (std < 0.1 after condition normalisation).
DEFAULT_SENSORS_TO_DROP = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]


class FeatureEngineer:
    def __init__(self, drop_sensors=None, window_sizes=None):
        self.drop_sensors = drop_sensors if drop_sensors is not None else DEFAULT_SENSORS_TO_DROP
        self.window_sizes = window_sizes if window_sizes is not None else [5, 10]
        self.sensor_cols = None
        self.max_cycle = None

    def fit(self, df):
        """
        Learn which sensors to keep and the max training cycle.
        Must be called on training data only.
        """
        all_sensor_cols = [c for c in df.columns if c.startswith("s") and c[1:].isdigit()]
        self.sensor_cols = [c for c in all_sensor_cols if c not in self.drop_sensors]
        self.max_cycle = df["time"].max()
        logger.info(
            "FeatureEngineer fit: keeping %d/%d sensors (dropped: %s), max_cycle=%d",
            len(self.sensor_cols),
            len(all_sensor_cols),
            self.drop_sensors,
            self.max_cycle,
        )
        return self

    def transform(self, df, dataset_id=None):
        """
        Apply rolling statistics, cycle normalisation, and optional dataset_id.
        Expects 'time' (cycle) and 'unit_no' columns to be present.
        Returns (augmented_df, feature_col_list).
        """
        if self.sensor_cols is None:
            raise RuntimeError("FeatureEngineer must be fit before transform")

        df = df.copy()

        for col in self.sensor_cols:
            for w in self.window_sizes:
                grouped = df.groupby("unit_no")[col]
                df[f"{col}_roll_mean_{w}"] = grouped.transform(
                    lambda x, w=w: x.rolling(w, min_periods=1).mean()
                )
                df[f"{col}_roll_std_{w}"] = grouped.transform(
                    lambda x, w=w: x.rolling(w, min_periods=1).std().fillna(0.0)
                )

        df["cycle_norm"] = df["time"] / self.max_cycle

        if dataset_id is not None:
            df["dataset_id"] = int(dataset_id)

        roll_cols = [c for c in df.columns if "_roll_" in c]
        extra_cols = ["cycle_norm"] + (["dataset_id"] if dataset_id is not None else [])
        feat_cols = self.sensor_cols + roll_cols + extra_cols

        logger.info(
            "FeatureEngineer.transform: %d rows, %d features "
            "(%d sensors + %d rolling + %d extra)",
            len(df),
            len(feat_cols),
            len(self.sensor_cols),
            len(roll_cols),
            len(extra_cols),
        )
        return df, feat_cols

    def fit_transform(self, df, dataset_id=None):
        return self.fit(df).transform(df, dataset_id=dataset_id)
