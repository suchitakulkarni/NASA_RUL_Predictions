# src/condition_normaliser.py
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

OP_COLS = ["op1", "op2", "op3"]


class ConditionNormaliser:
    def __init__(self, n_clusters=6, n_init=10, random_state=42):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.random_state = random_state
        self.kmeans = None
        self.sensor_cols = None
        self.cluster_stats = {}

    def fit(self, df, sensor_cols):
        logger.info(
            "Fitting ConditionNormaliser: n_clusters=%d, %d rows, %d sensors",
            self.n_clusters, len(df), len(sensor_cols),
        )
        self.sensor_cols = list(sensor_cols)

        self.kmeans = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state, n_init=self.n_init
        )
        labels = self.kmeans.fit_predict(df[OP_COLS])

        unique, counts = np.unique(labels, return_counts=True)
        logger.info("Cluster distribution: %s", dict(zip(unique.tolist(), counts.tolist())))

        df_tmp = df.assign(_cluster=labels)
        for c in range(self.n_clusters):
            subset = df_tmp.loc[df_tmp["_cluster"] == c, self.sensor_cols]
            mean = subset.mean().values
            std = subset.std().clip(lower=1e-6).values  # prevent zero division
            self.cluster_stats[c] = {"mean": mean, "std": std}
            logger.info("Cluster %d: %d rows, sensor mean range [%.3f, %.3f]",
                        c, len(subset), mean.min(), mean.max())

        return self

    def transform(self, df):
        if self.kmeans is None:
            raise RuntimeError("ConditionNormaliser must be fit before transform")

        df = df.copy()
        # Ensure sensor columns are float so in-place normalisation doesn't hit
        # pandas dtype mismatch when raw data is loaded as int64.
        df[self.sensor_cols] = df[self.sensor_cols].astype(float)
        labels = self.kmeans.predict(df[OP_COLS])
        df["_cluster"] = labels

        for c, stats in self.cluster_stats.items():
            mask = df["_cluster"] == c
            if not mask.any():
                logger.warning("Cluster %d has no rows in this split — skipping", c)
                continue
            df.loc[mask, self.sensor_cols] = (
                (df.loc[mask, self.sensor_cols].values - stats["mean"]) / stats["std"]
            )

        df.drop(columns=["_cluster"], inplace=True)
        unique_clusters = len(np.unique(labels))
        logger.info(
            "ConditionNormaliser.transform: %d rows assigned across %d clusters",
            len(df), unique_clusters,
        )
        return df

    def predict_clusters(self, df):
        """Return KMeans cluster label (int) for each row without normalising."""
        if self.kmeans is None:
            raise RuntimeError("ConditionNormaliser must be fit before predict_clusters")
        return self.kmeans.predict(df[OP_COLS]).astype(int)

    def fit_transform(self, df, sensor_cols):
        return self.fit(df, sensor_cols).transform(df)

    def save(self, path):
        joblib.dump(self, path)
        logger.info("ConditionNormaliser saved to %s", path)

    @classmethod
    def load(cls, path):
        obj = joblib.load(path)
        logger.info("ConditionNormaliser loaded from %s", path)
        return obj
