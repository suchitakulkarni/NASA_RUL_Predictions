# src/data.py
import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def load_datasets(data_dir="./CMAPSSData", dataset_id="FD001"):
    sfeatures = ["s" + str(i) for i in range(1, 22)]
    col_names = ["unit_no", "time", "op1", "op2", "op3"] + sfeatures

    train_path = os.path.join(data_dir, f"train_{dataset_id}.txt")
    test_path  = os.path.join(data_dir, f"test_{dataset_id}.txt")
    rul_path   = os.path.join(data_dir, f"RUL_{dataset_id}.txt")

    for path in [train_path, test_path, rul_path]:
        if not os.path.exists(path):
            logger.error("Expected data file not found: %s", path)
            raise FileNotFoundError(f"Data file missing: {path}")

    logger.info("Loading dataset %s from %s", dataset_id, data_dir)

    data_train = pd.read_csv(train_path, sep=r"\s+", header=None, names=col_names)
    data_test  = pd.read_csv(test_path,  sep=r"\s+", header=None, names=col_names)
    RUL_labels = pd.read_csv(rul_path,   sep=r"\s+", header=None, names=["RUL"])

    logger.info("Raw train: %d rows, %d units", len(data_train), data_train["unit_no"].nunique())
    logger.info("Raw test:  %d rows, %d units", len(data_test),  data_test["unit_no"].nunique())
    logger.info("RUL labels: %d entries", len(RUL_labels))

    # Validate no nulls in critical columns
    for name, df in [("train", data_train), ("test", data_test)]:
        null_counts = df[col_names].isnull().sum()
        if null_counts.any():
            logger.warning("Null values found in %s data:\n%s", name, null_counts[null_counts > 0])
        else:
            logger.debug("No null values in %s data", name)

    # Compute RUL for train
    max_times = data_train.groupby("unit_no")["time"].max().reset_index()
    data_train = data_train.merge(max_times, on="unit_no", suffixes=("", "_max"))
    data_train["RUL_calc"] = data_train["time_max"] - data_train["time"]
    data_train.drop(["time", "time_max"], axis=1, inplace=True)

    logger.info("Train RUL_calc: min=%.1f, max=%.1f, mean=%.1f",
                data_train["RUL_calc"].min(),
                data_train["RUL_calc"].max(),
                data_train["RUL_calc"].mean())

    # Compute RUL for test
    RUL_labels["unit_no"] = RUL_labels.index + 1
    max_times_test = data_test.groupby("unit_no")["time"].max().reset_index()
    data_test = data_test.merge(max_times_test, on="unit_no", suffixes=("", "_max"))
    data_test = data_test.merge(RUL_labels, on="unit_no")
    data_test["RUL_calc"] = data_test["time_max"] - data_test["time"] + data_test["RUL"]
    data_test.drop(["time", "time_max", "RUL"], axis=1, inplace=True)

    logger.info("Test RUL_calc: min=%.1f, max=%.1f, mean=%.1f",
                data_test["RUL_calc"].min(),
                data_test["RUL_calc"].max(),
                data_test["RUL_calc"].mean())

    feature_cols = ["op1", "op2", "op3"] + sfeatures
    logger.debug("Feature columns (%d total): %s", len(feature_cols), feature_cols)

    return data_train, data_test, RUL_labels, feature_cols


def normalize_data(data_train, data_test, feature_cols):
    logger.info("Normalizing %d feature columns with StandardScaler", len(feature_cols))

    # Validate expected columns are present
    for name, df in [("train", data_train), ("test", data_test)]:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            logger.error("Missing feature columns in %s data: %s", name, missing)
            raise ValueError(f"Missing columns in {name}: {missing}")

    scaler = StandardScaler()
    data_train = data_train.copy()
    data_test  = data_test.copy()

    data_train[feature_cols] = scaler.fit_transform(data_train[feature_cols])
    data_test[feature_cols]  = scaler.transform(data_test[feature_cols])

    logger.debug("Scaler mean (first 3 features): %s", scaler.mean_[:3].round(4))
    logger.debug("Scaler std  (first 3 features): %s", scaler.scale_[:3].round(4))
    logger.info("Normalization complete")

    return data_train, data_test, scaler
