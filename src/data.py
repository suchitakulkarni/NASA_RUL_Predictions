# src/data.py
import os
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def load_raw(data_dir="./CMAPSSData", dataset_id="FD001"):
    """
    Load raw train/test data and external RUL labels, keeping all columns
    including the time (cycle) column for use by the Phase 3/4 pipeline.
    """
    sfeatures = ["s" + str(i) for i in range(1, 22)]
    col_names = ["unit_no", "time", "op1", "op2", "op3"] + sfeatures

    train_path = os.path.join(data_dir, f"train_{dataset_id}.txt")
    test_path  = os.path.join(data_dir, f"test_{dataset_id}.txt")
    rul_path   = os.path.join(data_dir, f"RUL_{dataset_id}.txt")

    for path in [train_path, test_path, rul_path]:
        if not os.path.exists(path):
            logger.error("Expected data file not found: %s", path)
            raise FileNotFoundError(f"Data file missing: {path}")

    train_df   = pd.read_csv(train_path, sep=r"\s+", header=None, names=col_names)
    test_df    = pd.read_csv(test_path,  sep=r"\s+", header=None, names=col_names)
    rul_labels = pd.read_csv(rul_path,   sep=r"\s+", header=None, names=["RUL"])
    rul_labels["unit_no"] = rul_labels.index + 1

    logger.info(
        "Loaded %s: train=%d rows/%d units, test=%d rows/%d units",
        dataset_id,
        len(train_df), train_df["unit_no"].nunique(),
        len(test_df),  test_df["unit_no"].nunique(),
    )
    return train_df, test_df, rul_labels
