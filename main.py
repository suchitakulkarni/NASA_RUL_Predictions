# main.py
import logging
import numpy as np

from src.utils import setup_logging, RESULTS_DIR
from src.data import load_datasets, normalize_data
from src.model import split_train_test, create_sliding_window, run_all_quantile_models, predict
from src.evaluate import write_results_to_csv, plot_results

logger = logging.getLogger(__name__)


def run_pipeline(dataset_id = 'FD001'):
    logger.info("Pipeline started")

    # --- Data loading and preprocessing ---
    data_train, data_valid, RUL_labels, feature_cols = load_datasets(dataset_id=dataset_id)

    # Split by unit ID first so the scaler is fit only on training units.
    # Fitting on all data before splitting leaks test-unit statistics into the scaler.
    split_train_df, split_test_df = split_train_test(data_train)
    split_train_df, split_test_df, scaler = normalize_data(split_train_df, split_test_df, feature_cols)
    data_valid = data_valid.copy()
    data_valid[feature_cols] = scaler.transform(data_valid[feature_cols])

    # --- Sliding windows ---
    x_train, y_train, uid_train = create_sliding_window(split_train_df, feature_cols)
    x_test, y_test, uid_test = create_sliding_window(split_test_df, feature_cols)
    x_valid,  y_valid,  uid_valid  = create_sliding_window(data_valid, feature_cols)

    # --- Flatten for XGBoost ---
    X_train = x_train.reshape(x_train.shape[0], -1)
    X_test = x_test.reshape(x_test.shape[0], -1)
    X_valid  = x_valid.reshape(x_valid.shape[0], -1)

    logger.info("Flattened shapes -- X_train: %s, X_test: %s, X_valid: %s",
                X_train.shape, X_test.shape, X_valid.shape)

    # --- Optimisation ---
    quantiles = [0.05, 0.5, 0.95]
    models = run_all_quantile_models(X_train, y_train, X_test, y_test, X_valid, quantiles=quantiles, n_jobs = 4, model_dir = f"./saved_models_{dataset_id}")

    # --- Prediction ---
    ypreds = predict(X_train, y_train, X_valid, model_dir = f"./saved_models_{dataset_id}", quantiles=quantiles)

    # --- Evaluation and output ---
    df_preds = write_results_to_csv(uid_valid, ypreds, y_valid, quantiles=quantiles)
    fig = plot_results(df_preds, RUL_labels, title=f"pred_vs_true_{dataset_id}")

    logger.info("Pipeline complete. Results written to %s/", RESULTS_DIR)
    return df_preds, fig


if __name__ == "__main__":
    setup_logging(level=logging.INFO)
    run_pipeline(dataset_id = 'FD001')
