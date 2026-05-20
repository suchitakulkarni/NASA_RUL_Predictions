# src/model.py
import os
import logging, joblib
import numpy as np
import pandas as pd
import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


def split_train_test(data_train):
    unique_units = data_train["unit_no"].unique()
    logger.info("Splitting %d units into train/test (80/20)", len(unique_units))

    train_ids, test_ids = train_test_split(unique_units, test_size=0.2, random_state=42)
    train_df = data_train[data_train["unit_no"].isin(train_ids)]
    test_df = data_train[data_train["unit_no"].isin(test_ids)]

    #train_ids, test_valid_ids = train_test_split(unique_units, test_size=0.4, random_state=42)
    #test_ids, valid_ids = train_test_split(test_valid_ids, test_size=0.2, random_state=42)
    #train_df = data_train[data_train["unit_no"].isin(train_ids)]
    #test_df = data_train[data_train["unit_no"].isin(test_ids)]
    #valid_df = data_train[data_train["unit_no"].isin(valid_ids)]

    logger.info("Train set: %d units, %d rows", len(train_ids), len(train_df))
    logger.info("Test set: %d units, %d rows", len(test_ids), len(test_df))
    logger.debug("Train unit IDs: %s", sorted(train_ids.tolist()))

    return train_df, test_df


def create_sliding_window(df, feature_cols, window_length=30):
    logger.info("Creating sliding windows: window_length=%d, features=%d", window_length, len(feature_cols))

    x, y, unit_id = [], [], []
    skipped_units = []

    for unit in df["unit_no"].unique():
        unit_df = df[df["unit_no"] == unit]

        if len(unit_df) <= window_length:
            logger.warning(
                "Unit %s has only %d rows, shorter than window_length=%d -- skipping",
                unit, len(unit_df), window_length
            )
            skipped_units.append(unit)
            continue

        n_windows = len(unit_df) - window_length
        logger.debug("Unit %s: %d rows, generating %d windows", unit, len(unit_df), n_windows)

        for i in range(n_windows):
            window = unit_df.iloc[i:i + window_length][feature_cols].values
            label  = unit_df.iloc[i + window_length]["RUL_calc"]
            uid    = unit_df.iloc[i + window_length]["unit_no"]
            x.append(window)
            y.append(label)
            unit_id.append(uid)

    X, Y = np.array(x), np.array(y)
    logger.info("Sliding window complete: X shape=%s, y shape=%s", X.shape, Y.shape)
    logger.info("RUL range: min=%.1f, max=%.1f, mean=%.1f", Y.min(), Y.max(), Y.mean())

    if skipped_units:
        logger.warning("%d units skipped due to insufficient length: %s", len(skipped_units), skipped_units)

    return X, Y, np.asarray(unit_id)


def _optimize_single_quantile(X_train, y_train, X_test, y_test, trial, q):
    params = {
        "n_estimators":    trial.suggest_int("n_estimators", 40, 100),
        "max_depth":       trial.suggest_int("max_depth", 3, 10),
        "eta":             trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "verbosity": 0
    }

    logger.debug("Trial %d for q=%.2f: params=%s", trial.number, q, params)

    model = XGBRegressor(objective="reg:quantileerror", quantile_alpha=q, **params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    mae = mean_absolute_error(y_test, model.predict(X_test))
    logger.debug("Trial %d for q=%.2f: MAE=%.4f", trial.number, q, mae)
    return mae


def run_all_quantile_models(X_train, y_train, X_test, y_test, X_valid, quantiles=None, n_trials=50, n_jobs = 1, model_dir = './saved_models'):

    os.makedirs(model_dir, exist_ok = True)
    if quantiles is None:
        quantiles = [0.05, 0.5, 0.95]

    logger.info("Starting quantile optimisation: quantiles=%s, n_trials=%d", quantiles, n_trials)
    logger.info("Train size: %d, Test size: %d, Validation size: %d",
                len(X_train), len(X_test), len(X_valid))

    best_params = {}
    for q in quantiles:
        logger.info("Optimising quantile q=%.2f", q)
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: _optimize_single_quantile(X_train, y_train, X_test, y_test, trial, q),
            n_trials=n_trials, n_jobs=n_jobs
        )
        logger.info("Quantile q=%.2f done: best MAE=%.4f, best params=%s",
                    q, study.best_value, study.best_params)
        best_params[q] = study.best_params


        # Step 2: Train final model on X_train + X_test
        X_train_full = np.concatenate([X_train, X_test], axis=0)
        y_train_full = np.concatenate([y_train, y_test], axis=0)

        final_model = XGBRegressor(
            **study.best_params,
            objective='reg:quantileerror',
            quantile_alpha=q,
            random_state=42
        )
        final_model.fit(X_train_full, y_train_full) 


        model_filename = os.path.join(model_dir, f"quantile_{q:.2f}_model.pkl")
        joblib.dump(final_model, model_filename)
        logger.info(f"Saved final model (trained on X_train + X_valid) to {model_filename}")

    return best_params


def predict(X_train, y_train, X_valid, model_dir = './saved_models', quantiles=None):
    """
    Retrain each quantile model on X_train with the best params found during
    optimisation, then predict on X_valid.

    Returns a dict {quantile: predictions array}.
    """
    if quantiles is None:
        quantiles = [0.05, 0.5, 0.95]

    logger.info("Generating predictions for quantiles: %s", quantiles)
    logger.info("Retraining on full X_train (%d samples) before predicting", len(X_train))

    ypreds = {}
    for q in quantiles:
        model_path = os.path.join(model_dir, f"quantile_{q:.2f}_model.pkl")
        if not os.path.exists(model_path):
            logger.error("No trained params found for quantile q=%.2f", q)
            raise KeyError(f"Missing model params for quantile {q}")

        logger.debug("Retraining and predicting for q=%.2f", q)
        #model = XGBRegressor(
        #    objective="reg:quantileerror",
        #    quantile_alpha=q,
        #    **models[q]
        #)
        #model.fit(X_train, y_train)
        
        model = joblib.load(model_path)
        
        y_pred = model.predict(X_valid)
        ypreds[q] = y_pred

        logger.info("q=%.2f predictions: min=%.1f, max=%.1f, mean=%.1f",
                    q, y_pred.min(), y_pred.max(), y_pred.mean())

    return ypreds
