# src/model.py
import os
import logging, joblib
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.config import ModelConfig

optuna.logging.set_verbosity(optuna.logging.WARNING)


class _XGBPruningCallback(xgb.callback.TrainingCallback):
    """Report per-round val RMSE to Optuna so unpromising trials are cut early."""

    def __init__(self, trial: optuna.Trial) -> None:
        super().__init__()
        self._trial = trial

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        # evals_log: {"validation_0": {"rmse": [...]}}
        scores = next(iter(next(iter(evals_log.values())).values()))
        self._trial.report(scores[-1], epoch)
        if self._trial.should_prune():
            raise optuna.TrialPruned()
        return False

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


class RULModel:
    """XGBoost regressor for RUL estimation with Optuna hyperparameter search."""

    def __init__(self):
        self.model = None
        self.best_params = None
        self.val_rmse = None

    def train(self, X, y, cfg: ModelConfig, stratify=None, n_jobs=-1, feat_cols=None):
        """
        Find best hyperparameters via Optuna (cfg.val_split holdout), then fit
        final model on full (X, y).

        cfg:       ModelConfig driving trials, val_split, random_state, search bounds.
        stratify:  dataset_id array for joint mode so each dataset is equally
                   represented in the holdout.
        n_jobs:    XGBoost CPU threads; pass 1 when running multiple models in
                   parallel to avoid core oversubscription.
        feat_cols: column names matching X columns; used to enforce a monotone
                   decreasing constraint on cycle_norm (RUL must fall as the
                   engine ages).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        ss = cfg.search_space

        mono = None
        if feat_cols is not None and "cycle_norm" in feat_cols:
            mono = tuple(-1 if c == "cycle_norm" else 0 for c in feat_cols)
            logger.info(
                "Monotone constraint: cycle_norm (index %d) set to -1, all others 0",
                list(feat_cols).index("cycle_norm"),
            )

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=cfg.val_split, random_state=cfg.random_state, stratify=stratify
        )
        logger.info(
            "Optuna search: train=%d, val=%d, trials=%d, n_jobs=%d",
            len(X_tr), len(X_val), cfg.optuna_trials, n_jobs,
        )

        early_stop = cfg.early_stopping_rounds

        def objective(trial):
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", *ss.n_estimators),
                "max_depth":        trial.suggest_int("max_depth", *ss.max_depth),
                "learning_rate":    trial.suggest_float("learning_rate", *ss.learning_rate, log=True),
                "subsample":        trial.suggest_float("subsample", *ss.subsample),
                "colsample_bytree": trial.suggest_float("colsample_bytree", *ss.colsample_bytree),
                "reg_alpha":        trial.suggest_float("reg_alpha", *ss.reg_alpha, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", *ss.reg_lambda, log=True),
                "verbosity": 0,
                "random_state": cfg.random_state,
                "n_jobs": n_jobs,
                "early_stopping_rounds": early_stop,
            }
            if mono is not None:
                params["monotone_constraints"] = mono
            m = XGBRegressor(**params, callbacks=[_XGBPruningCallback(trial)])
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            rmse = np.sqrt(mean_squared_error(y_val, m.predict(X_val)))
            return rmse

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=cfg.pruner_startup_trials,
            n_warmup_steps=cfg.pruner_warmup_steps,
        )
        study = optuna.create_study(direction="minimize", pruner=pruner)
        study.optimize(objective, n_trials=cfg.optuna_trials)

        self.best_params = study.best_params
        self.val_rmse = study.best_value
        logger.info("Best val RMSE=%.4f, params=%s", self.val_rmse, self.best_params)

        final_params = {
            **self.best_params,
            "verbosity": 0,
            "random_state": cfg.random_state,
            "n_jobs": n_jobs,
        }
        if mono is not None:
            final_params["monotone_constraints"] = mono
        self.model = XGBRegressor(**final_params)
        self.model.fit(X, y, verbose=False)
        logger.info("Final model fitted on %d samples", len(X))
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("RULModel must be trained before predict")
        return self.model.predict(np.asarray(X, dtype=float))

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump(self, path)
        logger.info("RULModel saved to %s", path)

    @classmethod
    def load(cls, path):
        obj = joblib.load(path)
        logger.info("RULModel loaded from %s", path)
        return obj


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
