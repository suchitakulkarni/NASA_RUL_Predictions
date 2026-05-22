# src/model.py
import os
import logging, joblib
import numpy as np
import optuna
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import mean_squared_error
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


def _tilted_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Asymmetric (tilted) MAE: penalises over-prediction by alpha, under-prediction by 1-alpha."""
    e = y_pred - y_true
    return float(np.mean(np.where(e >= 0, alpha * e, (alpha - 1.0) * e)))


def _asymmetric_mae_obj(alpha: float):
    """XGBoost custom objective for tilted/pinball loss with constant hessian."""
    def objective(y_pred, dtrain):
        e = y_pred - dtrain.get_label()
        grad = np.where(e >= 0, alpha, alpha - 1.0)
        hess = np.ones_like(grad)
        return grad, hess
    return objective


class RULModel:
    """XGBoost regressor for RUL estimation with Optuna hyperparameter search."""

    def __init__(self):
        self.model = None
        self.best_params = None
        self.val_loss = None

    def train(self, X, y, cfg: ModelConfig, stratify=None, n_jobs=-1, feat_cols=None, groups=None):
        """
        Find best hyperparameters via Optuna (cfg.val_split holdout), then fit
        final model on full (X, y).

        cfg:       ModelConfig driving trials, val_split, random_state, search bounds.
        stratify:  dataset_id array for joint mode so each dataset is equally
                   represented in the holdout. Ignored when groups is provided.
        n_jobs:    XGBoost CPU threads; pass 1 when running multiple models in
                   parallel to avoid core oversubscription.
        feat_cols: column names matching X columns; used to enforce a monotone
                   decreasing constraint on cycle_norm (RUL must fall as the
                   engine ages).
        groups:    unit_no array (one entry per row of X). When provided, the
                   Optuna validation split is done at the unit level via
                   GroupShuffleSplit so no unit's rows appear in both train and
                   val — preventing data leakage in hyperparameter search.
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

        if groups is not None:
            groups = np.asarray(groups)
            gss = GroupShuffleSplit(
                n_splits=1, test_size=cfg.val_split, random_state=cfg.random_state
            )
            tr_idx, val_idx = next(gss.split(X, y, groups=groups))
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            logger.info(
                "Unit-level val split: %d train rows (%d units), %d val rows (%d units)",
                len(tr_idx), len(np.unique(groups[tr_idx])),
                len(val_idx), len(np.unique(groups[val_idx])),
            )
        else:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=cfg.val_split, random_state=cfg.random_state, stratify=stratify
            )
        logger.info(
            "Optuna search: train=%d, val=%d, trials=%d, n_jobs=%d",
            len(X_tr), len(X_val), cfg.optuna_trials, n_jobs,
        )

        early_stop = cfg.early_stopping_rounds
        use_asymmetric = cfg.loss_alpha > 0

        if use_asymmetric:
            logger.info(
                "Asymmetric MAE loss: alpha=%.2f (over-prediction penalty %.2fx under-prediction)",
                cfg.loss_alpha, cfg.loss_alpha / (1.0 - cfg.loss_alpha),
            )

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
            if use_asymmetric:
                m = XGBRegressor(
                    **params,
                    objective=_asymmetric_mae_obj(cfg.loss_alpha),
                    eval_metric="rmse",
                    callbacks=[_XGBPruningCallback(trial)],
                )
            else:
                m = XGBRegressor(**params, callbacks=[_XGBPruningCallback(trial)])
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            if use_asymmetric:
                return _tilted_loss(y_val, m.predict(X_val), cfg.loss_alpha)
            return np.sqrt(mean_squared_error(y_val, m.predict(X_val)))

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=cfg.pruner_startup_trials,
            n_warmup_steps=cfg.pruner_warmup_steps,
        )
        study = optuna.create_study(direction="minimize", pruner=pruner)
        study.optimize(objective, n_trials=cfg.optuna_trials)

        self.best_params = study.best_params
        self.val_loss = study.best_value
        if use_asymmetric:
            logger.info("Best val tilted_loss(alpha=%.2f)=%.4f, params=%s",
                        cfg.loss_alpha, self.val_loss, self.best_params)
        else:
            logger.info("Best val RMSE=%.4f, params=%s", self.val_loss, self.best_params)

        final_params = {
            **self.best_params,
            "verbosity": 0,
            "random_state": cfg.random_state,
            "n_jobs": n_jobs,
        }
        if mono is not None:
            final_params["monotone_constraints"] = mono
        if use_asymmetric:
            self.model = XGBRegressor(
                **final_params,
                objective=_asymmetric_mae_obj(cfg.loss_alpha),
            )
        else:
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


