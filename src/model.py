import pandas as pd
import os
import numpy as np
import optuna
from sklearn.preprocessing import StandardScaler
# We are going to compare two different regression models here
from xgboost import XGBRegressor
from optuna.integration import XGBoostPruningCallback
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import optuna
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def split_train_test(data_train):
    # --------------------
    # Proper Train/Validation Split by unit_no
    # --------------------
    train_ids, valid_ids = train_test_split(data_train['unit_no'].unique(), test_size=0.2, random_state=42)
    train_df = data_train[data_train['unit_no'].isin(train_ids)]
    valid_df = data_train[data_train['unit_no'].isin(valid_ids)]
    return train_df, valid_df

# --------------------
# Sliding Window
# --------------------
def create_sliding_window(df, feature_cols, window_length=30):
    x, y, unit_id = [], [], []
    for unit in df['unit_no'].unique():
        unit_df = df[df['unit_no'] == unit]
        for i in range(len(unit_df) - window_length):
            window = unit_df.iloc[i:i + window_length][feature_cols].values
            label = unit_df.iloc[i + window_length]['RUL_calc']
            uid = unit_df.iloc[i + window_length]['unit_no']
            unit_id.append(uid)
            x.append(window)
            y.append(label)
    return np.array(x), np.array(y), np.asarray(unit_id)

# --------------------
# Optimization and Training
# --------------------
def optimize_quantile_model(X_train, y_train, X_valid, y_valid, trial, q):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 40, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'eta': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'verbosity': 0
    }

    model = XGBRegressor(objective="reg:quantileerror", quantile_alpha=q, **params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )

    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

def run_all_quantile_models(X_train, y_train, X_valid, y_valid, X_test, quantiles = None):
    if quantiles == None: quantiles = [0.05, 0.5, 0.95]

    # Run for all quantiles
    models = {}

    for q in quantiles:
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: optimize_quantile_model(X_train, y_train, X_valid, y_valid, trial, q), n_trials=50)

        best_params = study.best_params

        xgb_regressor = XGBRegressor(**best_params, objective="reg:quantileerror", quantile_alpha=q)
        xgb_regressor.fit(X_train, y_train)
        #y_pred = xgb_regressor.predict(X_test)
        models[q] = best_params
    return models

def predict(X_train, y_train, X_test, models, quantiles = None):
    if quantiles == None: quantiles = [0.05, 0.5, 0.95]
    ypreds = {}
    for q in quantiles:
        xgb_regressor = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=q,
            **models[q]
        )
        xgb_regressor.fit(X_train, y_train)
        y_pred = xgb_regressor.predict(X_test)
        ypreds[q] = y_pred
    return ypreds[q]
