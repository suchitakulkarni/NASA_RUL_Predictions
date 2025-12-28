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
from src.model import *
from src.data import *
from src.evaluate import *
from src.utils import *

def run_pipeline():

    data_train, data_test, feature_cols = load_datasets()
    normalised_data_train_df, normalised_data_test_df = normalize_data(data_train, data_test, feature_cols)

    split_data_train_df, split_data_valid_df = split_train_test(normalised_data_train_df)

    x_train, y_train, uid_train = create_sliding_window(split_data_train_df, feature_cols)
    x_valid, y_valid, uid_valid = create_sliding_window(split_data_valid_df, feature_cols)
    x_test, y_test, uid_test = create_sliding_window(normalised_data_test_df, feature_cols)

    # Flatten
    X_train = x_train.reshape(x_train.shape[0], -1)
    X_valid = x_valid.reshape(x_valid.shape[0], -1)
    X_test = x_test.reshape(x_test.shape[0], -1)

    models = run_all_quantile_models(X_train, y_train, X_valid, y_valid, X_test, quantiles=None)

    predictions = predict(X_train, y_train, X_test, models, quantiles=None)
    df_preds = write_results_to_csv(uid_test, predictions)

    fig = plot_results(df_preds)
    fig.savefig('final_predictions.png')

if __name__ == "__main__":
    run_pipeline()