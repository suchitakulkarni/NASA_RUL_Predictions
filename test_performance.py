import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import VotingRegressor
from src.data import load_datasets, normalize_data
from src.model import create_sliding_window
#from combined_pred import load_and_ensemble_all_quantiles

# --- Config ---
dataset_ids = ["FD001", "FD002", "FD003", "FD004"]
quantiles = [0.05, 0.50, 0.95]

# --- Load individual models for a quantile (e.g., q=0.50) ---
def load_individual_models_for_quantile(quantile, dataset_ids):
    models = {}
    for ds_id in dataset_ids:
        model_path = f"results/saved_models_{ds_id}/quantile_{quantile:.2f}_model.pkl"
        model = joblib.load(model_path)
        models[ds_id] = model
    return models

# --- Create ensemble from loaded models ---
def create_ensemble_from_models(models):
    estimators = [(f"{ds_id}", model) for ds_id, model in models.items()]
    ensemble = VotingRegressor(estimators=estimators)
    ensemble.estimators_ = [model for _, model in estimators]
    return ensemble

# --- Load test data for a dataset ---
def load_test_data(dataset_id):
    test_path = f"CMAPSSData/{dataset_id}/test_{dataset_id}.txt"

    # --- Data loading and preprocessing ---
    data_train, data_valid, RUL_labels, feature_cols = load_datasets(dataset_id=dataset_id)
    data_train, data_valid, scaler = normalize_data(data_train, data_valid, feature_cols)

    # --- Sliding windows ---
    x_valid,  y_valid,  uid_valid  = create_sliding_window(data_valid, feature_cols)

    # --- Flatten for XGBoost ---
    X_valid  = x_valid.reshape(x_valid.shape[0], -1)

    #test_df = pd.read_csv(test_path, sep="\s+", header=None)
    #X_test = test_df.iloc[:, :-1].values  # All columns except last (RUL)
    #y_test = test_df.iloc[:, -1].values   # Last column is RUL
    return X_valid, y_valid

# --- Load an individual model ---
def load_individual_model(ds_id, quantile):
    model_dir = f"results/saved_models_{ds_id}"
    model_path = os.path.join(model_dir, f"quantile_{quantile:.2f}_model.pkl")
    return joblib.load(model_path)

# --- Load an ensemble model ---
def load_ensemble_model(quantile):
    model_path = f"ensemble_models/ensemble_quantile_{quantile:.2f}_model.pkl"
    ensemble = joblib.load(model_path)
    # Patch pkl files saved before estimators_ was set
    if not hasattr(ensemble, 'estimators_'):
        ensemble.estimators_ = [est for _, est in ensemble.estimators]
    return ensemble

def evaluate_model(model, X_test, y_test, quantile=None):
    """
    Evaluate a model on a test set.
    If the model is a quantile model (e.g., XGBoost with quantile_alpha),
    set the quantile before prediction.
    """
    #if quantile is not None and hasattr(model, 'set_params'):
    #    model.set_params(quantile_alpha=quantile)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return y_pred, mae

# --- Initialize results storage ---
results = {
    "individual": {q: [] for q in quantiles},  # {quantile: [mae1, mae2, ...]}
    "ensemble": {q: [] for q in quantiles}     # {quantile: [mae1, mae2, ...]}
}

# --- Loop over all datasets (FD001-FD004) ---
for test_ds in dataset_ids:
    X_test, y_test = load_test_data(test_ds)
    print(f"\nEvaluating on {test_ds} test set...")

    # --- Evaluate individual models ---
    for train_ds in dataset_ids:
        for q in quantiles:
            model = load_individual_model(train_ds, q)
            y_pred, mae = evaluate_model(model, X_test, y_test, q)
            results["individual"][q].append(mae)
            print(f"  Individual model (train={train_ds}, q={q}): MAE = {mae:.4f}")
            models = load_individual_models_for_quantile(q, dataset_ids)
            ensemble = create_ensemble_from_models(models)
    # --- Evaluate ensemble models ---
    #models = load_individual_models_for_quantile(quantile, dataset_ids)
    #ensemble = create_ensemble_from_models(models)
    for q in quantiles:
        ensemble = load_ensemble_model(q)
        y_pred, mae = evaluate_model(ensemble, X_test, y_test, q)
        results["ensemble"][q].append(mae)
        print(f"  Ensemble model (q={q}): MAE = {mae:.4f}")

# --- Calculate average MAE for individual and ensemble models ---
comparison = {}
for q in quantiles:
    avg_individual_mae = np.mean(results["individual"][q])
    avg_ensemble_mae = np.mean(results["ensemble"][q])
    comparison[q] = {
        "individual_avg_mae": avg_individual_mae,
        "ensemble_avg_mae": avg_ensemble_mae,
        "improvement": avg_individual_mae - avg_ensemble_mae
    }

# --- Print comparison table ---
print("\n=== Performance Comparison ===")
print(f"{'Quantile':<10} | {'Individual Avg MAE':<20} | {'Ensemble Avg MAE':<20} | {'Improvement':<15}")
print("-" * 70)
for q, metrics in comparison.items():
    print(f"{q:<10.2f} | {metrics['individual_avg_mae']:<20.4f} | {metrics['ensemble_avg_mae']:<20.4f} | {metrics['improvement']:<15.4f}")


#ensemble_models = load_and_ensemble_all_quantiles(dataset_ids, quantiles)

# --- Evaluate prediction intervals for ensemble ---
for test_ds in dataset_ids:
    X_test, y_test = load_test_data(test_ds)
    print(f"\nEvaluating prediction intervals on {test_ds}...")

    # Load ensemble models for 0.05, 0.50, 0.95
    lower_model = load_ensemble_model(0.05)
    median_model = load_ensemble_model(0.50)
    upper_model = load_ensemble_model(0.95)

    # Predict
    y_pred_lower = lower_model.predict(X_test)
    y_pred_median = median_model.predict(X_test)
    y_pred_upper = upper_model.predict(X_test)

    # Calculate metrics
    pi_width = np.mean(y_pred_upper - y_pred_lower)
    pi_coverage = np.mean((y_test >= y_pred_lower) & (y_test <= y_pred_upper))

    print(f"  90% PI Width: {pi_width:.4f}")
    print(f"  90% PI Coverage: {pi_coverage:.4f}")


