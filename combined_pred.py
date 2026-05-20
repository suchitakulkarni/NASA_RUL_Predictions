import os
import joblib
import json
from datetime import datetime
from sklearn.ensemble import VotingRegressor

# --- Config ---
dataset_ids = ["FD001", "FD002", "FD003", "FD004"]
quantiles = [0.05, 0.50, 0.95]
output_dir = "ensemble_models"

# --- Step 1: Load models for a single quantile ---
def load_models_for_quantile(quantile, dataset_ids):
    models = {}
    for ds_id in dataset_ids:
        model_dir = f"saved_models_{ds_id}"
        model_path = os.path.join(model_dir, f"quantile_{quantile:.2f}_model.pkl")
        if os.path.exists(model_path):
            models[ds_id] = joblib.load(model_path)
        else:
            print(f"Warning: Model not found at {model_path}")
    return models

# --- Step 2: Create ensemble for a single quantile ---
def create_ensemble_for_quantile(quantile, dataset_ids):
    models = load_models_for_quantile(quantile, dataset_ids)
    if not models:
        raise ValueError(f"No models found for quantile {quantile}")

    estimators = [(f"{ds_id}_q{quantile}", model) for ds_id, model in models.items()]
    ensemble = VotingRegressor(estimators=estimators)
    # Sub-estimators are already fitted; set estimators_ so sklearn treats this as fitted
    ensemble.estimators_ = [model for _, model in estimators]
    return ensemble

# --- Step 3: Load and ensemble all quantiles ---
def load_and_ensemble_all_quantiles(dataset_ids, quantiles):
    ensemble_models = {}
    for q in quantiles:
        ensemble = create_ensemble_for_quantile(q, dataset_ids)
        ensemble_models[q] = ensemble
        print(f"Created ensemble for quantile {q:.2f}")
    return ensemble_models

# --- Step 4: Save ensemble models ---
def save_ensemble_models(ensemble_models, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for q, ensemble in ensemble_models.items():
        model_path = os.path.join(output_dir, f"ensemble_quantile_{q:.2f}_model.pkl")
        joblib.dump(ensemble, model_path)
        print(f"Saved ensemble model for quantile {q:.2f} to {model_path}")

    # Save metadata
    metadata = {
        "quantiles": quantiles,
        "dataset_ids": dataset_ids,
        "training_date": datetime.now().isoformat(),
        "notes": "Ensemble of XGBoost models trained on NASA FD001-FD004 datasets"
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

# --- Run the workflow ---
if __name__ == "__main__":
    # Load and ensemble all models
    ensemble_models = load_and_ensemble_all_quantiles(dataset_ids, quantiles)

    # Save ensemble models
    save_ensemble_models(ensemble_models, output_dir)

    # Test the ensembles (optional)
    '''X_test = ...  # Your test data (14 features)
    for q, ensemble in ensemble_models.items():
        y_pred = ensemble.predict(X_test)
        print(f"Quantile {q:.2f} predictions: {y_pred[:5]}")  # Print first 5 predictions'''
