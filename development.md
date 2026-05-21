# CMAPSS RUL Estimation — Development Plan

## Execution Order

The phase ordering below (EDA → ablation → feature engineering → model) is correct for the
portfolio narrative. For getting better numbers first, execute in this order instead:

1. **Phase 3** — Condition normaliser + rolling statistics + RUL cap. This is the single
   highest-leverage change. FD002/FD004 have 6 operating conditions; without cluster-wise
   normalisation the degradation signal is buried in condition variance and XGBoost cannot
   learn it. Rolling mean/std over 5–10 cycles is what actually surfaces the trend.
   RUL cap at 125 prevents the model wasting capacity on unpredictable early-life cycles.

2. **Phase 4** — Joint model trained on all four datasets with `dataset_id` as a feature.
   Run this immediately after Phase 3 so the ablation in Phase 2 has a trained normalised
   pipeline to compare against.

3. **Phase 2** — Ablation heatmaps. With the normalised pipeline in hand, the 4×4 RMSE
   comparison (raw vs normalised, separate vs joint) becomes the quantitative backbone of
   the story.

4. **Phase 1** — EDA plots. The fitted `ConditionNormaliser` artefacts from Phase 3 can be
   reused here (clustering scatter, before/after sensor trajectories) so no recomputation
   is needed.

5. **Phase 5** — Conformal calibration. Run on a held-out split of the training data after
   the joint model is trained.

6. **Phase 6** — Metrics table, final prediction plots, feature importance.

7. **Phase 7** — Streamlit app refactor.

## Uncertainty Quantification Strategy

Keep the existing quantile regression infrastructure (0.05 / 0.5 / 0.95) for point
estimates and initial intervals. Add conformal calibration on top in Phase 5: the
nonconformity scores are computed on a calibration split using the quantile model's median
predictions, and the resulting coverage-guaranteed intervals replace the raw quantile
outputs for the final deliverable. This means:

- The 0.5 quantile model is the primary predictor (point estimate and business case input).
- The 0.05 / 0.95 quantile models serve as warm-start bounds for the conformal calibration.
- Coverage is formally validated (target 90%) rather than assumed.
- The story is: "we use quantile XGBoost for efficiency and conformal calibration for
  statistical rigour — no retraining required for the UQ step."

---

## Objective

Refactor the existing XGBoost + Optuna RUL pipeline into a generalisable,
justified, and deployment-ready system across all four CMAPSS subsets
(FD001–FD004). The final deliverable is a single deployable model with
proper uncertainty quantification and a self-contained justification
narrative backed by plots and numbers. Keep the framework FastAPI compliant.

---

## Dataset Reference

| Subset | Operating Conditions | Fault Modes | Train Units | Test Units |
|--------|---------------------|-------------|-------------|------------|
| FD001  | 1                   | 1           | 100         | 100        |
| FD002  | 6                   | 1           | 260         | 259        |
| FD003  | 1                   | 2           | 100         | 100        |
| FD004  | 6                   | 2           | 248         | 248        |

Columns: unit_id, cycle, op_setting_1, op_setting_2, op_setting_3,
sensors 1-21.

---

## Directory Structure

```
cmapss/
    data/
        raw/               # original .txt files
        processed/         # normalised, feature-engineered parquet files
    results/
        eda/               # EDA plots
        ablation/          # cross-dataset heatmaps
        models/            # saved XGBoost models and conformal calibration
        evaluation/        # final metrics and UQ plots
    src/
        data_loader.py
        feature_engineering.py
        condition_normaliser.py
        rul_target.py
        model.py
        conformal.py
        evaluation.py
        visualisation.py
    notebooks/             # scratch only, no production code here
    streamlit_app.py
    development.md
    requirements.txt
```

---

## Phase 1: EDA and Feature Engineering Justification

**Goal:** Motivate every engineering decision visually before touching
the model.

### 1.1 Raw Sensor Inspection

- Load all four subsets, concatenate with a `dataset_id` column.
- Plot raw sensor trajectories (sensors 2, 3, 4, 7, 8, 9, 11, 12, 13,
  14, 15, 17, 20, 21) per unit, coloured by operating condition cluster
  (use op_setting_1, op_setting_2, op_setting_3).
- Expected observation: multi-condition variance in FD002/FD004 masks
  degradation signal entirely on raw sensors.
- Save: `results/eda/raw_sensor_trajectories_fdXXX.png`

### 1.2 Near-Constant Sensor Identification

- Compute per-sensor standard deviation across all units and all
  datasets.
- Bar plot of sensor standard deviations.
- Sensors with std below threshold (empirically ~0.1 after normalisation)
  are candidates for dropping: typically sensors 1, 5, 6, 10, 16, 18, 19.
- Save: `results/eda/sensor_variance.png`
- Output: confirmed list of sensors to drop, logged.

### 1.3 Operating Condition Clustering

- Run k-means (k=6) on (op_setting_1, op_setting_2, op_setting_3)
  pooled across all four datasets.
- 3D scatter plot of operating conditions coloured by cluster.
- Verify FD001/FD003 collapse to a single cluster (they should).
- Save: `results/eda/operating_condition_clusters.png`
- Save cluster centroids for use in inference normalisation.

### 1.4 Before vs After Condition Normalisation

- For a representative sensor (sensor 11 or 12) on FD002:
  - Plot raw trajectories of 10 randomly sampled units.
  - Plot same trajectories after subtracting cluster-wise mean and
    dividing by cluster-wise std.
- Expected observation: degradation trend emerges clearly post-
  normalisation.
- Save: `results/eda/normalisation_effect_sensor_11.png`

### 1.5 Rolling Statistics Justification

- For a representative unit in FD004, plot:
  - Raw normalised sensor signal.
  - Rolling mean (window=5, 10, 20 cycles).
  - Rolling std (same windows).
- Show that rolling mean removes cycle-to-cycle noise while preserving
  the degradation trend.
- Save: `results/eda/rolling_stats_justification.png`

### 1.6 RUL Target Design

- Plot raw RUL (linear decay) vs piecewise linear RUL (capped at 125
  cycles) for a representative unit.
- Justification: XGBoost cannot extrapolate; capping prevents the model
  learning spurious early-life behaviour.
- Save: `results/eda/rul_target_comparison.png`

---

## Phase 2: Ablation Study — Cross-Dataset Generalisation

**Goal:** Produce the 4x4 RMSE heatmap that justifies Strategy B.

### 2.1 Baseline Cross-Dataset Heatmap (Raw Features)

- For each pair (train dataset i, test dataset j):
  - Train XGBoost with default hyperparameters on dataset i.
  - Evaluate RMSE on test split of dataset j.
- Plot 4x4 heatmap with RMSE values annotated.
- Expected: strong diagonal, poor off-diagonal — especially FD001
  trained, FD004 tested.
- Save: `results/ablation/cross_dataset_rmse_raw.png`

### 2.2 Post-Normalisation Cross-Dataset Heatmap

- Repeat 2.1 with condition-normalised features.
- Overlay or side-by-side comparison with 2.1.
- Expected: off-diagonal RMSE drops significantly, justifying
  normalisation as the generalisation mechanism.
- Save: `results/ablation/cross_dataset_rmse_normalised.png`

### 2.3 Per-Dataset vs Joint Model Comparison

- Train four separate XGBoost models (one per dataset) with Optuna.
- Train one joint XGBoost model on all four datasets combined, with
  `dataset_id` as a categorical feature, with Optuna.
- Bar chart: RMSE per dataset for separate models vs joint model.
- Expected: joint model is competitive with separate models on all four
  subsets after normalisation.
- Save: `results/ablation/separate_vs_joint_rmse.png`

---

## Phase 3: Feature Engineering Pipeline

**Goal:** Clean, modular, reusable feature engineering that slots into
the existing Optuna machinery.

### 3.1 Condition Normaliser (`src/condition_normaliser.py`)

- Class `ConditionNormaliser`
  - `fit(X)`: runs k-means on operational settings, stores centroids
    and per-cluster mean/std for each sensor.
  - `transform(X)`: assigns each cycle to nearest cluster, subtracts
    cluster mean, divides by cluster std.
  - `fit_transform(X)`: convenience method.
  - `save(path)` / `load(path)`: persist centroids for inference.
- Logging at fit and transform steps.

### 3.2 Feature Engineer (`src/feature_engineering.py`)

- Class `FeatureEngineer`
  - Drops near-constant sensors (configurable list).
  - Computes rolling mean and rolling std per unit per sensor
    (configurable window sizes, default [5, 10]).
  - Computes cycle-normalised time-to-now (cycles elapsed / max cycles
    in training — proxy for age).
  - Adds `dataset_id` as integer categorical.
  - Returns a flat feature matrix ready for XGBoost.
- All steps logged with feature counts before and after.

### 3.3 RUL Target (`src/rul_target.py`)

- Function `compute_rul(df, cap=125)`: computes piecewise linear RUL
  per unit.
- Function `compute_lead_time(rul_predictions, threshold)`: converts
  RUL predictions to lead time for maintenance scheduling — this is
  the business case number.

---

## Phase 4: Model Training with Optuna

**Goal:** Keep existing Optuna machinery, extend to joint training.

### 4.1 Model (`src/model.py`)

- Class `RULModel`
  - Wraps XGBoost regressor.
  - `train(X, y, optuna_trials=100)`: runs Optuna study, stores best
    params and fitted model.
  - `predict(X)`: returns point predictions.
  - `save(path)` / `load(path)`: persist model.
- Optuna search space: n_estimators, max_depth, learning_rate,
  subsample, colsample_bytree, reg_alpha, reg_lambda.
- Objective: RMSE on 20% holdout, stratified by dataset_id.

### 4.2 Training Script

- `train.py` with CLI arguments:
  - `--mode`: `separate` (four models) or `joint` (one model).
  - `--trials`: number of Optuna trials (default 100).
  - `--cap`: RUL cap value (default 125).
  - `--window`: rolling window sizes (default 5 10).
- Logs best params, RMSE per dataset, saves models to
  `results/models/`.

---

## Phase 5: Uncertainty Quantification via Conformal Prediction

**Goal:** Replace median estimates with statistically principled
prediction intervals.

### 5.1 Conformal Predictor (`src/conformal.py`)

- Class `ConformalPredictor`
  - `calibrate(model, X_cal, y_cal, alpha=0.1)`: computes nonconformity
    scores (absolute residuals) on calibration set, stores the
    (1-alpha) quantile per operating condition cluster.
  - `predict_interval(model, X)`: returns (lower, upper) bounds per
    prediction.
  - Stratified by operating condition cluster — intervals are tighter
    for well-represented conditions.
- Logging of coverage achieved on calibration set (should be ~90% for
  alpha=0.1).

### 5.2 Coverage Validation Plot

- Plot empirical coverage per dataset and per operating condition
  cluster vs nominal coverage (90%).
- Save: `results/evaluation/conformal_coverage.png`

### 5.3 Business Case: Cost Framing

- Function `cost_savings(rul_lower, rul_upper, cost_unplanned,
  cost_planned)`: computes expected savings from scheduling maintenance
  within the prediction interval vs reactive maintenance.
- Output as a summary table logged and saved to
  `results/evaluation/cost_summary.csv`.

---

## Phase 6: Evaluation and Final Plots

### 6.1 Metrics (`src/evaluation.py`)

- RMSE, MAE, and score function (standard CMAPSS asymmetric score
  penalising late predictions more heavily) per dataset.
- Summary table printed and saved to
  `results/evaluation/metrics_summary.csv`.

### 6.2 RUL Prediction Plots with Intervals

- For 5 randomly sampled test units per dataset:
  - Plot true RUL vs predicted RUL with conformal interval shaded.
  - Annotate lead time threshold.
- Save: `results/evaluation/rul_predictions_fdXXX.png`

### 6.3 Feature Importance

- XGBoost feature importance (gain) bar chart for joint model.
- Group by feature type: raw sensor, rolling mean, rolling std,
  dataset_id.
- Save: `results/evaluation/feature_importance.png`

---

## Phase 7: Streamlit App Refactor

**Goal:** Replace static display with interactive, business-case-framed
app.

### Tabs

1. **Dataset Explorer**
   - Select dataset (FD001–FD004).
   - Select unit ID from dropdown.
   - Show raw vs normalised sensor trajectories side by side.

2. **RUL Prediction**
   - Upload a new engine cycle history CSV or select a test unit.
   - Show predicted RUL with conformal interval.
   - Show lead time to recommended maintenance.

3. **Cost Calculator**
   - Sliders for cost of unplanned downtime and cost of planned
     maintenance.
   - Output expected savings from using the prediction interval.

4. **Model Justification**
   - Static display of the ablation heatmaps and feature importance
     plots generated during training.
   - Narrative text explaining each plot.

---

## Deliverables Checklist

- [ ] Phase 1: EDA plots (6 plots) saved to `results/eda/`
- [ ] Phase 2: Ablation heatmaps and bar chart (3 plots) saved to
      `results/ablation/`
- [ ] Phase 3: Feature engineering classes with logging
- [ ] Phase 4: Joint model trained and saved, separate models as
      ablation
- [ ] Phase 5: Conformal predictor calibrated, coverage validated
- [ ] Phase 6: Metrics table, prediction plots, feature importance
- [ ] Phase 7: Streamlit app with four tabs

---

## Development Order

Day 1: Phases 1 and 2 — get the justification narrative locked in
before writing any model code. The plots drive the story.

Day 2: Phases 3, 4, and 5 — feature engineering, joint model training,
conformal calibration.

Day 3: Phases 6 and 7 — evaluation, final plots, Streamlit refactor.

---

## Notes and Decisions

- XGBoost and Optuna machinery kept intact from existing codebase.
- Strategy B (joint model) is the primary result. Strategy A (separate
  models) is the ablation baseline only.
- Conformal prediction replaces median UQ. No retraining required —
  calibration runs on a held-out split of the training data.
- k=6 for operating condition clustering matches the number of
  conditions in FD002/FD004. FD001/FD003 units will collapse to one
  cluster naturally.
- dataset_id is passed as an integer feature, not one-hot, since
  XGBoost handles ordinal categoricals natively with
  `enable_categorical=True`.
- All plots saved to `results/` subdirectories. No plots displayed
  inline in scripts.
- Greek letters are not used in code, comments, or variable names.
- Ask for `setup_logging` function before writing any module.
