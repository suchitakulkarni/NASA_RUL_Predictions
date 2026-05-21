# Agent State — NASA RUL Predictions

## Current status

Three bugs fixed in this session; project on `develop` branch, no new commits yet.

## What works

- **New MLOps pipeline** (`train.py` → `calibrate.py` → `run_evaluation.py`): end-to-end for joint and separate models
- **Old quantile pipeline** (`main.py`): still functional, uses per-dataset `saved_models_{ds}/` directories
- `combined_pred.py`: evaluates separate models, now with optional conformal uncertainty on scatter plots
- All visualisation functions consolidated in `src/visualisation.py` per CLAUDE.md

## What was fixed this session

### 1. Unit-level train/val split in `RULModel.train()` (`src/model.py`)
- Added `groups` parameter; uses `GroupShuffleSplit` when provided so no unit's rows appear in both train and Optuna val set
- `train.py` now passes `unit_ids` from `_make_features()` to `model.train(..., groups=unit_ids)`

### 2. Uncertainty on final true-vs-predicted scatter plots (`src/visualisation.py`, `combined_pred.py`)
- `plot_pred_vs_true` moved from `src/evaluate.py` to `src/visualisation.py` (CLAUDE.md module boundary rule)
- New signature accepts optional `lower`/`upper` per dataset; plots conformal error bars when present
- `combined_pred.py` loads the joint model's `ConformalPredictor` (if found) and passes intervals through

### 3. Parallel training artifact namespacing (`train.py`, `calibrate.py`, `run_evaluation.py`, `combined_pred.py`)
- All shared artifacts now include `artifact_tag` in filename (default `{mode}_{features}`, e.g. `joint_engineered`)
  - `feature_engineer_{tag}.pkl`, `condition_normaliser_{tag}.pkl`, `{tag}_model.pkl`, `conformal_predictor_{tag}.pkl`
- `--run-tag` CLI arg on `train.py`, `calibrate.py`, `run_evaluation.py`, `combined_pred.py` for further disambiguation
- `--n-jobs` added to `train.py` CLI (default 1) to prevent XGBoost core oversubscription in parallel runs
- Fixed bug in `calibrate.py` and `run_evaluation.py` that loaded the nonexistent generic `joint_model.pkl`

### CLAUDE.md compliance fixes (`src/visualisation.py`, `src/evaluate.py`)
- All font sizes raised to ≥ 12 (labels/titles 14, ticks/legends 12)
- Hex colours replaced with named colours: blue, red, green, magenta, cyan
- Equal aspect ratio added to all scatter plots (`ax.set_aspect("equal")`)

## Next steps

- Retrain models using new namespaced artifact paths (existing `results/models/` files use old generic names)
  ```
  python train.py --mode joint --features engineered
  python calibrate.py --features engineered
  python run_evaluation.py --features engineered
  ```
- Add pytest coverage for `split_train_test` and `RULModel.train` (unit-level split correctness)
- Consider adding a scatter-style plot with conformal bands to `run_evaluation.py` in addition to trajectory plots
- `presentation.mpstyle` referenced in CLAUDE.md does not exist in repo — create or remove the reference
