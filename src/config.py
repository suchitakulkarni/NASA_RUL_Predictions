# src/config.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List
import yaml


@dataclass
class DataConfig:
    dir: str = "./CMAPSSData"
    datasets: List[str] = field(default_factory=lambda: ["FD001", "FD002", "FD003", "FD004"])
    rul_cap: int = 125


@dataclass
class PathsConfig:
    models_dir: str = "results/models"


@dataclass
class FeaturesConfig:
    sensors_to_drop: List[str] = field(
        default_factory=lambda: ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
    )
    window_sizes: List[int] = field(default_factory=lambda: [5, 10, 20, 30])


@dataclass
class ConditionNormaliserConfig:
    n_clusters: int = 6
    n_init: int = 10
    random_state: int = 42


@dataclass
class SearchSpaceConfig:
    """Optuna search bounds. Each entry is [min, max]."""
    n_estimators: List[int] = field(default_factory=lambda: [50, 500])
    max_depth: List[int] = field(default_factory=lambda: [3, 10])
    learning_rate: List[float] = field(default_factory=lambda: [0.01, 0.3])
    subsample: List[float] = field(default_factory=lambda: [0.5, 1.0])
    colsample_bytree: List[float] = field(default_factory=lambda: [0.5, 1.0])
    reg_alpha: List[float] = field(default_factory=lambda: [1.0e-6, 10.0])
    reg_lambda: List[float] = field(default_factory=lambda: [1.0e-6, 10.0])


@dataclass
class ModelConfig:
    optuna_trials: int = 100
    val_split: float = 0.2
    random_state: int = 42
    early_stopping_rounds: int = 50
    pruner_startup_trials: int = 10
    pruner_warmup_steps: int = 20
    search_space: SearchSpaceConfig = field(default_factory=SearchSpaceConfig)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    condition_normaliser: ConditionNormaliserConfig = field(default_factory=ConditionNormaliserConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    @classmethod
    def from_yaml(cls, path: str) -> Config:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        model_raw = raw.get("model", {})
        ss_raw = model_raw.pop("search_space", {})

        return cls(
            data=DataConfig(**raw.get("data", {})),
            paths=PathsConfig(**raw.get("paths", {})),
            features=FeaturesConfig(**raw.get("features", {})),
            condition_normaliser=ConditionNormaliserConfig(**raw.get("condition_normaliser", {})),
            model=ModelConfig(
                **model_raw,
                search_space=SearchSpaceConfig(**ss_raw),
            ),
        )

    def to_flat_dict(self) -> dict:
        """
        Flatten to dot-separated key/value pairs for experiment tracker logging.

            cfg.to_flat_dict()
            # {"data.rul_cap": 125, "model.optuna_trials": 100, ...}

            mlflow.log_params(cfg.to_flat_dict())   # MLflow
            wandb.config.update(cfg.to_flat_dict()) # Weights and Biases
        """
        def _flatten(d: dict, prefix: str = "") -> dict:
            out = {}
            for k, v in d.items():
                key = f"{prefix}{k}"
                if isinstance(v, dict):
                    out.update(_flatten(v, prefix=f"{key}."))
                else:
                    out[key] = v
            return out
        return _flatten(asdict(self))
