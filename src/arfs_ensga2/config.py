from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    outer_splits: int = 5
    outer_repeats: int = 5
    inner_splits: int = 3
    base_seed: int = 2026
    candidate_features: int = 30
    label_column: str = "label"
    group_column: str | None = None


@dataclass(frozen=True)
class OptimizerConfig:
    population_size: int = 12
    evaluation_budget: int = 240
    crossover_probability: float = 0.8
    mutation_probability: float = 0.12
    archive_capacity: int = 50
    window_size: int = 10
    rho: float = 0.2
    std_floor: float = 1e-6
    alpha: float = 0.35
    monitor_interval: int = 5
    mi_threshold: float = 0.0
    feature_ratio_denominator: str = "candidate"
    initialization_top_k: int = 10

    def validate(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.evaluation_budget < self.population_size:
            raise ValueError("evaluation_budget must cover the initial population")
        if self.feature_ratio_denominator not in {"candidate", "original"}:
            raise ValueError("feature_ratio_denominator must be candidate or original")
        for name in ("crossover_probability", "mutation_probability", "rho", "alpha"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")


def load_config(path: str | Path) -> tuple[ExperimentConfig, OptimizerConfig]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    experiment = ExperimentConfig(**raw.get("experiment", {}))
    optimizer = OptimizerConfig(**raw.get("optimizer", {}))
    optimizer.validate()
    return experiment, optimizer


def config_dict(experiment: ExperimentConfig, optimizer: OptimizerConfig) -> dict[str, Any]:
    return {"experiment": asdict(experiment), "optimizer": asdict(optimizer)}

