from __future__ import annotations

from dataclasses import replace
from itertools import product
from pathlib import Path

import pandas as pd

from .audit import audit_results
from .config import OptimizerConfig, load_config
from .experiment import run_experiment


PARAMETER_DATASETS = ("Colon", "SRBCT")
ONE_FACTOR_VALUES: dict[str, tuple[float | int, ...]] = {
    "mutation_probability": (0.05, 0.12, 0.2),
    "window_size": (3, 5, 10),
    "rho": (0.05, 0.2, 0.5),
    "alpha": (0.0, 0.35, 0.75),
    "monitor_interval": (3, 5, 10),
    "mi_threshold": (0.0, 0.5),
    "archive_capacity": (20, 50),
}
INTERACTIONS = (
    ("window_size", "rho"),
    ("alpha", "monitor_interval"),
)
SUMMARY_METRICS = (
    "balanced_accuracy",
    "macro_f1",
    "n_features",
    "redundancy",
    "hypervolume",
    "runtime_seconds",
)


def _token(value: float | int) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def parameter_settings() -> list[tuple[str, dict[str, float | int]]]:
    settings: list[tuple[str, dict[str, float | int]]] = []
    for field, values in ONE_FACTOR_VALUES.items():
        for value in values:
            settings.append((f"ofat/{field}/{_token(value)}", {field: value}))
    for first, second in INTERACTIONS:
        for first_value, second_value in product(
            ONE_FACTOR_VALUES[first], ONE_FACTOR_VALUES[second]
        ):
            setting_id = (
                f"interaction/{first}__{second}/"
                f"{_token(first_value)}__{_token(second_value)}"
            )
            settings.append(
                (setting_id, {first: first_value, second: second_value})
            )
    return settings


def run_parameter_analysis(
    data_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
) -> pd.DataFrame:
    experiment, default_optimizer = load_config(config_path)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    summary_rows: list[dict[str, object]] = []
    for dataset in PARAMETER_DATASETS:
        for setting_id, updates in parameter_settings():
            optimizer: OptimizerConfig = replace(default_optimizer, **updates)
            optimizer.validate()
            run_dir = output_dir / dataset / setting_id
            frame = run_experiment(
                data_dir / f"{dataset}.csv.gz",
                run_dir,
                experiment,
                optimizer,
                variants=("abc",),
            )
            report = audit_results(run_dir)
            if report["status"] != "PASS":
                raise RuntimeError(f"parameter run audit failed: {run_dir}")
            row: dict[str, object] = {
                "dataset": dataset,
                "setting": setting_id,
                **updates,
            }
            for metric in SUMMARY_METRICS:
                row[f"{metric}_mean"] = float(frame[metric].mean())
                row[f"{metric}_std"] = float(frame[metric].std(ddof=1))
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "parameter_analysis_summary.csv", index=False)
    return summary
