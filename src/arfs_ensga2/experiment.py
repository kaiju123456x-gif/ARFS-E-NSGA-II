from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from .config import ExperimentConfig, OptimizerConfig, config_dict
from .objectives import make_inner_splits
from .optimizer import ARFSENSGA2
from .pareto import hypervolume_3d


ABLATIONS: dict[str, tuple[bool, bool, bool]] = {
    "none": (False, False, False),
    "a": (True, False, False),
    "b": (False, True, False),
    "c": (False, False, True),
    "ab": (True, True, False),
    "ac": (True, False, True),
    "bc": (False, True, True),
    "abc": (True, True, True),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csv_dataset(
    path: str | Path, config: ExperimentConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[str]]:
    frame = pd.read_csv(path)
    if config.label_column not in frame:
        raise ValueError(f"missing label column: {config.label_column}")
    groups = None
    excluded = [config.label_column]
    if config.group_column:
        if config.group_column not in frame:
            raise ValueError(f"missing group column: {config.group_column}")
        groups = frame[config.group_column].astype(str).to_numpy()
        excluded.append(config.group_column)
    features = frame.drop(columns=excluded)
    non_numeric = features.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"feature columns must be numeric: {non_numeric[:5]}")
    y = LabelEncoder().fit_transform(frame[config.label_column].astype(str))
    return features.to_numpy(dtype=float), y, groups, features.columns.astype(str).tolist()


def outer_splits(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray | None, config: ExperimentConfig
) -> Iterable[tuple[int, int, np.ndarray, np.ndarray, int]]:
    for repeat in range(config.outer_repeats):
        seed = config.base_seed + repeat
        if groups is None:
            splitter = StratifiedKFold(
                n_splits=config.outer_splits, shuffle=True, random_state=seed
            )
            iterator = splitter.split(X, y)
        else:
            splitter = StratifiedGroupKFold(
                n_splits=config.outer_splits, shuffle=True, random_state=seed
            )
            iterator = splitter.split(X, y, groups=groups)
        for fold, (train, test) in enumerate(iterator):
            yield repeat, fold, train, test, seed


def preprocess_fold(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    candidate_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SimpleImputer, MinMaxScaler]:
    imputer = SimpleImputer(strategy="median")
    train_imputed = imputer.fit_transform(X_train)
    test_imputed = imputer.transform(X_test)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_imputed)
    test_scaled = scaler.transform(test_imputed)
    k = min(candidate_features, train_scaled.shape[1])
    selector = SelectKBest(f_classif, k=k).fit(train_scaled, y_train)
    candidate_indices = selector.get_support(indices=True)
    return (
        train_scaled[:, candidate_indices],
        test_scaled[:, candidate_indices],
        candidate_indices,
        imputer,
        scaler,
    )


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(type(value).__name__)


def run_experiment(
    data_path: str | Path,
    output_dir: str | Path,
    experiment_config: ExperimentConfig,
    optimizer_config: OptimizerConfig,
    *,
    variants: Iterable[str] = ("abc",),
) -> pd.DataFrame:
    data_path = Path(data_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y, groups, feature_names = load_csv_dataset(data_path, experiment_config)
    variants = [variant.lower() for variant in variants]
    invalid = sorted(set(variants) - set(ABLATIONS))
    if invalid:
        raise ValueError(f"unknown variants: {invalid}")
    rows: list[dict[str, object]] = []
    split_records: list[dict[str, object]] = []
    archive_records: list[dict[str, object]] = []
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(exist_ok=True)
    for repeat, fold, train, test, split_seed in outer_splits(
        X, y, groups, experiment_config
    ):
        if groups is not None and set(groups[train]) & set(groups[test]):
            raise RuntimeError("group leakage detected in outer split")
        X_train, X_test, candidate_indices, _, _ = preprocess_fold(
            X[train], X[test], y[train], experiment_config.candidate_features
        )
        fold_groups = groups[train] if groups is not None else None
        inner_seed = experiment_config.base_seed + repeat * 100 + fold
        inner = make_inner_splits(
            X_train, y[train], experiment_config.inner_splits, inner_seed, fold_groups
        )
        if fold_groups is not None:
            for inner_train, inner_valid in inner:
                if set(fold_groups[inner_train]) & set(fold_groups[inner_valid]):
                    raise RuntimeError("group leakage detected in inner split")
        split_records.append(
            {
                "repeat": repeat,
                "fold": fold,
                "split_seed": split_seed,
                "inner_seed": inner_seed,
                "train_indices": train.tolist(),
                "test_indices": test.tolist(),
                "inner_splits_relative_to_outer_train": [
                    {"train": a.tolist(), "validation": b.tolist()} for a, b in inner
                ],
                "candidate_indices": candidate_indices.tolist(),
                "candidate_names": [feature_names[index] for index in candidate_indices],
            }
        )
        for variant in variants:
            guided, dpim, drfm = ABLATIONS[variant]
            # Paired comparisons use exactly the same stochastic seed for every variant.
            run_seed = experiment_config.base_seed + repeat * 10_000 + fold
            optimizer = ARFSENSGA2(
                optimizer_config,
                guided_initialization=guided,
                dpim=dpim,
                drfm=drfm,
                random_state=run_seed,
            )
            started = time.perf_counter()
            result = optimizer.fit(
                X_train,
                y[train],
                inner,
                original_dimension=X.shape[1],
            )
            selected_local = np.flatnonzero(result.endpoint_mask)
            model = KNeighborsClassifier(n_neighbors=min(3, len(train)), weights="distance")
            model.fit(X_train[:, selected_local], y[train])
            prediction = model.predict(X_test[:, selected_local])
            transformed_archive = result.archive_objectives.copy()
            transformed_archive[:, 2] = transformed_archive[:, 2] / (
                1.0 + transformed_archive[:, 2]
            )
            hv = hypervolume_3d(transformed_archive, np.array([1.1, 1.1, 1.1]))
            runtime = time.perf_counter() - started
            selected_original = candidate_indices[selected_local]
            row = {
                "variant": variant,
                "repeat": repeat,
                "fold": fold,
                "seed": run_seed,
                "accuracy": accuracy_score(y[test], prediction),
                "balanced_accuracy": balanced_accuracy_score(y[test], prediction),
                "macro_f1": f1_score(y[test], prediction, average="macro", zero_division=0),
                "n_features": len(selected_local),
                "redundancy": result.endpoint_objectives[2],
                "hypervolume": hv,
                "runtime_seconds": runtime,
                "evaluations": result.evaluations,
                "selected_candidate_indices": json.dumps(selected_local.tolist()),
                "selected_original_indices": json.dumps(selected_original.tolist()),
                "selected_feature_names": json.dumps(
                    [feature_names[index] for index in selected_original]
                ),
            }
            rows.append(row)
            for archive_index, (mask, objective) in enumerate(
                zip(result.archive_masks, result.archive_objectives)
            ):
                archive_records.append(
                    {
                        "variant": variant,
                        "repeat": repeat,
                        "fold": fold,
                        "archive_index": archive_index,
                        "classification_error": objective[0],
                        "feature_ratio": objective[1],
                        "redundancy": objective[2],
                        "selected_candidate_indices": json.dumps(
                            np.flatnonzero(mask).tolist()
                        ),
                    }
                )
            diagnostic_path = diagnostics_dir / f"{variant}_r{repeat}_f{fold}.json"
            diagnostic_path.write_text(
                json.dumps(result.diagnostics, indent=2, default=_json_default),
                encoding="utf-8",
            )
    result_frame = pd.DataFrame(rows)
    result_frame.to_csv(output_dir / "fold_results.csv", index=False)
    pd.DataFrame(archive_records).to_csv(output_dir / "archives.csv", index=False)
    (output_dir / "splits.json").write_text(
        json.dumps(split_records, indent=2), encoding="utf-8"
    )
    summary = (
        result_frame.groupby("variant")[
            [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "n_features",
                "redundancy",
                "hypervolume",
                "runtime_seconds",
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in summary.columns
    ]
    summary.to_csv(output_dir / "summary.csv", index=False)
    manifest = {
        "data_path": str(data_path),
        "data_sha256": file_sha256(data_path),
        "shape": {"samples": X.shape[0], "features": X.shape[1], "classes": int(len(np.unique(y)))},
        "class_counts": {str(label): int(np.sum(y == label)) for label in np.unique(y)},
        "missing_values": int(np.isnan(X).sum()),
        "grouped": groups is not None,
        "variants": variants,
        "config": config_dict(experiment_config, optimizer_config),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return result_frame


def create_demo_dataset(path: str | Path, seed: int = 2026) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    X, y = make_classification(
        n_samples=80,
        n_features=24,
        n_informative=8,
        n_redundant=6,
        n_classes=2,
        random_state=seed,
        shuffle=True,
    )
    frame = pd.DataFrame(X, columns=[f"feature_{index:03d}" for index in range(X.shape[1])])
    frame["label"] = y
    frame.to_csv(path, index=False)
    return path

