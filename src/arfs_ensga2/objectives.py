from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier


def make_inner_splits(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    seed: int,
    groups: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(X, y))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(X, y, groups=groups))


@dataclass
class ObjectiveEvaluator:
    X: np.ndarray
    y: np.ndarray
    splits: list[tuple[np.ndarray, np.ndarray]]
    original_dimension: int
    ratio_denominator: str = "candidate"
    seed: int = 0

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        self.y = np.asarray(self.y)
        self.calls = 0
        self._pair_mi: dict[tuple[int, int], float] = {}
        self.class_mi = mutual_info_classif(self.X, self.y, random_state=self.seed)

    def _pairwise_mi(self, i: int, j: int) -> float:
        key = (min(i, j), max(i, j))
        if key not in self._pair_mi:
            value = mutual_info_regression(
                self.X[:, [key[0]]], self.X[:, key[1]], random_state=self.seed
            )[0]
            self._pair_mi[key] = float(max(0.0, value))
        return self._pair_mi[key]

    def redundancy(self, selected: np.ndarray) -> float:
        if len(selected) <= 1:
            return 0.0
        values = [
            self._pairwise_mi(int(selected[i]), int(selected[j]))
            for i in range(len(selected))
            for j in range(i + 1, len(selected))
        ]
        return float(np.mean(values))

    def evaluate(self, mask: np.ndarray) -> np.ndarray:
        self.calls += 1
        selected = np.flatnonzero(mask)
        if not len(selected):
            raise ValueError("empty feature masks must be repaired before evaluation")
        scores: list[float] = []
        for train, valid in self.splits:
            neighbors = min(3, len(train))
            model = KNeighborsClassifier(n_neighbors=neighbors, weights="distance")
            model.fit(self.X[train][:, selected], self.y[train])
            prediction = model.predict(self.X[valid][:, selected])
            scores.append(balanced_accuracy_score(self.y[valid], prediction))
        denominator = (
            self.X.shape[1]
            if self.ratio_denominator == "candidate"
            else self.original_dimension
        )
        return np.array(
            [1.0 - float(np.mean(scores)), len(selected) / denominator, self.redundancy(selected)],
            dtype=float,
        )

    def evaluate_population(self, masks: np.ndarray) -> np.ndarray:
        return np.vstack([self.evaluate(mask) for mask in masks])

