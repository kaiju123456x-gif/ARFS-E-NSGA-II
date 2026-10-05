from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import minmax_scale


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    low, high = float(np.min(values)), float(np.max(values))
    if high <= low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def relieff_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Small-sample ReliefF implementation using one nearest hit/miss per class."""
    n, d = X.shape
    ranges = np.ptp(X, axis=0)
    ranges[ranges == 0] = 1.0
    scaled = X / ranges
    scores = np.zeros(d, dtype=float)
    classes, counts = np.unique(y, return_counts=True)
    priors = dict(zip(classes.tolist(), (counts / n).tolist()))
    for i in range(n):
        distances = np.sum(np.abs(scaled - scaled[i]), axis=1)
        distances[i] = np.inf
        same = np.flatnonzero(y == y[i])
        same = same[same != i]
        if len(same):
            hit = same[np.argmin(distances[same])]
            scores -= np.abs(scaled[i] - scaled[hit]) / n
        denominator = max(1e-12, 1.0 - priors[y[i]])
        for cls in classes:
            if cls == y[i]:
                continue
            candidates = np.flatnonzero(y == cls)
            miss = candidates[np.argmin(distances[candidates])]
            weight = priors[cls] / denominator
            scores += weight * np.abs(scaled[i] - scaled[miss]) / n
    return scores


def feature_score_matrix(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, list[str]]:
    rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    rf.fit(X, y)
    scores = [
        relieff_scores(X, y),
        mutual_info_classif(X, y, random_state=seed),
        chi2(np.clip(X, 0.0, None), y)[0],
        f_classif(X, y)[0],
        rf.feature_importances_,
    ]
    names = ["relieff", "mutual_information", "chi_square", "f_score", "random_forest"]
    return np.column_stack([_normalize(score) for score in scores]), names


def validation_weights(
    X: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    performances = np.zeros(scores.shape[1], dtype=float)
    top_k = min(max(1, top_k), X.shape[1])
    for column in range(scores.shape[1]):
        selected = np.argsort(-scores[:, column], kind="stable")[:top_k]
        fold_scores = []
        for train, valid in splits:
            model = KNeighborsClassifier(n_neighbors=min(3, len(train)), weights="distance")
            model.fit(X[train][:, selected], y[train])
            prediction = model.predict(X[valid][:, selected])
            fold_scores.append(balanced_accuracy_score(y[valid], prediction))
        performances[column] = np.mean(fold_scores)
    shifted = performances - np.max(performances)
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    return weights, performances


def guided_population(
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    population_size: int,
    top_k: int,
    rng: np.random.Generator,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    score_matrix, names = feature_score_matrix(X, y, seed)
    weights, performances = validation_weights(X, y, score_matrix, splits, top_k)
    combined = score_matrix @ weights
    centered = combined - np.median(combined)
    probabilities = expit(centered)
    population = rng.random((population_size, X.shape[1])) < probabilities
    best = int(np.argmax(combined))
    empty = np.flatnonzero(~population.any(axis=1))
    population[empty, best] = True
    diagnostics = {
        "sources": names,
        "source_weights": weights.tolist(),
        "source_validation_balanced_accuracy": performances.tolist(),
        "activation_probabilities": probabilities.tolist(),
    }
    return population.astype(bool), combined, diagnostics


def random_sparse_population(
    population_size: int, dimension: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    probability = min(0.5, max(1.0 / dimension, 0.1))
    population = rng.random((population_size, dimension)) < probability
    empty = np.flatnonzero(~population.any(axis=1))
    population[empty, rng.integers(0, dimension, size=len(empty))] = True
    weights = np.ones(dimension, dtype=float)
    return population.astype(bool), weights, {"random_activation_probability": probability}

