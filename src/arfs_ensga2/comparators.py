"""Literature comparator cores adapted to the repository's shared evaluator.

The implementations in this module deliberately accept an already preprocessed
outer-training matrix and already constructed inner folds.  They therefore
cannot create method-specific data splits or inspect an outer test fold.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import Lasso
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from .config import OptimizerConfig
from .initialization import relieff_scores
from .objectives import ObjectiveEvaluator
from .optimizer import OptimizationResult
from .pareto import (
    crowding_distance,
    dominates,
    environmental_selection,
    fast_nondominated_sort,
    truncate_archive,
)


EVOLUTIONARY_COMPARATORS = (
    "mopso",
    "nsga3",
    "ftgga",
    "mmode",
    "momogs-pce",
    "xgboost-moga",
    "hmo-nsga-ii",
)
SINGLE_SUBSET_COMPARATORS = ("hsic-lasso", "block-hsic-lasso")
AVAILABLE_COMPARATORS = EVOLUTIONARY_COMPARATORS + SINGLE_SUBSET_COMPARATORS


def _repair(masks: np.ndarray, preferred: int = 0) -> np.ndarray:
    masks = np.asarray(masks, dtype=bool)
    empty = np.flatnonzero(~masks.any(axis=1))
    masks[empty, preferred] = True
    return masks


def _initial_population(size: int, dimensions: int, rng: np.random.Generator) -> np.ndarray:
    population = np.zeros((size, dimensions), dtype=bool)
    upper = max(2, min(dimensions, max(3, dimensions // 3)))
    for row in population:
        count = int(rng.integers(1, upper + 1))
        row[rng.choice(dimensions, size=count, replace=False)] = True
    return population


def _evaluate_to_budget(
    evaluator: ObjectiveEvaluator, masks: np.ndarray, budget: int
) -> tuple[np.ndarray, np.ndarray]:
    remaining = budget - evaluator.calls
    masks = np.asarray(masks, dtype=bool)[:remaining]
    if not len(masks):
        return masks, np.empty((0, 3), dtype=float)
    return masks, evaluator.evaluate_population(masks)


def _result(
    evaluator: ObjectiveEvaluator,
    masks: np.ndarray,
    objectives: np.ndarray,
    config: OptimizerConfig,
    method: str,
    diagnostics: dict[str, object] | None = None,
) -> OptimizationResult:
    archive_masks, archive_objectives = truncate_archive(
        masks, objectives, config.archive_capacity
    )
    endpoint = min(
        range(len(archive_objectives)),
        key=lambda i: tuple(archive_objectives[i].tolist())
        + tuple(np.flatnonzero(archive_masks[i]).tolist()),
    )
    information = {
        "method": method,
        "config": asdict(config),
        "budget_type": "objective_evaluations",
    }
    information.update(diagnostics or {})
    return OptimizationResult(
        archive_masks,
        archive_objectives,
        archive_masks[endpoint].copy(),
        archive_objectives[endpoint].copy(),
        evaluator.calls,
        information,
    )


def _evaluator(
    X: np.ndarray,
    y: np.ndarray,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    config: OptimizerConfig,
    original_dimension: int,
    seed: int,
) -> ObjectiveEvaluator:
    return ObjectiveEvaluator(
        X,
        y,
        inner_splits,
        original_dimension,
        config.feature_ratio_denominator,
        seed,
    )


def run_mopso(X, y, inner_splits, config, original_dimension, seed):
    """Binary MOPSO with a nondominated leader archive."""
    rng = np.random.default_rng(seed)
    evaluator = _evaluator(X, y, inner_splits, config, original_dimension, seed)
    positions = _initial_population(config.population_size, X.shape[1], rng)
    velocities = np.zeros(positions.shape, dtype=float)
    positions, objectives = _evaluate_to_budget(evaluator, positions, config.evaluation_budget)
    personal_positions = positions.copy()
    personal_objectives = objectives.copy()
    all_masks, all_objectives = positions.copy(), objectives.copy()
    archive_masks, archive_objectives = truncate_archive(
        positions, objectives, config.archive_capacity
    )
    while evaluator.calls < config.evaluation_budget:
        proposals = []
        for i in range(len(positions)):
            leader = archive_masks[int(rng.integers(len(archive_masks)))]
            velocities[i] = (
                0.7 * velocities[i]
                + 1.5
                * rng.random(X.shape[1])
                * (personal_positions[i].astype(float) - positions[i].astype(float))
                + 1.5
                * rng.random(X.shape[1])
                * (leader.astype(float) - positions[i].astype(float))
            )
            probability = 1.0 / (1.0 + np.exp(-np.clip(velocities[i], -30, 30)))
            proposals.append(rng.random(X.shape[1]) < probability)
        proposed = _repair(np.asarray(proposals), int(np.argmax(evaluator.class_mi)))
        proposed, proposed_objectives = _evaluate_to_budget(
            evaluator, proposed, config.evaluation_budget
        )
        for i, objective in enumerate(proposed_objectives):
            if dominates(objective, personal_objectives[i]):
                personal_positions[i] = proposed[i]
                personal_objectives[i] = objective
        positions[: len(proposed)] = proposed
        objectives[: len(proposed)] = proposed_objectives
        all_masks = np.vstack([all_masks, proposed])
        all_objectives = np.vstack([all_objectives, proposed_objectives])
        archive_masks, archive_objectives = truncate_archive(
            all_masks, all_objectives, config.archive_capacity
        )
    return _result(evaluator, all_masks, all_objectives, config, "mopso")


def _tournament(objectives: np.ndarray, rng: np.random.Generator) -> int:
    ranks = np.empty(len(objectives), dtype=int)
    distances = np.zeros(len(objectives))
    for rank, front in enumerate(fast_nondominated_sort(objectives)):
        ranks[front] = rank
        distances[front] = crowding_distance(objectives[front])
    a, b = rng.integers(len(objectives), size=2)
    return int(a if (ranks[a], -distances[a], a) < (ranks[b], -distances[b], b) else b)


def _das_dennis(divisions: int = 12) -> np.ndarray:
    points = []
    for first in range(divisions + 1):
        for second in range(divisions - first + 1):
            third = divisions - first - second
            points.append([first, second, third])
    return np.asarray(points, dtype=float) / divisions


def _nsga3_select(masks: np.ndarray, objectives: np.ndarray, size: int, rng) -> tuple[np.ndarray, np.ndarray]:
    selected: list[int] = []
    last_front: list[int] = []
    for front in fast_nondominated_sort(objectives):
        if len(selected) + len(front) <= size:
            selected.extend(front)
        else:
            last_front = front
            break
    if len(selected) < size and last_front:
        lower = objectives.min(axis=0)
        span = np.where(objectives.max(axis=0) > lower, objectives.max(axis=0) - lower, 1.0)
        normalized = (objectives - lower) / span
        refs = _das_dennis()
        ref_norm = np.linalg.norm(refs, axis=1)
        ref_norm[ref_norm == 0] = 1.0
        association = np.empty(len(objectives), dtype=int)
        distance = np.empty(len(objectives), dtype=float)
        for i, point in enumerate(normalized):
            projection = (refs @ point) / (ref_norm**2)
            residual = point[None, :] - projection[:, None] * refs
            d = np.linalg.norm(residual, axis=1)
            association[i] = int(np.argmin(d))
            distance[i] = d[association[i]]
        counts = np.bincount(association[selected], minlength=len(refs))
        candidates = list(last_front)
        while candidates and len(selected) < size:
            available_refs = np.unique(association[candidates])
            minimum = counts[available_refs].min()
            ref = int(rng.choice(available_refs[counts[available_refs] == minimum]))
            local = [i for i in candidates if association[i] == ref]
            chosen = min(local, key=lambda i: (distance[i], i)) if counts[ref] == 0 else int(rng.choice(local))
            selected.append(chosen)
            candidates.remove(chosen)
            counts[ref] += 1
    idx = np.asarray(selected[:size], dtype=int)
    return masks[idx].copy(), objectives[idx].copy()


def run_nsga3(X, y, inner_splits, config, original_dimension, seed):
    """NSGA-III with Das-Dennis reference directions and niching."""
    rng = np.random.default_rng(seed)
    evaluator = _evaluator(X, y, inner_splits, config, original_dimension, seed)
    population = _initial_population(config.population_size, X.shape[1], rng)
    population, objectives = _evaluate_to_budget(evaluator, population, config.evaluation_budget)
    all_masks, all_objectives = population.copy(), objectives.copy()
    while evaluator.calls < config.evaluation_budget:
        children = []
        while len(children) < config.population_size:
            first = population[_tournament(objectives, rng)].copy()
            second = population[_tournament(objectives, rng)].copy()
            if X.shape[1] > 1 and rng.random() < config.crossover_probability:
                point = int(rng.integers(1, X.shape[1]))
                first[point:] = second[point:]
            mutation = rng.random(X.shape[1]) < config.mutation_probability
            first[mutation] = ~first[mutation]
            children.append(first)
        children = _repair(np.asarray(children), int(np.argmax(evaluator.class_mi)))
        children, child_objectives = _evaluate_to_budget(evaluator, children, config.evaluation_budget)
        combined_masks = np.vstack([population, children])
        combined_objectives = np.vstack([objectives, child_objectives])
        population, objectives = _nsga3_select(
            combined_masks, combined_objectives, config.population_size, rng
        )
        all_masks = np.vstack([all_masks, children])
        all_objectives = np.vstack([all_objectives, child_objectives])
    return _result(evaluator, all_masks, all_objectives, config, "nsga3")


def run_ftgga(X, y, inner_splits, config, original_dimension, seed):
    """Feature-threshold-guided genetic algorithm."""
    rng = np.random.default_rng(seed)
    evaluator = _evaluator(X, y, inner_splits, config, original_dimension, seed)
    thresholds = evaluator.class_mi.copy()
    thresholds /= max(float(thresholds.max()), 1e-12)
    population = _initial_population(config.population_size, X.shape[1], rng)
    population, objectives = _evaluate_to_budget(evaluator, population, config.evaluation_budget)
    all_masks, all_objectives = population.copy(), objectives.copy()
    generations = 0
    while evaluator.calls < config.evaluation_budget:
        generations += 1
        elite_count = max(1, config.population_size // 4)
        elite_idx = sorted(range(len(population)), key=lambda i: tuple(objectives[i]))[:elite_count]
        frequency = population[elite_idx].mean(axis=0)
        thresholds = np.clip(0.8 * thresholds + 0.2 * frequency, 0.02, 0.98)
        children = []
        while len(children) < config.population_size:
            a = population[_tournament(objectives, rng)]
            b = population[_tournament(objectives, rng)]
            source = np.where(rng.random(X.shape[1]) < thresholds, a, b)
            child = source.copy()
            mutation = rng.random(X.shape[1]) < config.mutation_probability
            child[mutation] = rng.random(np.sum(mutation)) < thresholds[mutation]
            children.append(child)
        children = _repair(np.asarray(children), int(np.argmax(thresholds)))
        children, child_objectives = _evaluate_to_budget(evaluator, children, config.evaluation_budget)
        population, objectives = environmental_selection(
            np.vstack([population, children]),
            np.vstack([objectives, child_objectives]),
            config.population_size,
        )
        all_masks = np.vstack([all_masks, children])
        all_objectives = np.vstack([all_objectives, child_objectives])
    return _result(
        evaluator, all_masks, all_objectives, config, "ftgga", {"generations": generations}
    )


def _unit_scale(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    lower, upper = float(values.min()), float(values.max())
    if upper <= lower:
        return np.zeros_like(values)
    return (values - lower) / (upper - lower)


def _feature_tournament(
    candidates: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
    *,
    prefer_high: bool,
) -> int:
    if len(candidates) == 1:
        return int(candidates[0])
    first, second = rng.choice(candidates, size=2, replace=False)
    if weights[first] == weights[second]:
        return int(min(first, second))
    if prefer_high:
        return int(first if weights[first] > weights[second] else second)
    return int(first if weights[first] < weights[second] else second)


def _hmo_feature_weights(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Aggregate the paper's Information Gain, RF, and ReliefF rankers.

    The paper names SVM-Rank as its aggregator but does not specify training
    preferences or a reproducible threshold.  Equal-weight normalized rank
    averaging is the declared fair-protocol completion used here.
    """
    information_gain = mutual_info_classif(X, y, random_state=seed)
    forest = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    forest.fit(X, y)
    score_matrix = np.column_stack(
        [
            _unit_scale(information_gain),
            _unit_scale(forest.feature_importances_),
            _unit_scale(relieff_scores(X, y)),
        ]
    )
    return score_matrix.mean(axis=1)


def _hmo_initial_population(
    size: int, dimensions: int, weights: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    population = np.zeros((size, dimensions), dtype=bool)
    all_features = np.arange(dimensions)
    for row in population:
        if dimensions == 1:
            row[0] = True
            continue
        count = int(rng.integers(2, dimensions)) if dimensions > 2 else 1
        available = all_features.copy()
        for _ in range(count):
            selected = _feature_tournament(available, weights, rng, prefer_high=True)
            row[selected] = True
            available = available[available != selected]
    return population


def _hmo_crossover(
    parents: np.ndarray, weights: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Three-parent, feature-weight-guided crossover from the paper description."""
    votes = parents.astype(int).sum(axis=0)
    child = votes >= 2
    singly_selected = votes == 1
    child[singly_selected] = rng.random(np.sum(singly_selected)) < weights[singly_selected]
    return child


def _hmo_mutation(
    child: np.ndarray, weights: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Remove a low-weight selected feature or add a high-weight absent feature."""
    child = child.copy()
    if rng.random() < 0.5 and np.sum(child) > 1:
        selected = np.flatnonzero(child)
        child[_feature_tournament(selected, weights, rng, prefer_high=False)] = False
    else:
        absent = np.flatnonzero(~child)
        if len(absent):
            child[_feature_tournament(absent, weights, rng, prefer_high=True)] = True
    return child


def run_hmo_nsga2(X, y, inner_splits, config, original_dimension, seed):
    """Paper-guided reimplementation of Vijai and P.'s HMO-NSGA-II.

    Paper-defined components are the three filter rankers, score-guided
    population initialization, three-parent crossover, balanced add/remove
    score-guided mutation, and NSGA-II environmental selection.  The shared
    repository protocol supplies unspecified operating parameters and the
    common three-objective evaluator.
    """
    rng = np.random.default_rng(seed)
    evaluator = _evaluator(X, y, inner_splits, config, original_dimension, seed)
    weights = _hmo_feature_weights(X, y, seed)
    population = _hmo_initial_population(config.population_size, X.shape[1], weights, rng)
    population, objectives = _evaluate_to_budget(evaluator, population, config.evaluation_budget)
    all_masks, all_objectives = population.copy(), objectives.copy()
    while evaluator.calls < config.evaluation_budget:
        children = []
        while len(children) < config.population_size:
            parent_indices = [_tournament(objectives, rng) for _ in range(3)]
            parents = population[parent_indices]
            if rng.random() < config.crossover_probability:
                child = _hmo_crossover(parents, weights, rng)
            else:
                child = parents[0].copy()
            if rng.random() < config.mutation_probability:
                child = _hmo_mutation(child, weights, rng)
            children.append(child)
        children = _repair(np.asarray(children), int(np.argmax(weights)))
        children, child_objectives = _evaluate_to_budget(evaluator, children, config.evaluation_budget)
        population, objectives = environmental_selection(
            np.vstack([population, children]),
            np.vstack([objectives, child_objectives]),
            config.population_size,
        )
        all_masks = np.vstack([all_masks, children])
        all_objectives = np.vstack([all_objectives, child_objectives])
    return _result(
        evaluator,
        all_masks,
        all_objectives,
        config,
        "hmo-nsga-ii",
        {
            "source_doi": "10.1016/j.dajour.2025.100550",
            "source_url": "https://doi.org/10.1016/j.dajour.2025.100550",
            "paper_defined_components": [
                "information_gain_random_forest_relieff_weights",
                "weight_guided_initialization",
                "three_parent_crossover",
                "balanced_weight_guided_mutation",
                "nsga2_environmental_selection",
            ],
            "fair_protocol_completions": [
                "equal_weight_normalized_score_aggregation",
                "continuous_fused_scores_in_place_of_unspecified_rank_threshold",
                "majority_then_score_probability_three_parent_crossover",
                "shared_crossover_probability",
                "shared_mutation_probability",
                "shared_three_objective_evaluator",
                "shared_evaluation_budget",
            ],
        },
    )


def run_mmode(X, y, inner_splits, config, original_dimension, seed):
    """Multifactorial multi-objective differential evolution on two train-only views."""
    rng = np.random.default_rng(seed)
    evaluator = _evaluator(X, y, inner_splits, config, original_dimension, seed)
    f_values = np.nan_to_num(f_classif(X, y)[0], nan=0.0)
    mi_values = mutual_info_classif(X, y, random_state=seed)
    branch_size = max(1, X.shape[1] // 2)
    allowed = [np.argsort(-f_values)[:branch_size], np.argsort(-mi_values)[:branch_size]]
    population = _initial_population(config.population_size, X.shape[1], rng)
    for i, row in enumerate(population):
        keep = np.zeros(X.shape[1], dtype=bool)
        keep[allowed[i % 2]] = True
        row &= keep
    population = _repair(population, int(np.argmax(mi_values)))
    population, objectives = _evaluate_to_budget(evaluator, population, config.evaluation_budget)
    skills = np.arange(len(population)) % 2
    all_masks, all_objectives = population.copy(), objectives.copy()
    while evaluator.calls < config.evaluation_budget:
        children = []
        child_skills = []
        for i in range(config.population_size):
            skill = int(skills[i % len(skills)])
            pool = np.flatnonzero(skills == skill)
            if len(pool) < 3:
                pool = np.arange(len(population))
            a, b, c = rng.choice(pool, size=3, replace=False)
            donor_probability = np.clip(
                population[a].astype(float)
                + 0.5 * (population[b].astype(float) - population[c].astype(float)),
                0.0,
                1.0,
            )
            child = population[i % len(population)].copy()
            crossover = rng.random(X.shape[1]) < 0.9
            child[crossover] = rng.random(np.sum(crossover)) < donor_probability[crossover]
            if rng.random() < 0.3:
                other = population[int(rng.integers(len(population)))]
                transfer = rng.random(X.shape[1]) < 0.1
                child[transfer] = other[transfer]
            keep = np.zeros(X.shape[1], dtype=bool)
            keep[allowed[skill]] = True
            child &= keep
            children.append(child)
            child_skills.append(skill)
        children = _repair(np.asarray(children), int(np.argmax(mi_values)))
        children, child_objectives = _evaluate_to_budget(evaluator, children, config.evaluation_budget)
        child_skills = np.asarray(child_skills[: len(children)])
        combined_masks = np.vstack([population, children])
        combined_objectives = np.vstack([objectives, child_objectives])
        combined_skills = np.r_[skills, child_skills]
        population, objectives = environmental_selection(
            combined_masks, combined_objectives, config.population_size
        )
        skills = np.asarray([
            combined_skills[np.flatnonzero(np.all(combined_masks == row, axis=1))[0]]
            for row in population
        ])
        all_masks = np.vstack([all_masks, children])
        all_objectives = np.vstack([all_objectives, child_objectives])
    return _result(
        evaluator,
        all_masks,
        all_objectives,
        config,
        "mmode",
        {"branch_size": branch_size, "views": ["anova_f", "mutual_information"]},
    )


def run_momogs_pce(X, y, inner_splits, config, original_dimension, seed):
    """Multi-subspace evolutionary search with population co-evolution."""
    rng = np.random.default_rng(seed)
    evaluator = _evaluator(X, y, inner_splits, config, original_dimension, seed)
    subspaces = [part for part in np.array_split(np.arange(X.shape[1]), 4) if len(part)]
    population = _initial_population(config.population_size, X.shape[1], rng)
    for i, row in enumerate(population):
        keep = np.zeros(X.shape[1], dtype=bool)
        keep[subspaces[i % len(subspaces)]] = True
        row &= keep
    population = _repair(population, int(np.argmax(evaluator.class_mi)))
    population, objectives = _evaluate_to_budget(evaluator, population, config.evaluation_budget)
    all_masks, all_objectives = population.copy(), objectives.copy()
    exploration_limit = int(0.7 * config.evaluation_budget)
    while evaluator.calls < config.evaluation_budget:
        children = []
        for i in range(config.population_size):
            first = population[_tournament(objectives, rng)]
            second = population[_tournament(objectives, rng)]
            child = np.where(rng.random(X.shape[1]) < 0.5, first, second).copy()
            mutation = rng.random(X.shape[1]) < config.mutation_probability
            child[mutation] = ~child[mutation]
            if evaluator.calls < exploration_limit:
                keep = np.zeros(X.shape[1], dtype=bool)
                keep[subspaces[i % len(subspaces)]] = True
                child &= keep
            elif rng.random() < 0.5:
                donor = population[int(rng.integers(len(population)))]
                trait = rng.random(X.shape[1]) < 0.15
                child[trait] = donor[trait]
            children.append(child)
        children = _repair(np.asarray(children), int(np.argmax(evaluator.class_mi)))
        children, child_objectives = _evaluate_to_budget(evaluator, children, config.evaluation_budget)
        population, objectives = environmental_selection(
            np.vstack([population, children]),
            np.vstack([objectives, child_objectives]),
            config.population_size,
        )
        all_masks = np.vstack([all_masks, children])
        all_objectives = np.vstack([all_objectives, child_objectives])
    return _result(
        evaluator,
        all_masks,
        all_objectives,
        config,
        "momogs-pce",
        {"subspaces": [part.tolist() for part in subspaces], "exploration_fraction": 0.7},
    )


def _xgboost_importance(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError(
            "xgboost-moga requires the optional 'xgboost' package; no surrogate is used"
        ) from exc
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=seed,
        n_jobs=1,
        eval_metric="logloss",
    )
    model.fit(X, y)
    return np.asarray(model.feature_importances_, dtype=float)


def run_xgboost_moga(X, y, inner_splits, config, original_dimension, seed):
    """XGBoost importance-prefiltered multi-objective genetic algorithm."""
    rng = np.random.default_rng(seed)
    evaluator = _evaluator(X, y, inner_splits, config, original_dimension, seed)
    importance = _xgboost_importance(X, y, seed)
    eligible = np.flatnonzero(importance > 0)
    if not len(eligible):
        eligible = np.array([int(np.argmax(evaluator.class_mi))])
    population = _initial_population(config.population_size, X.shape[1], rng)
    keep = np.zeros(X.shape[1], dtype=bool)
    keep[eligible] = True
    population &= keep
    population = _repair(population, int(eligible[np.argmax(importance[eligible])]))
    population, objectives = _evaluate_to_budget(evaluator, population, config.evaluation_budget)
    all_masks, all_objectives = population.copy(), objectives.copy()
    while evaluator.calls < config.evaluation_budget:
        children = []
        while len(children) < config.population_size:
            a = population[_tournament(objectives, rng)]
            b = population[_tournament(objectives, rng)]
            child = np.where(rng.random(X.shape[1]) < 0.5, a, b).copy()
            mutation = (rng.random(X.shape[1]) < config.mutation_probability) & keep
            child[mutation] = ~child[mutation]
            child &= keep
            children.append(child)
        children = _repair(np.asarray(children), int(eligible[np.argmax(importance[eligible])]))
        children, child_objectives = _evaluate_to_budget(evaluator, children, config.evaluation_budget)
        population, objectives = environmental_selection(
            np.vstack([population, children]),
            np.vstack([objectives, child_objectives]),
            config.population_size,
        )
        all_masks = np.vstack([all_masks, children])
        all_objectives = np.vstack([all_objectives, child_objectives])
    return _result(
        evaluator,
        all_masks,
        all_objectives,
        config,
        "xgboost-moga",
        {"positive_importance_features": eligible.tolist()},
    )


def _center(kernel: np.ndarray) -> np.ndarray:
    return kernel - kernel.mean(axis=0) - kernel.mean(axis=1)[:, None] + kernel.mean()


def _gaussian_kernel(values: np.ndarray, bandwidth: float) -> np.ndarray:
    distances = squareform(pdist(values.reshape(-1, 1), metric="sqeuclidean"))
    return _center(np.exp(-distances / (2.0 * bandwidth**2)))


def _hsic_rank(X: np.ndarray, y: np.ndarray, alpha: float, bandwidth: float, block: int | None) -> np.ndarray:
    target = _gaussian_kernel(y.astype(float), bandwidth).ravel()
    design = np.empty((X.shape[0] ** 2, X.shape[1]), dtype=float)
    for feature in range(X.shape[1]):
        if block is None:
            design[:, feature] = _gaussian_kernel(X[:, feature], bandwidth).ravel()
        else:
            column = np.zeros((X.shape[0], X.shape[0]), dtype=float)
            for start in range(0, X.shape[0], block):
                stop = min(start + block, X.shape[0])
                column[start:stop, start:stop] = _gaussian_kernel(
                    X[start:stop, feature], bandwidth
                )
            design[:, feature] = column.ravel()
    model = Lasso(alpha=alpha, positive=True, max_iter=5000, random_state=0)
    model.fit(design, target)
    return np.argsort(-np.asarray(model.coef_))


def _run_hsic(X, y, inner_splits, config, original_dimension, seed, block_mode: bool):
    evaluator = _evaluator(X, y, inner_splits, config, original_dimension, seed)
    alphas = (1e-4, 1e-2, 1.0)
    subset_sizes = tuple(sorted(set(min(value, X.shape[1]) for value in (5, 10, 20))))
    blocks = (5, 10) if block_mode else (None,)
    candidates = []
    for alpha in alphas:
        for block in blocks:
            for size in subset_sizes:
                scores = []
                for train, valid in inner_splits:
                    ranking = _hsic_rank(X[train], y[train], alpha, 1.0, block)
                    selected = ranking[:size]
                    model = KNeighborsClassifier(n_neighbors=min(3, len(train)), weights="distance")
                    model.fit(X[train][:, selected], y[train])
                    scores.append(
                        balanced_accuracy_score(y[valid], model.predict(X[valid][:, selected]))
                    )
                candidates.append((float(np.mean(scores)), -size, -alpha, block, alpha, size))
    best = max(candidates, key=lambda row: row[:3])
    _, _, _, block, alpha, size = best
    ranking = _hsic_rank(X, y, alpha, 1.0, block)
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[ranking[:size]] = True
    objective = evaluator.evaluate(mask)
    method = "block-hsic-lasso" if block_mode else "hsic-lasso"
    return OptimizationResult(
        mask[None, :],
        objective[None, :],
        mask,
        objective,
        1,
        {
            "method": method,
            "budget_type": "single_subset",
            "selected_alpha": alpha,
            "selected_size": size,
            "selected_block_size": block,
            "inner_selection_score": best[0],
        },
    )


def run_hsic_lasso(X, y, inner_splits, config, original_dimension, seed):
    return _run_hsic(X, y, inner_splits, config, original_dimension, seed, False)


def run_block_hsic_lasso(X, y, inner_splits, config, original_dimension, seed):
    return _run_hsic(X, y, inner_splits, config, original_dimension, seed, True)


RUNNERS: dict[str, Callable[..., OptimizationResult]] = {
    "mopso": run_mopso,
    "nsga3": run_nsga3,
    "ftgga": run_ftgga,
    "mmode": run_mmode,
    "momogs-pce": run_momogs_pce,
    "xgboost-moga": run_xgboost_moga,
    "hmo-nsga-ii": run_hmo_nsga2,
    "hsic-lasso": run_hsic_lasso,
    "block-hsic-lasso": run_block_hsic_lasso,
}


def run_comparator(method, X, y, inner_splits, config, original_dimension, seed):
    normalized = method.lower()
    if normalized not in RUNNERS:
        raise ValueError(f"unknown comparator: {method}")
    return RUNNERS[normalized](
        np.asarray(X, dtype=float),
        np.asarray(y),
        inner_splits,
        config,
        int(original_dimension),
        int(seed),
    )


__all__ = [
    "AVAILABLE_COMPARATORS",
    "EVOLUTIONARY_COMPARATORS",
    "SINGLE_SUBSET_COMPARATORS",
    "run_comparator",
]
