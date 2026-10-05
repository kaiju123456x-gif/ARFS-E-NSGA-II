from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import rankdata

from .config import OptimizerConfig
from .initialization import guided_population, random_sparse_population
from .objectives import ObjectiveEvaluator
from .pareto import (
    crowding_distance,
    dominates,
    environmental_selection,
    fast_nondominated_sort,
    igd,
    truncate_archive,
)


@dataclass
class OptimizationResult:
    archive_masks: np.ndarray
    archive_objectives: np.ndarray
    endpoint_mask: np.ndarray
    endpoint_objectives: np.ndarray
    evaluations: int
    diagnostics: dict[str, object]


class ARFSENSGA2:
    """ARFS-E-NSGA-II with independently switchable ablation components."""

    def __init__(
        self,
        config: OptimizerConfig | None = None,
        *,
        guided_initialization: bool = True,
        dpim: bool = True,
        drfm: bool = True,
        random_state: int = 0,
    ) -> None:
        self.config = config or OptimizerConfig()
        self.config.validate()
        self.guided_initialization = guided_initialization
        self.dpim = dpim
        self.drfm = drfm
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self._none_counter: dict[bytes, int] = {}

    @staticmethod
    def _repair(population: np.ndarray, preferred: int) -> None:
        empty = np.flatnonzero(~population.any(axis=1))
        population[empty, preferred] = True

    @staticmethod
    def _rank_and_crowding(objectives: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ranks = np.empty(len(objectives), dtype=int)
        distances = np.zeros(len(objectives), dtype=float)
        for rank, front in enumerate(fast_nondominated_sort(objectives)):
            ranks[front] = rank
            distances[front] = crowding_distance(objectives[front])
        return ranks, distances

    def _tournament(self, objectives: np.ndarray) -> int:
        ranks, distances = self._rank_and_crowding(objectives)
        a, b = self.rng.integers(0, len(objectives), size=2)
        if ranks[a] != ranks[b]:
            return int(a if ranks[a] < ranks[b] else b)
        if distances[a] != distances[b]:
            return int(a if distances[a] > distances[b] else b)
        return int(min(a, b))

    def _partition(
        self,
        population: np.ndarray,
        objectives: np.ndarray,
        archive_masks: np.ndarray,
        archive_objectives: np.ndarray,
    ) -> tuple[list[int], list[int], list[int]]:
        conv: list[int] = []
        div: list[int] = []
        none: list[int] = []
        for i, objective in enumerate(objectives):
            if any(dominates(objective, archive) for archive in archive_objectives):
                conv.append(i)
            elif not any(dominates(archive, objective) for archive in archive_objectives):
                div.append(i)
            else:
                none.append(i)
        for i in conv + div:
            self._none_counter[population[i].tobytes()] = 0
        for i in none:
            key = population[i].tobytes()
            self._none_counter[key] = self._none_counter.get(key, 0) + 1
            if self._none_counter[key] > 2 and len(archive_masks):
                source = int(self.rng.integers(0, len(archive_masks)))
                population[i] = archive_masks[source]
                objectives[i] = archive_objectives[source]
                self._none_counter[population[i].tobytes()] = 0
        return conv, div, none

    def _mutation_state(self, igd_history: list[float]) -> tuple[float, str, bool]:
        cfg = self.config
        if len(igd_history) < 3:
            return cfg.mutation_probability, "bidirectional", False
        increments = np.diff(igd_history[-cfg.window_size :])
        scale = max(float(np.std(increments)), cfg.std_floor)
        stagnant = abs(float(increments[-1])) < cfg.rho * scale
        if stagnant:
            return min(cfg.mutation_probability * 1.5, 0.2), "bidirectional", True
        return max(cfg.mutation_probability * 0.5, 0.05), "sparsity", False

    def _crossover_and_mutate(
        self,
        parents: np.ndarray,
        mutation_probability: float,
        mutation_bias: str,
        preferred: int,
    ) -> np.ndarray:
        order = self.rng.permutation(len(parents))
        parents = parents[order]
        children: list[np.ndarray] = []
        for pos in range(0, len(parents), 2):
            first = parents[pos].copy()
            second = parents[(pos + 1) % len(parents)].copy()
            if self.rng.random() < self.config.crossover_probability:
                exchange = self.rng.random(first.size) < 0.5
                first[exchange], second[exchange] = second[exchange].copy(), first[exchange].copy()
            children.extend([first, second])
        offspring = np.asarray(children[: len(parents)], dtype=bool)
        events = self.rng.random(offspring.shape) < mutation_probability
        if mutation_bias == "sparsity":
            # The paper says "prioritize" 1->0 but gives no probability. We use 1.0 vs 0.25.
            additions = (~offspring) & events & (self.rng.random(offspring.shape) < 0.25)
            removals = offspring & events
            offspring[removals] = False
            offspring[additions] = True
        else:
            offspring[events] = ~offspring[events]
        self._repair(offspring, preferred)
        return offspring

    def _standard_offspring(
        self, population: np.ndarray, objectives: np.ndarray, preferred: int
    ) -> tuple[np.ndarray, dict[str, object]]:
        indices = [self._tournament(objectives) for _ in range(len(population))]
        offspring = self._crossover_and_mutate(
            population[indices], self.config.mutation_probability, "bidirectional", preferred
        )
        return offspring, {"mutation_probability": self.config.mutation_probability, "stagnant": False}

    def _dpim_offspring(
        self,
        population: np.ndarray,
        objectives: np.ndarray,
        archive_masks: np.ndarray,
        archive_objectives: np.ndarray,
        igd_history: list[float],
        preferred: int,
    ) -> tuple[np.ndarray, dict[str, object]]:
        conv, div, none = self._partition(
            population, objectives, archive_masks, archive_objectives
        )
        distances = crowding_distance(objectives)
        parents: list[np.ndarray] = []
        for _ in range(len(population)):
            if not conv and not div:
                chosen = self._tournament(objectives)
            elif (self.rng.random() < 0.5 and conv) or not div:
                chosen = min(conv, key=lambda idx: (objectives[idx, 0], idx))
            else:
                max_distance = max(distances[idx] for idx in div)
                chosen = min(idx for idx in div if distances[idx] == max_distance)
            parents.append(population[chosen])
        probability, bias, stagnant = self._mutation_state(igd_history)
        offspring = self._crossover_and_mutate(
            np.asarray(parents), probability, bias, preferred
        )
        return offspring, {
            "groups": {"convergence": len(conv), "diversity": len(div), "none": len(none)},
            "mutation_probability": probability,
            "mutation_bias": bias,
            "stagnant": stagnant,
        }

    @staticmethod
    def _correlation_matrix(X: np.ndarray) -> np.ndarray:
        pearson = np.nan_to_num(np.corrcoef(X, rowvar=False), nan=0.0)
        ranks = np.apply_along_axis(rankdata, 0, X)
        spearman = np.nan_to_num(np.corrcoef(ranks, rowvar=False), nan=0.0)
        matrix = 0.5 * np.abs(pearson) + 0.5 * np.abs(spearman)
        np.fill_diagonal(matrix, 0.0)
        return matrix

    def _drfm_refine(
        self,
        population: np.ndarray,
        class_mi: np.ndarray,
        correlations: np.ndarray,
        generation: int,
    ) -> tuple[np.ndarray, dict[str, object]]:
        refined = population.copy()
        dormant: list[int] = []
        safe: list[int] = []
        if generation % self.config.monitor_interval == 0:
            unused = np.flatnonzero(~refined.any(axis=0))
            safe = unused[class_mi[unused] > self.config.mi_threshold].tolist()
            dormant = sorted(set(unused.tolist()) - set(safe))
        swaps = 0
        for mask in refined:
            selected = np.flatnonzero(mask)
            unselected = np.flatnonzero(~mask)
            if not len(selected) or not len(unselected):
                continue
            selected_merit = []
            for feature in selected:
                neighbors = selected[selected != feature]
                penalty = float(np.mean(correlations[feature, neighbors])) if len(neighbors) else 0.0
                selected_merit.append(class_mi[feature] - self.config.alpha * penalty)
            worst_pos = int(np.argmin(selected_merit))
            worst = int(selected[worst_pos])
            retained = selected[selected != worst]
            candidate_merit = []
            for feature in unselected:
                penalty = float(np.mean(correlations[feature, retained])) if len(retained) else 0.0
                candidate_merit.append(class_mi[feature] - self.config.alpha * penalty)
            best_pos = int(np.argmax(candidate_merit))
            best = int(unselected[best_pos])
            if candidate_merit[best_pos] > selected_merit[worst_pos]:
                mask[worst] = False
                mask[best] = True
                swaps += 1
        return refined, {"swaps": swaps, "safe_features": safe, "dormant_features": dormant}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        inner_splits: list[tuple[np.ndarray, np.ndarray]],
        *,
        original_dimension: int | None = None,
    ) -> OptimizationResult:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        original_dimension = int(original_dimension or X.shape[1])
        evaluator = ObjectiveEvaluator(
            X,
            y,
            inner_splits,
            original_dimension,
            self.config.feature_ratio_denominator,
            self.random_state,
        )
        if self.guided_initialization:
            population, feature_weights, init_diagnostics = guided_population(
                X,
                y,
                inner_splits,
                self.config.population_size,
                self.config.initialization_top_k,
                self.rng,
                self.random_state,
            )
        else:
            population, feature_weights, init_diagnostics = random_sparse_population(
                self.config.population_size, X.shape[1], self.rng
            )
        preferred = int(np.argmax(feature_weights))
        objectives = evaluator.evaluate_population(population)
        archive_masks, archive_objectives = truncate_archive(
            population, objectives, self.config.archive_capacity
        )
        reference_points = archive_objectives.copy()
        lower = np.min(objectives, axis=0)
        upper = np.max(objectives, axis=0)
        igd_history = [igd(archive_objectives, reference_points, lower, upper)]
        correlations = self._correlation_matrix(X)
        generations: list[dict[str, object]] = []
        generation = 0
        while evaluator.calls < self.config.evaluation_budget:
            generation += 1
            if self.dpim:
                offspring, search_info = self._dpim_offspring(
                    population,
                    objectives,
                    archive_masks,
                    archive_objectives,
                    igd_history,
                    preferred,
                )
            else:
                offspring, search_info = self._standard_offspring(
                    population, objectives, preferred
                )
            drfm_info: dict[str, object] = {"swaps": 0}
            if self.drfm:
                offspring, drfm_info = self._drfm_refine(
                    offspring, evaluator.class_mi, correlations, generation
                )
            remaining = self.config.evaluation_budget - evaluator.calls
            offspring = offspring[:remaining]
            offspring_objectives = evaluator.evaluate_population(offspring)
            population, objectives = environmental_selection(
                np.vstack([population, offspring]),
                np.vstack([objectives, offspring_objectives]),
                self.config.population_size,
            )
            archive_masks, archive_objectives = truncate_archive(
                np.vstack([archive_masks, offspring]),
                np.vstack([archive_objectives, offspring_objectives]),
                self.config.archive_capacity,
            )
            reference_points = np.unique(
                np.vstack([reference_points, archive_objectives]), axis=0
            )
            lower = np.minimum(lower, np.min(offspring_objectives, axis=0))
            upper = np.maximum(upper, np.max(offspring_objectives, axis=0))
            igd_history.append(igd(archive_objectives, reference_points, lower, upper))
            generations.append(
                {
                    "generation": generation,
                    "evaluations": evaluator.calls,
                    "archive_size": len(archive_masks),
                    "igd": igd_history[-1],
                    "search": search_info,
                    "drfm": drfm_info,
                }
            )
        endpoint_index = min(
            range(len(archive_objectives)),
            key=lambda idx: tuple(archive_objectives[idx].tolist()) + tuple(np.flatnonzero(archive_masks[idx]).tolist()),
        )
        diagnostics = {
            "config": asdict(self.config),
            "components": {
                "guided_initialization": self.guided_initialization,
                "dpim": self.dpim,
                "drfm": self.drfm,
            },
            "initialization": init_diagnostics,
            "igd_history": igd_history,
            "generations": generations,
            "pairwise_mi_cache_size": len(evaluator._pair_mi),
        }
        return OptimizationResult(
            archive_masks,
            archive_objectives,
            archive_masks[endpoint_index].copy(),
            archive_objectives[endpoint_index].copy(),
            evaluator.calls,
            diagnostics,
        )


__all__ = ["ARFSENSGA2", "OptimizationResult", "OptimizerConfig"]

