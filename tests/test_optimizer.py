import numpy as np
from sklearn.datasets import make_classification

from arfs_ensga2.config import OptimizerConfig
from arfs_ensga2.objectives import ObjectiveEvaluator, make_inner_splits
from arfs_ensga2.optimizer import ARFSENSGA2
from arfs_ensga2.pareto import igd


def test_optimizer_is_deterministic_and_respects_budget():
    X, y = make_classification(
        n_samples=48,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=7,
    )
    splits = make_inner_splits(X, y, 2, 9)
    config = OptimizerConfig(
        population_size=6,
        evaluation_budget=18,
        archive_capacity=12,
        initialization_top_k=4,
    )
    first = ARFSENSGA2(config, random_state=11).fit(X, y, splits)
    second = ARFSENSGA2(config, random_state=11).fit(X, y, splits)
    assert first.evaluations == 18
    assert second.evaluations == 18
    assert first.endpoint_mask.any()
    np.testing.assert_array_equal(first.archive_masks, second.archive_masks)
    np.testing.assert_allclose(first.archive_objectives, second.archive_objectives)


def test_aligned_igd_increment_uses_one_reference_set_and_scaling():
    previous = np.array([[0.2, 0.7, 0.5], [0.7, 0.2, 0.5]])
    current = np.array([[0.15, 0.65, 0.45], [0.65, 0.15, 0.45]])
    reference = np.vstack([previous, current, [[0.4, 0.4, 0.6]]])
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([1.0, 1.0, 1.0])

    previous_igd, current_igd, increment = ARFSENSGA2._aligned_igd_increment(
        previous, current, reference, lower, upper
    )

    assert previous_igd == igd(previous, reference, lower, upper)
    assert current_igd == igd(current, reference, lower, upper)
    assert increment == current_igd - previous_igd


def test_mutation_state_uses_aligned_increments_not_raw_igd_differences():
    config = OptimizerConfig(mutation_probability=0.12, rho=0.5, window_size=3)
    optimizer = ARFSENSGA2(config, random_state=11)

    probability, bias, stagnant = optimizer._mutation_state(
        [0.0, 100.0, 100.0], [0.1, 1.0]
    )

    assert stagnant is False
    assert probability == 0.06
    assert bias == "sparsity"


def test_optimizer_records_aligned_igd_evidence():
    X, y = make_classification(
        n_samples=48,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=7,
    )
    splits = make_inner_splits(X, y, 2, 9)
    config = OptimizerConfig(
        population_size=6,
        evaluation_budget=24,
        archive_capacity=12,
        initialization_top_k=4,
    )

    result = ARFSENSGA2(config, random_state=11).fit(X, y, splits)
    records = result.diagnostics["igd_alignment_history"]

    assert len(records) == len(result.diagnostics["generations"])
    assert result.diagnostics["aligned_igd_increments"] == [
        record["increment"] for record in records
    ]
    for record in records:
        assert record["increment"] == record["current_igd"] - record["previous_igd"]


def test_pairwise_mutual_information_is_symmetric(monkeypatch):
    X = np.column_stack(
        [np.arange(12, dtype=float), np.arange(12, dtype=float) ** 2]
    )
    y = np.tile([0, 1], 6)
    splits = make_inner_splits(X, y, 3, 7)
    calls = []

    def directional_estimator(predictor, response, random_state):
        calls.append((predictor[:, 0].copy(), response.copy(), random_state))
        value = 1.0 if len(calls) == 1 else 3.0
        return np.array([value])

    monkeypatch.setattr(
        "arfs_ensga2.objectives.mutual_info_regression", directional_estimator
    )
    evaluator = ObjectiveEvaluator(X, y, splits, original_dimension=2, seed=11)

    assert evaluator._pairwise_mi(0, 1) == 2.0
    assert evaluator._pairwise_mi(1, 0) == 2.0
    assert len(calls) == 2

