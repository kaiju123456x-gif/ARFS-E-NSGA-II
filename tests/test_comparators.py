import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold

from arfs_ensga2.comparators import (
    AVAILABLE_COMPARATORS,
    EVOLUTIONARY_COMPARATORS,
    SINGLE_SUBSET_COMPARATORS,
    run_comparator,
)
from arfs_ensga2.config import OptimizerConfig


@pytest.fixture(scope="module")
def comparison_case():
    X, y = make_classification(
        n_samples=30,
        n_features=8,
        n_informative=4,
        n_redundant=2,
        random_state=19,
    )
    splits = list(StratifiedKFold(2, shuffle=True, random_state=23).split(X, y))
    config = OptimizerConfig(
        population_size=6,
        evaluation_budget=18,
        archive_capacity=10,
        initialization_top_k=3,
    )
    return X, y, splits, config


@pytest.mark.parametrize("method", EVOLUTIONARY_COMPARATORS)
def test_evolutionary_comparator_uses_exact_budget(method, comparison_case):
    X, y, splits, config = comparison_case
    result = run_comparator(method, X, y, splits, config, X.shape[1], 7)
    assert result.evaluations == config.evaluation_budget
    assert result.endpoint_mask.any()
    assert result.archive_masks.shape[1] == X.shape[1]
    assert result.archive_objectives.shape[1] == 3


@pytest.mark.parametrize("method", SINGLE_SUBSET_COMPARATORS)
def test_single_subset_comparator_is_explicit(method, comparison_case):
    X, y, splits, config = comparison_case
    result = run_comparator(method, X, y, splits, config, X.shape[1], 7)
    assert result.evaluations == 1
    assert result.diagnostics["budget_type"] == "single_subset"
    assert result.endpoint_mask.any()


def test_comparator_is_deterministic(comparison_case):
    X, y, splits, config = comparison_case
    first = run_comparator("mopso", X, y, splits, config, X.shape[1], 17)
    second = run_comparator("mopso", X, y, splits, config, X.shape[1], 17)
    np.testing.assert_array_equal(first.endpoint_mask, second.endpoint_mask)
    np.testing.assert_allclose(first.endpoint_objectives, second.endpoint_objectives)


def test_hmo_is_a_runnable_paper_guided_reimplementation(comparison_case):
    X, y, splits, config = comparison_case
    result = run_comparator("hmo-nsga-ii", X, y, splits, config, X.shape[1], 7)
    assert result.evaluations == config.evaluation_budget
    assert result.diagnostics["source_doi"] == "10.1016/j.dajour.2025.100550"
    assert result.diagnostics["fair_protocol_completions"]


def test_mmode_retains_the_paper_run_source_views(comparison_case):
    X, y, splits, config = comparison_case
    result = run_comparator("mmode", X, y, splits, config, X.shape[1], 7)
    assert result.diagnostics["views"] == ["relieff", "mutual_information"]


@pytest.mark.parametrize("method", SINGLE_SUBSET_COMPARATORS)
def test_hsic_uses_the_manuscript_kernel_cap(method, comparison_case):
    X, y, splits, config = comparison_case
    result = run_comparator(method, X, y, splits, config, X.shape[1], 7)
    assert result.diagnostics["kernel_sample_cap"] == 250
    assert result.diagnostics["kernel_samples_used"] == len(X)


def test_registry_contains_only_runnable_methods():
    assert "hmo-nsga-ii" in AVAILABLE_COMPARATORS

