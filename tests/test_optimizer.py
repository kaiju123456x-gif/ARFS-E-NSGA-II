import numpy as np
from sklearn.datasets import make_classification

from arfs_ensga2.config import OptimizerConfig
from arfs_ensga2.objectives import make_inner_splits
from arfs_ensga2.optimizer import ARFSENSGA2


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

