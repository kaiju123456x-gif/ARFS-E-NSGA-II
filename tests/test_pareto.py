import numpy as np

from arfs_ensga2.pareto import hypervolume_3d, nondominated_mask


def test_nondominated_mask_removes_dominated_point():
    objectives = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.2, 0.1, 0.5]])
    assert nondominated_mask(objectives).tolist() == [True, False, True]


def test_hypervolume_single_box():
    point = np.array([[0.2, 0.3, 0.4]])
    reference = np.array([1.0, 1.0, 1.0])
    assert np.isclose(hypervolume_3d(point, reference), 0.8 * 0.7 * 0.6)


def test_hypervolume_ignores_dominated_duplicate():
    points = np.array([[0.2, 0.3, 0.4], [0.4, 0.5, 0.6], [0.2, 0.3, 0.4]])
    reference = np.ones(3)
    assert np.isclose(hypervolume_3d(points, reference), 0.8 * 0.7 * 0.6)

