from __future__ import annotations

import numpy as np


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True when minimization vector a Pareto-dominates b."""
    return bool(np.all(a <= b) and np.any(a < b))


def nondominated_mask(objectives: np.ndarray) -> np.ndarray:
    n = len(objectives)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i != j and dominates(objectives[j], objectives[i]):
                keep[i] = False
                break
    return keep


def fast_nondominated_sort(objectives: np.ndarray) -> list[list[int]]:
    n = len(objectives)
    dominated: list[list[int]] = [[] for _ in range(n)]
    counts = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(objectives[p], objectives[q]):
                dominated[p].append(q)
            elif dominates(objectives[q], objectives[p]):
                counts[p] += 1
        if counts[p] == 0:
            fronts[0].append(p)
    rank = 0
    while fronts[rank]:
        next_front: list[int] = []
        for p in fronts[rank]:
            for q in dominated[p]:
                counts[q] -= 1
                if counts[q] == 0:
                    next_front.append(q)
        rank += 1
        fronts.append(next_front)
    return fronts[:-1]


def crowding_distance(objectives: np.ndarray) -> np.ndarray:
    n, m = objectives.shape
    distance = np.zeros(n, dtype=float)
    if n <= 2:
        distance[:] = np.inf
        return distance
    for column in range(m):
        order = np.argsort(objectives[:, column], kind="stable")
        distance[order[[0, -1]]] = np.inf
        span = objectives[order[-1], column] - objectives[order[0], column]
        if span <= 0:
            continue
        for pos in range(1, n - 1):
            distance[order[pos]] += (
                objectives[order[pos + 1], column]
                - objectives[order[pos - 1], column]
            ) / span
    return distance


def environmental_selection(
    masks: np.ndarray, objectives: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray]:
    selected: list[int] = []
    for front in fast_nondominated_sort(objectives):
        if len(selected) + len(front) <= size:
            selected.extend(front)
            continue
        local = objectives[front]
        distance = crowding_distance(local)
        order = np.argsort(-distance, kind="stable")
        selected.extend(np.asarray(front)[order[: size - len(selected)]].tolist())
        break
    idx = np.asarray(selected, dtype=int)
    return masks[idx].copy(), objectives[idx].copy()


def truncate_archive(
    masks: np.ndarray, objectives: np.ndarray, capacity: int
) -> tuple[np.ndarray, np.ndarray]:
    # Keep the first copy of duplicate genotypes, then remove dominated points.
    _, unique_idx = np.unique(masks, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    masks, objectives = masks[unique_idx], objectives[unique_idx]
    keep = nondominated_mask(objectives)
    masks, objectives = masks[keep], objectives[keep]
    if len(masks) <= capacity:
        return masks.copy(), objectives.copy()
    distance = crowding_distance(objectives)
    idx = np.argsort(-distance, kind="stable")[:capacity]
    return masks[idx].copy(), objectives[idx].copy()


def igd(archive: np.ndarray, reference: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    if not len(archive) or not len(reference):
        return float("inf")
    span = np.where(upper > lower, upper - lower, 1.0)
    a = (archive - lower) / span
    r = (reference - lower) / span
    distances = np.linalg.norm(r[:, None, :] - a[None, :, :], axis=2)
    return float(np.mean(np.min(distances, axis=1)))


def _hypervolume_2d(points: np.ndarray, reference: np.ndarray) -> float:
    points = points[np.all(points < reference, axis=1)]
    if not len(points):
        return 0.0
    ys = np.unique(np.r_[points[:, 0], reference[0]])
    area = 0.0
    for left, right in zip(ys[:-1], ys[1:]):
        active = points[points[:, 0] <= left]
        if len(active):
            area += (right - left) * max(0.0, reference[1] - np.min(active[:, 1]))
    return float(area)


def hypervolume_3d(points: np.ndarray, reference: np.ndarray) -> float:
    """Exact dominated hypervolume for three minimization objectives."""
    points = np.asarray(points, dtype=float)
    reference = np.asarray(reference, dtype=float)
    points = points[np.all(points < reference, axis=1)]
    if not len(points):
        return 0.0
    xs = np.unique(np.r_[points[:, 0], reference[0]])
    volume = 0.0
    for left, right in zip(xs[:-1], xs[1:]):
        active = points[points[:, 0] <= left, 1:]
        volume += (right - left) * _hypervolume_2d(active, reference[1:])
    return float(volume)

