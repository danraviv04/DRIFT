"""Exact equal-N residue KS for many saved selections, with tie-group handling."""
import numpy as np
from numba import njit


@njit(cache=True)
def _evaluate(order, group_end, selected, n, pool):
    result = np.empty(selected.shape[0])
    for draw in range(selected.shape[0]):
        total = 0
        for residue in range(order.shape[0]):
            gap = 0
            largest = 0
            for j in range(order.shape[1]):
                idx = order[residue, j]
                if selected[draw, idx]:
                    gap += 1 if idx < pool else -1
                if group_end[residue, j]:
                    largest = max(largest, abs(gap))
            total += largest
        result[draw] = total / (n * order.shape[0])
    return result


class ResampledKS:
    def __init__(self, wt, mutant):
        assert wt.shape[0] == mutant.shape[0] and np.isfinite(wt).all() and np.isfinite(mutant).all()
        values = np.concatenate([wt, mutant], axis=1)
        self.order = np.argsort(values, axis=1, kind="stable").astype(np.int16)
        ordered = np.take_along_axis(values, self.order, axis=1)
        self.end = np.ones_like(ordered, dtype=np.bool_)
        self.end[:, :-1] = ordered[:, :-1] != ordered[:, 1:]
        self.pool = wt.shape[1]
        self.total = values.shape[1]

    def __call__(self, left, right):
        assert left.shape == right.shape
        selected = np.zeros((len(left), self.total), dtype=np.bool_)
        rows = np.arange(len(left))[:, None]
        selected[rows, left] = True
        selected[rows, right + self.pool] = True
        assert (selected.sum(1) == left.shape[1] * 2).all()
        return _evaluate(self.order, self.end, selected, left.shape[1], self.pool)
