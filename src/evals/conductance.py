"""Conductance of a 2-way graph partition.

phi(S) = cut(S, V \\ S) / min(vol(S), vol(V \\ S))
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from core.graph import Graph
from core.registry import register_eval
from evals.base import Eval


def compute_conductance(A: sp.csr_matrix, labels: np.ndarray) -> float:
    """Conductance of the 2-way partition induced by ``labels``.

    Labels are coerced to {False, True} via the predicate ``labels != labels[0]``
    so cluster IDs need not be 0/1 specifically; only the binary partition
    matters. Returns ``inf`` if either side is empty (degenerate partition).
    """
    A = A.tocsr() if not sp.isspmatrix_csr(A) else A
    deg = np.asarray(A.sum(axis=1)).ravel()
    total_vol = float(deg.sum())

    unique = np.unique(labels)
    if unique.size != 2:
        # Degenerate partition (single cluster). Conductance is undefined;
        # report inf so downstream comparisons treat it as the worst possible.
        return float("inf")

    side = labels == unique[0]
    vol_S = float(deg[side].sum())
    vol_other = total_vol - vol_S
    denom = min(vol_S, vol_other)
    if denom == 0:
        return float("inf")

    # Cut weight: sum of A[i, j] over i in S, j not in S. A is symmetric, so
    # this equals the off-block sum of the row-sliced submatrix.
    A_rows = A[side]
    cut = float(A_rows[:, ~side].sum())
    return cut / denom


@register_eval("conductance")
class Conductance(Eval):
    """Eval wrapper around :func:`compute_conductance`."""

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        graph: Graph,
        predicted: np.ndarray,
        target: np.ndarray,
    ) -> float:
        return compute_conductance(graph.adjacency, predicted)
