"""Adjusted Rand Index for comparing two clusterings of the same n nodes.

The Rand Index counts the fraction of pairs of points that two clusterings
agree on (same cluster in both, or different in both). Adjusting for chance
subtracts off the expected RI under a fixed-marginal hypergeometric null and
rescales so that 1.0 is perfect agreement and 0.0 is the chance level:

    ARI = (RI - E[RI]) / (max(RI) - E[RI])

In contingency-table form, with n_ij = |U_i ∩ V_j|, a_i = sum_j n_ij,
b_j = sum_i n_ij, and N = sum_ij n_ij,

                sum_ij C(n_ij, 2)  -  [sum_i C(a_i, 2)] [sum_j C(b_j, 2)] / C(N, 2)
    ARI = -----------------------------------------------------------------------------
          (1/2)(sum_i C(a_i, 2) + sum_j C(b_j, 2)) - [sum_i C(a_i, 2)] [sum_j C(b_j, 2)] / C(N, 2)

Reference: Hubert & Arabie (1985), "Comparing partitions".
"""

from __future__ import annotations

import numpy as np

from core.graph import Graph
from core.registry import register_eval
from evals.base import Eval


def _contingency(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Dense contingency table indexed by reordered cluster ids."""
    if u.shape != v.shape:
        raise ValueError(f"shape mismatch: {u.shape} vs {v.shape}")
    _, u_idx = np.unique(u, return_inverse=True)
    _, v_idx = np.unique(v, return_inverse=True)
    R = u_idx.max() + 1
    C = v_idx.max() + 1
    table = np.zeros((R, C), dtype=np.int64)
    np.add.at(table, (u_idx, v_idx), 1)
    return table


def _comb2(x: np.ndarray | int) -> np.ndarray | int:
    """C(x, 2) = x*(x-1)/2, vectorized."""
    return x * (x - 1) // 2


def adjusted_rand_index(u: np.ndarray, v: np.ndarray) -> float:
    """ARI between two integer label arrays of the same length."""
    table = _contingency(u, v)
    N = int(table.sum())
    if N < 2:
        return 1.0  # 0 or 1 points: trivially identical

    a = table.sum(axis=1)
    b = table.sum(axis=0)
    sum_comb_n = _comb2(table).sum()
    sum_comb_a = _comb2(a).sum()
    sum_comb_b = _comb2(b).sum()
    comb_N = _comb2(N)

    expected = sum_comb_a * sum_comb_b / comb_N
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    denom = max_index - expected
    if denom == 0.0:
        # Both clusterings degenerate to one big cluster (or all singletons,
        # all matching). Convention: identical => 1.0.
        return 1.0
    return float((sum_comb_n - expected) / denom)


@register_eval("ari")
class AdjustedRandIndexEval(Eval):
    """Adjusted Rand Index between predicted and target clusterings.

    Ignores ``graph``. Returns 1.0 for identical clusterings and ~0 under
    chance; can go slightly negative for worse-than-chance partitions.
    """

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        graph: Graph,
        predicted: np.ndarray,
        target: np.ndarray,
    ) -> float:
        return adjusted_rand_index(predicted, target)


if __name__ == "__main__":
    # Identical labelings (up to relabeling) → 1.0.
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([5, 5, 9, 9, 7, 7])
    print(f"ARI(identical relabel): {adjusted_rand_index(a, b):.4f}")
    # One swap → still high but < 1.
    c = np.array([0, 1, 1, 1, 2, 2])
    print(f"ARI(one swap): {adjusted_rand_index(a, c):.4f}")
    # Random vs. truth → near 0.
    rng = np.random.default_rng(0)
    rand = rng.integers(0, 3, size=600)
    truth = np.repeat([0, 1, 2], 200)
    print(f"ARI(random vs truth): {adjusted_rand_index(rand, truth):.4f}")
