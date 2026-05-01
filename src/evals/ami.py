"""Adjusted Mutual Information between two clusterings.

NMI normalizes mutual information by the entropies, but raw NMI is biased
upward when one of the clusterings has many small clusters: a finer
partition gets credit for the agreements it racks up by chance. AMI fixes
this by subtracting off the expectation of MI under a fixed-marginal
permutation null:

    AMI(U, V) = (MI(U, V) - E[MI(U, V)]) / (avg(H(U), H(V)) - E[MI(U, V)])

where E[MI] is computed exactly via the hypergeometric closed form from
Vinh, Epps & Bailey (2010), "Information Theoretic Measures for Clusterings
Comparison: Variants, Properties, Normalization and Correction for Chance".
We use the arithmetic-mean normalizer (sklearn's default) so the score is 1
iff the clusterings are identical, ~0 under chance, and possibly negative
for adversarial pairings.
"""

from __future__ import annotations

import numpy as np
from math import lgamma

from core.graph import Graph
from core.registry import register_eval
from evals.base import Eval


def _contingency(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    if u.shape != v.shape:
        raise ValueError(f"shape mismatch: {u.shape} vs {v.shape}")
    _, u_idx = np.unique(u, return_inverse=True)
    _, v_idx = np.unique(v, return_inverse=True)
    R = u_idx.max() + 1
    C = v_idx.max() + 1
    table = np.zeros((R, C), dtype=np.int64)
    np.add.at(table, (u_idx, v_idx), 1)
    return table


def _entropy(counts: np.ndarray, N: int) -> float:
    counts = counts[counts > 0]
    p = counts / N
    return float(-(p * np.log(p)).sum())


def _mutual_info(table: np.ndarray, N: int) -> float:
    a = table.sum(axis=1)
    b = table.sum(axis=0)
    mi = 0.0
    nz_r, nz_c = np.nonzero(table)
    for i, j in zip(nz_r, nz_c):
        n_ij = table[i, j]
        # MI term: (n_ij / N) * log( N * n_ij / (a_i * b_j) ).
        mi += (n_ij / N) * np.log(N * n_ij / (a[i] * b[j]))
    return float(mi)


def _expected_mutual_info(table: np.ndarray, N: int) -> float:
    """E[MI] under the permutation model with fixed marginals (Vinh 2010).

    For each (i, j), n_ij ranges over [max(1, a_i + b_j - N), min(a_i, b_j)]
    and the contribution is
        (n_ij / N) * log(N * n_ij / (a_i * b_j)) * P(n_ij | a_i, b_j, N)
    with P given by the hypergeometric pmf
        a_i! b_j! (N - a_i)! (N - b_j)! / (N! n_ij! (a_i - n_ij)! (b_j - n_ij)! (N - a_i - b_j + n_ij)!)
    The lower bound starts at 1 because n_ij = 0 contributes 0 (0 log 0 = 0).
    Factorials live in log-space via lgamma to stay numerically stable.
    """
    a = table.sum(axis=1)
    b = table.sum(axis=0)
    R, C = table.shape
    log_N_fact = lgamma(N + 1)

    emi = 0.0
    for i in range(R):
        a_i = int(a[i])
        if a_i == 0:
            continue
        log_ai_fact = lgamma(a_i + 1)
        log_Nminus_ai_fact = lgamma(N - a_i + 1)
        for j in range(C):
            b_j = int(b[j])
            if b_j == 0:
                continue
            log_bj_fact = lgamma(b_j + 1)
            log_Nminus_bj_fact = lgamma(N - b_j + 1)

            n_lo = max(1, a_i + b_j - N)
            n_hi = min(a_i, b_j)
            for n_ij in range(n_lo, n_hi + 1):
                log_p = (
                    log_ai_fact + log_bj_fact
                    + log_Nminus_ai_fact + log_Nminus_bj_fact
                    - log_N_fact
                    - lgamma(n_ij + 1)
                    - lgamma(a_i - n_ij + 1)
                    - lgamma(b_j - n_ij + 1)
                    - lgamma(N - a_i - b_j + n_ij + 1)
                )
                term = (n_ij / N) * np.log(N * n_ij / (a_i * b_j))
                emi += term * np.exp(log_p)
    return float(emi)


def adjusted_mutual_info(u: np.ndarray, v: np.ndarray) -> float:
    """AMI between two integer label arrays of the same length."""
    table = _contingency(u, v)
    N = int(table.sum())
    if N < 2:
        return 1.0

    a = table.sum(axis=1)
    b = table.sum(axis=0)
    H_u = _entropy(a, N)
    H_v = _entropy(b, N)

    # If either side is a trivial single cluster, MI ≡ 0 and the normalizer
    # is 0; the standard convention is AMI = 1 iff both labelings agree
    # (which they trivially do when one side has a single cluster only when
    # the other does too). We follow sklearn: return 1.0 only when both are
    # trivial and identical, else 0.0.
    if H_u == 0.0 and H_v == 0.0:
        return 1.0
    if H_u == 0.0 or H_v == 0.0:
        return 0.0

    mi = _mutual_info(table, N)
    emi = _expected_mutual_info(table, N)
    normalizer = 0.5 * (H_u + H_v)
    denom = normalizer - emi
    if denom == 0.0:
        return 1.0 if mi == emi else 0.0
    return float((mi - emi) / denom)


@register_eval("ami")
class AdjustedMutualInfoEval(Eval):
    """Adjusted Mutual Information between predicted and target clusterings.

    Ignores ``graph``. 1.0 for identical clusterings, ~0 under chance.
    """

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        graph: Graph,
        predicted: np.ndarray,
        target: np.ndarray,
    ) -> float:
        return adjusted_mutual_info(predicted, target)


if __name__ == "__main__":
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([5, 5, 9, 9, 7, 7])
    print(f"AMI(identical relabel): {adjusted_mutual_info(a, b):.4f}")
    c = np.array([0, 1, 1, 1, 2, 2])
    print(f"AMI(one swap): {adjusted_mutual_info(a, c):.4f}")
    rng = np.random.default_rng(0)
    rand = rng.integers(0, 3, size=600)
    truth = np.repeat([0, 1, 2], 200)
    print(f"AMI(random vs truth): {adjusted_mutual_info(rand, truth):.4f}")
