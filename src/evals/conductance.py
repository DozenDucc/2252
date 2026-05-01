"""Graph conductance of a cut and of a predicted clustering.

For a vertex subset S ⊂ V with complement S̄ = V \\ S, the conductance is
    phi(S) = w(E(S, S̄)) / min(vol(S), vol(S̄))
where w(E(S, S̄)) is the total weight of edges crossing the cut and
vol(T) = sum_{v ∈ T} deg(v). The graph conductance phi(G) = min_S phi(S) is
NP-hard to compute exactly, so as a clustering metric we report the
worst-cluster conductance max_i phi(C_i) over the predicted clusters C_i —
this is the quantity bounded above by higher-order Cheeger inequalities.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from core.graph import Graph
from core.registry import register_eval
from evals.base import Eval


def cut_conductance(adjacency: sp.spmatrix, S: np.ndarray) -> float:
    """Conductance phi(S) of the cut (S, V \\ S).

    Parameters
    ----------
    adjacency : sparse matrix, shape (n, n)
        Symmetric (weighted) adjacency. Diagonal entries are ignored.
    S : ndarray
        Either a boolean mask of length n, or an int array of vertex indices.

    Returns
    -------
    phi(S). Returns 1.0 (the worst possible value) for the degenerate cases
    S = ∅ or S = V, since the textbook ratio is 0/0 there.
    """
    A = adjacency.tocsr()
    n = A.shape[0]

    mask = np.zeros(n, dtype=bool)
    S = np.asarray(S)
    if S.dtype == bool:
        if S.shape[0] != n:
            raise ValueError(f"boolean mask length {S.shape[0]} != n={n}")
        mask = S
    else:
        mask[S.astype(int)] = True

    if not mask.any() or mask.all():
        return 1.0

    deg = np.asarray(A.sum(axis=1)).ravel()
    vol_S = float(deg[mask].sum())
    vol_Sbar = float(deg[~mask].sum())
    denom = min(vol_S, vol_Sbar)
    if denom == 0.0:
        return 1.0

    # Cut weight: rows in S, columns in S̄. Symmetric A means we only count
    # each edge once across the partition by construction.
    cut = float(A[mask][:, ~mask].sum())
    return cut / denom


def clustering_conductances(
    adjacency: sp.spmatrix, labels: np.ndarray
) -> np.ndarray:
    """Per-cluster conductance phi(C_i) for every cluster id present in ``labels``.

    Returns
    -------
    ndarray of shape (k,) where k is the number of distinct labels, in the
    order produced by ``np.unique(labels)``.
    """
    A = adjacency.tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    total_vol = float(deg.sum())
    unique = np.unique(labels)
    out = np.empty(unique.shape[0], dtype=float)
    for i, lab in enumerate(unique):
        mask = labels == lab
        vol_C = float(deg[mask].sum())
        vol_rest = total_vol - vol_C
        denom = min(vol_C, vol_rest)
        if denom == 0.0:
            out[i] = 1.0
            continue
        cut = float(A[mask][:, ~mask].sum())
        out[i] = cut / denom
    return out


@register_eval("conductance")
class ConductanceEval(Eval):
    """Conductance of the predicted clustering.

    For a k-way partition into clusters C_1, ..., C_k we compute phi(C_i) for
    each cluster and aggregate. ``aggregate="max"`` (the default) reports the
    worst (bottleneck) cluster, matching the higher-order Cheeger quantity
    ρ_k = max_i phi(C_i). ``"mean"`` averages, ``"min"`` reports the best
    cluster (mostly useful as a sanity check).

    Ignores ``target``.
    """

    def __init__(self, aggregate: str = "max") -> None:
        if aggregate not in {"max", "mean", "min"}:
            raise ValueError(
                f"aggregate must be one of 'max', 'mean', 'min'; got {aggregate!r}"
            )
        self.aggregate = aggregate

    def __call__(
        self,
        graph: Graph,
        predicted: np.ndarray,
        target: np.ndarray,
    ) -> float:
        phis = clustering_conductances(graph.adjacency, predicted)
        if phis.size == 0:
            return 1.0
        if self.aggregate == "max":
            return float(phis.max())
        if self.aggregate == "min":
            return float(phis.min())
        return float(phis.mean())


if __name__ == "__main__":
    # Two 20-cliques bridged by a single edge: the natural 2-way cut has
    # cut weight 1 and side volumes 20*19 + 1 each, so phi ≈ 1/381.
    n_per = 20
    block = np.ones((n_per, n_per)) - np.eye(n_per)
    A_dense = np.block([[block, np.zeros_like(block)],
                        [np.zeros_like(block), block]])
    A_dense[0, n_per] = A_dense[n_per, 0] = 1.0
    A = sp.csr_matrix(A_dense)

    S = np.zeros(2 * n_per, dtype=bool)
    S[:n_per] = True
    print(f"phi(S) for the bridge cut: {cut_conductance(A, S):.6f}")

    labels = np.concatenate([np.zeros(n_per, dtype=int), np.ones(n_per, dtype=int)])
    print(f"per-cluster phi: {clustering_conductances(A, labels)}")
