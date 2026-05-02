"""Sweep cut on the k-th L_sym eigenvector (rather than just the Fiedler vector).

Useful for testing whether the recovery of an optimum cut on a Cartesian-product
graph is about *which* eigenvector is rounded, independent of the rounding
mechanism. Identical to ``CheegerSweepCut`` except the eigenvector index is a
constructor argument.
"""

from __future__ import annotations

import numpy as np

from core.graph import Graph
from core.registry import register_algorithm
from algorithms.base import Algorithm
from algorithms._spectral import spectral_embedding


@register_algorithm("sweep_on_kth")
class SweepOnKth(Algorithm):
    """Sweep cut on the ``k_eigvec``-th L_sym eigenvector (1-indexed).

    Index 1 is the trivial constant; index 2 is the Fiedler vector; index 3
    is the next non-trivial eigenvector; etc.
    """

    def __init__(self, k_eigvec: int = 2, seed: int = 42) -> None:
        if k_eigvec < 2:
            raise ValueError(f"k_eigvec must be >= 2; got {k_eigvec}")
        self.k_eigvec = k_eigvec
        self.seed = seed

    def fit_predict(self, graph: Graph, k: int) -> np.ndarray:
        if k != 2:
            raise ValueError(f"SweepOnKth is 2-way only; got k={k}")

        A = graph.adjacency
        n = A.shape[0]

        emb = spectral_embedding(A, k=self.k_eigvec, row_normalize=False, seed=self.seed)
        v = emb.embedding[:, self.k_eigvec - 1]
        deg = emb.degrees

        with np.errstate(divide="ignore"):
            x = np.where(deg > 0, v / np.sqrt(deg), 0.0)

        order = np.argsort(x, kind="stable")

        total_vol = float(deg.sum())
        in_S = np.zeros(n, dtype=bool)
        vol_S = 0.0
        cut_S = 0.0
        best_phi = np.inf
        best_i = 0

        indptr, indices, data = A.indptr, A.indices, A.data
        for i in range(n - 1):
            u = order[i]
            row = slice(indptr[u], indptr[u + 1])
            nbrs = indices[row]
            w = data[row]
            mask_in = in_S[nbrs]
            cut_S += float(w[~mask_in].sum()) - float(w[mask_in].sum())
            vol_S += float(deg[u])
            in_S[u] = True

            denom = min(vol_S, total_vol - vol_S)
            if denom > 0:
                phi = cut_S / denom
                if phi < best_phi:
                    best_phi = phi
                    best_i = i

        labels = np.zeros(n, dtype=int)
        labels[order[: best_i + 1]] = 1
        return labels
