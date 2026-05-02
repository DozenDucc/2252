"""Cheeger sweep-cut: 2-way partitioning by rounding the Fiedler vector.

Computes the eigenvector of the random-walk Laplacian for the second-smallest
eigenvalue, sorts vertices by it, and returns the prefix cut with minimum
conductance. The Cheeger inequality guarantees the chosen cut satisfies
    phi(S) <= sqrt(2 * lambda_2)
where lambda_2 is the second eigenvalue of L_sym.
"""

from __future__ import annotations

import numpy as np

from core.graph import Graph
from core.registry import register_algorithm
from algorithms.base import Algorithm
from algorithms._spectral import spectral_embedding


@register_algorithm("cheeger_sweep_cut")
class CheegerSweepCut(Algorithm):
    """2-way partitioning via the Fiedler-vector sweep cut."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def fit_predict(self, graph: Graph, k: int) -> np.ndarray:
        if k != 2:
            raise ValueError(f"CheegerSweepCut is 2-way only; got k={k}")

        A = graph.adjacency
        n = A.shape[0]

        # Bottom-2 eigenvectors of L_sym; column 1 is the Fiedler vector.
        emb = spectral_embedding(A, k=2, row_normalize=False, seed=self.seed)
        v = emb.embedding[:, 1]
        deg = emb.degrees

        # Convert to L_rw eigenvector so the Cheeger bound is tight.
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
        # Sweep over prefixes S_i = {order[0], ..., order[i]} for i in [0, n-2].
        # Skip i = n-1 (S = V gives an empty complement).
        for i in range(n - 1):
            u = order[i]
            row = slice(indptr[u], indptr[u + 1])
            nbrs = indices[row]
            w = data[row]
            mask_in = in_S[nbrs]
            # Edges u->S̄ become new cut edges; edges u->S stop being cut.
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
