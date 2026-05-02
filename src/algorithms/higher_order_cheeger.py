"""Higher-order Cheeger: k-way partitioning via the L_rw spectral embedding.

The higher-order Cheeger inequality (Lee-Oveis Gharan-Trevisan 2014) says
    lambda_k / 2  <=  rho(k)  <=  O(k^2) sqrt(lambda_k)
where rho(k) = min over k-partitions {S_1,...,S_k} of max_i phi(S_i), and
lambda_k is the k-th smallest eigenvalue of L_sym. The upper bound is
constructive: cluster the bottom-k spectral embedding.

Concretely we take the bottom-k eigenvectors phi_1,...,phi_k of L_sym and
convert to the random-walk Laplacian's eigenvectors psi_i(v) = phi_i(v) /
sqrt(deg(v)) -- this is the "diffusion" coordinate where phi(S) translates
into a clean isoperimetric quantity. Each vertex gets a point in R^k and
we run k-means.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from core.graph import Graph
from core.registry import register_algorithm
from algorithms.base import Algorithm
from algorithms._spectral import spectral_embedding


@register_algorithm("higher_order_cheeger")
class HigherOrderCheeger(Algorithm):
    """k-way partitioning via k-means on the bottom-k L_rw eigenvectors."""

    def __init__(self, seed: int = 42, n_init: int = 10) -> None:
        self.seed = seed
        self.n_init = n_init

    def fit_predict(self, graph: Graph, k: int) -> np.ndarray:
        if k == 1:
            return np.zeros(graph.num_nodes, dtype=int)

        emb = spectral_embedding(
            graph.adjacency, k=k, row_normalize=False, seed=self.seed
        )
        deg = emb.degrees
        # psi_i(v) = phi_i(v) / sqrt(deg(v)) maps L_sym eigvecs to L_rw eigvecs.
        # Isolated nodes (deg == 0) are placed at the origin.
        d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        F = emb.embedding * d_inv_sqrt[:, None]

        return KMeans(
            n_clusters=k, n_init=self.n_init, random_state=self.seed
        ).fit_predict(F)
