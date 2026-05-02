"""Standard spectral clustering algorithm."""

import numpy as np
from sklearn.cluster import KMeans

from core.graph import Graph
from core.registry import register_algorithm
from algorithms.base import Algorithm
from algorithms._spectral import spectral_embedding


@register_algorithm("standard_spectral_clustering")
class StandardSpectralClustering(Algorithm):
    """Standard spectral clustering algorithm."""

    def __init__(self, seed: int = 42, n_init: int = 10) -> None:
        self.seed = seed
        self.n_init = n_init

    def fit_predict(self, graph: Graph, k: int) -> np.ndarray:
        if k == 1:
            return np.zeros(graph.num_nodes, dtype=int)
        # NJW: drop the sqrt(deg)-proportional first eigenvector (it encodes
        # degree, not community), THEN row-normalize. Doing it the other way
        # round contaminates the row norms with the degree component.
        emb = spectral_embedding(graph.adjacency, k + 1, row_normalize=False, seed=self.seed)
        X = emb.embedding[:, 1:]
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms
        return KMeans(n_clusters=k, n_init=self.n_init, random_state=self.seed).fit_predict(X)
