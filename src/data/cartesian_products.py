"""Cartesian-product graph factories for the spectral-dimension experiments.

Each factory builds G = H_1 square H_2 with consecutive-integer vertex labels
ordered as ``i * |H_2| + j`` for ``i in V(H_1), j in V(H_2)``. This matches
the layout convention used by ``data.tree_cross_path`` so that the factor
structure can be reasoned about cleanly (e.g., "tree copy at path position j"
lives at indices ``i * q + j`` for fixed ``j``).

Each factory returns ``(adjacency, info)`` where info carries:
- ``H1_size``, ``H2_size``: the factor sizes
- ``H1_factory``, ``H2_factory``: callables building the factor adjacencies
  (used to compute factor eigenvalues)
- ``slab_axis``: 0 if the optimum slab cut is along H1, 1 if along H2, None
  if there is no expected slab structure
- a ``slab_labels`` callable returning the optimum slab partition (or None)
- Optionally, structural metadata in extra fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import networkx as nx
import numpy as np
import scipy.sparse as sp

from core.graph import Graph
from core.registry import register_dataset
from data.base import Dataset
from data.tree_cross_path import make_double_tree


# ----- factor builders -----------------------------------------------------


def path_factor(n: int) -> sp.csr_matrix:
    return nx.to_scipy_sparse_array(nx.path_graph(n), format="csr", dtype=float)


def cycle_factor(n: int) -> sp.csr_matrix:
    return nx.to_scipy_sparse_array(nx.cycle_graph(n), format="csr", dtype=float)


def double_tree_factor(tree_height: int) -> sp.csr_matrix:
    G, _, _ = make_double_tree(tree_height)
    return nx.to_scipy_sparse_array(G, format="csr", dtype=float)


def double_tree_factor_for_size(p: int) -> tuple[sp.csr_matrix, int]:
    """Pick the smallest balanced-double-tree of size >= p; returns (A, h_used)."""
    h = 0
    while 2 * (2 ** (h + 1) - 1) < p:
        h += 1
    return double_tree_factor(h), h


# ----- product assembly ----------------------------------------------------


def cartesian_product_adj(A1: sp.csr_matrix, A2: sp.csr_matrix) -> sp.csr_matrix:
    """Adjacency of H1 square H2 in the i*|H2| + j vertex order.

    A(H1 sq H2) = A1 kron I_{|H2|} + I_{|H1|} kron A2.
    """
    n1, n2 = A1.shape[0], A2.shape[0]
    return (sp.kron(A1, sp.identity(n2), format="csr")
            + sp.kron(sp.identity(n1), A2, format="csr")).tocsr()


def slab_along_axis_0(n1: int, n2: int) -> np.ndarray:
    """Slab cut splitting the H1 axis at i = n1 // 2."""
    half = n1 // 2
    labels = np.empty(n1 * n2, dtype=int)
    for i in range(n1):
        labels[i * n2 : (i + 1) * n2] = 0 if i < half else 1
    return labels


def slab_along_axis_1(n1: int, n2: int) -> np.ndarray:
    """Slab cut splitting the H2 axis at j = n2 // 2."""
    half = n2 // 2
    labels = np.empty(n1 * n2, dtype=int)
    for i in range(n1):
        for j in range(n2):
            labels[i * n2 + j] = 0 if j < half else 1
    return labels


def double_tree_slab_along_axis_0(p: int, q: int) -> np.ndarray:
    """For G = T_p square H_2 (n2=q), the analytic optimum is to cut the
    root-to-root edge in each tree copy. With our index convention
    (i indexes the double tree, j indexes the second factor), nodes 0..p/2-1
    belong to the left half and p/2..p-1 to the right half. The function is
    just ``slab_along_axis_0`` but kept as a name for documentation.
    """
    return slab_along_axis_0(p, q)


# ----- info dataclass ------------------------------------------------------


@dataclass
class CartesianProductInfo:
    name: str
    H1_size: int
    H2_size: int
    n: int
    factor_H1_label: str
    factor_H2_label: str
    slab_axis: Optional[int]
    extra: dict = field(default_factory=dict)


# ----- dataset bases -------------------------------------------------------


class _CartesianProductBase(Dataset):
    """Common loader. Subclasses set the factor builders and slab axis."""

    H1_size: int = 0
    H2_size: int = 0
    factor_H1_label: str = ""
    factor_H2_label: str = ""
    slab_axis: Optional[int] = None  # 0 = cut along H1, 1 = along H2

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._cache: tuple[Graph, np.ndarray, CartesianProductInfo, sp.csr_matrix, sp.csr_matrix] | None = None

    def _build_factors(self) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        raise NotImplementedError

    def load(self) -> tuple[Graph, np.ndarray]:
        if self._cache is None:
            A1, A2 = self._build_factors()
            n1, n2 = A1.shape[0], A2.shape[0]
            A = cartesian_product_adj(A1, A2)
            n = n1 * n2
            if self.slab_axis == 0:
                target = slab_along_axis_0(n1, n2)
            elif self.slab_axis == 1:
                target = slab_along_axis_1(n1, n2)
            else:
                target = np.zeros(n, dtype=int)  # no analytic optimum

            info = CartesianProductInfo(
                name=self.name or self.__class__.__name__,
                H1_size=n1,
                H2_size=n2,
                n=n,
                factor_H1_label=self.factor_H1_label,
                factor_H2_label=self.factor_H2_label,
                slab_axis=self.slab_axis,
            )
            graph = Graph(
                adjacency=A,
                num_nodes=n,
                name=self.name or self.__class__.__name__,
                metadata={
                    "H1_size": n1,
                    "H2_size": n2,
                    "n": n,
                    "factor_H1_label": self.factor_H1_label,
                    "factor_H2_label": self.factor_H2_label,
                    "slab_axis": -1 if self.slab_axis is None else self.slab_axis,
                },
            )
            self._cache = (graph, target, info, A1, A2)
        graph, target, _, _, _ = self._cache
        return graph, target

    @property
    def num_clusters(self) -> int:
        return 2

    @property
    def info(self) -> CartesianProductInfo:
        if self._cache is None:
            self.load()
        assert self._cache is not None
        return self._cache[2]

    @property
    def factors(self) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        if self._cache is None:
            self.load()
        assert self._cache is not None
        return self._cache[3], self._cache[4]


# ----- concrete registered datasets ---------------------------------------


@register_dataset("path_x_cycle")
class PathCrossCycle(_CartesianProductBase):
    """P_q square C_p, with q < p. Slab cut goes along the path axis."""

    factor_H1_label = "path"
    factor_H2_label = "cycle"
    slab_axis = 0

    def __init__(self, q: int = 20, p: int = 40, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.q = q
        self.p = p

    def _build_factors(self) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        return path_factor(self.q), cycle_factor(self.p)


@register_dataset("cycle_x_doubletree")
class CycleCrossDoubleTree(_CartesianProductBase):
    """C_q square T_p. The double tree's small lambda_2 means the optimum slab
    cut goes along the *tree* axis (cutting the root-root edges per cycle copy)."""

    factor_H1_label = "cycle"
    factor_H2_label = "doubletree"
    slab_axis = 1  # split H2 (tree) at its midpoint

    def __init__(self, q: int = 20, tree_height: int = 4, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.q = q
        self.tree_height = tree_height

    def _build_factors(self) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        return cycle_factor(self.q), double_tree_factor(self.tree_height)


@register_dataset("path_x_path_unequal")
class PathCrossPathUnequal(_CartesianProductBase):
    """P_q square P_p with q << p. Slab cut along the longer-path axis."""

    factor_H1_label = "path"
    factor_H2_label = "path"
    slab_axis = 1  # the longer factor is H2; its lambda_2 is smaller

    def __init__(self, q: int = 10, p: int = 50, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.q = q
        self.p = p

    def _build_factors(self) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        return path_factor(self.q), path_factor(self.p)


@register_dataset("doubletree_x_doubletree")
class DoubleTreeCrossDoubleTree(_CartesianProductBase):
    """T_{p1} square T_{p2}. Equal-size case: lambda_2 ties exactly, so neither
    axis is dominant; we mark slab_axis=0 by convention but expect d* = 2."""

    factor_H1_label = "doubletree"
    factor_H2_label = "doubletree"
    slab_axis = 0  # arbitrary, ties in lambda_2

    def __init__(self, h1: int = 4, h2: int = 4, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.h1 = h1
        self.h2 = h2

    def _build_factors(self) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        return double_tree_factor(self.h1), double_tree_factor(self.h2)


@register_dataset("doubletree_x_doubletree_unequal")
class DoubleTreeCrossDoubleTreeUnequal(_CartesianProductBase):
    """T_{p1} square T_{p2} with p1 < p2 (smaller h1 vs h2). Slab cut along the
    larger-tree axis (smaller lambda_2)."""

    factor_H1_label = "doubletree"
    factor_H2_label = "doubletree"
    slab_axis = 1

    def __init__(self, h1: int = 3, h2: int = 5, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.h1 = h1
        self.h2 = h2

    def _build_factors(self) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        return double_tree_factor(self.h1), double_tree_factor(self.h2)


@register_dataset("path_x_doubletree_perturbed")
class PathCrossDoubleTreePerturbed(_CartesianProductBase):
    """P_q square T_p plus random extra edges with probability ``epsilon``
    inserted between non-adjacent pairs. Used to test perturbation-robustness
    of the framework's predictions.
    """

    factor_H1_label = "path"
    factor_H2_label = "doubletree"
    slab_axis = 1  # tree axis still has the smallest lambda_2 for small epsilon

    def __init__(
        self,
        q: int = 25,
        tree_height: int = 4,
        epsilon: float = 0.001,
        seed: int = 0,
    ) -> None:
        super().__init__(seed=seed)
        self.q = q
        self.tree_height = tree_height
        self.epsilon = epsilon

    def _build_factors(self) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        return path_factor(self.q), double_tree_factor(self.tree_height)

    def load(self) -> tuple[Graph, np.ndarray]:
        if self._cache is None:
            A1, A2 = self._build_factors()
            n1, n2 = A1.shape[0], A2.shape[0]
            A = cartesian_product_adj(A1, A2)
            n = n1 * n2

            if self.epsilon > 0:
                rng = np.random.default_rng(self._seed)
                # Sample edges with rejection: for each potential extra edge
                # (i, j), include with prob epsilon. To keep cost manageable
                # for n on the order of 1500 (n^2 ~ 2.3M cells), iterate
                # explicitly only when n is small; otherwise use a
                # negative-binomial-style sampler that picks a Bernoulli
                # count and then random pairs.
                n_pairs_total = n * (n - 1) // 2
                expected_extra = self.epsilon * n_pairs_total
                # Sample number of extra edges from Poisson(expected) for speed.
                n_extra = int(rng.poisson(expected_extra))
                if n_extra > 0:
                    # Sample without replacement-ish from upper triangle.
                    idx = rng.integers(0, n_pairs_total, size=n_extra)
                    # decode linear -> (i, j) with i < j: use the standard
                    # triangular inversion. To keep this simple for
                    # moderate n, materialize via np.triu_indices once and
                    # gather. n=1550 -> ~1.2M pairs, fine.
                    rows, cols = np.triu_indices(n, k=1)
                    pick_r = rows[idx]
                    pick_c = cols[idx]
                    # Build a sparse perturbation matrix and add (deduplicated
                    # via sum_duplicates + clipping to {0, 1}).
                    new_data = np.ones(len(pick_r), dtype=A.dtype)
                    P = sp.coo_matrix(
                        (new_data, (pick_r, pick_c)), shape=(n, n)
                    )
                    P = (P + P.T).tocsr()
                    P.sum_duplicates()
                    A = (A + P).tocsr()
                    # Clamp to {0, 1}: any cell with weight > 1 due to overlap
                    # with existing edges becomes 1.
                    A.data = np.clip(A.data, 0.0, 1.0)
                    # Drop any zero entries.
                    A.eliminate_zeros()

            target = slab_along_axis_1(n1, n2)
            info = CartesianProductInfo(
                name=self.name or self.__class__.__name__,
                H1_size=n1,
                H2_size=n2,
                n=n,
                factor_H1_label=self.factor_H1_label,
                factor_H2_label=self.factor_H2_label,
                slab_axis=self.slab_axis,
                extra={"epsilon": self.epsilon, "seed": self._seed},
            )
            graph = Graph(
                adjacency=A,
                num_nodes=n,
                name=self.name or f"path_x_doubletree_perturbed_eps{self.epsilon}",
                metadata={
                    "H1_size": n1,
                    "H2_size": n2,
                    "n": n,
                    "factor_H1_label": self.factor_H1_label,
                    "factor_H2_label": self.factor_H2_label,
                    "slab_axis": self.slab_axis,
                    "epsilon": self.epsilon,
                },
            )
            self._cache = (graph, target, info, A1, A2)
        graph, target, _, _, _ = self._cache
        return graph, target


# Tunable variant of path-x-doubletree at arbitrary q for the interleaving sweep.
# We do *not* register a dataset class per (q, h) pair since prediction 3 needs
# many; the script builds them ad-hoc via this factory.
def make_path_x_doubletree(q: int, tree_height: int) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """Return (A_path, A_dt, A_product, slab_labels) for P_q square T_h."""
    A_path = path_factor(q)
    A_dt = double_tree_factor(tree_height)
    A = cartesian_product_adj(A_path, A_dt)
    p = A_dt.shape[0]
    # Vertex i*p + j has path-position i, tree-vertex j. The optimum slab cut
    # splits the tree axis (cut the root-to-root edge in each path-copy of the
    # double tree).
    labels = slab_along_axis_1(q, p)
    return A_path, A_dt, A, labels
