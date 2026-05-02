"""Tree-cross-path graph from Guattery & Miller (1998), §6.

The tree-cross-path graph is the Cartesian product of:
  - a *double tree* on p = 2 * (2^(h+1) - 1) vertices: two complete balanced
    binary trees of height h, joined by a single edge between their roots;
  - a *path* on q vertices.

Total vertices n = p * q. Total edges = q * (p - 1) + p * (q - 1) = 2pq - p - q.

Choosing q > pi * sqrt(p) puts the path's spectral gap below the double
tree's (lambda_2(P_q) ~ pi^2 / q^2, lambda_2(double tree) >= 1/p), so the
Fiedler vector of the product is constant on each copy of the double tree
and varies only along the path. This is the structural ingredient that
makes sweep cut return |partial S| ~ p (cutting between two whole double-tree
copies) instead of the optimum |partial S| = q (cutting the q root-to-root
edges, one per path copy). See Guattery-Miller Theorem 6.3.

The optimum cut: split each double tree at its root-to-root edge. One side
holds the q copies of the "left" component tree, the other holds the q
copies of the "right" component tree. Each side has q * (2^(h+1) - 1)
vertices and they are joined to one another only through the q root-to-root
edges. Cut size = q.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from core.graph import Graph
from core.registry import register_dataset
from data.base import Dataset


@dataclass
class TreeCrossPathInfo:
    """Structural metadata about a tree-cross-path graph."""

    tree_height: int
    path_length: int
    p: int                      # double-tree size
    q: int                      # path size
    n: int                      # p * q
    component_tree_size: int    # 2^(h+1) - 1
    left_root_index: int        # node index of the "left" tree's root in the double tree
    right_root_index: int       # node index of the "right" tree's root in the double tree
    optimum_cut_size: int       # = q (cut the q root-to-root edges)


def _balanced_tree_with_root(h: int) -> tuple[nx.Graph, int]:
    """A balanced binary tree of height h. Root index is 0."""
    return nx.balanced_tree(r=2, h=h), 0


def make_double_tree(tree_height: int) -> tuple[nx.Graph, int, int]:
    """Two balanced binary trees of height ``tree_height`` joined at the roots.

    Returns ``(G, left_root, right_root)``. Vertex indices 0..N-1 with
    N = 2 * (2^(h+1) - 1) = 2^(h+2) - 2. The first half holds the left tree
    (root at 0); the second half holds the right tree (root at N // 2).
    """
    if tree_height < 0:
        raise ValueError(f"tree_height must be >= 0; got {tree_height}")
    left, left_root = _balanced_tree_with_root(tree_height)
    right, right_root = _balanced_tree_with_root(tree_height)
    n_left = left.number_of_nodes()
    right = nx.relabel_nodes(right, {v: v + n_left for v in right.nodes})
    right_root = right_root + n_left
    G = nx.compose(left, right)
    G.add_edge(left_root, right_root)
    return G, left_root, right_root


def make_tree_cross_path(
    tree_height: int, path_length: int
) -> tuple[sp.csr_matrix, TreeCrossPathInfo]:
    """Build the tree-cross-path graph and return ``(A, info)``.

    Vertex layout: networkx's ``cartesian_product`` returns nodes labeled
    ``(u, v)`` for ``u`` in the double tree and ``v`` in the path; we relabel
    them to consecutive integers using the ordering ``u * q + v`` so that
    "tree copy index" floor-divides cleanly out of the linear index.
    """
    if tree_height < 0:
        raise ValueError(f"tree_height must be >= 0; got {tree_height}")
    if path_length < 2:
        raise ValueError(f"path_length must be >= 2; got {path_length}")

    double_tree, left_root, right_root = make_double_tree(tree_height)
    p = double_tree.number_of_nodes()
    q = path_length
    path = nx.path_graph(q)

    product = nx.cartesian_product(double_tree, path)
    # Relabel (u, v) -> u*q + v for consecutive integer indices and a
    # predictable "first axis = tree, second axis = path" layout.
    relabel = {(u, v): u * q + v for u, v in product.nodes}
    product = nx.relabel_nodes(product, relabel, copy=True)

    A = nx.to_scipy_sparse_array(product, nodelist=range(p * q), format="csr", dtype=np.float64)

    n = p * q
    expected_edges = q * (p - 1) + p * (q - 1)
    actual_edges = int(A.nnz // 2)
    if actual_edges != expected_edges:
        raise AssertionError(
            f"edge count mismatch: expected {expected_edges}, got {actual_edges}"
        )
    if (A != A.T).nnz != 0:
        raise AssertionError("tree-cross-path adjacency is not symmetric")
    if A.diagonal().sum() != 0:
        raise AssertionError("tree-cross-path adjacency has self-loops")
    n_cc, _ = connected_components(A, directed=False)
    if n_cc != 1:
        raise AssertionError(f"tree-cross-path is not connected: {n_cc} components")

    component_tree_size = (2 ** (tree_height + 1)) - 1
    info = TreeCrossPathInfo(
        tree_height=tree_height,
        path_length=q,
        p=p,
        q=q,
        n=n,
        component_tree_size=component_tree_size,
        left_root_index=left_root,
        right_root_index=right_root,
        optimum_cut_size=q,
    )
    return A, info


def optimum_partition(info: TreeCrossPathInfo) -> np.ndarray:
    """Labels for the optimum bisection: cut the q root-to-root edges.

    Vertex i*q + j (i in [0, p), j in [0, q)) is on the "left" side
    iff its tree-vertex index ``i`` belongs to the left component tree
    (i.e., 0 <= i < p // 2). The two sides each contain q copies of one
    component tree, joined to the other side only through the root-root
    edges (one per path index).
    """
    n = info.n
    half_p = info.p // 2
    labels = np.empty(n, dtype=int)
    for i in range(info.p):
        side = 0 if i < half_p else 1
        labels[i * info.q : (i + 1) * info.q] = side
    return labels


class _TreeCrossPathBase(Dataset):
    """Common base. Subclasses fix tree_height and path_length."""

    tree_height: int = 0
    path_length: int = 0

    def __init__(self, tree_height: Optional[int] = None, path_length: Optional[int] = None) -> None:
        # Allow YAML-driven overrides; otherwise use the class defaults set
        # by the registered subclasses below.
        if tree_height is not None:
            self.tree_height = tree_height
        if path_length is not None:
            self.path_length = path_length
        self._cache: tuple[Graph, np.ndarray, TreeCrossPathInfo] | None = None

    def load(self) -> tuple[Graph, np.ndarray]:
        if self._cache is None:
            A, info = make_tree_cross_path(self.tree_height, self.path_length)
            graph = Graph(
                adjacency=A,
                num_nodes=A.shape[0],
                name=self.name or f"tree_cross_path_h{info.tree_height}_q{info.path_length}",
                metadata={
                    "p": info.p,
                    "q": info.q,
                    "n": info.n,
                    "tree_height": info.tree_height,
                    "path_length": info.path_length,
                    "component_tree_size": info.component_tree_size,
                    "left_root_index": info.left_root_index,
                    "right_root_index": info.right_root_index,
                    "optimum_cut_size": info.optimum_cut_size,
                },
            )
            target = optimum_partition(info)
            self._cache = (graph, target, info)
        graph, target, _ = self._cache
        return graph, target

    @property
    def num_clusters(self) -> int:
        return 2

    @property
    def info(self) -> TreeCrossPathInfo:
        if self._cache is None:
            self.load()
        assert self._cache is not None
        return self._cache[2]


@register_dataset("tree_cross_path_small")
class TreeCrossPathSmall(_TreeCrossPathBase):
    """h=3, q=20. p=30, n=600.

    Note: the task spec suggested q=12, but at q=12 the path's lambda_2 is
    *larger* than the double tree's (0.068 vs 0.057), inverting the
    construction's spectral premise — sweep cut then accidentally finds the
    optimum because the path is no longer the bottleneck. q=20 puts the
    factor-eigenvalue ratio at ~0.43 (path / double_tree), comfortably
    inside the regime where Guattery-Miller's analysis applies.
    """

    tree_height = 3
    path_length = 20


@register_dataset("tree_cross_path_medium")
class TreeCrossPathMedium(_TreeCrossPathBase):
    """h=4, q=25. p=62, n=1550."""

    tree_height = 4
    path_length = 25


@register_dataset("tree_cross_path_large")
class TreeCrossPathLarge(_TreeCrossPathBase):
    """h=5, q=50. p=126, n=6300."""

    tree_height = 5
    path_length = 50


# A smoke check / construction-verification entrypoint lives in
# scripts/guattery_miller_check.py to avoid the double-registration trap that
# `python -m data.tree_cross_path` triggers (the `data` package imports this
# module during boot, then __main__ re-executes the module body).
