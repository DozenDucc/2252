"""Dumbbell graph: two K_m cliques joined by a path of ell edges.

The dumbbell is the standard worst case for spectral graph partitioning:
the bridge cut has conductance ~1/m^2 while any cut inside a clique has
conductance >= 1/2. Sweep cut is known to find the bridge; whether
k-means does is the empirical question this file supports.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

# Vertex class codes returned alongside the adjacency matrix.
CLIQUE_A = 0
CLIQUE_B = 1
PATH_INTERNAL = 2


def make_dumbbell(m: int, ell: int) -> tuple[sp.csr_matrix, np.ndarray]:
    """Two K_m cliques joined by a path of ``ell`` edges.

    Vertex layout:
      0 .. m-1         clique A; vertex 0 is the path endpoint.
      m .. 2m-1        clique B; vertex m is the path endpoint.
      2m .. 2m+ell-2   internal path vertices, ordered along the path
                       from the clique-A side toward the clique-B side.
    Total n = 2m + ell - 1. For ell == 1 there are no internal path vertices
    and the path collapses to the single edge (0, m).

    Returns (A, vertex_class) where vertex_class[i] is one of CLIQUE_A,
    CLIQUE_B, PATH_INTERNAL.
    """
    if m < 2:
        raise ValueError(f"m must be >= 2; got {m}")
    if ell < 1:
        raise ValueError(f"ell must be >= 1; got {ell}")

    n = 2 * m + ell - 1

    rows: list[int] = []
    cols: list[int] = []

    # Clique A: complete graph on 0..m-1.
    for i in range(m):
        for j in range(i + 1, m):
            rows.append(i); cols.append(j)
            rows.append(j); cols.append(i)

    # Clique B: complete graph on m..2m-1.
    for i in range(m, 2 * m):
        for j in range(i + 1, 2 * m):
            rows.append(i); cols.append(j)
            rows.append(j); cols.append(i)

    # Path of ell edges joining vertex 0 (in A) to vertex m (in B), passing
    # through internal vertices 2m, 2m+1, ..., 2m+ell-2 in order.
    path_seq = [0] + list(range(2 * m, 2 * m + ell - 1)) + [m]
    for u, v in zip(path_seq[:-1], path_seq[1:]):
        rows.append(u); cols.append(v)
        rows.append(v); cols.append(u)

    data = np.ones(len(rows), dtype=np.float64)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    # Validation. These should be cheap and catch off-by-ones immediately.
    if (A != A.T).nnz != 0:
        raise AssertionError("dumbbell adjacency is not symmetric")
    if A.diagonal().sum() != 0:
        raise AssertionError("dumbbell adjacency has self-loops")
    n_components, _ = connected_components(A, directed=False)
    if n_components != 1:
        raise AssertionError(f"dumbbell is not connected: {n_components} components")

    vertex_class = np.empty(n, dtype=np.int8)
    vertex_class[:m] = CLIQUE_A
    vertex_class[m:2 * m] = CLIQUE_B
    vertex_class[2 * m:] = PATH_INTERNAL

    return A, vertex_class


def path_position(m: int, ell: int) -> np.ndarray:
    """Position of each vertex along the path, or -1 if not on the path.

    Position 0 is the clique-A endpoint (vertex 0), position ell is the
    clique-B endpoint (vertex m), and positions 1..ell-1 are the internal
    path vertices in order. All other vertices get -1.
    """
    n = 2 * m + ell - 1
    pos = np.full(n, -1, dtype=np.int64)
    pos[0] = 0
    pos[m] = ell
    for k, v in enumerate(range(2 * m, 2 * m + ell - 1), start=1):
        pos[v] = k
    return pos


if __name__ == "__main__":
    # Reproduces the smoke check baked into algorithms/_spectral.py: two K_20
    # joined by a single bridging edge between vertex 0 and vertex 20.
    A, cls = make_dumbbell(20, 1)
    assert A.shape == (40, 40)
    assert A.nnz == 2 * 20 * 19 + 2  # 2 * (K_20 directed edges) + (0,20) + (20,0)
    assert cls.tolist() == [CLIQUE_A] * 20 + [CLIQUE_B] * 20
    print(f"make_dumbbell(20, 1): n={A.shape[0]}, nnz={A.nnz}, ok")

    A, cls = make_dumbbell(20, 2)
    assert A.shape == (41, 41)
    assert cls[40] == PATH_INTERNAL
    assert A[0, 40] == 1.0 and A[40, 20] == 1.0
    print(f"make_dumbbell(20, 2): n={A.shape[0]}, nnz={A.nnz}, internal=[{40}], ok")
