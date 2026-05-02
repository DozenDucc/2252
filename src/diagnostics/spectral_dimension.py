"""Spectral dimension of a cut.

Given a graph G with symmetric normalized Laplacian L_sym = I - D^{-1/2} A D^{-1/2}
and an indicator chi_S of S subset V, define the *normalized cut indicator*

    g_bar_S = D^{1/2} chi_S / || D^{1/2} chi_S ||

(a unit vector in R^n). If f_1, f_2, ... are the L_sym eigenvectors sorted by
eigenvalue, the *spectral dimension at tolerance epsilon* is the smallest d
such that the residual

    r(d) = || g_bar_S - sum_{i <= d} <g_bar_S, f_i> f_i ||^2

falls below epsilon.

This module exposes:
- ``normalized_cut_indicator(A, labels)`` -> g_bar_S (unit vector).
- ``spectral_profile(A, labels, eigvecs)`` -> array r(d) for d = 1..k.
- ``spectral_dimension(A, labels, eigvecs, eps)`` -> smallest d with r(d) <= eps,
  or None if k eigenvectors are not enough to drop the residual that low.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp


def normalized_cut_indicator(A: sp.csr_matrix, labels: np.ndarray) -> np.ndarray:
    """Return g_bar_S = D^{1/2} chi_S / ||D^{1/2} chi_S|| for one side of a 2-way cut.

    ``labels`` may carry any two distinct integer labels; we treat the side
    matching ``labels[0]`` as S. (For a 2-way cut, choice of side doesn't
    matter for the spectral profile because g_bar_{V\\S} is just g_bar_S
    rotated; all inner products with f_i pick up an overall sign flip and
    squared inner products are invariant.)
    """
    deg = np.asarray(A.sum(axis=1)).ravel()
    sqrt_deg = np.sqrt(deg)
    side = labels == labels[0]
    chi = side.astype(float)
    g = sqrt_deg * chi
    norm = float(np.linalg.norm(g))
    if norm == 0:
        raise ValueError("normalized cut indicator has zero norm (empty side?)")
    return g / norm


def spectral_profile(
    A: sp.csr_matrix,
    labels: np.ndarray,
    eigvecs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the cumulative residual norm r(d) for d = 1..k.

    Parameters
    ----------
    A : sparse adjacency
    labels : 0/1-style label vector for a 2-way partition
    eigvecs : (n, k) array; columns are L_sym eigenvectors sorted by eigenvalue

    Returns
    -------
    inner_products : (k,) array of <g_bar_S, f_i>
    residuals : (k,) array of r(d) = 1 - sum_{i<=d} <g_bar_S, f_i>^2

    Note that we use the identity || g_bar_S ||^2 = 1 to compute the residual
    in O(k) rather than projecting and re-subtracting.
    """
    g = normalized_cut_indicator(A, labels)
    inner = eigvecs.T @ g  # shape (k,)
    cum_proj_sq = np.cumsum(inner**2)
    residuals = np.maximum(0.0, 1.0 - cum_proj_sq)
    return inner, residuals


def spectral_dimension(
    A: sp.csr_matrix,
    labels: np.ndarray,
    eigvecs: np.ndarray,
    eps: float = 0.01,
) -> Optional[int]:
    """Smallest d such that r(d) <= eps. Returns None if no such d <= k exists."""
    _, residuals = spectral_profile(A, labels, eigvecs)
    below = np.flatnonzero(residuals <= eps)
    if below.size == 0:
        return None
    return int(below[0]) + 1  # 1-indexed
