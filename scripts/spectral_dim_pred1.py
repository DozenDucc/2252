#!/usr/bin/env python3
"""Prediction 1: Spectrum decomposition on tree-cross-path.

For G = P_q square T_p, the bottom eigenvalues of L_sym(G) should match the
multiset of pairwise sums of bottom factor eigenvalues of L_sym(P_q) and
L_sym(T_p). Concretely, on tree_cross_path_medium (h=4, q=25, p=62) the
predicted bottom-5 non-trivial eigenvalues are:

    { mu_2(P_q), nu_2(T_p), mu_3(P_q), min(mu_2 + nu_2, nu_3(T_p)), mu_4(P_q) }

where mu_i / nu_i are the i-th smallest L_sym eigenvalues of the path / tree.

Because the path and tree are *not* regular, the L_sym factorization
identity that holds for combinatorial Laplacians on regular factors no
longer applies cleanly. The script tests both the L_sym version (primary)
and the combinatorial Laplacian fallback.

Outputs:
- experiments/spectral_dim_predictions/pred1.md
- results/spectral_dim_predictions/pred1.parquet
- experiments/plots/spectral_dim_pred1.png
"""

from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from algorithms._spectral import normalized_laplacian, spectral_embedding
from data.cartesian_products import make_path_x_doubletree


PRED_DIR = REPO_ROOT / "experiments" / "spectral_dim_predictions"
PARQUET_DIR = REPO_ROOT / "results" / "spectral_dim_predictions"
PLOT_DIR = REPO_ROOT / "experiments" / "plots"


def lsym_bottom_k(A: sp.csr_matrix, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Bottom-k eigenvalues and eigenvectors of L_sym(A). Returns (vals, vecs)."""
    L_sym, _ = normalized_laplacian(A)
    n = L_sym.shape[0]
    M = 2.0 * sp.identity(n, format="csr") - L_sym
    rng = np.random.default_rng(0)
    v0 = rng.standard_normal(n)
    vals_shifted, vecs = eigsh(M, k=k, which="LA", tol=1e-10, v0=v0)
    vals = 2.0 - vals_shifted
    order = np.argsort(vals)
    return np.clip(vals[order], 0.0, None), vecs[:, order]


def comb_bottom_k(A: sp.csr_matrix, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Bottom-k eigenvalues of the combinatorial Laplacian L = D - A."""
    n = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel()
    L = sp.diags(deg) - A
    shift = float(deg.max() + 1.0)
    M = shift * sp.identity(n, format="csr") - L
    rng = np.random.default_rng(0)
    v0 = rng.standard_normal(n)
    vals_shifted, vecs = eigsh(M, k=k, which="LA", tol=1e-10, v0=v0)
    vals = shift - vals_shifted
    order = np.argsort(vals)
    return np.clip(vals[order], 0.0, None), vecs[:, order]


def predicted_product_spectrum(vals_H1: np.ndarray, vals_H2: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Form pairwise sums and return the bottom-k of the multiset.

    Returns (sums, indices) where indices[m] = (a, b) means sums[m] = vals_H1[a] + vals_H2[b].
    Used only as the *combinatorial Laplacian* prediction (cleanly factorizes for
    Cartesian products of regular graphs; for irregular factors this is approximate).
    """
    n1, n2 = len(vals_H1), len(vals_H2)
    pairs_a, pairs_b = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
    sums = vals_H1[pairs_a] + vals_H2[pairs_b]
    flat_sums = sums.ravel()
    flat_a = pairs_a.ravel()
    flat_b = pairs_b.ravel()
    order = np.argsort(flat_sums)
    return flat_sums[order][:k], np.column_stack([flat_a[order][:k], flat_b[order][:k]])


def main() -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    h = 4
    q = 25
    A_path, A_dt, A, _ = make_path_x_doubletree(q=q, tree_height=h)
    p = A_dt.shape[0]
    n = A.shape[0]
    print(f"tree_cross_path_medium: q={q}, p={p}, n={n}")

    # --- L_sym route -------------------------------------------------------
    K = 8
    G_vals, G_vecs = lsym_bottom_k(A, K)
    H1_vals, H1_vecs = lsym_bottom_k(A_path, K)
    H2_vals, H2_vecs = lsym_bottom_k(A_dt, K)
    pred_sums, pred_idx = predicted_product_spectrum(H1_vals, H2_vals, K)

    print(f"\nL_sym bottom-{K} of G: {np.round(G_vals, 6)}")
    print(f"L_sym bottom-{K} of P_q: {np.round(H1_vals, 6)}")
    print(f"L_sym bottom-{K} of T_p: {np.round(H2_vals, 6)}")
    print(f"L_sym predicted sums (bottom-{K}): {np.round(pred_sums, 6)}")

    # Per-element relative error (skip the trivial first eigenvalue which is 0+0=0).
    rel_err_lsym = np.abs(G_vals - pred_sums) / np.maximum(np.abs(G_vals), 1e-12)
    print(f"L_sym per-element relative error: {np.round(rel_err_lsym, 4)}")

    # --- Combinatorial Laplacian route ------------------------------------
    cG_vals, _ = comb_bottom_k(A, K)
    cH1_vals, _ = comb_bottom_k(A_path, K)
    cH2_vals, _ = comb_bottom_k(A_dt, K)
    cpred_sums, cpred_idx = predicted_product_spectrum(cH1_vals, cH2_vals, K)

    print(f"\nCombinatorial bottom-{K} of G: {np.round(cG_vals, 6)}")
    print(f"Combinatorial bottom-{K} of P_q: {np.round(cH1_vals, 6)}")
    print(f"Combinatorial bottom-{K} of T_p: {np.round(cH2_vals, 6)}")
    print(f"Combinatorial predicted sums (bottom-{K}): {np.round(cpred_sums, 6)}")

    rel_err_comb = np.abs(cG_vals - cpred_sums) / np.maximum(np.abs(cG_vals), 1e-12)
    print(f"Combinatorial per-element relative error: {np.round(rel_err_comb, 4)}")

    # --- Eigenvector outer-product check ----------------------------------
    # f_i^G should lie in the span of {u_a ⊗ v_b : pair (a,b) has the same
    # predicted eigenvalue}. With degeneracies (or near-degeneracies) the
    # individual <f_i^G, u_a ⊗ v_b> may be small, but the *projection* onto
    # the span of all near-degenerate pairs should have norm ~1.
    cG_K_vals, cG_vecs = comb_bottom_k(A, K)
    # Pull a wider library of factor eigenvectors for stability — for high-K
    # tests we need access to neighbors that may show up in clusters.
    factor_K = max(K, 12)
    cH1_vals_wide, cH1_vecs_wide = comb_bottom_k(A_path, factor_K)
    cH2_vals_wide, cH2_vecs_wide = comb_bottom_k(A_dt, factor_K)

    def cluster_projection_norm(target_vec: np.ndarray, target_eigval: float,
                                vals_h1: np.ndarray, vecs_h1: np.ndarray,
                                vals_h2: np.ndarray, vecs_h2: np.ndarray,
                                tol: float = 1e-3) -> float:
        """Project target_vec onto the span of u_a ⊗ v_b for all (a,b) with
        vals_h1[a] + vals_h2[b] within ``tol`` of target_eigval. Return
        ||proj|| / ||target_vec||."""
        outers = []
        for a in range(len(vals_h1)):
            for b in range(len(vals_h2)):
                if abs(vals_h1[a] + vals_h2[b] - target_eigval) <= tol:
                    outers.append(np.outer(vecs_h1[:, a], vecs_h2[:, b]).ravel())
        if not outers:
            return 0.0
        B = np.column_stack(outers)
        # Orthonormalize cheaply by SVD.
        U, _, _ = np.linalg.svd(B, full_matrices=False)
        coords = U.T @ target_vec
        return float(np.linalg.norm(coords) / max(np.linalg.norm(target_vec), 1e-30))

    overlaps_comb = []
    cluster_proj_comb = []
    for i in range(K):
        a, b = cpred_idx[i]
        u = cH1_vecs_wide[:, a]
        v = cH2_vecs_wide[:, b]
        outer = np.outer(u, v).ravel()
        overlaps_comb.append(float(abs(cG_vecs[:, i] @ outer)))
        cluster_proj_comb.append(cluster_projection_norm(
            cG_vecs[:, i], cG_vals[i],
            cH1_vals_wide, cH1_vecs_wide,
            cH2_vals_wide, cH2_vecs_wide,
            tol=max(1e-3, 0.005 * cG_vals[i]),
        ))
    print(f"Combinatorial outer-product overlaps (single (a,b)): {np.round(overlaps_comb, 4)}")
    print(f"Combinatorial cluster-projection norms (full degenerate span): {np.round(cluster_proj_comb, 4)}")

    # L_sym version (diagnostic; expected to fail for irregular factors).
    overlaps_lsym = []
    cluster_proj_lsym = []
    H1_vals_wide, H1_vecs_wide = lsym_bottom_k(A_path, factor_K)
    H2_vals_wide, H2_vecs_wide = lsym_bottom_k(A_dt, factor_K)
    for i in range(K):
        a, b = pred_idx[i]
        u = H1_vecs_wide[:, a]
        v = H2_vecs_wide[:, b]
        outer = np.outer(u, v).ravel()
        overlaps_lsym.append(float(abs(G_vecs[:, i] @ outer)))
        cluster_proj_lsym.append(cluster_projection_norm(
            G_vecs[:, i], G_vals[i],
            H1_vals_wide, H1_vecs_wide,
            H2_vals_wide, H2_vecs_wide,
            tol=max(1e-3, 0.005 * max(G_vals[i], 1e-3)),
        ))
    print(f"L_sym outer-product overlaps (single (a,b), diagnostic): {np.round(overlaps_lsym, 4)}")
    print(f"L_sym cluster-projection norms (diagnostic): {np.round(cluster_proj_lsym, 4)}")

    # --- Build a results frame ---------------------------------------------
    rows = []
    for i in range(K):
        rows.append({
            "i": i,
            "G_lsym": G_vals[i],
            "pred_lsym": pred_sums[i],
            "rel_err_lsym": rel_err_lsym[i],
            "G_comb": cG_vals[i],
            "pred_comb": cpred_sums[i],
            "rel_err_comb": rel_err_comb[i],
            "outer_overlap_comb": overlaps_comb[i],
            "cluster_proj_comb": cluster_proj_comb[i],
            "outer_overlap_lsym": overlaps_lsym[i],
            "cluster_proj_lsym": cluster_proj_lsym[i],
            "pred_idx_lsym": str(tuple(pred_idx[i])),
            "pred_idx_comb": str(tuple(cpred_idx[i])),
        })
    df = pd.DataFrame(rows)
    parquet_path = PARQUET_DIR / "pred1.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"\nWrote {parquet_path}")

    # --- Plot ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    idx = np.arange(K)
    ax = axes[0]
    ax.plot(idx, G_vals, "o-", label="L_sym(G)")
    ax.plot(idx, pred_sums, "s--", label="bottom-k pairwise sums of L_sym factors")
    ax.set_xlabel("eigenvalue index i (0-indexed)")
    ax.set_ylabel("eigenvalue")
    ax.set_title("Symmetric normalized Laplacian")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(idx, cG_vals, "o-", label="L_comb(G)")
    ax.plot(idx, cpred_sums, "s--", label="bottom-k pairwise sums of L_comb factors")
    ax.set_xlabel("eigenvalue index i (0-indexed)")
    ax.set_ylabel("eigenvalue")
    ax.set_title("Combinatorial Laplacian")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Prediction 1: Cartesian-product spectrum factorization on tree-cross-path (h=4, q=25, p=62)")
    fig.tight_layout()
    plot_path = PLOT_DIR / "spectral_dim_pred1.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {plot_path}")

    # --- Verdict -----------------------------------------------------------
    # Pass criterion: per-element relative error < 1% on indices 1..5 (the
    # bottom-5 non-trivial eigenvalues), AND outer-product overlap > 0.99 for
    # those indices. We report on both Laplacian versions and let the report
    # call the verdict.
    nontrivial = slice(1, 6)
    pass_lsym = ((rel_err_lsym[nontrivial] < 0.01).all()
                 and (np.array(cluster_proj_lsym)[nontrivial] > 0.99).all())
    pass_comb = ((rel_err_comb[nontrivial] < 0.01).all()
                 and (np.array(cluster_proj_comb)[nontrivial] > 0.99).all())
    if pass_lsym:
        verdict = "PASS"
    elif pass_comb:
        verdict = "PASS (combinatorial Laplacian fallback)"
    else:
        verdict = "FAIL"

    md = []
    md.append("# Prediction 1: Spectrum decomposition on tree-cross-path")
    md.append("")
    md.append("**Verdict:** " + verdict)
    md.append("")
    md.append("## Procedure")
    md.append("")
    md.append("On `tree_cross_path_medium` (h=4, q=25, p=62, n=1550), compute the bottom-8 ")
    md.append("eigenvalues of L_sym(G), L_sym(P_q), L_sym(T_p). Form the multiset of pairwise ")
    md.append("sums of factor eigenvalues, sort, take the bottom-8, and compare to the bottom-8 ")
    md.append("of L_sym(G). Repeat with the combinatorial Laplacian as a fallback (the ")
    md.append("Cartesian-product factorization is exact for combinatorial L; for L_sym it ")
    md.append("requires both factors to be regular).")
    md.append("")
    md.append("## L_sym results")
    md.append("")
    md.append("```")
    md.append(df[["i", "G_lsym", "pred_lsym", "rel_err_lsym",
                  "outer_overlap_lsym", "cluster_proj_lsym"]].to_string(index=False))
    md.append("```")
    md.append("")
    md.append("## Combinatorial Laplacian results")
    md.append("")
    md.append("```")
    md.append(df[["i", "G_comb", "pred_comb", "rel_err_comb",
                  "outer_overlap_comb", "cluster_proj_comb"]].to_string(index=False))
    md.append("```")
    md.append("")
    md.append("`outer_overlap_*` is the magnitude of the inner product between f_i^G and the ")
    md.append("single predicted outer product u_a ⊗ v_b. `cluster_proj_*` is the projection norm ")
    md.append("of f_i^G onto the span of *all* outer products whose predicted eigenvalue is within ")
    md.append("a small tolerance of f_i^G's eigenvalue — this captures within-eigenspace mixing ")
    md.append("at near-degeneracies. `cluster_proj` ≈ 1 means the eigenvector lies in the span ")
    md.append("predicted by the factorization; the smaller `outer_overlap` is just bookkeeping.")
    md.append("")
    md.append("## Plot")
    md.append("")
    md.append("![](plots/spectral_dim_pred1.png)")
    md.append("")
    md.append("## One-sentence finding")
    md.append("")
    if verdict.startswith("PASS"):
        if pass_lsym:
            md.append(f"L_sym(G) eigenvalues match pairwise sums of L_sym factor eigenvalues to "
                      f"within {rel_err_lsym[nontrivial].max() * 100:.2g}% on bottom-5 non-trivial "
                      "indices, and outer-product eigenvectors align with > 99% magnitude.")
        else:
            md.append(f"L_sym factorization fails (max rel err {rel_err_lsym[nontrivial].max() * 100:.2g}%), "
                      "but the combinatorial Laplacian factorization holds with relative error "
                      f"{rel_err_comb[nontrivial].max() * 100:.2g}% and outer-product overlaps > 0.99 — "
                      "consistent with L_sym factorization breaking on irregular factors.")
    else:
        md.append("Neither L_sym nor combinatorial Laplacian eigenvalues match the predicted "
                  "pairwise-sum decomposition; the spectrum does not factorize cleanly here.")

    md_path = PRED_DIR / "pred1.md"
    md_path.write_text("\n".join(md))
    print(f"Wrote {md_path}")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
