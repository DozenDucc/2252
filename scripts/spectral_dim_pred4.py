#!/usr/bin/env python3
"""Prediction 4: Generalization across Cartesian products.

Test the framework on a small library of Cartesian product graphs, predicting
- d* = NJW dim at which the slab cut is first recovered (= eigvec index of
  first eigvec aligned with the slab indicator, minus 1 for f_1 drop)
- whether sweep cut succeeds or fails

Test cases:
- path_x_doubletree (q=25, h=4): from existing experiment, control
- path_x_cycle (q=20, p=40)
- cycle_x_doubletree (q=20, h=4)
- path_x_path_unequal (q=10, p=50)
- doubletree_x_doubletree (h1=h2=4)
- doubletree_x_doubletree_unequal (h1=3, h2=5)
- path_x_doubletree_perturbed (eps in {0.001, 0.01})

Outputs:
- experiments/spectral_dim_predictions/pred4.md
- results/spectral_dim_predictions/pred4.parquet
- experiments/plots/spectral_dim_pred4.png
"""

from __future__ import annotations

import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

from algorithms._spectral import normalized_laplacian, spectral_embedding
from algorithms.cheeger_sweep_cut import CheegerSweepCut
from core.graph import Graph
from core.registry import DATASETS  # noqa: F401
from data import cartesian_products  # noqa: F401
from data.cartesian_products import (
    PathCrossCycle, CycleCrossDoubleTree, PathCrossPathUnequal,
    DoubleTreeCrossDoubleTree, DoubleTreeCrossDoubleTreeUnequal,
    PathCrossDoubleTreePerturbed, make_path_x_doubletree,
)
from data.tree_cross_path import TreeCrossPathMedium, optimum_partition
from diagnostics.spectral_dimension import normalized_cut_indicator, spectral_profile
from evals.conductance import compute_conductance


PRED_DIR = REPO_ROOT / "experiments" / "spectral_dim_predictions"
PARQUET_DIR = REPO_ROOT / "results" / "spectral_dim_predictions"
PLOT_DIR = REPO_ROOT / "experiments" / "plots"


DIMENSIONS = [2, 3, 4, 5, 6, 8, 10]
SEED = 0


def lsym_bottom_k(A: sp.csr_matrix, k: int) -> tuple[np.ndarray, np.ndarray]:
    L_sym, _ = normalized_laplacian(A)
    n = L_sym.shape[0]
    M = 2.0 * sp.identity(n, format="csr") - L_sym
    rng = np.random.default_rng(0)
    v0 = rng.standard_normal(n)
    vals_shifted, vecs = eigsh(M, k=k, which="LA", tol=1e-9, v0=v0)
    vals = 2.0 - vals_shifted
    order = np.argsort(vals)
    return np.clip(vals[order], 0.0, None), vecs[:, order]


def slab_template_along_axis(A1: sp.csr_matrix, A2: sp.csr_matrix, slab_axis: int) -> np.ndarray:
    """Build the predicted slab indicator template up to normalization.

    For G = H1 sq H2 in path-major (i*|H2| + j) layout:
    - slab_axis = 0 (split H1): template = v_2(H1) ⊗ 1_H2
    - slab_axis = 1 (split H2): template = 1_H1 ⊗ v_2(H2)

    Both factor Fiedlers are computed as the L_sym v_2 of the factor.
    """
    if slab_axis == 0:
        v2_h1, _ = lsym_bottom_k(A1, 3)
        _, vecs = lsym_bottom_k(A1, 3)
        v = vecs[:, 1]
        n2 = A2.shape[0]
        out = np.kron(v, np.ones(n2))
    elif slab_axis == 1:
        _, vecs = lsym_bottom_k(A2, 3)
        v = vecs[:, 1]
        n1 = A1.shape[0]
        out = np.kron(np.ones(n1), v)
    else:
        raise ValueError("slab_axis must be 0 or 1")
    return out / np.linalg.norm(out)


def run_one_case(name: str, A: sp.csr_matrix, A1: sp.csr_matrix, A2: sp.csr_matrix,
                 slab_axis: int | None, slab_labels: np.ndarray | None,
                 phi_proxy_optimum: float | None = None) -> dict:
    """Run all algorithms and return measurements."""
    n = A.shape[0]
    n1, n2 = A1.shape[0], A2.shape[0]
    K = 12
    # Use the canonical spectral_embedding for the product graph (matches
    # the algorithm wrappers); use the local lsym_bottom_k for the small
    # factor graphs (cost is trivial there).
    g_emb = spectral_embedding(A, k=K, row_normalize=False, seed=SEED)
    G_vals = g_emb.eigenvalues
    G_vecs = g_emb.embedding
    H1_vals, _ = lsym_bottom_k(A1, min(4, A1.shape[0] - 1))
    H2_vals, _ = lsym_bottom_k(A2, min(4, A2.shape[0] - 1))

    # Predicted d* eigvec index (1-indexed) = first eigvec aligned with slab template.
    if slab_axis is not None:
        template = slab_template_along_axis(A1, A2, slab_axis)
        overlaps = np.abs(G_vecs.T @ template)
        d_star_eigvec = None
        for i in range(K):
            if overlaps[i] > 0.5:
                d_star_eigvec = i + 1
                break
        d_star_njw = None if d_star_eigvec is None else d_star_eigvec - 1
    else:
        overlaps = np.zeros(K)
        d_star_eigvec = None
        d_star_njw = None

    # Phi(slab) — when slab_axis defined.
    phi_slab = None
    if slab_labels is not None:
        try:
            phi_slab = float(compute_conductance(A, slab_labels))
        except Exception:
            phi_slab = None

    # Sweep cut.
    graph = Graph(adjacency=A, num_nodes=n, name=name)
    sweep_labels = CheegerSweepCut(seed=SEED).fit_predict(graph, k=2)
    phi_sweep = float(compute_conductance(A, sweep_labels))

    # k-means at varying NJW dim.
    deg = np.asarray(A.sum(axis=1)).ravel()
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    F = G_vecs * d_inv_sqrt[:, None]

    kmeans_phi = {}
    kmeans_labels_dict = {}
    for d in DIMENSIONS:
        if d + 1 > F.shape[1]:
            break
        X = F[:, 1:d + 1]
        labels = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit_predict(X)
        kmeans_phi[d] = float(compute_conductance(A, labels))
        kmeans_labels_dict[d] = labels

    # Best-of-all-runs as proxy optimum (used when no analytic slab is provided).
    candidates = [phi_sweep] + list(kmeans_phi.values())
    if phi_slab is not None:
        candidates.append(phi_slab)
    if phi_proxy_optimum is not None:
        candidates.append(phi_proxy_optimum)
    phi_best = float(min(candidates))

    # Spectral profile of the slab cut (or proxy).
    sp_resid = None
    if slab_labels is not None:
        try:
            _, sp_resid = spectral_profile(A, slab_labels, G_vecs)
            sp_resid = sp_resid.tolist()
        except Exception:
            sp_resid = None

    # Empirical slab-recovery dim: smallest NJW dim with phi <= 1.05 * phi_slab.
    # If no slab, use phi_best as denominator.
    denom = phi_slab if phi_slab is not None else phi_best
    empirical_recovery = None
    for d in sorted(kmeans_phi.keys()):
        if denom > 0 and kmeans_phi[d] / denom <= 1.05:
            empirical_recovery = d
            break

    return {
        "name": name,
        "n": n,
        "n1": n1,
        "n2": n2,
        "G_eigvals_top4": G_vals[:4].tolist(),
        "H1_eigvals_top4": H1_vals[:4].tolist(),
        "H2_eigvals_top4": H2_vals[:4].tolist(),
        "lambda2_H1": float(H1_vals[1]),
        "lambda2_H2": float(H2_vals[1]),
        "slab_axis": slab_axis,
        "slab_template_overlaps": overlaps.tolist(),
        "d_star_eigvec_predicted": d_star_eigvec,
        "d_star_njw_predicted": d_star_njw,
        "d_star_njw_empirical": empirical_recovery,
        "phi_slab": phi_slab,
        "phi_sweep": phi_sweep,
        "phi_sweep_ratio": phi_sweep / denom if denom else None,
        "phi_best_overall": phi_best,
        "kmeans_phi": kmeans_phi,
        "kmeans_ratio": {d: phi / denom for d, phi in kmeans_phi.items()} if denom else {},
        "spectral_residual": sp_resid,
    }


def main() -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    cases = []

    # 1. path_x_doubletree (control). Use TreeCrossPathMedium-style.
    A_path, A_dt, A, slab_lbl = make_path_x_doubletree(q=25, tree_height=4)
    cases.append({
        "name": "path_x_doubletree (q=25, h=4)",
        "A": A, "A1": A_path, "A2": A_dt,
        "slab_axis": 1, "slab_labels": slab_lbl,
        "predicted_d_star_njw": 2,  # from prediction 1/2: f_3 → NJW d=2
        "predicted_sweep": "fail",
    })

    # 2. path_x_cycle.
    ds = PathCrossCycle(q=20, p=40)
    g, t = ds.load()
    A1, A2 = ds.factors
    cases.append({
        "name": "path_x_cycle (q=20, p=40)",
        "A": g.adjacency, "A1": A1, "A2": A2,
        "slab_axis": 0, "slab_labels": t,
        "predicted_d_star_njw": 2,  # task spec said d*=3 (eigvec) → NJW d=2
        "predicted_sweep": "fail",
    })

    # 3. cycle_x_doubletree.
    ds = CycleCrossDoubleTree(q=20, tree_height=4)
    g, t = ds.load()
    A1, A2 = ds.factors
    cases.append({
        "name": "cycle_x_doubletree (q=20, h=4)",
        "A": g.adjacency, "A1": A1, "A2": A2,
        "slab_axis": 1, "slab_labels": t,
        "predicted_d_star_njw": 2,
        "predicted_sweep": "fail or ill-defined",
    })

    # 4. path_x_path_unequal.
    ds = PathCrossPathUnequal(q=10, p=50)
    g, t = ds.load()
    A1, A2 = ds.factors
    cases.append({
        "name": "path_x_path_unequal (q=10, p=50)",
        "A": g.adjacency, "A1": A1, "A2": A2,
        "slab_axis": 1, "slab_labels": t,
        "predicted_d_star_njw": 2,
        "predicted_sweep": "fail",
    })

    # 5. doubletree x doubletree (equal sizes).
    ds = DoubleTreeCrossDoubleTree(h1=4, h2=4)
    g, t = ds.load()
    A1, A2 = ds.factors
    cases.append({
        "name": "doubletree_x_doubletree (h1=h2=4)",
        "A": g.adjacency, "A1": A1, "A2": A2,
        "slab_axis": 0, "slab_labels": t,
        "predicted_d_star_njw": 1,  # eigvec f_2 (NJW d=1) — both axes are tree
        "predicted_sweep": "succeed",
    })

    # 6. doubletree x doubletree (unequal: h1=3, h2=5 → p1=14, p2=62).
    ds = DoubleTreeCrossDoubleTreeUnequal(h1=3, h2=5)
    g, t = ds.load()
    A1, A2 = ds.factors
    cases.append({
        "name": "doubletree_x_doubletree_unequal (h1=3, h2=5)",
        "A": g.adjacency, "A1": A1, "A2": A2,
        "slab_axis": 1, "slab_labels": t,
        "predicted_d_star_njw": 2,
        "predicted_sweep": "fail",
    })

    # 7a. path_x_doubletree_perturbed (eps=0.001).
    ds = PathCrossDoubleTreePerturbed(q=25, tree_height=4, epsilon=0.001, seed=0)
    g, t = ds.load()
    A1, A2 = ds.factors
    cases.append({
        "name": "path_x_doubletree_perturbed (eps=0.001)",
        "A": g.adjacency, "A1": A1, "A2": A2,
        "slab_axis": 1, "slab_labels": t,
        "predicted_d_star_njw": None,
        "predicted_sweep": "TBD",
    })

    # 7b. path_x_doubletree_perturbed (eps=0.01).
    ds = PathCrossDoubleTreePerturbed(q=25, tree_height=4, epsilon=0.01, seed=0)
    g, t = ds.load()
    A1, A2 = ds.factors
    cases.append({
        "name": "path_x_doubletree_perturbed (eps=0.01)",
        "A": g.adjacency, "A1": A1, "A2": A2,
        "slab_axis": 1, "slab_labels": t,
        "predicted_d_star_njw": None,
        "predicted_sweep": "TBD",
    })

    print(f"Running {len(cases)} test cases...")
    results = []
    for case in cases:
        t0 = time.perf_counter()
        r = run_one_case(case["name"], case["A"], case["A1"], case["A2"],
                          case["slab_axis"], case["slab_labels"])
        r["predicted_d_star_njw_explicit"] = case["predicted_d_star_njw"]
        r["predicted_sweep"] = case["predicted_sweep"]
        elapsed = time.perf_counter() - t0
        print(f"  {case['name']:<55} d*_pred(slab)={r['d_star_njw_predicted']} "
              f"d*_emp={r['d_star_njw_empirical']} sweep_ratio="
              f"{r['phi_sweep_ratio']:.3f} ({elapsed:.1f}s)")
        results.append(r)

    # Long-form parquet.
    rows = []
    for r in results:
        for d, phi in r["kmeans_phi"].items():
            rows.append({
                "name": r["name"],
                "n": r["n"],
                "method": "njw_kmeans",
                "d": d,
                "phi": phi,
                "phi_slab": r["phi_slab"],
                "phi_ratio": r["kmeans_ratio"].get(d),
                "d_star_njw_predicted": r["d_star_njw_predicted"],
                "d_star_njw_empirical": r["d_star_njw_empirical"],
            })
        rows.append({
            "name": r["name"],
            "n": r["n"],
            "method": "sweep_cut",
            "d": 2,
            "phi": r["phi_sweep"],
            "phi_slab": r["phi_slab"],
            "phi_ratio": r["phi_sweep_ratio"],
            "d_star_njw_predicted": r["d_star_njw_predicted"],
            "d_star_njw_empirical": r["d_star_njw_empirical"],
        })
    df = pd.DataFrame(rows)
    df.to_parquet(PARQUET_DIR / "pred4.parquet", index=False)

    # Plot.
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(results):
        c = cmap(i % 10)
        ds = sorted(r["kmeans_phi"].keys())
        ratios = [r["kmeans_ratio"].get(d) for d in ds]
        ax.plot(ds, ratios, marker="o", color=c,
                label=f"{r['name']} (d*_pred={r['d_star_njw_predicted']}, "
                      f"emp={r['d_star_njw_empirical']})")
    ax.axhline(1.0, color="grey", linestyle="--")
    ax.set_xlabel("NJW embedding dim d (uses f_2..f_{d+1})")
    ax.set_ylabel("Φ(k-means) / Φ(slab or proxy)")
    ax.set_yscale("log")
    ax.set_title("Prediction 4: NJW k-means conductance ratio across Cartesian products")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    plot_path = PLOT_DIR / "spectral_dim_pred4.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {plot_path}")

    # Verdict: count how many of the 6 non-perturbed cases match (predicted d*
    # within ±1 of empirical). Pass if at least 4 match.
    non_perturbed = [r for r in results if "perturbed" not in r["name"]]
    matches = []
    for r in non_perturbed:
        pred = r["d_star_njw_predicted"]
        emp = r["d_star_njw_empirical"]
        if pred is None or emp is None:
            matches.append(False)
        else:
            matches.append(abs(pred - emp) <= 1)
    n_matches = sum(matches)
    n_total = len(non_perturbed)
    if n_matches >= 4:
        verdict = f"PASS ({n_matches}/{n_total} non-perturbed cases match)"
    elif n_matches >= 2:
        verdict = f"PARTIAL ({n_matches}/{n_total} non-perturbed cases match)"
    else:
        verdict = f"FAIL ({n_matches}/{n_total} non-perturbed cases match)"

    md = []
    md.append("# Prediction 4: Generalization across Cartesian products")
    md.append("")
    md.append("**Verdict:** " + verdict)
    md.append("")
    md.append("## Procedure")
    md.append("")
    md.append("Build a small library of Cartesian-product graphs. For each, compute the predicted ")
    md.append("d* (NJW dim where the slab cut is first recovered) by aligning the slab template ")
    md.append("(1 ⊗ v_2(H_2) for slab along axis 1, v_2(H_1) ⊗ 1 for axis 0) with G's L_sym ")
    md.append("eigenvectors. Compare to the empirical recovery dim across NJW k-means at d ∈ {2,3,...,10}.")
    md.append("")
    md.append("## Per-case summary")
    md.append("")
    md.append("| name | n | λ_2(H_1) | λ_2(H_2) | slab axis | d*_njw pred | d*_njw emp | match (±1) | sweep ratio | sweep predicted |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for r, m in zip(non_perturbed, matches):
        md.append(
            f"| {r['name']} | {r['n']} | {r['lambda2_H1']:.4f} | {r['lambda2_H2']:.4f} | "
            f"{r['slab_axis']} | {r['d_star_njw_predicted']} | {r['d_star_njw_empirical']} | "
            f"{'✓' if m else '✗'} | {r['phi_sweep_ratio']:.3f} | {r['predicted_sweep']} |"
        )
    md.append("")
    md.append("Perturbed cases (no analytic prediction):")
    md.append("")
    perturbed = [r for r in results if "perturbed" in r["name"]]
    md.append("| name | n | d*_njw pred | d*_njw emp | sweep ratio |")
    md.append("|---|---|---|---|---|")
    for r in perturbed:
        md.append(
            f"| {r['name']} | {r['n']} | {r['d_star_njw_predicted']} | "
            f"{r['d_star_njw_empirical']} | {r['phi_sweep_ratio']:.3f} |"
        )
    md.append("")
    md.append("## NJW k-means conductance ratios")
    md.append("")
    pivot = df[df["method"] == "njw_kmeans"].pivot(index="name", columns="d", values="phi_ratio")
    md.append("```")
    md.append(pivot.round(3).to_string())
    md.append("```")
    md.append("")
    md.append("## Plot")
    md.append("")
    md.append("![](plots/spectral_dim_pred4.png)")
    md.append("")
    md.append("## One-sentence finding")
    md.append("")
    md.append(f"{n_matches}/{n_total} of the non-perturbed Cartesian-product test cases have ")
    md.append("predicted d*_njw matching empirical d*_njw within ±1; see table for which ")
    md.append("families violate the prediction.")

    md_path = PRED_DIR / "pred4.md"
    md_path.write_text("\n".join(md))
    print(f"Wrote {md_path}")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
