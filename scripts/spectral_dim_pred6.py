#!/usr/bin/env python3
"""Prediction 6: Sweep on f_3 escapes (rounding-vs-dimensionality).

The framework says the escape on tree-cross-path is about *which* eigenvector
is rounded, not the rounding strategy. Sweep cut on f_3 should recover the
optimum; sweep on f_4 should not; 1D k-means on f_3 should also recover.

Outputs:
- experiments/spectral_dim_predictions/pred6.md
- results/spectral_dim_predictions/pred6.parquet
"""

from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from algorithms._spectral import spectral_embedding
from algorithms.cheeger_sweep_cut import CheegerSweepCut
from algorithms.sweep_on_kth import SweepOnKth
from data.tree_cross_path import TreeCrossPathMedium, optimum_partition
from evals.conductance import compute_conductance


PRED_DIR = REPO_ROOT / "experiments" / "spectral_dim_predictions"
PARQUET_DIR = REPO_ROOT / "results" / "spectral_dim_predictions"


def classify_axis(graph_meta: dict, labels: np.ndarray, q: int, p: int) -> str:
    """Heuristic: does the cut split by tree-vertex (good) or by path-position (bad)?

    Tree-major layout: vertex i*q + j has tree-vertex i in [0,p), path-position j
    in [0,q). M[i, j] is the label. M.var(axis=0) measures variance over tree
    vertices (for fixed path position) — large means labels depend on tree
    vertex, i.e., the cut is the optimum tree-axis cut. M.var(axis=1) measures
    variance over path positions — large means labels depend on path position,
    i.e., the bad sweep cut.
    """
    M = labels.reshape(p, q)  # rows: tree, cols: path
    var_over_tree_idx = M.var(axis=0).mean()    # large → cut depends on tree index
    var_over_path_idx = M.var(axis=1).mean()    # large → cut depends on path index
    if var_over_tree_idx > 0.1 and var_over_path_idx < 0.05:
        return "split by tree-vertex (optimum cut)"
    if var_over_path_idx > 0.1 and var_over_tree_idx < 0.05:
        return "split by path-position (Guattery-Miller bad cut)"
    return "mixed"


def main() -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    ds = TreeCrossPathMedium()
    graph, _ = ds.load()
    info = ds.info
    A = graph.adjacency
    opt = optimum_partition(info)
    phi_opt = compute_conductance(A, opt)
    print(f"phi_opt = {phi_opt:.6f}")

    rows = []

    # Sweep on f_2 (default cheeger sweep cut).
    sw2 = CheegerSweepCut(seed=0).fit_predict(graph, k=2)
    phi_sw2 = compute_conductance(A, sw2)
    rows.append({
        "method": "sweep_on_f2",
        "phi": phi_sw2,
        "phi_ratio": phi_sw2 / phi_opt,
        "cut_axis": classify_axis(graph.metadata, sw2, info.q, info.p),
    })

    # Sweep on f_3 and f_4.
    for k_eigvec in [3, 4]:
        sw = SweepOnKth(k_eigvec=k_eigvec, seed=0).fit_predict(graph, k=2)
        phi = compute_conductance(A, sw)
        rows.append({
            "method": f"sweep_on_f{k_eigvec}",
            "phi": phi,
            "phi_ratio": phi / phi_opt,
            "cut_axis": classify_axis(graph.metadata, sw, info.q, info.p),
        })

    # 1D k-means on f_2 and f_3.
    emb = spectral_embedding(A, k=4, row_normalize=False, seed=0)
    deg = emb.degrees
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    F = emb.embedding * d_inv_sqrt[:, None]  # L_rw eigvecs

    for k_eigvec in [2, 3]:
        X = F[:, k_eigvec - 1:k_eigvec]  # one column
        labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
        phi = compute_conductance(A, labels)
        rows.append({
            "method": f"kmeans_1d_on_f{k_eigvec}",
            "phi": phi,
            "phi_ratio": phi / phi_opt,
            "cut_axis": classify_axis(graph.metadata, labels, info.q, info.p),
        })

    # 2D k-means on (f_2, f_3) for completeness.
    X23 = F[:, 1:3]
    labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X23)
    phi = compute_conductance(A, labels)
    rows.append({
        "method": "kmeans_2d_f2_f3",
        "phi": phi,
        "phi_ratio": phi / phi_opt,
        "cut_axis": classify_axis(graph.metadata, labels, info.q, info.p),
    })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_parquet(PARQUET_DIR / "pred6.parquet", index=False)

    # Pass criteria.
    sw3_ratio = float(df.loc[df["method"] == "sweep_on_f3", "phi_ratio"].iloc[0])
    sw4_ratio = float(df.loc[df["method"] == "sweep_on_f4", "phi_ratio"].iloc[0])
    km1d_f3_ratio = float(df.loc[df["method"] == "kmeans_1d_on_f3", "phi_ratio"].iloc[0])
    pass_sw3 = sw3_ratio < 1.05
    pass_sw4 = sw4_ratio > 1.5
    pass_km3 = km1d_f3_ratio < 1.05

    if pass_sw3 and pass_sw4 and pass_km3:
        verdict = "PASS"
    elif pass_sw3 and pass_km3:
        verdict = "PARTIAL"  # sweep on f_4 also recovered, surprising but not fatal
    else:
        verdict = "FAIL"

    md = []
    md.append("# Prediction 6: Sweep on f_3 escapes")
    md.append("")
    md.append("**Verdict:** " + verdict)
    md.append("")
    md.append("## Procedure")
    md.append("")
    md.append("On `tree_cross_path_medium`, run sweep cut on f_2, f_3, f_4 (each is the L_rw ")
    md.append("eigenvector for the corresponding L_sym eigenvalue). Also run 1D k-means on f_2, f_3 ")
    md.append("and 2D k-means on (f_2, f_3) for comparison. Compare conductance ratio to the ")
    md.append("analytic optimum.")
    md.append("")
    md.append("## Results")
    md.append("")
    md.append("```")
    md.append(df.to_string(index=False))
    md.append("```")
    md.append("")
    md.append("## One-sentence finding")
    md.append("")
    if verdict == "PASS":
        md.append(f"Sweep on f_3 reaches Φ_ratio = {sw3_ratio:.3f}, sweep on f_4 fails with "
                  f"Φ_ratio = {sw4_ratio:.3f}, and 1D k-means on f_3 reaches "
                  f"Φ_ratio = {km1d_f3_ratio:.3f} — confirming the escape is about which "
                  "eigenvector is rounded, not the rounding scheme.")
    else:
        md.append(f"Sweep f_3 ratio = {sw3_ratio:.3f}, sweep f_4 ratio = {sw4_ratio:.3f}, "
                  f"1D k-means on f_3 ratio = {km1d_f3_ratio:.3f}.")

    md_path = PRED_DIR / "pred6.md"
    md_path.write_text("\n".join(md))
    print(f"Wrote {md_path}")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
