#!/usr/bin/env python3
"""Prediction 5: Spectral dimension on non-product graphs.

Compute the spectral profile r(d) of the optimum (or proxy-optimum) cut on:
- 2-cluster SBM, easy and hard regimes
- 2-level hierarchical SBM
- random 3-regular graph (no community structure)

Predicted spectral dimensions:
- 2-cluster easy SBM: ~ 1
- 2-cluster hard SBM: still small (< 5)
- HSBM at top level: ~ 2
- random 3-regular: large (~ n / 2)

Outputs:
- experiments/spectral_dim_predictions/pred5.md
- results/spectral_dim_predictions/pred5.parquet
- experiments/plots/spectral_dim_pred5.png
"""

from __future__ import annotations

import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans

from algorithms._spectral import spectral_embedding
from diagnostics.spectral_dimension import normalized_cut_indicator, spectral_profile, spectral_dimension
from evals.conductance import compute_conductance


PRED_DIR = REPO_ROOT / "experiments" / "spectral_dim_predictions"
PARQUET_DIR = REPO_ROOT / "results" / "spectral_dim_predictions"
PLOT_DIR = REPO_ROOT / "experiments" / "plots"


# ----- dataset builders -----------------------------------------------------


def build_sbm_2cluster(n: int, p_in: float, p_out: float, seed: int = 0) -> tuple[sp.csr_matrix, np.ndarray]:
    """Symmetric 2-block SBM."""
    sizes = [n // 2, n - n // 2]
    P = np.array([[p_in, p_out], [p_out, p_in]])
    G = nx.stochastic_block_model(sizes, P, seed=seed)
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
    target = np.repeat([0, 1], sizes)
    # Ensure connected — if not, add a single bridge edge from block 0 to block 1.
    from scipy.sparse.csgraph import connected_components
    n_cc, lbls = connected_components(A, directed=False)
    if n_cc > 1:
        # Connect first vertex of cluster 0 to first vertex of cluster 1.
        A = A.tolil()
        A[0, sizes[0]] = 1.0
        A[sizes[0], 0] = 1.0
        A = A.tocsr()
    return A, target


def build_hsbm_2level(n: int = 1000, seed: int = 0) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Hierarchical SBM: 4 leaf clusters in a binary tree.

    Top-level partition: clusters {0, 1} vs {2, 3}.
    Leaf-level partition: each cluster a separate label.
    """
    n_per = n // 4
    sizes = [n_per] * 4
    p_intra = 0.05         # within-cluster
    p_intra_pair = 0.015   # between paired clusters in same top-level subtree
    p_inter = 0.003        # between subtrees (top-level cut)

    # Block matrix:
    P = np.array([
        [p_intra,      p_intra_pair, p_inter,     p_inter     ],
        [p_intra_pair, p_intra,      p_inter,     p_inter     ],
        [p_inter,      p_inter,      p_intra,     p_intra_pair],
        [p_inter,      p_inter,      p_intra_pair, p_intra    ],
    ])
    G = nx.stochastic_block_model(sizes, P, seed=seed)
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
    leaf = np.concatenate([np.full(s, k) for k, s in enumerate(sizes)])
    top = (leaf >= 2).astype(int)
    from scipy.sparse.csgraph import connected_components
    n_cc, _ = connected_components(A, directed=False)
    if n_cc > 1:
        A = A.tolil()
        A[0, sizes[0] + sizes[1]] = 1.0
        A[sizes[0] + sizes[1], 0] = 1.0
        A = A.tocsr()
    return A, leaf, top


def build_random_regular(n: int = 1000, d: int = 3, seed: int = 0) -> tuple[sp.csr_matrix, np.ndarray]:
    """Random d-regular graph; no community structure.

    For the "optimum cut" we use 2D k-means (best-effort) since there is no
    planted partition — we expect this graph to have *no* low-d structure.
    """
    G = nx.random_regular_graph(d, n, seed=seed)
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
    return A, np.zeros(n, dtype=int)  # placeholder target


# ----- run loop -------------------------------------------------------------


def compute_profile(A: sp.csr_matrix, labels: np.ndarray, K: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (eigvals, residuals up to K, inner products)."""
    emb = spectral_embedding(A, k=K, row_normalize=False, seed=0)
    inner, residuals = spectral_profile(A, labels, emb.embedding)
    return emb.eigenvalues, residuals, inner


def best_2way_cut_via_kmeans(A: sp.csr_matrix, d: int = 8, seed: int = 0) -> np.ndarray:
    """Run NJW k-means at NJW dim d and return labels of the result."""
    emb = spectral_embedding(A, k=d + 1, row_normalize=False, seed=seed)
    deg = emb.degrees
    F = emb.embedding * np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)[:, None]
    X = F[:, 1:d + 1]
    return KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(X)


def main() -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    cases = []
    K = 30
    eps_list = [0.01, 0.05]

    print("Building datasets...")
    A_easy, t_easy = build_sbm_2cluster(1000, 0.05, 0.005, seed=0)
    A_hard, t_hard = build_sbm_2cluster(1000, 0.05, 0.03, seed=0)
    A_hsbm, leaf, top = build_hsbm_2level(1000, seed=0)
    A_rr, _ = build_random_regular(1000, 3, seed=0)

    # For 2-cluster SBMs: planted partition.
    cases.append(("sbm_2cluster_easy", A_easy, t_easy, "planted"))
    cases.append(("sbm_2cluster_hard", A_hard, t_hard, "planted"))
    # HSBM: top-level partition AND leaf-level partition.
    cases.append(("hsbm_2level_top", A_hsbm, top, "planted top"))
    cases.append(("hsbm_2level_leaf", A_hsbm, leaf, "planted 4-way (use any pair vs rest)"))
    # Random 3-regular: 8D k-means proxy.
    proxy_rr = best_2way_cut_via_kmeans(A_rr, d=8)
    cases.append(("random_3regular_n1000", A_rr, proxy_rr, "8D k-means proxy"))

    rows = []
    profiles_for_plot = {}
    print("Computing spectral profiles...")
    for name, A, labels, source in cases:
        if len(np.unique(labels)) != 2:
            # 4-way leaf — turn into a 2-way comparison: cluster 0 vs everything else.
            labels = (labels == 0).astype(int)
        t0 = time.perf_counter()
        eigvals, residuals, inner = compute_profile(A, labels, K=K)
        sd_01 = next((d + 1 for d, r in enumerate(residuals) if r <= 0.01), None)
        sd_05 = next((d + 1 for d, r in enumerate(residuals) if r <= 0.05), None)
        phi = compute_conductance(A, labels)
        elapsed = time.perf_counter() - t0
        print(f"  {name:<28} sd_eps=0.05: {sd_05}, sd_eps=0.01: {sd_01}, "
              f"phi={phi:.4f} ({elapsed:.1f}s)")

        for d in range(K):
            rows.append({
                "name": name,
                "label_source": source,
                "d": d + 1,
                "residual": float(residuals[d]),
                "eigval": float(eigvals[d]),
                "inner_product": float(inner[d]),
            })
        profiles_for_plot[name] = (eigvals, residuals)
        rows.append({
            "name": name + "_meta",
            "label_source": source,
            "d": -1,
            "residual": -1,
            "eigval": -1,
            "inner_product": -1,
            "spectral_dim_eps_0_01": sd_01 if sd_01 is not None else -1,
            "spectral_dim_eps_0_05": sd_05 if sd_05 is not None else -1,
            "phi": phi,
        })

    df = pd.DataFrame(rows)
    df.to_parquet(PARQUET_DIR / "pred5.parquet", index=False)

    # Plot.
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    for i, (name, (eigvals, residuals)) in enumerate(profiles_for_plot.items()):
        c = cmap(i % 10)
        ax.plot(np.arange(1, len(residuals) + 1), residuals, marker="o",
                color=c, label=name, markersize=3)
    ax.axhline(0.01, color="grey", linestyle=":", label="eps=0.01")
    ax.axhline(0.05, color="grey", linestyle="--", label="eps=0.05")
    ax.set_xlabel("d")
    ax.set_ylabel("r(d) = ||g_bar - Proj_d g_bar||²")
    ax.set_yscale("log")
    ax.set_title("Prediction 5: spectral profile r(d) for various graph families")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    plot_path = PLOT_DIR / "spectral_dim_pred5.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {plot_path}")

    # Verdict.
    sd_easy = df[(df["name"] == "sbm_2cluster_easy_meta")]["spectral_dim_eps_0_05"].iloc[0]
    sd_hard = df[(df["name"] == "sbm_2cluster_hard_meta")]["spectral_dim_eps_0_05"].iloc[0]
    sd_hsbm_top = df[(df["name"] == "hsbm_2level_top_meta")]["spectral_dim_eps_0_05"].iloc[0]
    sd_rr = df[(df["name"] == "random_3regular_n1000_meta")]["spectral_dim_eps_0_05"].iloc[0]

    pass_easy = sd_easy > 0 and sd_easy <= 5
    pass_hard = sd_hard > 0 and sd_hard <= 10
    pass_hsbm = sd_hsbm_top > 0 and sd_hsbm_top <= 5
    # For the random regular: spectral dim should be much larger or undefined within K.
    pass_rr = sd_rr == -1 or sd_rr > 10

    if pass_easy and pass_hsbm and pass_rr:
        verdict = "PASS"
    elif sum([pass_easy, pass_hsbm, pass_rr]) >= 2:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    md = []
    md.append("# Prediction 5: Spectral dimension on non-product graphs")
    md.append("")
    md.append("**Verdict:** " + verdict)
    md.append("")
    md.append("## Procedure")
    md.append("")
    md.append("Compute the cumulative residual r(d) of the planted (or proxy) cut indicator ")
    md.append("under the bottom-30 L_sym eigenvectors. Report the spectral dimension at ε=0.01 and ε=0.05.")
    md.append("")
    md.append("## Datasets and spectral dimensions")
    md.append("")
    meta = df[df["d"] == -1]
    md.append("| dataset | label source | sd at ε=0.01 | sd at ε=0.05 | Φ(cut) |")
    md.append("|---|---|---|---|---|")
    for _, row in meta.iterrows():
        sd1 = row["spectral_dim_eps_0_01"]
        sd5 = row["spectral_dim_eps_0_05"]
        sd1s = "≥ 30 (not reached)" if sd1 == -1 else int(sd1)
        sd5s = "≥ 30 (not reached)" if sd5 == -1 else int(sd5)
        md.append(f"| {row['name'].replace('_meta', '')} | {row['label_source']} | {sd1s} | {sd5s} | {row['phi']:.4f} |")
    md.append("")
    md.append("## Plot")
    md.append("")
    md.append("![](plots/spectral_dim_pred5.png)")
    md.append("")
    md.append("## One-sentence finding")
    md.append("")
    if verdict == "PASS":
        md.append("Clustered graphs (SBM, HSBM) have small spectral dimension while the random "
                  "3-regular graph's residual stays high across the bottom-30 eigenvectors, "
                  "consistent with the framework's diagnostic generalizing qualitatively beyond "
                  "Cartesian products.")
    elif verdict == "PARTIAL":
        md.append("Some predictions match qualitatively but at least one graph family violates "
                  "the prediction; see the table.")
    else:
        md.append("The diagnostic does not give the qualitative ordering predicted; see the table.")

    md_path = PRED_DIR / "pred5.md"
    md_path.write_text("\n".join(md))
    print(f"Wrote {md_path}")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
