#!/usr/bin/env python3
"""Empirical check: does k-means rounding escape sweep cut's failure on the
tree-cross-path graph from Guattery & Miller (1998)?

For each registered tree-cross-path size, run sweep cut once and three
k-means variants over five seeds against the bottom-k eigenvectors of
L_sym. Record conductance, cut size, and (where it makes sense) ARI vs.
the optimum bisection. Emit:

  results/tree_cross_path.parquet
  experiments/plots/guattery_miller_conductance.png
  experiments/guattery_miller_check.md     (auto-generated draft)

Run from repo root:
    PYTHONPATH=src python3 scripts/guattery_miller_check.py
"""

from __future__ import annotations

import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from algorithms._spectral import normalized_laplacian, spectral_embedding
from algorithms.cheeger_sweep_cut import CheegerSweepCut
from core.graph import Graph
from core.registry import DATASETS  # noqa: F401  -- import side-effects register datasets
from data import tree_cross_path  # noqa: F401  -- side-effect: register the three sizes
from data.tree_cross_path import (
    TreeCrossPathInfo,
    make_double_tree,
    make_tree_cross_path,
    optimum_partition,
)
from evals.conductance import compute_conductance


SIZES = ["tree_cross_path_small", "tree_cross_path_medium", "tree_cross_path_large"]
SEEDS = [0, 1, 2, 3, 4]
KMEANS_VARIANTS = ["kmeans_unnormalized", "kmeans_njw", "kmeans_unnormalized_1d"]
N_INIT = 10


# ----- factor-eigenvalue check ---------------------------------------------


def _comb_lambda2(A: sp.csr_matrix) -> float:
    """Second-smallest eigenvalue of the *combinatorial* Laplacian.

    Used only for the factor-ordering verification (Check 3); the algorithms
    themselves use L_sym via spectral_embedding.
    """
    n = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel()
    L = sp.diags(deg) - A
    M = (deg.max() + 1) * sp.identity(n) - L
    vals_shifted, _ = eigsh(M, k=2, which="LA", tol=1e-9, v0=np.ones(n))
    vals = (deg.max() + 1) - vals_shifted
    return float(np.sort(vals)[1])


def factor_eigenvalues(info: TreeCrossPathInfo) -> tuple[float, float]:
    """Return (lambda_2 of double tree, lambda_2 of path), combinatorial L."""
    DT, _, _ = make_double_tree(info.tree_height)
    A_dt = nx.to_scipy_sparse_array(DT, format="csr", dtype=float)
    A_path = nx.to_scipy_sparse_array(nx.path_graph(info.q), format="csr", dtype=float)
    return _comb_lambda2(A_dt), _comb_lambda2(A_path)


def fiedler_constancy_ratio(emb_v2: np.ndarray, info: TreeCrossPathInfo) -> float:
    """How constant is v_2 within each tree-copy, relative to its variation
    across path positions? <<1 means the Fiedler vector is essentially a
    function of the path coordinate alone — the Guattery-Miller regime."""
    M = emb_v2.reshape(info.p, info.q)
    var_within_tree_copy = M.var(axis=0).mean()
    var_across_path = M.var(axis=1).mean()
    return float(var_within_tree_copy / max(var_across_path, 1e-30))


# ----- cut classification --------------------------------------------------


def _cut_edges_mask(A: sp.csr_matrix, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (rows, cols) of upper-triangle cut edges."""
    coo = A.tocoo()
    upper = coo.row < coo.col
    rows = coo.row[upper]
    cols = coo.col[upper]
    cut = labels[rows] != labels[cols]
    return rows[cut], cols[cut]


def classify_cut(
    A: sp.csr_matrix, labels: np.ndarray, info: TreeCrossPathInfo
) -> tuple[str, int, int, int]:
    """Classify the cut. Returns (cut_class, n_cut, n_cross_copy, n_within_copy).

    cut_class:
      "between_copies"   the cut is between path positions only (no edge inside
                         a single double-tree copy is cut). |partial S| ~ p
                         per cut path position. This is the Guattery-Miller
                         bad cut.
      "within_copies"    every cut edge lies inside some double-tree copy.
                         Includes the optimum (cut q root-to-root edges, one
                         per copy).
      "mixed"            both kinds appear.
      "no_cut"           degenerate.
    """
    rows, cols = _cut_edges_mask(A, labels)
    n_cut = len(rows)
    if n_cut == 0:
        return "no_cut", 0, 0, 0
    # A "copy of the double tree" lives at a fixed path-position j. Vertex
    # u*q + j has path-position = vertex_index % q. An edge goes *between*
    # copies (from path position j to j' != j) iff `% q` differs across its
    # endpoints. Tree edges within copy j have the same `% q`; path-edge
    # rungs between copies have differing `% q`.
    copy_r = rows % info.q
    copy_c = cols % info.q
    cross_copy = copy_r != copy_c
    n_cross = int(cross_copy.sum())
    n_within = n_cut - n_cross
    if n_cross > 0 and n_within == 0:
        return "between_copies", n_cut, n_cross, n_within
    if n_within > 0 and n_cross == 0:
        return "within_copies", n_cut, n_cross, n_within
    return "mixed", n_cut, n_cross, n_within


# ----- algorithm wrappers --------------------------------------------------


def kmeans_labels(X: np.ndarray, seed: int) -> np.ndarray:
    return KMeans(
        n_clusters=2, n_init=N_INIT, random_state=seed
    ).fit_predict(X)


def row_l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


# ----- main sweep ----------------------------------------------------------


def run_one_size(name: str) -> tuple[list[dict], dict]:
    """Run all algorithms on a single registered tree-cross-path dataset.

    Returns (rows, summary) where summary holds per-size metadata used by the
    verification checks in the report.
    """
    cls = DATASETS[name]
    ds = cls()
    graph, target = ds.load()
    info: TreeCrossPathInfo = ds.info

    A = graph.adjacency

    # Optimum bisection: cut the q root-to-root edges. Computed analytically
    # from the structure (no algorithm involved).
    opt_labels = optimum_partition(info)
    phi_opt = compute_conductance(A, opt_labels)
    opt_class, n_opt, n_opt_cross, n_opt_within = classify_cut(A, opt_labels, info)
    assert n_opt == info.q, (n_opt, info.q)
    assert opt_class == "within_copies"

    # One eigendecomposition; reused across algorithms so the variance comes
    # only from k-means initialization.
    t0 = time.perf_counter()
    emb = spectral_embedding(A, k=3, row_normalize=False, seed=0)
    t_eig = time.perf_counter() - t0

    # Factor-eigenvalue check (the gating Guattery-Miller condition).
    l2_dt, l2_path = factor_eigenvalues(info)
    fiedler_ratio = fiedler_constancy_ratio(emb.embedding[:, 1], info)

    summary = {
        "name": name,
        "h": info.tree_height,
        "p": info.p,
        "q": info.q,
        "n": info.n,
        "phi_opt": phi_opt,
        "lambda2_double_tree": l2_dt,
        "lambda2_path": l2_path,
        "factor_ratio_path_over_tree": l2_path / l2_dt,
        "fiedler_constancy_ratio": fiedler_ratio,
        "bottom_eigenvalues_Lsym": emb.eigenvalues.tolist(),
        "t_eigendecomp_s": t_eig,
    }

    rows: list[dict] = []

    # Sweep cut: deterministic.
    t0 = time.perf_counter()
    sweep_labels = CheegerSweepCut(seed=0).fit_predict(graph, k=2)
    t_sweep = time.perf_counter() - t0
    sweep_phi = compute_conductance(A, sweep_labels)
    sweep_cls, sweep_n_cut, sweep_n_cross, sweep_n_within = classify_cut(A, sweep_labels, info)
    rows.append({
        "dataset": name, "h": info.tree_height, "p": info.p, "q": info.q, "n": info.n,
        "algorithm": "sweep_cut", "seed": -1,
        "phi": sweep_phi,
        "phi_opt": phi_opt,
        "phi_ratio_to_opt": sweep_phi / phi_opt if phi_opt > 0 else float("inf"),
        "n_cut_edges": sweep_n_cut,
        "n_cross_copy_cuts": sweep_n_cross,
        "n_within_copy_cuts": sweep_n_within,
        "cut_class": sweep_cls,
        "ari_vs_opt": float(adjusted_rand_score(opt_labels, sweep_labels)),
        "ari_vs_sweep": 1.0,
        "side_size_min": int(min((sweep_labels == 0).sum(), (sweep_labels == 1).sum())),
        "runtime_s": t_sweep,
    })

    X_unnorm = emb.embedding[:, 1:3]
    X_njw = row_l2_normalize(emb.embedding[:, 1:3].copy())
    X_1d = emb.embedding[:, 1:2]
    X_for = {
        "kmeans_unnormalized": X_unnorm,
        "kmeans_njw": X_njw,
        "kmeans_unnormalized_1d": X_1d,
    }

    for variant in KMEANS_VARIANTS:
        for seed in SEEDS:
            t0 = time.perf_counter()
            labels = kmeans_labels(X_for[variant], seed)
            t_alg = time.perf_counter() - t0
            phi = compute_conductance(A, labels)
            cls_, n_cut, n_cross, n_within = classify_cut(A, labels, info)
            rows.append({
                "dataset": name, "h": info.tree_height, "p": info.p, "q": info.q, "n": info.n,
                "algorithm": variant, "seed": seed,
                "phi": phi,
                "phi_opt": phi_opt,
                "phi_ratio_to_opt": phi / phi_opt if phi_opt > 0 else float("inf"),
                "n_cut_edges": n_cut,
                "n_cross_copy_cuts": n_cross,
                "n_within_copy_cuts": n_within,
                "cut_class": cls_,
                "ari_vs_opt": float(adjusted_rand_score(opt_labels, labels)),
                "ari_vs_sweep": float(adjusted_rand_score(sweep_labels, labels)),
                "side_size_min": int(min((labels == 0).sum(), (labels == 1).sum())),
                "runtime_s": t_alg,
            })
    return rows, summary


def run_grid() -> tuple[pd.DataFrame, list[dict]]:
    rows: list[dict] = []
    summaries: list[dict] = []
    t_start = time.perf_counter()
    for name in SIZES:
        t0 = time.perf_counter()
        size_rows, summary = run_one_size(name)
        rows.extend(size_rows)
        summaries.append(summary)
        elapsed = time.perf_counter() - t0
        print(
            f"  {name:<28} h={summary['h']} p={summary['p']:>4} q={summary['q']:>3} "
            f"n={summary['n']:>5} phi_opt={summary['phi_opt']:.5f} "
            f"factor_ratio={summary['factor_ratio_path_over_tree']:.3f} "
            f"v2_const_ratio={summary['fiedler_constancy_ratio']:.3g} ({elapsed:.1f}s)"
        )
    print(f"  total elapsed: {time.perf_counter() - t_start:.1f}s")
    return pd.DataFrame(rows), summaries


def make_plot(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel: phi/phi_opt by algorithm vs n.
    ax = axes[0]
    algs = ["sweep_cut"] + KMEANS_VARIANTS
    cmap = plt.get_cmap("tab10")
    for i, alg in enumerate(algs):
        sub = df[df["algorithm"] == alg]
        d = sub.groupby("n")["phi_ratio_to_opt"]
        mean = d.mean()
        lo = d.min()
        hi = d.max()
        ax.plot(mean.index, mean.values, marker="o", color=cmap(i), label=alg)
        ax.fill_between(mean.index, lo.values, hi.values, alpha=0.15, color=cmap(i))
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1, label="optimum")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n = p · q")
    ax.set_ylabel("phi(algorithm) / phi(optimum)")
    ax.set_title("Conductance ratio to the optimum cut\n(line = seed mean, band = seed min/max)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9)

    # Right panel: |partial S| / p across algorithms — does the cut size
    # match Guattery-Miller's prediction (~p) or the optimum (~q)?
    ax = axes[1]
    for i, alg in enumerate(algs):
        sub = df[df["algorithm"] == alg]
        # Group by n; report mean |partial S|
        d = sub.groupby(["n", "p", "q"])["n_cut_edges"].mean().reset_index()
        ax.plot(d["n"], d["n_cut_edges"], marker="o", color=cmap(i), label=alg)
    # Reference lines: y=p (G-M prediction) and y=q (optimum) for each n.
    refs = df.groupby("n")[["p", "q"]].first().reset_index()
    ax.plot(refs["n"], refs["p"], color="black", linestyle="--", linewidth=1, label="p (G-M prediction)")
    ax.plot(refs["n"], refs["q"], color="black", linestyle=":", linewidth=1, label="q (optimum)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n = p · q")
    ax.set_ylabel("|∂S|  (cut edge count)")
    ax.set_title("Cut size: between-copy (~p) vs within-copy (~q)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9)

    fig.suptitle("Tree-cross-path: does k-means escape sweep cut's failure?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_report(
    df: pd.DataFrame, summaries: list[dict], out_path: pathlib.Path
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Tree-cross-path empirical check (auto-draft)")
    lines.append("")
    lines.append("_Auto-generated by `scripts/guattery_miller_check.py`._")
    lines.append("")

    lines.append("## Construction sanity")
    lines.append("")
    lines.append("| dataset | h | p | q | n | λ₂(tree) | λ₂(path) | path/tree | v2 const. ratio |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for s in summaries:
        lines.append(
            f"| {s['name']} | {s['h']} | {s['p']} | {s['q']} | {s['n']} "
            f"| {s['lambda2_double_tree']:.4g} | {s['lambda2_path']:.4g} "
            f"| {s['factor_ratio_path_over_tree']:.3f} "
            f"| {s['fiedler_constancy_ratio']:.3g} |"
        )
    lines.append("")
    lines.append("`v2 const. ratio` = (variance across tree-vertices for fixed path-position) / "
                 "(variance across path-vertices for fixed tree-vertex). <<1 means the "
                 "Fiedler vector is approximately constant within each tree copy — the "
                 "Guattery-Miller regime.")
    lines.append("")

    lines.append("## Per-algorithm conductance ratio to the optimum")
    lines.append("")
    pivot = (
        df.groupby(["dataset", "algorithm"])["phi_ratio_to_opt"]
        .agg(["mean", "min", "max"]).round(3)
    )
    lines.append("```")
    lines.append(pivot.to_string())
    lines.append("```")
    lines.append("")

    lines.append("## Cut size and class")
    lines.append("")
    cut_summary = (
        df.groupby(["dataset", "algorithm"])
        .agg(
            n_cut_mean=("n_cut_edges", "mean"),
            n_cut_min=("n_cut_edges", "min"),
            n_cut_max=("n_cut_edges", "max"),
        )
        .round(2)
    )
    lines.append("```")
    lines.append(cut_summary.to_string())
    lines.append("```")
    lines.append("")
    lines.append("Cut-class distribution per algorithm:")
    lines.append("")
    cls_summary = (
        df.groupby(["dataset", "algorithm", "cut_class"])
        .size()
        .unstack(fill_value=0)
    )
    lines.append("```")
    lines.append(cls_summary.to_string())
    lines.append("```")
    lines.append("")

    lines.append("## Plot")
    lines.append("")
    lines.append("![](plots/guattery_miller_conductance.png)")
    lines.append("")
    lines.append("## Raw results")
    lines.append("")
    lines.append("`results/tree_cross_path.parquet` — one row per (dataset, algorithm, seed).")
    out_path.write_text("\n".join(lines))


def main() -> None:
    print("Running guattery_miller_check sweep…")
    df, summaries = run_grid()

    out_parquet = REPO_ROOT / "results" / "tree_cross_path.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    print(f"Wrote {len(df)} rows to {out_parquet}")

    plot_path = REPO_ROOT / "experiments" / "plots" / "guattery_miller_conductance.png"
    make_plot(df, plot_path)
    print(f"Wrote plot to {plot_path}")

    report_path = REPO_ROOT / "experiments" / "guattery_miller_check.md"
    write_report(df, summaries, report_path)
    print(f"Wrote draft report to {report_path}")


if __name__ == "__main__":
    main()
