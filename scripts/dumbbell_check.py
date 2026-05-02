#!/usr/bin/env python3
"""Empirical check: do k-means and sweep cut differ on the dumbbell?

For each (m, ell) in a small grid, builds a dumbbell graph, runs sweep cut
once and three k-means variants over five seeds, and records conductance,
cut location, and ARI vs. sweep cut. Outputs:

  results/dumbbell.parquet
  experiments/plots/dumbbell_conductance.png
  experiments/dumbbell_check.md      (auto-generated draft summary)

Run from the repo root:
    PYTHONPATH=src python3 scripts/dumbbell_check.py
"""

from __future__ import annotations

import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover
    raise SystemExit(f"matplotlib required: {e}") from e

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
except ImportError as e:  # pragma: no cover
    raise SystemExit(f"scikit-learn required: {e}") from e

from algorithms._spectral import spectral_embedding
from algorithms.cheeger_sweep_cut import CheegerSweepCut
from core.graph import Graph
from data.dumbbell import (
    CLIQUE_A,
    CLIQUE_B,
    PATH_INTERNAL,
    make_dumbbell,
    path_position,
)
from evals.conductance import compute_conductance


M_VALUES = [10, 20, 50, 100]
ELL_VALUES = [2, 5, 10, 20, 50]
SEEDS = [0, 1, 2, 3, 4]
KMEANS_VARIANTS = ["kmeans_unnormalized", "kmeans_njw", "kmeans_unnormalized_1d"]
N_INIT = 10  # sklearn's standard "best of 10 inits" — realistic deploy.


def classify_cut(
    A,
    labels: np.ndarray,
    vertex_class: np.ndarray,
    pos: np.ndarray,
) -> tuple[str, int]:
    """Classify where the cut falls on the dumbbell.

    Returns (cut_class, n_cut_edges).

    cut_class is one of:
      "bridge_<i>"        single cut edge at path position i↔i+1
      "bridge_multi"      multiple cuts but all on the path
      "clique_interior_A" cuts entirely inside clique A (and possibly path)
      "clique_interior_B" cuts entirely inside clique B (and possibly path)
      "mixed"             cut edges in both cliques, or other combinations
    """
    A_coo = A.tocoo()
    upper = A_coo.row < A_coo.col
    rows = A_coo.row[upper]
    cols = A_coo.col[upper]
    cut_mask = labels[rows] != labels[cols]
    n_cut = int(cut_mask.sum())
    if n_cut == 0:
        return "no_cut", 0

    cut_rows = rows[cut_mask]
    cut_cols = cols[cut_mask]

    # An edge is a "path edge" iff both endpoints lie on the path
    # (clique endpoints have pos == 0 or pos == ell; internals have pos in
    # 1..ell-1; non-path vertices have pos == -1).
    on_path_r = pos[cut_rows] >= 0
    on_path_c = pos[cut_cols] >= 0
    is_path_edge = on_path_r & on_path_c

    # An edge is "inside clique A" iff both endpoints are in clique A.
    is_clique_A = (vertex_class[cut_rows] == CLIQUE_A) & (vertex_class[cut_cols] == CLIQUE_A)
    is_clique_B = (vertex_class[cut_rows] == CLIQUE_B) & (vertex_class[cut_cols] == CLIQUE_B)

    n_path = int(is_path_edge.sum())
    n_A = int(is_clique_A.sum())
    n_B = int(is_clique_B.sum())

    if n_path == n_cut:
        if n_cut == 1:
            # Identify which path edge: position i ↔ i+1.
            i = int(min(pos[cut_rows[0]], pos[cut_cols[0]]))
            return f"bridge_{i}", n_cut
        return "bridge_multi", n_cut
    if n_A > 0 and n_B == 0:
        return "clique_interior_A", n_cut
    if n_B > 0 and n_A == 0:
        return "clique_interior_B", n_cut
    return "mixed", n_cut


def kmeans_labels(X: np.ndarray, seed: int) -> np.ndarray:
    return KMeans(
        n_clusters=2, n_init=N_INIT, random_state=seed
    ).fit_predict(X)


def row_l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def run_grid() -> pd.DataFrame:
    rows = []
    t0 = time.perf_counter()
    for m in M_VALUES:
        for ell in ELL_VALUES:
            A, vertex_class = make_dumbbell(m, ell)
            n = A.shape[0]
            pos = path_position(m, ell)
            graph = Graph(adjacency=A, num_nodes=n, name=f"dumbbell_m{m}_ell{ell}")

            # One eigendecomposition per (m, ell); reused across all algorithms
            # so eigendecomposition variance is not a confound between them.
            emb = spectral_embedding(A, k=3, row_normalize=False, seed=0)

            X_unnorm = emb.embedding[:, 1:3]
            X_njw = row_l2_normalize(emb.embedding[:, 1:3].copy())
            X_1d = emb.embedding[:, 1:2]

            sweep_labels = CheegerSweepCut(seed=0).fit_predict(graph, k=2)
            sweep_phi = compute_conductance(A, sweep_labels)
            sweep_class, sweep_n_cut = classify_cut(A, sweep_labels, vertex_class, pos)

            rows.append({
                "m": m, "ell": ell, "n": n,
                "algorithm": "sweep_cut", "seed": -1,
                "phi": sweep_phi, "phi_ratio": 1.0,
                "cut_class": sweep_class, "n_cut_edges": sweep_n_cut,
                "ari_vs_sweep": 1.0,
            })

            for seed in SEEDS:
                for variant, X in zip(KMEANS_VARIANTS, [X_unnorm, X_njw, X_1d]):
                    labels = kmeans_labels(X, seed)
                    phi = compute_conductance(A, labels)
                    cls, n_cut = classify_cut(A, labels, vertex_class, pos)
                    rows.append({
                        "m": m, "ell": ell, "n": n,
                        "algorithm": variant, "seed": seed,
                        "phi": phi,
                        "phi_ratio": phi / sweep_phi if sweep_phi > 0 else float("inf"),
                        "cut_class": cls, "n_cut_edges": n_cut,
                        "ari_vs_sweep": float(adjusted_rand_score(sweep_labels, labels)),
                    })
            print(f"  m={m:>4}  ell={ell:>2}  n={n:>4}  sweep phi={sweep_phi:.4g}  "
                  f"cut={sweep_class}  ({time.perf_counter() - t0:.1f}s)")
    return pd.DataFrame(rows)


def make_plot(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(KMEANS_VARIANTS), figsize=(15, 4.5), sharey=True)
    cmap = plt.get_cmap("viridis")
    color_for = {ell: cmap(i / max(1, len(ELL_VALUES) - 1)) for i, ell in enumerate(ELL_VALUES)}

    for ax, alg in zip(axes, KMEANS_VARIANTS):
        sub = df[df["algorithm"] == alg]
        for ell in ELL_VALUES:
            d = sub[sub["ell"] == ell].groupby("m")["phi_ratio"]
            mean = d.mean()
            lo = d.min()
            hi = d.max()
            ax.fill_between(mean.index, lo.values, hi.values, alpha=0.18, color=color_for[ell])
            ax.plot(mean.index, mean.values, marker="o", color=color_for[ell], label=f"ell={ell}")
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("m (clique size)")
        ax.set_title(alg)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"$\phi_{\rm alg} / \phi_{\rm sweep}$")
    axes[-1].legend(loc="best", fontsize=9)
    fig.suptitle("Conductance ratio vs sweep cut on the dumbbell graph "
                 "(band = seed min/max, line = seed mean)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_report(df: pd.DataFrame, out_path: pathlib.Path) -> str:
    """Write a draft markdown report with the auto-emitted verdict."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    km = df[df["algorithm"].isin(KMEANS_VARIANTS)]

    # Decision rule from the plan.
    max_ratio = float(km["phi_ratio"].max())
    min_seed_ratio_per_cell = (
        km.groupby(["m", "ell", "algorithm"])["phi_ratio"].min()
    )
    any_cell_above_1p5_all_seeds = bool((min_seed_ratio_per_cell > 1.5).any())

    median_per_cell = km.groupby(["m", "ell", "algorithm"])["phi_ratio"].median().reset_index()
    monotone_in_m = False
    monotone_in_ell = False
    for alg in KMEANS_VARIANTS:
        sub = median_per_cell[median_per_cell["algorithm"] == alg]
        for ell in ELL_VALUES:
            row = sub[sub["ell"] == ell].sort_values("m")["phi_ratio"].values
            if len(row) > 1 and np.all(np.diff(row) >= -1e-9) and row[-1] > row[0] + 0.05:
                monotone_in_m = True
        for m in M_VALUES:
            row = sub[sub["m"] == m].sort_values("ell")["phi_ratio"].values
            if len(row) > 1 and np.all(np.diff(row) >= -1e-9) and row[-1] > row[0] + 0.05:
                monotone_in_ell = True

    if max_ratio <= 1.05:
        verdict = "DEAD"
        verdict_text = (
            "k-means matches sweep cut to within 5% across the entire grid. "
            "The dumbbell does not separate the two algorithms. "
            "Recommend pivoting to the Guattery–Miller path-tree-product graph "
            "or to SBM near the Kesten–Stigum threshold."
        )
    elif any_cell_above_1p5_all_seeds and (monotone_in_m or monotone_in_ell):
        verdict = "ALIVE"
        verdict_text = (
            "k-means is consistently worse than sweep cut by >1.5x at some "
            "(m, ell), and the gap grows with at least one of m or ell. The "
            "direction is alive — proceed with attempting an analytical proof."
        )
    else:
        verdict = "NOISY / INCONCLUSIVE"
        verdict_text = (
            "Some gap exists, but it is either seed-dependent or non-monotone "
            "in (m, ell). The 'separation' likely reflects k-means initialization "
            "variance rather than a fundamental algorithmic difference. Report "
            "this as a finding and pivot to a different candidate graph."
        )

    # Per-algorithm summary table: median phi_ratio over seeds.
    pivot = (
        km.groupby(["algorithm", "m", "ell"])["phi_ratio"].median()
        .unstack("ell")
    )

    cut_class_summary = (
        km.groupby(["algorithm", "cut_class"]).size().unstack(fill_value=0)
    )

    seed_var = (
        km.groupby(["m", "ell", "algorithm"])["phi_ratio"]
        .agg(["min", "max", "std"])
    )
    max_seed_std = float(seed_var["std"].max())

    lines: list[str] = []
    lines.append("# Dumbbell empirical check: k-means vs sweep cut")
    lines.append("")
    lines.append(f"_Auto-generated draft. Run via `PYTHONPATH=src python3 scripts/dumbbell_check.py`._")
    lines.append("")
    lines.append(f"## Verdict: **{verdict}**")
    lines.append("")
    lines.append(verdict_text)
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Grid: m ∈ {M_VALUES}, ell ∈ {ELL_VALUES}, seeds ∈ {SEEDS}.")
    lines.append(f"- ell convention: number of path edges (so ell=1 is a single bridge edge,")
    lines.append(f"  ell=2 has one internal path vertex, …; n = 2m + ell − 1).")
    lines.append(f"- Algorithms: sweep_cut (deterministic), kmeans_unnormalized (KMeans on")
    lines.append(f"  bottom-2 non-trivial L_sym eigenvectors, no row normalization),")
    lines.append(f"  kmeans_njw (same with post-drop row L2 normalization),")
    lines.append(f"  kmeans_unnormalized_1d (KMeans on Fiedler vector only).")
    lines.append(f"- KMeans uses n_init={N_INIT}, random_state=seed.")
    lines.append("")
    lines.append("## Q1. Are partitions different?")
    lines.append("")
    lines.append(f"- Maximum phi_ratio observed across the grid: **{max_ratio:.3f}**.")
    lines.append(f"- Cells where every seed gives phi_ratio > 1.5: "
                 f"{int((min_seed_ratio_per_cell > 1.5).sum())} of {len(min_seed_ratio_per_cell)}.")
    lines.append("")
    lines.append("### Median phi_ratio (over seeds), pivoted by ell:")
    lines.append("")
    lines.append("```")
    lines.append(pivot.to_string(float_format=lambda x: f"{x:.3f}"))
    lines.append("```")
    lines.append("")
    lines.append("## Q2. Where does k-means cut?")
    lines.append("")
    lines.append("Distribution of cut classes per algorithm:")
    lines.append("")
    lines.append("```")
    lines.append(cut_class_summary.to_string())
    lines.append("```")
    lines.append("")
    lines.append("## Q3. Does row normalization (NJW) matter?")
    lines.append("")
    for alg in KMEANS_VARIANTS:
        med = km[km["algorithm"] == alg]["phi_ratio"].median()
        mx = km[km["algorithm"] == alg]["phi_ratio"].max()
        lines.append(f"- **{alg}**: median ratio = {med:.3f}, max ratio = {mx:.3f}")
    lines.append("")
    lines.append("## Q4. Scaling")
    lines.append("")
    lines.append(f"- Median phi_ratio is monotone increasing in m (some ell): {monotone_in_m}")
    lines.append(f"- Median phi_ratio is monotone increasing in ell (some m): {monotone_in_ell}")
    lines.append("")
    lines.append("## Q5. Seed variance")
    lines.append("")
    lines.append(f"- Maximum across-seed std of phi_ratio in any cell: **{max_seed_std:.3f}**.")
    lines.append("- Per-cell (min, max, std) of phi_ratio across seeds:")
    lines.append("")
    lines.append("```")
    lines.append(seed_var.round(3).to_string())
    lines.append("```")
    lines.append("")
    lines.append("## Plot")
    lines.append("")
    lines.append("![](plots/dumbbell_conductance.png)")
    lines.append("")
    lines.append("## Raw results")
    lines.append("")
    lines.append("`results/dumbbell.parquet` — one row per (m, ell, algorithm, seed).")
    lines.append("")
    out_path.write_text("\n".join(lines))
    return verdict


def main() -> None:
    print("Running dumbbell_check sweep…")
    df = run_grid()
    out_parquet = REPO_ROOT / "results" / "dumbbell.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    print(f"Wrote {len(df)} rows to {out_parquet}")

    plot_path = REPO_ROOT / "experiments" / "plots" / "dumbbell_conductance.png"
    make_plot(df, plot_path)
    print(f"Wrote plot to {plot_path}")

    report_path = REPO_ROOT / "experiments" / "dumbbell_check.md"
    verdict = write_report(df, report_path)
    print(f"Wrote draft report to {report_path}")
    print(f"\nVerdict: {verdict}")


if __name__ == "__main__":
    main()
