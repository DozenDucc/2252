#!/usr/bin/env python3
"""Prediction 2: optimum cut indicator avoids f_2.

For the analytic optimum cut S* on tree_cross_path_medium, compute
<g_bar_{S*}, f_i> for i = 1..8 and the cumulative residual r(d).
The framework predicts:
- |<g_bar, f_2>| < 0.05 (Fiedler is path-axis, optimum is tree-axis)
- |<g_bar, f_3>| > 0.5  (the first tree-axis eigenvector picks up the optimum)
- r(3) < 0.05

Outputs:
- experiments/spectral_dim_predictions/pred2.md
- results/spectral_dim_predictions/pred2.parquet
- experiments/plots/spectral_dim_pred2.png
"""

from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from algorithms._spectral import spectral_embedding
from data.tree_cross_path import optimum_partition, TreeCrossPathInfo
from data.cartesian_products import make_path_x_doubletree
from diagnostics.spectral_dimension import normalized_cut_indicator, spectral_profile


PRED_DIR = REPO_ROOT / "experiments" / "spectral_dim_predictions"
PARQUET_DIR = REPO_ROOT / "results" / "spectral_dim_predictions"
PLOT_DIR = REPO_ROOT / "experiments" / "plots"


def main() -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    h = 4
    q = 25
    A_path, A_dt, A, _ = make_path_x_doubletree(q=q, tree_height=h)
    p = A_dt.shape[0]
    n = A.shape[0]
    info = TreeCrossPathInfo(
        tree_height=h, path_length=q, p=p, q=q, n=n,
        component_tree_size=(2 ** (h + 1)) - 1,
        left_root_index=0,
        right_root_index=p // 2,
        optimum_cut_size=q,
    )
    # Note: tree_cross_path uses i*q + j (tree-major), but our generic
    # cartesian_products factory uses i*p + j (path-major). We need the
    # optimum partition under the path-major layout used by `make_path_x_doubletree`.
    # Path-major: vertex i*p + j has path index i, tree index j.
    # Optimum: split on the tree axis (cut root-to-root edge in each path-copy
    # of the double tree). Tree index j in [0, p/2) is left side.
    target = np.empty(n, dtype=int)
    for i in range(q):
        for j in range(p):
            target[i * p + j] = 0 if j < p // 2 else 1

    K = 8
    emb = spectral_embedding(A, k=K, row_normalize=False, seed=0)
    eigvals = emb.eigenvalues
    eigvecs = emb.embedding

    inner, residuals = spectral_profile(A, target, eigvecs)
    print(f"L_sym eigvals (bottom-{K}): {np.round(eigvals, 6)}")
    print(f"<g_bar_S*, f_i> for i=1..{K}: {np.round(inner, 4)}")
    print(f"r(d) for d=1..{K}: {np.round(residuals, 6)}")

    rows = []
    for i in range(K):
        rows.append({
            "i": i + 1,
            "eigval": float(eigvals[i]),
            "inner_product": float(inner[i]),
            "abs_inner_product": float(abs(inner[i])),
            "residual_at_d": float(residuals[i]),
        })
    df = pd.DataFrame(rows)
    df.to_parquet(PARQUET_DIR / "pred2.parquet", index=False)

    abs_inner = np.abs(inner)
    pass_f2 = abs_inner[1] < 0.05
    pass_f3 = abs_inner[2] > 0.5
    pass_residual = residuals[2] < 0.05
    pass_others = (abs_inner[3:8] < 0.1).all()

    if pass_f2 and pass_f3 and pass_residual:
        verdict = "PASS"
    elif pass_f3 and pass_residual:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.bar(np.arange(1, K + 1), abs_inner)
    ax.axhline(0.05, color="grey", linestyle=":", label="0.05 threshold")
    ax.set_xlabel("eigenvector index i (1-indexed)")
    ax.set_ylabel("|<g_bar_S*, f_i>|")
    ax.set_title("Inner products of optimum cut indicator with bottom-8 eigvecs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(np.arange(1, K + 1), residuals, "o-")
    ax.axhline(0.05, color="grey", linestyle=":", label="0.05 threshold")
    ax.axhline(0.01, color="black", linestyle=":", label="0.01 threshold")
    ax.set_xlabel("d")
    ax.set_ylabel("r(d) = ||g_bar - Proj_d g_bar||²")
    ax.set_yscale("log")
    ax.set_title("Cumulative residual of optimum cut indicator")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Prediction 2: optimum cut indicator on tree_cross_path_medium")
    fig.tight_layout()
    plot_path = PLOT_DIR / "spectral_dim_pred2.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {plot_path}")

    md = []
    md.append("# Prediction 2: optimum cut indicator avoids f_2")
    md.append("")
    md.append("**Verdict:** " + verdict)
    md.append("")
    md.append("## Procedure")
    md.append("")
    md.append("On `tree_cross_path_medium`, compute `g_bar_S*` for the analytic optimum cut, ")
    md.append("then take inner products with the bottom-8 L_sym eigenvectors. Report the inner ")
    md.append("products and the cumulative residual r(d).")
    md.append("")
    md.append("## Inner products and residuals")
    md.append("")
    md.append("```")
    md.append(df.to_string(index=False))
    md.append("```")
    md.append("")
    md.append("## Plot")
    md.append("")
    md.append("![](plots/spectral_dim_pred2.png)")
    md.append("")
    md.append("## One-sentence finding")
    md.append("")
    if verdict == "PASS":
        md.append(f"|⟨g_bar, f_2⟩| = {abs_inner[1]:.4f} (predicted < 0.05), "
                  f"|⟨g_bar, f_3⟩| = {abs_inner[2]:.4f} (predicted > 0.5), and "
                  f"r(3) = {residuals[2]:.4f} (predicted < 0.05) — the cut indicator "
                  "lives almost entirely in span(f_1, f_3), not in span(f_1, f_2).")
    elif verdict == "PARTIAL":
        md.append(f"|⟨g_bar, f_3⟩| = {abs_inner[2]:.4f} and r(3) = {residuals[2]:.4f}, but "
                  f"|⟨g_bar, f_2⟩| = {abs_inner[1]:.4f} which exceeds the 0.05 threshold — "
                  "the cut is mostly in span(f_1, f_3) but with a non-negligible f_2 component.")
    else:
        md.append("The optimum cut indicator does not concentrate as predicted on f_3; "
                  f"|⟨g_bar, f_3⟩| = {abs_inner[2]:.4f}, r(3) = {residuals[2]:.4f}.")

    md_path = PRED_DIR / "pred2.md"
    md_path.write_text("\n".join(md))
    print(f"Wrote {md_path}")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
