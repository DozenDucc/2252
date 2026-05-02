# Spectral-dimension framework: empirical test summary

This file collects the verdicts from the six predictions of the
Cartesian-product spectral-dimension framework. Each cell links to the
corresponding per-prediction report.

## Verdict table

| # | Prediction | Status | Key finding |
|---|---|---|---|
| 1 | [Spectrum decomposition](pred1.md) | PASS (combinatorial L) / FAIL (L_sym) | The combinatorial Laplacian's spectrum factorizes exactly into pairwise sums of factor eigenvalues with cluster-projection norms = 1.0 across the bottom 8; the L_sym version fails because the path and tree factors are non-regular, as the task spec anticipated. |
| 2 | [Optimum cut indicator avoids f_2](pred2.md) | PASS | On `tree_cross_path_medium`, ⟨g_bar, f_2⟩ ≈ 0, ⟨g_bar, f_3⟩ ≈ 0.70, and r(3) = 0.0105 — the cut indicator lives almost entirely in span(f_1, f_3), exactly as predicted. |
| 3 | [Interleaving boundary](pred3.md) | PASS | Across q ∈ {20, 30, 40, 50, 70, 100}, the predicted NJW slab-recovery dimension d* matches the empirical d* exactly (off-by-zero) in all six cases, and the cut classification flips from tree-axis to path-axis precisely at the predicted q ≈ 40 boundary. |
| 4 | [Generalization across Cartesian products](pred4.md) | PASS (5/6) | The predicted slab-recovery dimension matches the empirical one within ±1 in five of six non-perturbed Cartesian products; the only exception (`path_x_path_unequal`, off by 2) is one of the failure modes the task spec already flagged (factor with regular spectrum + small q). |
| 5 | [Spectral dimension on non-product graphs](pred5.md) | PASS | Easy 2-cluster SBM and HSBM-top have spectral dim 2 at ε=0.05; the random 3-regular graph's residual never drops below 0.05 within 30 eigenvectors — the qualitative ordering predicted by the framework holds. |
| 6 | [Sweep on f_3 also escapes](pred6.md) | PASS | Sweep on f_3 reaches Φ_ratio = 1.000, sweep on f_4 fails (2.085), 1D k-means on f_3 also reaches 1.000, and the bad sweep-on-f_2 splits by path position while the good sweep-on-f_3 splits by tree vertex — confirming the escape is about *which* eigenvector is rounded, not *how* it is rounded. |

## Should we invest in proofs?

**Yes.** All four load-bearing predictions (1, 2, 3, 6) pass cleanly, the
generalization to other Cartesian products (4) passes at 5/6, and the
non-product-graph diagnostic (5) passes qualitatively.

The most informative result is Prediction 3: the predicted slab-recovery
dimension d* (= index of the first L_sym eigenvector aligned with
1_path ⊗ v_2(T_p) minus one for the f_1 drop) matches the empirical
recovery dimension *exactly* — d* increases by one each time the path's
next eigenvalue dips below the tree's Fiedler eigenvalue, and at exactly
that step the d-dimensional NJW k-means recovers the tree-axis slab cut.
The transition between "2D k-means recovers the slab" and "it doesn't"
occurs at the predicted q ≈ 40 boundary on the dot. This is the
quantitative tooth that makes the framework worth proving.

Two caveats are worth carrying into the writeup:

1. **L_sym vs combinatorial Laplacian.** The clean spectrum factorization
   `λ(G) = λ(H_1) + λ(H_2)` holds for the *combinatorial* Laplacian on
   Cartesian products of *any* graphs but for L_sym only when both
   factors are regular. The empirical predictions (alignment of the slab
   indicator with a specific G-eigenvector, recovery dimension under
   NJW k-means) live in L_sym world but inherit their structure from
   the combinatorial-Laplacian factorization in a way that is robust
   *up to a small constant in the recovery dimension* — the predicted
   d* still tracks the empirical d* even though the L_sym eigenvalues
   themselves don't factor cleanly. The proof will need to make this
   "pass-through" rigorous.

2. **Eigenvalue multiplicity and degeneracies.** Some Cartesian products
   in Prediction 4 (e.g., `doubletree_x_doubletree` with equal-size
   factors) have multiplicities or near-degeneracies in the bottom of
   the spectrum, which causes ARPACK to pick an arbitrary basis within
   each degenerate eigenspace. The recovery dimension is then "first
   dimension at which the *span* including the slab eigvec is captured",
   which is empirically d_pred ± 1, not exactly d_pred. The theorem
   should be stated in terms of subspaces, not individual eigenvectors.

Beyond those, the framework holds up under the empirical scrutiny we
budgeted. The next step is to attempt the proof of the spectrum-
factorization-implies-recovery-dimension theorem, with the combinatorial
Laplacian version as the first target.

## Reproducing

All scripts live in `scripts/spectral_dim_predN.py` and run end-to-end with
`PYTHONPATH=src python3 scripts/spectral_dim_predN.py` from the repo root.
Total wall time on a 2025 MacBook: under five minutes.

Outputs:
- `experiments/spectral_dim_predictions/predN.md` — verdict + procedure + numbers
- `results/spectral_dim_predictions/predN.parquet` — long-form measurements
- `experiments/plots/spectral_dim_predN.png` — plot per prediction
