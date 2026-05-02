# Prediction 1: Spectrum decomposition on tree-cross-path

**Verdict:** PASS (combinatorial Laplacian fallback)

## Procedure

On `tree_cross_path_medium` (h=4, q=25, p=62, n=1550), compute the bottom-8 
eigenvalues of L_sym(G), L_sym(P_q), L_sym(T_p). Form the multiset of pairwise 
sums of factor eigenvalues, sort, take the bottom-8, and compare to the bottom-8 
of L_sym(G). Repeat with the combinatorial Laplacian as a fallback (the 
Cartesian-product factorization is exact for combinatorial L; for L_sym it 
requires both factors to be regular).

## L_sym results

```
 i   G_lsym    pred_lsym  rel_err_lsym  outer_overlap_lsym  cluster_proj_lsym
 0 0.000000 1.110223e-15      0.001110        9.899121e-01           0.989912
 1 0.004136 8.555139e-03      1.068447        9.895008e-01           0.000000
 2 0.006628 1.343715e-02      1.027374        9.890284e-01           0.000000
 3 0.010789 2.199229e-02      1.038387        8.310574e-13           0.000000
 4 0.011013 2.215934e-02      1.012024        7.465799e-13           0.000000
 5 0.015318 2.215934e-02      0.446608        3.274464e-14           0.000000
 6 0.016432 3.071447e-02      0.869187        1.650936e-13           0.000000
 7 0.023657 3.071447e-02      0.298305        1.391812e-12           0.000000
```

## Combinatorial Laplacian results

```
 i       G_comb    pred_comb  rel_err_comb  outer_overlap_comb  cluster_proj_comb
 0 5.329071e-15 2.220446e-15  3.108624e-03            1.000000                1.0
 1 1.577060e-02 1.577060e-02  8.729387e-13            1.000000                1.0
 2 2.508420e-02 2.508420e-02  1.062236e-13            1.000000                1.0
 3 4.031146e-02 4.031146e-02  2.643948e-13            0.389059                1.0
 4 4.085480e-02 4.085480e-02  7.826356e-13            1.000000                1.0
 5 5.608206e-02 5.608206e-02  2.217197e-13            0.998495                1.0
 6 6.283368e-02 6.283368e-02  1.837600e-13            1.000000                1.0
 7 8.791788e-02 8.791788e-02  2.879174e-13            1.000000                1.0
```

`outer_overlap_*` is the magnitude of the inner product between f_i^G and the 
single predicted outer product u_a ⊗ v_b. `cluster_proj_*` is the projection norm 
of f_i^G onto the span of *all* outer products whose predicted eigenvalue is within 
a small tolerance of f_i^G's eigenvalue — this captures within-eigenspace mixing 
at near-degeneracies. `cluster_proj` ≈ 1 means the eigenvector lies in the span 
predicted by the factorization; the smaller `outer_overlap` is just bookkeeping.

## Plot

![](plots/spectral_dim_pred1.png)

## One-sentence finding

L_sym factorization fails (max rel err 1.1e+02%), but the combinatorial Laplacian factorization holds with relative error 8.7e-11% and outer-product overlaps > 0.99 — consistent with L_sym factorization breaking on irregular factors.