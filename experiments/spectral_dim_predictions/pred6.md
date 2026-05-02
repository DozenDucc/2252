# Prediction 6: Sweep on f_3 escapes

**Verdict:** PASS

## Procedure

On `tree_cross_path_medium`, run sweep cut on f_2, f_3, f_4 (each is the L_rw 
eigenvector for the corresponding L_sym eigenvalue). Also run 1D k-means on f_2, f_3 
and 2D k-means on (f_2, f_3) for comparison. Compare conductance ratio to the 
analytic optimum.

## Results

```
         method      phi  phi_ratio                                         cut_axis
    sweep_on_f2 0.021277   2.564255 split by path-position (Guattery-Miller bad cut)
    sweep_on_f3 0.008297   1.000000               split by tree-vertex (optimum cut)
    sweep_on_f4 0.017301   2.085121               split by tree-vertex (optimum cut)
kmeans_1d_on_f2 0.021453   2.585550 split by path-position (Guattery-Miller bad cut)
kmeans_1d_on_f3 0.008297   1.000000               split by tree-vertex (optimum cut)
kmeans_2d_f2_f3 0.008297   1.000000               split by tree-vertex (optimum cut)
```

## One-sentence finding

Sweep on f_3 reaches Φ_ratio = 1.000, sweep on f_4 fails with Φ_ratio = 2.085, and 1D k-means on f_3 reaches Φ_ratio = 1.000 — confirming the escape is about which eigenvector is rounded, not the rounding scheme.