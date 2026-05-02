# Prediction 2: optimum cut indicator avoids f_2

**Verdict:** PASS

## Procedure

On `tree_cross_path_medium`, compute `g_bar_S*` for the analytic optimum cut, 
then take inner products with the bottom-8 L_sym eigenvectors. Report the inner 
products and the cumulative residual r(d).

## Inner products and residuals

```
 i   eigval  inner_product  abs_inner_product  residual_at_d
 1 0.000000   7.071068e-01       7.071068e-01       0.500000
 2 0.004136  -5.025331e-15       5.025331e-15       0.500000
 3 0.006628  -6.996532e-01       6.996532e-01       0.010485
 4 0.010789   1.262879e-15       1.262879e-15       0.010485
 5 0.010789   1.618601e-13       1.618601e-13       0.010485
 6 0.011013   5.687708e-14       5.687708e-14       0.010485
 7 0.015318   1.341982e-14       1.341982e-14       0.010485
 8 0.015318  -7.421364e-17       7.421364e-17       0.010485
```

## Plot

![](plots/spectral_dim_pred2.png)

## One-sentence finding

|⟨g_bar, f_2⟩| = 0.0000 (predicted < 0.05), |⟨g_bar, f_3⟩| = 0.6997 (predicted > 0.5), and r(3) = 0.0105 (predicted < 0.05) — the cut indicator lives almost entirely in span(f_1, f_3), not in span(f_1, f_2).