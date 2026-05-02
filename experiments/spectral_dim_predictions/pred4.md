# Prediction 4: Generalization across Cartesian products

**Verdict:** PASS (5/6 non-perturbed cases match)

## Procedure

Build a small library of Cartesian-product graphs. For each, compute the predicted 
d* (NJW dim where the slab cut is first recovered) by aligning the slab template 
(1 ⊗ v_2(H_2) for slab along axis 1, v_2(H_1) ⊗ 1 for axis 0) with G's L_sym 
eigenvectors. Compare to the empirical recovery dim across NJW k-means at d ∈ {2,3,...,10}.

## Per-case summary

| name | n | λ_2(H_1) | λ_2(H_2) | slab axis | d*_njw pred | d*_njw emp | match (±1) | sweep ratio | sweep predicted |
|---|---|---|---|---|---|---|---|---|---|
| path_x_doubletree (q=25, h=4) | 1550 | 0.0086 | 0.0134 | 1 | 2 | 2 | ✓ | 2.564 | fail |
| path_x_cycle (q=20, p=40) | 800 | 0.0136 | 0.0123 | 0 | 3 | 2 | ✓ | 1.000 | fail |
| cycle_x_doubletree (q=20, h=4) | 1240 | 0.0489 | 0.0134 | 1 | 1 | 2 | ✓ | 1.000 | fail or ill-defined |
| path_x_path_unequal (q=10, p=50) | 500 | 0.0603 | 0.0021 | 1 | 1 | 3 | ✗ | 1.000 | fail |
| doubletree_x_doubletree (h1=h2=4) | 3844 | 0.0134 | 0.0134 | 0 | 1 | 2 | ✓ | 2.032 | succeed |
| doubletree_x_doubletree_unequal (h1=3, h2=5) | 3780 | 0.0323 | 0.0060 | 1 | 1 | 2 | ✓ | 1.000 | fail |

Perturbed cases (no analytic prediction):

| name | n | d*_njw pred | d*_njw emp | sweep ratio |
|---|---|---|---|---|
| path_x_doubletree_perturbed (eps=0.001) | 1550 | None | None | 1.153 |
| path_x_doubletree_perturbed (eps=0.01) | 1550 | None | 2 | 0.887 |

## NJW k-means conductance ratios

```
d                                                2      3      4      5      6      8      10
name                                                                                         
cycle_x_doubletree (q=20, h=4)                1.000  2.085  4.800  4.800  4.556  4.556  4.556
doubletree_x_doubletree (h1=h2=4)             1.000  1.000  1.000  2.068  1.832  2.355  2.027
doubletree_x_doubletree_unequal (h1=3, h2=5)  1.000  2.041  2.041  2.759  4.255  4.255  4.255
path_x_cycle (q=20, p=40)                     1.000  1.000  1.000  1.237  1.596  1.637  2.707
path_x_doubletree (q=25, h=4)                 1.000  1.000  2.085  2.085  2.085  2.085  2.915
path_x_doubletree_perturbed (eps=0.001)       2.294  1.632  2.294  1.676  1.559  1.615  1.525
path_x_doubletree_perturbed (eps=0.01)        1.000  0.920  0.931  0.882  0.939  0.907  0.899
path_x_path_unequal (q=10, p=50)              1.478  1.042  2.831  3.672  1.679  3.197  4.754
```

## Plot

![](plots/spectral_dim_pred4.png)

## One-sentence finding

5/6 of the non-perturbed Cartesian-product test cases have 
predicted d*_njw matching empirical d*_njw within ±1; see table for which 
families violate the prediction.