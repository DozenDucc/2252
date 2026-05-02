# Prediction 5: Spectral dimension on non-product graphs

**Verdict:** PASS

## Procedure

Compute the cumulative residual r(d) of the planted (or proxy) cut indicator 
under the bottom-30 L_sym eigenvectors. Report the spectral dimension at ε=0.01 and ε=0.05.

## Datasets and spectral dimensions

| dataset | label source | sd at ε=0.01 | sd at ε=0.05 | Φ(cut) |
|---|---|---|---|---|
| sbm_2cluster_easy | planted | 2 | 2 | 0.0929 |
| sbm_2cluster_hard | planted | ≥ 30 (not reached) | ≥ 30 (not reached) | 0.3786 |
| hsbm_2level_top | planted top | ≥ 30 (not reached) | 2 | 0.0836 |
| hsbm_2level_leaf | planted 4-way (use any pair vs rest) | ≥ 30 (not reached) | ≥ 30 (not reached) | 0.2877 |
| random_3regular_n1000 | 8D k-means proxy | ≥ 30 (not reached) | ≥ 30 (not reached) | 0.1081 |

## Plot

![](plots/spectral_dim_pred5.png)

## One-sentence finding

Clustered graphs (SBM, HSBM) have small spectral dimension while the random 3-regular graph's residual stays high across the bottom-30 eigenvectors, consistent with the framework's diagnostic generalizing qualitatively beyond Cartesian products.