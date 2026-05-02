# path_x_path_unequal diagnosis (Section 3.4.3)

Graph: $P_{10} \,\square\, P_{50}$, $n=500$, $\Phi(\text{slab}) = 0.010638$. Slab axis = 1 (the longer-path $P_{50}$ axis); slab cut size = $|V(P_{10})| \cdot 1 = 10$ edges.

**Convention note up front.** The codebase's `pred4.py` computes its 'd=2 NJW' embedding by *dropping the trivial eigenvector $f_1$* and using $(f_2, f_3)$. The task-spec's mechanism (A) is stated for the $(f_1, f_2)$ embedding (i.e. INCLUDING $f_1$). Both conventions are tested here; the actual failing embedding is $(f_2, f_3)$, not $(f_1, f_2)$.

**Reproduction.** 2D unnormalized k-means on $(f_2, f_3)$ reproduces pred4's $\Phi$-ratio $\approx 1.478$ (measured here: 1.4780). 2D NJW row-normalized on $(f_2, f_3)$ gives ratio 1.3224. Sweep cut on $f_2$ gives ratio 1.000.

## Sweep-cut control

```
             mean  std  min  max
variant                         
sweep_on_f2   1.0  0.0  1.0  1.0
```

## Test 1 — row-normalize vs unnormalized, on both candidate embeddings

Five 2-cluster variants, n_init=10, k-means++ init, 10 seeds.

| variant | embedding | rounding |
|---|---|---|
| njw_f1f2 | $(f_1, f_2)$ | row-normalized + 2D k-means |
| unnormalized_f1f2 | $(f_1, f_2)$ | 2D k-means (no normalize) |
| njw_f2f3 | $(f_2, f_3)$ | row-normalized + 2D k-means |
| unnormalized_f2f3 | $(f_2, f_3)$ | 2D k-means (no normalize) |
| drop_trivial_1d_f2 | $f_2$ alone | 1D k-means |

```
                      mean    std     min     max
variant                                          
drop_trivial_1d_f2  1.0000  0.000  1.0000  1.0000
njw_f1f2            1.0000  0.000  1.0000  1.0000
njw_f2f3            1.3224  0.055  1.1929  1.3947
unnormalized_f1f2   1.0000  0.000  1.0000  1.0000
unnormalized_f2f3   1.4780  0.000  1.4780  1.4780
```

**Test 1 verdict.** **(A) row-normalization REFUTED**, in the form the task spec stated. Both NJW and unnormalized 2D k-means succeed on $(f_1, f_2)$ — including the trivial eigenvector and row-normalizing does NOT break the geometry. The actual failure happens on the $(f_2, f_3)$ embedding (NJW: 1.322; unnormalized: 1.478); both fail similarly, so row-normalization is not the lever. The 1D drop-trivial recovers the optimum, consistent with the framework's $\ell^* = 1$ prediction.

## Test 2 — k-means initialization sensitivity on the failing embedding

All runs at d=2 on the failing embedding $(f_2, f_3)$, both NJW and unnormalized. Vary `n_init` and `init`.

```
                                    mean     std     min     max
variant                                                         
njw_f2f3 / n_init=1, k-means++    1.4893  0.4770  1.0000  2.1510
njw_f2f3 / n_init=1, random       1.8202  0.4297  1.2533  2.1510
njw_f2f3 / n_init=10, k-means++   1.3224  0.0550  1.1929  1.3947
njw_f2f3 / n_init=100, k-means++  1.3202  0.0000  1.3202  1.3202
unn_f2f3 / n_init=1, k-means++    1.6092  0.4049  1.0000  2.0215
unn_f2f3 / n_init=1, random       1.7424  0.2419  1.4780  2.0215
unn_f2f3 / n_init=10, k-means++   1.4780  0.0000  1.4780  1.4780
unn_f2f3 / n_init=100, k-means++  1.4780  0.0000  1.4780  1.4780
```

**Test 2 verdict.** **(B) k-means local optima REFUTED.** With `n_init=100` the ratio is still 1.320 (NJW) and 1.478 (unnormalized) — well above the 1.05 success threshold. Increasing `n_init` reduces seed-to-seed variance (std drops from ~0.43 at n_init=1 to 0.0 at n_init=100), so k-means is *converging* across restarts, but it converges to a non-recovery partition. This is k-means' true 2D optimum on this point cloud — it just doesn't coincide with the graph-cut optimum, because the 2D k-means objective (within-cluster variance) is not aligned with conductance when the embedding includes a second irrelevant axis.

## Test 3 — eigenvalue degeneracy diagnostic

Bottom-10 L_sym eigenvalues:

```
  lambda_ 1 = 0.0000000
  lambda_ 2 = 0.0010607
  lambda_ 3 = 0.0042376
  lambda_ 4 = 0.0095151
  lambda_ 5 = 0.0168675
  lambda_ 6 = 0.0262592
  lambda_ 7 = 0.0273186
  lambda_ 8 = 0.0287187
  lambda_ 9 = 0.0321042
  lambda_10 = 0.0376452
```

Consecutive gaps $\lambda_{i+1} - \lambda_i$ (small min gap ⇒ near-degeneracy):

```
  gap[ 1- 2] = 0.0010607
  gap[ 2- 3] = 0.0031769
  gap[ 3- 4] = 0.0052775
  gap[ 4- 5] = 0.0073524
  gap[ 5- 6] = 0.0093917
  gap[ 6- 7] = 0.0010594
  gap[ 7- 8] = 0.0014001
  gap[ 8- 9] = 0.0033855
  gap[ 9-10] = 0.0055409
  min gap = 0.0010594
```

Predicted bottom-10 from L_sym factor sums (irregular path factors break exact factorization, but the order is roughly preserved):

```
  predicted_ 1 = 0.0000000
  predicted_ 2 = 0.0020546
  predicted_ 3 = 0.0082100
  predicted_ 4 = 0.0184408
  predicted_ 5 = 0.0327051
  predicted_ 6 = 0.0509443
  predicted_ 7 = 0.0603074
  predicted_ 8 = 0.0623620
  predicted_ 9 = 0.0685174
  predicted_10 = 0.0730832
```

Per-eigvec alignment with two slab templates:
- $T_{\text{slab}} = \mathbf{1}_{H_1} \otimes f_2(H_2)$ (the bottleneck-slab template, $H_2 = P_{50}$)
- $T_{\text{cross}} = f_2(H_1) \otimes \mathbf{1}_{H_2}$ (the non-bottleneck slab template, $H_1 = P_{10}$)

| i | overlap with $T_{\text{slab}}$ | overlap with $T_{\text{cross}}$ |
|---|---|---|
| $f_{1}$ | 0.0000 | 0.0000 |
| $f_{2}$ | 0.9975 | 0.0000 |
| $f_{3}$ | 0.0000 | 0.0000 |
| $f_{4}$ | 0.0131 | 0.0000 |
| $f_{5}$ | 0.0000 | 0.0000 |
| $f_{6}$ | 0.0114 | 0.0000 |
| $f_{7}$ | 0.0000 | 0.9897 |
| $f_{8}$ | 0.0000 | 0.0000 |
| $f_{9}$ | 0.0000 | 0.0875 |
| $f_{10}$ | 0.0000 | 0.0000 |

**Test 3 verdict.** **(C) ARPACK basis arbitrariness in a near-degenerate subspace REFUTED.** $f_2$ aligns cleanly with $T_{\text{slab}}$ (overlap 0.997) and $f_3$ has near-zero overlap with BOTH $T_{\text{slab}}$ (0.000) and $T_{\text{cross}}$ (0.000) — i.e. $f_3$ is a *third* path mode (a higher harmonic), not a near-degenerate mixture of the two slab templates. The min eigenvalue gap (0.00106) involves $\lambda_6$–$\lambda_7$, well above any embedding dimension we are using at d=2 or d=3.

## Test 4 — kmeans_no_trivial on (f_2, f_X)

Drop $f_1$ first, then run NJW (and unnormalized) on $(f_2, f_X)$ for $X \in \{3, 4, 7\}$. $f_7$ is the eigvec aligned with $T_{\text{cross}}$ per Test 3 (so $(f_2, f_7)$ is the embedding we'd expect to bisect cleanly along $f_2$, since the second axis is orthogonal to the bottleneck slab).

```
            mean    std     min     max
variant                                
njw_f2f3  1.3224  0.055  1.1929  1.3947
njw_f2f4  1.0000  0.000  1.0000  1.0000
njw_f2f7  5.0000  0.000  5.0000  5.0000
unn_f2f3  1.4780  0.000  1.4780  1.4780
unn_f2f4  1.0000  0.000  1.0000  1.0000
unn_f2f7  5.0000  0.000  5.0000  5.0000
```

**Test 4 verdict.** **The 'second axis' is decisive.** Three different choices of the second eigenvector paired with $f_2$ produce three qualitatively different outcomes: $(f_2, f_3)$ gives NJW ratio 1.322 (the original Pred4 failure); $(f_2, f_4)$ gives 1.000 (clean recovery); $(f_2, f_7)$ gives 5.000 (k-means bisects decisively along $f_7$, the *non-bottleneck* slab template, finding the wrong slab — ratio ≈ $p/q = 5$). Both $f_3$ and $f_4$ are orthogonal to the two slab templates, but $f_3$'s component pulls 2D k-means off the Fiedler axis enough to give a hybrid bisection, while $f_4$'s does not. None of the original mechanisms (A, B, C) predicts this dependence on the second axis; the mechanism is **(D) — embedding over-parameterization**, where adding a second axis at $d > \ell^* = 1$ may distort or even override the Fiedler bisection depending on what that axis encodes.

## Summary

- **Test 1**: (A) row-normalization is NOT the mechanism. $(f_1, f_2)$ NJW *succeeds* (ratio 1.000); $(f_2, f_3)$ NJW and unnormalized *both* fail similarly. Including the trivial eigvec is helpful, not harmful. 1D drop-trivial on $f_2$ alone recovers the optimum, confirming the framework's $\ell^* = 1$ prediction.
- **Test 2**: (B) k-means local optima are NOT the mechanism. `n_init=100` does not move the ratio off ~1.48 / ~1.32.
- **Test 3**: (C) ARPACK basis arbitrariness in a near-degenerate subspace is NOT the mechanism. $f_2$ has overlap 0.997 with $T_{\text{slab}}$ — clean. $f_3$ has essentially zero overlap with EITHER slab template — it is a third path mode (the next path harmonic), not a degenerate mixture of slab axes.
- **Test 4**: The actual mechanism is **(D) — embedding over-parameterization**. The standard 'drop trivial, use bottom-2 non-trivial' recipe at d=2 includes $f_3$, orthogonal to both slab templates. 2D k-means on $(f_2, f_3)$ balances variance along this irrelevant axis against $f_2$ and lands on a 'hybrid' bisection at $\Phi$-ratio ≈ 1.32–1.48. Swapping $f_3$ for $f_4$ removes the disruption (ratio 1.000). Swapping $f_3$ for $f_7$ (the $T_{\text{cross}}$-aligned eigvec) makes things much *worse* (ratio 5.0): k-means bisects decisively along $f_7$, finding the *wrong* slab. The second axis is not benign at any choice — only the 1D Fiedler embedding is robust.

## What this means for Section 3.4.3

The path × path 'discrepancy' is consistent with the framework's $\ell^* = 1$ prediction once the recipe is stated precisely: **one** non-trivial eigenvector ($f_2$) suffices, and 1D methods (sweep cut, 1D k-means, NJW with the trivial included) all achieve $\Phi$-ratio = 1.000. The 1.478 / 1.042 numbers reported in Pred4 reflect a *methodological* artifact of the standard 'drop trivial, stack the next $d$ eigvecs' recipe at $d > \ell^*$ — the extra eigvecs are neither slab templates ($f_3, f_4$ have near-zero overlap with $T_{\text{slab}}$ and $T_{\text{cross}}$) nor harmless: their variance contributes to k-means' objective and pulls the bisection off the Fiedler axis. None of the three originally hypothesized mechanisms (A, B, C) is responsible.

Suggested rewrite for Section 3.4.3: drop the three candidate-mechanism speculation; instead state that 'the predicted $\ell^* = 1$ holds: any 1D method on $f_2$ recovers the optimum. Standard NJW with $d=2$ (i.e. $(f_2, f_3)$) overshoots $\ell^*$ and includes a non-slab eigvec whose variance perturbs k-means; this is a generic over-parameterization issue when $d$ exceeds the spectral dimension of the cut, not a framework failure.'
