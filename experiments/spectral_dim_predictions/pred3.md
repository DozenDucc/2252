# Prediction 3: interleaving boundary across q values

**Verdict:** PASS

## Procedure

Build P_q × T_p with h=4 (p=62) for q in {20, 30, 40, 50, 70, 100}. 
For each, compute bottom-10 L_sym eigenvectors of G, identify d* (the first 
eigenvector with > 0.5 absolute alignment with the slab template 
1_path ⊗ v_2(T_p)), and run sweep cut + d-dim NJW k-means + higher-order 
Cheeger rounding for d ∈ {2,3,4,5,6}.

## Per-q summary

| q | p | n | Φ(slab) | Φ(sweep)/Φ(slab) | d*_eigvec | d*_njw pred | slab-recov d* | first low-Φ d | r(2) | r(3) | r(5) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 62 | 1240 | 0.0083 | 3.100 | 3 | 2 | 2 | 2 | 0.5 | 0.0105 | 0.0105 |
| 30 | 62 | 1860 | 0.0083 | 2.067 | 3 | 2 | 2 | 2 | 0.5 | 0.0105 | 0.0105 |
| 40 | 62 | 2480 | 0.0082 | 1.550 | 4 | 3 | 3 | 3 | 0.5 | 0.5 | 0.0106 |
| 50 | 62 | 3100 | 0.0082 | 1.240 | 4 | 3 | 3 | 3 | 0.5 | 0.5 | 0.0107 |
| 70 | 62 | 4340 | 0.0082 | 0.886 | 5 | 4 | 4 | 3 | 0.5 | 0.5 | 0.0109 |
| 100 | 62 | 6200 | 0.0082 | 0.620 | 7 | 6 | 6 | 2 | 0.5 | 0.5 | 0.5 |

*slab-recov d*\* = smallest NJW dim where the cut found is **tree-axis**
 *and* Φ ≤ 1.05·Φ(slab). first low-Φ d = smallest NJW dim where Φ ≤ 1.05·Φ(slab)
regardless of cut shape (a path-axis cut may be lower-conductance than the tree-axis
slab when q is large, since cut size scales as q (tree-axis slab) vs p (path-axis)).*

## NJW k-means conductance ratio Φ/Φ(slab) (d = number of non-trivial eigvecs)

```
d        2      3      4      5      6      7      8
q                                                   
20   1.000  1.000  1.000  1.000  1.000  3.964  5.668
30   1.000  1.000  1.000  2.067  1.000  4.616  2.096
40   2.401  1.000  1.000  1.000  1.000  3.645  3.574
50   1.832  1.000  1.000  2.591  2.085  2.085  2.847
70   1.353  0.912  1.000  1.844  2.087  1.000  2.085
100  0.942  0.633  1.645  1.003  1.000  1.000  1.280
```

Cut-axis classification of NJW k-means cuts:

```
             d=2        d=3        d=4        d=5        d=6        d=7        d=8
q=20   tree-axis  tree-axis  tree-axis  tree-axis  tree-axis      mixed      mixed
q=30   tree-axis  tree-axis  tree-axis  tree-axis  tree-axis      mixed  tree-axis
q=40   path-axis  tree-axis  tree-axis  tree-axis  tree-axis      mixed      mixed
q=50   path-axis  tree-axis  tree-axis  path-axis  tree-axis  tree-axis      mixed
q=70   path-axis  path-axis  tree-axis  path-axis  path-axis  tree-axis  tree-axis
q=100  path-axis  path-axis  path-axis  path-axis  tree-axis  tree-axis      mixed
```

## Higher-order Cheeger conductance ratio

```
d        2      3      4      5      6      7      8
q                                                   
20   3.100  1.000  1.000  1.000  1.000  1.000  3.964
30   2.067  1.000  1.000  1.000  2.067  1.000  4.616
40   1.550  2.401  1.000  1.000  1.000  1.000  3.645
50   1.240  1.832  1.000  1.000  2.591  2.085  2.085
70   0.886  1.353  0.912  1.000  1.844  2.087  1.000
100  0.620  0.942  0.633  1.645  1.003  1.000  1.000
```

## Plot

![](plots/spectral_dim_pred3.png)

## One-sentence finding

Small-q 2D k-means recovers the tree-axis slab; large-q 2D k-means does not (it either fails or finds a cheaper *path-axis* cut, since for q ≳ p the path-axis slab has lower conductance than the tree-axis slab); and the predicted recovery dimension d* (= first eigenvector aligned with the tree-axis slab template) matches the empirical slab-recovery dimension within ±1.