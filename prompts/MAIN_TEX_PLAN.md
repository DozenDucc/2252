# Plan for main.tex edits — CS 2252 spectral dimension framework

This document is the source of truth for the three new sections to be added to
`main.tex`. Every citation, theorem statement, and predicted/observed value
below should be considered load-bearing. Do not summarize — copy verbatim
when writing the LaTeX.

## Goal

Add three sections to `main.tex` between Section 7 ("Fallback Plan") and the
appendix:

1. **Section: Motivation and Related Work** — motivate the spectral-dimension
   framework, give a complete chronological walk through results and bounds.
2. **Section: A Spectral-Dimension Framework for Cartesian Products** —
   formal definitions, the four-part theorem (a)/(b)/(c)/(d), what's proven
   vs. conjectured, and the predictions each part makes.
3. **Section: Empirical Verification** — report the six predictions, the
   tree-cross-path experiment, and what we conclude.

Style target: NeurIPS-quality formal writing. Theorem environments. Honest
about what's new and what's a clean repackaging of classical results.

## Key honesty constraints

These are non-negotiable framings, derived from prior conversation:

- **The framework is not new in its core ideas.** It crystallizes implicit
  phenomena from Macgregor-Sun 2022, Lee-Oveis Gharan-Trevisan 2014, Fiedler
  1973, and Chung-Tetali 1998. The contribution is (i) a clean structural
  special case (Cartesian products) where matching upper/lower bounds are
  tight, (ii) the *spectral dimension of a cut* as a unifying diagnostic,
  (iii) empirical verification.
- **L_sym factorization fails on irregular factors.** The clean Cartesian
  product spectrum factorization holds for the *combinatorial* Laplacian on
  Cartesian products of any graphs, but for L_sym only when both factors are
  regular. We measure with L_sym (because that's what algorithms use), but
  the structural predictions inherit from the combinatorial-Laplacian
  factorization in a way that is robust *up to a small constant in the
  recovery dimension*. This must be stated explicitly.
- **Prediction 4 passed 5/6, not 6/6.** Failures must be reported.
- **Prediction 5 is a qualitative pass, not quantitative.** Hard SBM, HSBM-leaf,
  and random 3-regular all have spectral dimension ≥ 30 at ε = 0.05. Only easy
  SBM and HSBM-top pass cleanly at ε = 0.05. Easy SBM passes at ε = 0.01.
- **The "escape" on tree-cross-path is dimensionality, not rounding.**
  Sweep on f_3 also escapes (Prediction 6). The interesting axis is which
  eigenvector you use, not how you round.

# Section 1: Motivation and Related Work — chronology

## 1.1 Definitions to introduce up front (so the chronology has language)

- **Combinatorial Laplacian** $L = D - A$.
- **Normalized Laplacian** $\mathcal{N} = I - D^{-1/2} A D^{-1/2}$. Eigenvalues
  $0 = \lambda_1 \leq \lambda_2 \leq \ldots \leq \lambda_n \leq 2$.
- **Conductance** of $S$: $\Phi(S) = w(S, V \setminus S) / \min(\text{vol}(S),
  \text{vol}(V \setminus S))$ where vol$(S) = \sum_{u \in S} d(u)$.
- **Graph conductance** $\Phi(G) = \min_S \Phi(S)$.
- **k-way expansion constant** $\rho(k) = \min_{\text{partition } A_1, \ldots, A_k}
  \max_i \Phi(A_i)$.
- **Indicator vectors** $\chi_S$, normalized indicator $\bar g_S = D^{1/2}\chi_S
  / \|D^{1/2}\chi_S\|$.
- **Cartesian product** $G_1 \square G_2$: vertex set $V_1 \times V_2$, edges
  $(u,v) \sim (u',v')$ iff ($u=u'$ and $v\sim v'$ in $G_2$) or ($v=v'$ and
  $u\sim u'$ in $G_1$). NOT the same as direct product or strong product.
  We must be careful: Guattery-Miller call this the "crossproduct" but they
  mean Cartesian product (they define it explicitly).

## 1.2 Chronology — the citations and what each contributed

The chronology should be presented as flowing text, but here are the papers in
order with full citations and what each contributed. The LaTeX writeup should
include each of these (some can be grouped into a single sentence; the bibtex
entries below are all needed).

### Cheeger inequality and continuous origins

- **Cheeger, J. (1970).** "A lower bound for the smallest eigenvalue of the
  Laplacian." In *Problems in Analysis (Princeton, 1969)*, pp. 195-199,
  Princeton University Press.
  — Original Cheeger inequality on Riemannian manifolds.

### Algebraic connectivity (the foundation)

- **Fiedler, M. (1973).** "Algebraic connectivity of graphs." *Czechoslovak
  Mathematical Journal*, 23(2):298-305.
  — Defines $\lambda_2$ as algebraic connectivity. Proves $\lambda_2 > 0$ iff
  $G$ connected. Critically: proves Cartesian product spectrum factorization
  ($L(G \square H)$ has eigenvalues that are pairwise sums of factor
  eigenvalues, eigenvectors are outer products). **This is the engine of our
  theorem.**

- **Fiedler, M. (1975).** "A property of eigenvectors of nonnegative symmetric
  matrices and its application to graph theory." *Czechoslovak Mathematical
  Journal*, 25(4):619-633.
  — Properties of $f_2$, including monotonicity along paths in trees.

### Discrete Cheeger and isoperimetric inequalities

- **Donath, W. E., and Hoffman, A. J. (1973).** "Lower bounds for the
  partitioning of graphs." *IBM Journal of Research and Development*,
  17(5):420-425.
  — First lower bound on bisection size using eigenvalues.

- **Alon, N., and Milman, V. D. (1985).** "$\lambda_1$, isoperimetric
  inequalities for graphs, and superconcentrators." *Journal of Combinatorial
  Theory, Series B*, 38(1):73-88.
  — Discrete Cheeger inequality (one direction).

- **Alon, N. (1986).** "Eigenvalues and expanders." *Combinatorica*, 6(1):83-96.
  — Other direction of discrete Cheeger.

- **Mohar, B. (1989).** "Isoperimetric numbers of graphs." *Journal of
  Combinatorial Theory, Series B*, 47(3):274-291.
  — The form of Cheeger's inequality we use:
  $\lambda_2/2 \leq i(G) \leq \sqrt{\lambda_2(2\Delta - \lambda_2)}$.
  Constructive proof via sweep cut.

### Early spectral partitioning algorithms

- **Hagen, L., and Kahng, A. B. (1992).** "New spectral methods for ratio cut
  partitioning and clustering." *IEEE Transactions on Computer-Aided Design*,
  11(9):1074-1085.
  — "Best threshold cut" idea; ratio cut.

- **Pothen, A., Simon, H. D., and Liou, K.-P. (1990).** "Partitioning sparse
  matrices with eigenvectors of graphs." *SIAM Journal on Matrix Analysis and
  Applications*, 11(3):430-452.
  — Standard reference for spectral bisection in numerical linear algebra
  community.

### Negative results — Guattery-Miller

- **Guattery, S., and Miller, G. L. (1995).** "On the performance of spectral
  graph partitioning methods." In *Proceedings of the Sixth Annual ACM-SIAM
  Symposium on Discrete Algorithms (SODA '95)*, pp. 233-242.
  — Original conference version.

- **Guattery, S., and Miller, G. L. (1998).** "On the quality of spectral
  separators." *SIAM Journal on Matrix Analysis and Applications*,
  19(3):701-719.
  — Journal version. The cockroach (roach graph) defeats spectral bisection;
  the tree-cross-path defeats best threshold cut. §7.2 conjectures that
  geometric/k-means rounding using bottom-d eigenvectors *also* fails on
  tree-cross-path for any constant d. This conjecture is what we falsify
  empirically in a specific regime.

### Cartesian product isoperimetry

- **Houdré, C., and Tetali, P. (1996).** "Isoperimetric invariants for Markov
  chains and graphs." Preprint / later in *Annals of Probability*.
  — Markov chain version.

- **Chung, F. R. K., and Tetali, P. (1998).** "Isoperimetric inequalities for
  Cartesian products of graphs." *Combinatorics, Probability and Computing*,
  7(2):141-148.
  — The conductance of $G \square H$ is bounded in terms of conductances of
  factors. **This is the engine for the slab-cut optimality (theorem part a).**

### Spielman-Teng — recursive spectral, planar graphs

- **Spielman, D. A., and Teng, S.-H. (1996).** "Spectral partitioning works:
  Planar graphs and finite element meshes." In *37th Annual Symposium on
  Foundations of Computer Science (FOCS '96)*, pp. 96-105.
  Journal version: *Linear Algebra and its Applications*, 421(2-3):284-305,
  2007.
  — $\lambda_2 = O(1/n)$ for bounded-degree planar graphs. Recursive spectral
  partitioning algorithm. Proves recursive spectral bisection produces $O(\sqrt{n})$
  bisectors on planar graphs, $O(n^{1-1/d})$ on $d$-dim well-shaped meshes.

### NJW and the modern spectral clustering recipe

- **Ng, A. Y., Jordan, M. I., and Weiss, Y. (2001).** "On spectral clustering:
  Analysis and an algorithm." In *Advances in Neural Information Processing
  Systems 14 (NeurIPS '01)*, pp. 849-856.
  — The k-eigenvector + row-normalize + k-means recipe that became standard.

- **von Luxburg, U. (2007).** "A tutorial on spectral clustering." *Statistics
  and Computing*, 17(4):395-416.
  — Standard tutorial reference, already cited in main.tex.

### Higher-order Cheeger

- **Louis, A., Raghavendra, P., Tetali, P., and Vempala, S. S. (2012).** "Many
  sparse cuts via higher eigenvalues." In *Proceedings of the 44th Annual ACM
  Symposium on Theory of Computing (STOC '12)*, pp. 1131-1140.
  — Earlier multi-way version.

- **Lee, J. R., Oveis Gharan, S., and Trevisan, L. (2014).** "Multiway spectral
  partitioning and higher-order Cheeger inequalities." *Journal of the ACM*,
  61(6):37:1-37:30. (Conference version: STOC '12, pp. 1117-1130.)
  arXiv:1111.1055.
  — **The foundational result.** Higher-order Cheeger:
  $\rho(k)/2 \leq \lambda_k/2 \leq \rho(k) \leq O(k^2) \sqrt{\lambda_k}$.
  Algorithm: bottom-k eigenvector embedding + randomized geometric rounding.
  This is our theoretical framework.

- **Kwok, T. C., Lau, L. C., Lee, Y. T., Oveis Gharan, S., and Trevisan, L.
  (2013).** "Improved Cheeger's inequality: Analysis of spectral partitioning
  algorithms through higher order spectral gap." In *Proceedings of the 45th
  Annual ACM Symposium on Theory of Computing (STOC '13)*, pp. 11-20.
  arXiv:1301.5584.
  — $\Phi(G) \leq O(k) \cdot \lambda_2 / \sqrt{\lambda_k}$. Sweep cut achieves
  this bound. Sharper than $\sqrt{\lambda_2}$ when $\lambda_k$ is bounded
  away from zero. Tight up to constants.

### Spectral clustering theory under structural assumptions

- **Peng, R., Sun, H., and Zanetti, L. (2015).** "Partitioning well-clustered
  graphs: Spectral clustering works!" In *Proceedings of the 28th Conference
  on Learning Theory (COLT '15)*, pp. 1423-1455. Journal: *SIAM Journal on
  Computing*, 46(2):710-743, 2017. arXiv:1411.2021.
  — Defines $\Upsilon(k) = \lambda_{k+1}/\rho(k)$. Proves: if $\Upsilon(k) \geq
  \Omega(k^3)$, then standard spectral clustering (k-means on bottom-k
  embedding) recovers the optimal k-way partition up to small error. Already
  cited in main.tex.

- **Kolev, P., and Mehlhorn, K. (2016).** "A note on spectral clustering."
  *Algorithmica* / ESA 2016.
  — Corrects a technical issue in PSZ; weakens to $\Upsilon(k) = \Omega(k^2)$.

- **Mizutani, T. (2021).** "Improved analysis of spectral algorithm for
  clustering." *Optimization Letters*, 15(4):1303-1325.
  — $\Upsilon(k) = \Omega(k)$ suffices.

- **Macgregor, P., and Sun, H. (2022).** "A tighter analysis of spectral
  clustering, and beyond." In *Proceedings of the 39th International
  Conference on Machine Learning (ICML '22)*, pp. 14717-14742.
  arXiv:2208.01724.
  — **The most directly relevant result for our work.** Two contributions:
  1. *Tighter analysis*: $\Upsilon(k) = \Omega(1)$ suffices for spectral
     clustering with k almost-balanced clusters.
  2. *Fewer than k eigenvectors*: introduces the **meta-graph** of optimal
     clusters, defines **$(\theta, \ell)$-distinguishability**, proves that
     if the meta-graph $M$ on $k$ vertices is $(\theta, \ell)$-distinguishable
     for some $\ell < k$ and a refined gap $\Psi(\ell) = \sum_{i=1}^\ell
     \gamma_i / \lambda_{\ell+1}$ is small, then spectral clustering with
     just $\ell$ eigenvectors recovers the $k$ clusters.
  Empirically confirmed on BSDS image segmentation, MNIST, USPS.
  **Our framework specializes Macgregor-Sun's meta-graph mechanism to
  Cartesian products, where the meta-graph has algebraic origins (Fiedler
  1973).**

## 1.3 The motivation for spectral dimension itself

This is the conceptual through-line for Section 1. The chronology shows that
spectral algorithms differ in two independent axes:

1. **Embedding dimensionality**: how many eigenvectors does the algorithm see?
   - Sweep cut: 1 (just $f_2$).
   - Standard k-way spectral clustering with $k$ clusters: $k$.
   - Macgregor-Sun's algorithm: $\ell$, with $\ell < k$ allowed.
   - Higher-order Cheeger rounding: $k$.
2. **Rounding strategy**: how is the embedding partitioned?
   - Sweep on a single eigenvector.
   - k-means on multi-dim embedding.
   - Randomized geometric partitioning (Lee-Oveis Gharan-Trevisan).
   - Spectral rotation, etc.

The interesting question — when does each algorithm work? — really lives on
*axis 1* (dimensionality). The optimum cut indicator $\bar g_{S^*}$ is some
specific vector in $\mathbb{R}^n$. A spectral algorithm using $d$ eigenvectors
can only recover cuts whose indicators are well-approximated by linear
combinations of $f_1, \ldots, f_d$. If $\bar g_{S^*}$ has significant mass on
$f_{d+1}$ and beyond, no rounding can recover $S^*$ from a $d$-dim embedding.

**This motivates the spectral dimension of a cut.**

# Section 2: Theory

## 2.1 Definitions

**Definition (Spectral dimension at tolerance $\epsilon$).**
For a graph $G$ with normalized Laplacian eigenvectors $f_1, \ldots, f_n$ and
a subset $S \subseteq V(G)$, define
$$d_\epsilon(S) := \min\left\{d : \left\|\bar g_S - \sum_{i=1}^d \langle \bar g_S, f_i \rangle f_i\right\|^2 \leq \epsilon\right\}$$
where $\bar g_S = D^{1/2}\chi_S / \|D^{1/2}\chi_S\|$ is the normalized
indicator vector. Default tolerance $\epsilon = 0.01$.

The spectral dimension of $G$ (with respect to 2-way cuts) is $d_\epsilon(S^*)$
where $S^*$ is conductance-optimal.

**Definition (Slab cut).** For $G = H_1 \square H_2$ and a 2-way cut $T \subseteq
V(H_2)$, the slab cut along the $H_2$-axis is $S = V(H_1) \times T$. Symmetric
definition for slab along $H_1$-axis.

**Definition (Spectral interleaving condition).** $G = H_1 \square H_2$ satisfies
the interleaving condition (with $H_1$ as the bottleneck factor) if
$\mu_2(H_1) < \nu_2(H_2) < \mu_3(H_1)$, where $\mu_i$ are the eigenvalues of
the *combinatorial* Laplacian of $H_1$ and $\nu_j$ of $H_2$. (We use
combinatorial Laplacian for the regime statement; the algorithmic statements
work with $L_{\text{sym}}$, with caveats below.)

## 2.2 Three corollaries and a conjecture

I'll state each part with the appropriate framing: Corollaries 1-3 derive
from existing results, and Conjecture 1 is genuinely open. The LaTeX writeup
should use `\begin{corollary}` for the first three and `\begin{conjecture}`
for the fourth. Each corollary should explicitly cite the prior result it
derives from and include a derivation (proof sketch).

### Corollary 1 (Slab-cut optimality on Cartesian products).

**Prior result we derive from:** *Chung & Tetali (1998), "Isoperimetric
Inequalities for Cartesian Products of Graphs," Combinatorics, Probability
and Computing, 7(2):141-148, Theorem 3.1.* Their theorem establishes that the
isoperimetric number of $G_1 \square G_2$ is bounded by linear combinations
of the factors' isoperimetric numbers, with the optimal sets being slab
cuts (Cartesian products of one factor's optimal cut with the other factor's
full vertex set).

**Statement.** *Let $G = H_1 \square H_2$ with $H_1$, $H_2$ connected. Suppose
$\mu_2(H_1) < \nu_2(H_2)$ (call $H_1$ the **bottleneck factor**) and*
$$|V(H_1)| \cdot |\partial T^*_{H_1}| > |V(H_2)| \cdot |\partial T^*_{H_2}|, \quad (\star)$$
*where $T^*_{H_i}$ is the conductance-optimal 2-way cut of $H_i$. Then the
conductance-optimal 2-way cut of $G$ is the slab*
$$S^* = V(H_1) \times T^*_{H_2}$$
*(along the non-bottleneck factor's axis), with*
$$\Phi_G(S^*) = \Theta(\Phi_{H_2}(T^*_{H_2})) = \Theta(\nu_2(H_2)).$$

**Derivation.** The two candidate slabs are:
- *Slab A* (bottleneck axis): $|V(H_2)| \cdot |\partial T^*_{H_1}|$ edges.
- *Slab B* (non-bottleneck axis): $|V(H_1)| \cdot |\partial T^*_{H_2}|$ edges.

Hypothesis $(\star)$ says slab B has fewer edges. Both slabs have volume
$\Theta(|V(H_1)| \cdot |V(H_2)|)$, so slab B has lower conductance.
Chung-Tetali's Theorem 3.1 implies these slab cuts are within constants of
optimal among *all* 2-way cuts (not just slabs), so $S^* = $ slab B.

**Why both hypotheses are needed (the size condition $(\star)$).** The
bottleneck-vs-non-bottleneck distinction is *separate* from the size
distinction. The bottleneck condition $\mu_2(H_1) < \nu_2(H_2)$ determines
which axis sweep cut explores (sweep follows $f_2(G)$, which by Fiedler 1973
lives on the bottleneck factor). The size condition $(\star)$ determines
which axis is the *true optimum*. Sweep cut fails iff these two axes disagree.

**Numerical check (medium tree-cross-path, $q=25$, $p=62$):** Slab A: $p
\cdot 1 = 62$ edges. Slab B: $q \cdot 1 = 25$ edges. $(\star)$ holds:
$62 > 25$. Empirical optimum cut size = $q = 25$, $\Phi(\text{opt}) = 0.0083$.
Sweep cut size $\approx 64 \approx p$ (slab A). Confirms.

**When $(\star)$ fails: large-$q$ regime.** Pushing $q$ above $p$ in
tree-cross-path inverts the size condition. At $q=100$, $p=62$: slab A has
62 edges, slab B has 100 edges, so slab A is now the actual optimum. Sweep
cut still returns slab A (its output is unchanged by $q$, since sweep
follows the bottleneck factor regardless), and now sweep cut **succeeds**
because slab A has become the optimum. Pred3 confirms this empirically:
at $q=100$, the sweep ratio is $0.620$ (sweep beats the tree-axis slab),
and 2D NJW k-means at $d=2$ finds a path-axis cut with ratio $0.942$ (also
better than the tree-axis "slab"). The framework's "sweep fails" prediction
holds only when both bottleneck *and* size conditions are met — i.e., in
the regime studied empirically ($q \lesssim p$).

**Status:** Direct corollary of Chung-Tetali (1998), Theorem 3.1. The
explicit size condition $(\star)$ is added to disambiguate which slab is
optimal. Not a contribution — we use it.

### Corollary 2 (Spectral dimension factorization).

**Prior result we derive from:** *Fiedler, M. (1973), "Algebraic connectivity
of graphs," Czechoslovak Mathematical Journal, 23(2):298-305, Theorem 2.3
(Cartesian product spectrum).* Fiedler proves that the eigenvalues of the
combinatorial Laplacian $L(G_1 \square G_2)$ are all pairwise sums
$\{\mu_i + \nu_j : i \in [|V(G_1)|], j \in [|V(G_2)|]\}$, and that the
corresponding eigenvectors are outer products $u_i \otimes v_j$ of the factor
eigenvectors.

**Statement.** *Suppose the hypotheses of Corollary 1 hold ($\mu_2(H_1) <
\nu_2(H_2)$ and $(\star)$). Suppose additionally that both factors are
regular, so the L_sym spectrum factorizes (when factors are non-regular,
the statement holds for the combinatorial Laplacian; see caveat below).
Let $S^* = V(H_1) \times T^*_{H_2}$ be the optimal 2-way cut from Corollary 1.
Then for sufficiently small $\epsilon$,*
$$d_\epsilon(S^*; G) = d_\epsilon(T^*_{H_2}; H_2).$$

**Derivation.** The normalized indicator of the slab cut is
$$\bar g_{S^*} = \frac{\mathbf{1}_{H_1}}{\sqrt{|V(H_1)|}} \otimes \bar g_{T^*_{H_2}}$$
(constant in the $H_1$-direction, varying in the $H_2$-direction). By
Fiedler's Theorem 2.3, eigenvectors of the form $\mathbf{1}_{H_1} \otimes v_j$
(constant in the $H_1$-direction) correspond exactly to eigenvalues of
$L(G)$ that come from the $H_2$-spectrum (specifically, the pairwise sums
$\mu_1 + \nu_j = \nu_j$). Therefore $\bar g_{S^*}$ has the same expansion
in the $L(G)$-eigenbasis as $\bar g_{T^*_{H_2}}$ has in the $L(H_2)$-eigenbasis,
just embedded into the larger space. The number of eigenvectors needed to
reach $\epsilon$-residual is the same.

**Caveat (irregular factors).** Fiedler's theorem is stated for the
combinatorial Laplacian $L = D - A$, where the factorization is exact
regardless of factor regularity. For $L_{\text{sym}} = I - D^{-1/2} A
D^{-1/2}$, the factorization holds exactly only when both factors are regular.
On non-regular factors (e.g., the tree-cross-path graph, where the double
tree has degrees $\{1, 2, 3\}$), the L_sym factorization fails — Pred1
measures relative errors of $\approx 110\%$ between predicted and actual
L_sym eigenvalues. However, the *structural* prediction (which eigenvector
of $G$ aligns with the slab template) is preserved approximately: Pred1
reports cluster-projection norm $> 0.99$ for L_sym, meaning the predicted
eigenvector lives within the right invariant subspace even though the
eigenvalues don't sum exactly. This is enough to make Corollaries 2-3
empirically robust on non-regular factors.

**Status:** Direct corollary of Fiedler (1973), Theorem 2.3, on regular
Cartesian products. The combinatorial-vs-L_sym distinction is a known
technical caveat; we report it honestly.

### Corollary 3 (Cheeger tightness on Cartesian products in the asymmetric regime).

**Prior results we derive from:**
- *Mohar, B. (1989), "Isoperimetric numbers of graphs," Journal of
  Combinatorial Theory Series B, 47(3):274-291, Theorem 4.2.* Cheeger's
  upper bound: $i(G) \leq \sqrt{\lambda_2 (2\Delta - \lambda_2)}$, achieved
  constructively by sweep cut.
- *Guattery, S. and Miller, G. L. (1998), "On the quality of spectral
  separators," SIAM Journal on Matrix Analysis and Applications,
  19(3):701-719, Theorem 6.3.* Sweep cut on the tree-cross-path graph
  $P_q \square T_p$ with $q \approx c\sqrt{p}$ produces a cut with cut
  quotient bigger than $i(G)$ by $\Omega(p^{1/2})$.
- *Corollary 1* (above).

**Statement.** *Suppose $G = H_1 \square H_2$ satisfies the hypotheses of
Corollary 1 (in particular, the size condition $(\star)$). Then:*

*(i) Sweep cut achieves $\Phi_{\text{sweep}}(G) = \Theta(\sqrt{\mu_2(H_1)})$,
saturating Cheeger's upper bound on $\lambda_2(G) = \mu_2(H_1)$.*

*(ii) The optimum is $\Phi(G) = \Theta(\nu_2(H_2))$.*

*(iii) Therefore*
$$\frac{\Phi_{\text{sweep}}(G)}{\Phi(G)} = \Theta\!\left(\frac{\sqrt{\mu_2(H_1)}}{\nu_2(H_2)}\right),$$
*which diverges as $\mu_2(H_1) \to 0$ at fixed $\nu_2(H_2)$.*

**Derivation.**
- *(i):* By Fiedler 1973, $f_2(G) = f_2(H_1) \otimes \mathbf{1}_{H_2}/\sqrt{|V(H_2)|}$,
  constant in the $H_2$-direction. Sweep cut on $G$ therefore produces slab A
  (along the bottleneck axis $H_1$). The conductance of slab A in $G$ equals
  the conductance of the corresponding factor cut in $H_1$ (by direct
  computation: cut size $|V(H_2)| \cdot |\partial T|$, volume $\Theta(|V(H_2)|
  \cdot \text{vol}_{H_1}(T))$). Sweep cut on $H_1$ achieves Cheeger's upper
  bound $\Theta(\sqrt{\mu_2(H_1)})$ by Mohar 1989, Theorem 4.2.
- *(ii):* By Corollary 1, $\Phi(G) = \Theta(\Phi_{H_2}(T^*_{H_2})) =
  \Theta(\nu_2(H_2))$ (Cheeger lower bound applied within $H_2$).
- *(iii):* Take the ratio of (i) and (ii).

**Subtle framing.** Cheeger applied to $G$ as a whole gives
$\lambda_2(G)/2 \leq \Phi(G) \leq \sqrt{2\lambda_2(G)}$ where $\lambda_2(G)
= \mu_2(H_1)$ (since $H_1$ is the bottleneck). Sweep cut saturates the
Cheeger *upper* bound on $\lambda_2(G)$. However, the actual optimum
$\Phi(G) = \Theta(\nu_2(H_2))$ is much **larger** than the Cheeger *lower*
bound $\lambda_2(G)/2 = \Theta(\mu_2(H_1))$ — i.e., the lower Cheeger bound
on $G$ is loose, because the optimum cut "lives in" a different invariant
subspace than $f_2(G)$. The right eigenvalue for understanding the optimum
is $\lambda_3(G) = \nu_2(H_2)$, not $\lambda_2(G)$.

This is the cleanest framing of the whole framework: sweep cut saturates
Cheeger's upper bound applied to $\lambda_2(G)$, but $\lambda_2(G)$ is the
wrong eigenvalue — the optimum is governed by $\lambda_3(G)$, and only an
algorithm with access to $f_3(G)$ can find it.

**Numerical verification (tree-cross-path empirical numbers):** predicted
ratio $= \Theta(\sqrt{\mu_2(P_q)}/\nu_2(T_p)) = \Theta((1/q)/(1/p)) = \Theta(p/q)$.
- Small ($q=20$, $p=30$): $p/q = 1.5$. Empirical: 1.500. ✓
- Medium ($q=25$, $p=62$): $p/q = 2.48$. Empirical: 2.564. ✓
- Large ($q=50$, $p=126$): $p/q = 2.52$. Empirical: 2.520. ✓

In the symmetric scaling regime $q \sim \sqrt{p}$: $p/q \sim \sqrt{p}$,
$n = pq \sim p^{3/2}$, so $p/q \sim n^{1/3}$. This is the Guattery-Miller
$n^{1/3}$ gap.

**Status:** Direct corollary of Mohar 1989 + Guattery-Miller 1998 +
Corollary 1. Guattery-Miller proved the lower bound for tree-cross-path
specifically; here we generalize the argument to arbitrary Cartesian
products satisfying the bottleneck condition $(\star)$. The generalization
is small (the proof technique is identical: Fiedler 1973 forces sweep cut
into the bottleneck axis), so we still call this a corollary.

### Conjecture 1 (Recovery via Cartesian-product meta-graphs).

**Prior result we conjecturally extend:** *Macgregor, P. and Sun, H. (2022),
"A tighter analysis of spectral clustering, and beyond," ICML '22, Theorem 4.*
For an input graph $G$ with $k$ almost-balanced clusters whose meta-graph $M$
is $(\theta, \ell)$-distinguishable and satisfies $\Psi(\ell) \leq \theta^3 /
1600(1+\text{APT})$, standard spectral clustering with $\ell$ eigenvectors
recovers the $k$ clusters up to misclassification volume $\leq 2176(1+\text{APT})
\Psi(\ell) \text{vol}(V_G) / (k\theta^2)$.

**Conjecture (informal).** *Under the hypotheses of Corollaries 1-2, the
meta-graph of the natural product partition refinement of $G = H_1 \square H_2$
is itself the Cartesian product $M_{H_1} \square M_{H_2}$, with
$(\theta', \ell^*)$-distinguishability inherited from the factor meta-graphs
where $\ell^* = \ell_1 + \ell_2$ (or possibly $\ell_1 + \ell_2 - 1$ accounting
for the trivial eigenvector) and $\theta' = \min(\theta_1, \theta_2) / O(1)$.
Consequently, standard spectral clustering with $\ell^* = 1 + d_\epsilon(T^*_{H_2}; H_2)$
eigenvectors of $G$ recovers the slab cut $S^*$ from Corollary 1 up to small
misclassification volume.*

**Empirical evidence (Pred3).** Across $q \in \{20, 30, 40, 50, 70, 100\}$ on
the medium-tree tree-cross-path family ($p = 62$), the *predicted* recovery
dimension $\ell^*$ — defined as the index of the first L_sym eigenvector
aligned with the slab template $\mathbf{1}_{P_q} \otimes f_2(T_p)$ — matches
the *empirical* recovery dimension (smallest $d$ at which NJW $d$-dimensional
$k$-means recovers the tree-axis slab) exactly across all six values. As $q$
grows past the boundary $q^* \approx 40$ where path's $\mu_3$ crosses tree's
$\nu_2$, $\ell^*$ increases by one each time another path eigenvalue dips
below $\nu_2(T_p)$, and the empirical recovery dimension increases
correspondingly. This is the strongest empirical evidence we have that the
framework predicts algorithm behavior quantitatively, not just qualitatively.

**What proving Conjecture 1 would require.**
1. Show that the meta-graph of the natural product partition is exactly
   $M_{H_1} \square M_{H_2}$ as edge-weighted graphs. This is plausible by
   direct computation of inter-cluster cut weights but needs checking.
2. Show that the $(\theta', \ell^*)$-distinguishability of $M_{H_1} \square
   M_{H_2}$ follows from the factor distinguishabilities, with explicit
   constants. This is the technical heart.
3. Apply Macgregor-Sun's Theorem 4 to conclude.

**Status:** Conjecture, with strong empirical support from Pred3 on
tree-cross-path and Pred4 on five other Cartesian product families. Stated
as "future work" in the writeup.

## 2.3 Predictions made by the framework

For each empirical prediction tested, link to its corresponding theorem part:

- **Prediction 1** (spectrum decomposition) ← Corollary 2's input (Fiedler factorization).
- **Prediction 2** (optimum cut indicator avoids $f_2$) ← Corollary 2's conclusion.
- **Prediction 3** (interleaving boundary across $q$) ← Corollary 2 + Conjecture 1 boundary.
- **Prediction 4** (generalization across Cartesian products) ← Corollary 2's universality.
- **Prediction 5** (spectral dimension on non-product graphs) ← Definition's diagnostic value.
- **Prediction 6** (sweep on $f_3$) ← Corollary 2's mechanism (which axis is $\bar g_{S^*}$ aligned with?).

# Section 3: Empirical Verification

The structure of Section 3 should be:

## 3.1 Methodology

- Hardware/software stack: Python, scipy ARPACK for eigenvectors, sklearn KMeans
  with `n_init=10`, k-means++ init.
- Algorithms compared: sweep_cut, kmeans_njw, kmeans_unnormalized,
  kmeans_unnormalized_1d, sweep_on_f_k, higher_order_cheeger.
- Default $\epsilon = 0.01$ for spectral dimension.

## 3.2 Tree-cross-path: the canonical example

Refer to and incorporate findings from `guattery_miller_check.md`:

- Construction: $P_q \square T_p$ where $T_p$ is double tree.
- Three sizes: small (h=3, q=20, n=600), medium (h=4, q=25, n=1550), large
  (h=5, q=50, n=6300).
- **Headline numbers:**
  - Sweep cut: $\Phi$ ratio 1.50, 2.56, 2.52 across sizes (tracks $\sqrt{p}$).
  - 1D k-means: identical to sweep cut.
  - 2D k-means (NJW or unnormalized): 1.000 across all sizes, all seeds.
  - Cut size: sweep + 1D k-means hit $p$ (G-M bad cut); 2D k-means hits $q$
    (optimum).
- Per-eigenvector structure (medium): $f_1$ lives on path (tree-var/path-var
  ratio = 0.018), $f_2$ lives on tree (ratio = 457), $f_3, f_4$ lives on tree
  (ratio = 365).

**Insert figure**: `experiments/plots/guattery_miller_conductance.png`
(left: log-log conductance ratio vs n; right: cut-edge count vs n).

## 3.3 Six predictions — results

For each prediction, summarize the verdict in 1-2 paragraphs and reference the
corresponding figure.

### Prediction 1 (PASS for combinatorial L; FAIL for L_sym)

- **Method:** Compute bottom-8 eigenvalues of L(G), compare to multiset of
  pairwise sums of factor eigenvalues.
- **Combinatorial L result:** rel error 8.7e-11% across all 8 eigenvalues;
  cluster_proj norms = 1.000.
- **L_sym result:** max rel error 110%; factorization fails because path and
  tree factors are non-regular.
- **What this shows:** The framework's spectrum factorization is *exactly* the
  combinatorial-Laplacian Cartesian product spectrum theorem. For algorithms
  that use L_sym (which is most of them), the predictions are inherited
  approximately, with the structural alignment (which eigenvector aligns with
  the slab) preserved even though eigenvalues themselves don't factor.
- **Insert figure:** `experiments/plots/spectral_dim_pred1.png`.

### Prediction 2 (PASS)

- **Method:** Compute inner products $\langle \bar g_{S^*}, f_i \rangle$ for
  bottom-8 L_sym eigenvectors of `tree_cross_path_medium`.
- **Result:** $\langle \bar g, f_2 \rangle = 5 \times 10^{-15}$ (essentially
  zero). $\langle \bar g, f_3 \rangle = 0.700$. Residual at $d=3$ is 0.0105.
- **What this shows:** The optimum cut indicator lives essentially entirely
  in span($f_1, f_3$), not in span($f_1, f_2$). This is the smoking gun for
  why sweep cut (which uses only $f_2$) cannot find the optimum. Any algorithm
  that doesn't include $f_3$ in its embedding cannot recover $S^*$.
- **Insert figure:** `experiments/plots/spectral_dim_pred2.png`.

### Prediction 3 (PASS — load-bearing)

- **Method:** Vary $q \in \{20, 30, 40, 50, 70, 100\}$ at fixed $h = 4$ ($p = 62$).
  For each, identify the predicted recovery dimension $d^*$ (= first
  L_sym eigenvector aligned with the slab template), then check empirical
  recovery via NJW k-means at $d \in \{2, 3, \ldots, 8\}$.
- **Result:** Predicted $d^*$ matches empirical $d^*$ exactly across all six
  values of $q$ (within ±0). The sweep-cut conductance ratio decreases as $q$
  grows ($q=20$: 3.10, $q=30$: 2.07, $q=40$: 1.55, $q=50$: 1.24, $q=70$:
  0.886, $q=100$: 0.620), reflecting that for large $q$ the path-axis slab
  becomes lower-conductance than the tree-axis slab.
- **Subtlety:** For $q \geq 70$, the path-axis slab has lower conductance
  than the tree-axis slab (since cut size scales as $q$ for tree-axis but as
  $p$ for path-axis). So 2D k-means may "succeed" at low conductance but find
  the *wrong* slab axis. The "slab-recovery dim" must specifically check that
  the cut found is *tree-axis*; this is what we report.
- **What this shows:** The framework predicts not just *whether* spectral
  algorithms succeed, but *with how many eigenvectors* they need to. The
  prediction is quantitative and matches across all tested cases.
- **Insert figure:** `experiments/plots/spectral_dim_pred3.png`.

### Prediction 4 (PASS, 5/6)

- **Method:** Test on path×doubletree, path×cycle, cycle×doubletree,
  path×path_unequal, doubletree×doubletree, doubletree×doubletree_unequal.
- **Result:** 5/6 within ±1 of predicted $d^*$. Failure case:
  `path_x_path_unequal` (q=10, p=50) predicted $d^* = 1$, empirical $d^* = 3$.
  Failure mechanism: when both factors are paths, eigenvalue multiplicities
  and degeneracies in the bottom of the spectrum cause ARPACK to pick an
  arbitrary basis within each near-degenerate eigenspace, and the
  slab-template alignment is split across multiple eigenvectors rather than
  concentrated on one.
- **What this shows:** The framework generalizes beyond tree-cross-path to
  most Cartesian products, but breaks down in the presence of eigenvalue
  multiplicities. The fix (in future work) is to state the theorem in terms
  of *invariant subspaces* rather than individual eigenvectors.
- **Insert figure:** `experiments/plots/spectral_dim_pred4.png`.

### Prediction 5 (PASS, qualitative only)

- **Method:** Compute spectral dimension of planted/proxy cuts on
  sbm_2cluster_easy, sbm_2cluster_hard, hsbm_2level (top + leaf),
  random_3regular.
- **Result:**
  - sbm_2cluster_easy: $d_{0.01} = 2$, $d_{0.05} = 2$, $\Phi = 0.093$. **PASS.**
  - sbm_2cluster_hard: $d_{0.01} \geq 30$, $d_{0.05} \geq 30$, $\Phi = 0.379$.
    Spectral dimension is large because the cut is barely discernible at
    this SNR.
  - hsbm_2level_top: $d_{0.01} \geq 30$, $d_{0.05} = 2$. The top-level cut is
    captured at moderate tolerance but not at tight tolerance, consistent
    with hierarchical structure.
  - hsbm_2level_leaf: $d_{0.01} \geq 30$, $d_{0.05} \geq 30$.
  - random_3regular_n1000: $d_{0.01} \geq 30$, $d_{0.05} \geq 30$. **PASS** (no
    cluster structure, so no low-dim signal exists).
- **What this shows:** The spectral-dimension diagnostic gives interpretable
  values on graphs where structure exists at the right SNR (easy SBM, top of
  HSBM at moderate tolerance), and correctly indicates absence of structure
  (random 3-regular). It is *not* a perfect diagnostic across all real graphs;
  hard SBM and HSBM-leaf show the limitation. The qualitative ordering
  (clustered < random) holds.
- **Insert figure:** `experiments/plots/spectral_dim_pred5.png`.

### Prediction 6 (PASS)

- **Method:** Run sweep cut using $f_2$, $f_3$, $f_4$ separately on
  `tree_cross_path_medium`. Also 1D k-means on $f_2$ vs $f_3$.
- **Result:**
  - sweep_on_f2: $\Phi$ ratio 2.56 (path-position cut, GM bad cut).
  - sweep_on_f3: $\Phi$ ratio 1.000 (tree-vertex cut, optimum).
  - sweep_on_f4: $\Phi$ ratio 2.09 (tree-vertex cut, suboptimal).
  - kmeans_1d_on_f2: $\Phi$ ratio 2.59 (path-position cut, GM bad cut).
  - kmeans_1d_on_f3: $\Phi$ ratio 1.000 (tree-vertex cut, optimum).
  - kmeans_2d_f2_f3: $\Phi$ ratio 1.000.
- **What this shows:** The "escape" of 2D k-means is *not* about k-means
  rounding being magical. It's about having access to $f_3$, the tree-Fiedler
  eigenvector. Sweep cut on $f_3$ alone (a 1D method) escapes equally well.
  The interesting axis is which eigenvector is used (which is governed by
  Fiedler 1973 + interleaving), not how it's rounded.

## 3.4 Discrepancies and open puzzles

The framework is broadly empirically supported, but several observations
do not line up cleanly with the corollaries as stated. We catalog each
honestly here, since the reader will encounter all of them in the prediction
data.

### 3.4.1 L_sym factorization fails on irregular factors (Pred1)

Corollary 2's spectrum factorization is exact for the *combinatorial*
Laplacian on Cartesian products of any graphs (this is Fiedler 1973). For
$L_{\text{sym}}$, factorization is exact only when both factors are regular.
On tree-cross-path, neither factor is regular (path has degrees $\{1, 2\}$,
double tree has degrees $\{1, 2, 3\}$), so the L_sym spectrum does not
factor: Pred1 measures pairwise-sum errors of $\approx 110\%$.

What survives: the *structural* prediction (which $G$-eigenvector aligns
with the slab template) is preserved with cluster-projection norm $> 0.99$
across the bottom-8 L_sym eigenvectors. So Corollaries 2-3, while formally
stated for the regular case, empirically hold approximately on irregular
factors. The writeup should clearly note that the corollaries are stated
for the combinatorial Laplacian (where they are exact) and that the L_sym
versions are inherited approximately.

### 3.4.2 Corollary 1 needs a size hypothesis to characterize the optimum slab (Pred3, large $q$)

The version of Corollary 1 we state requires *two* hypotheses: $\mu_2(H_1)
< \nu_2(H_2)$ (bottleneck condition) and $(\star)$ (size condition). Pred3
exposes why both are needed: as $q$ grows past $\sim 70$ (with $p$ fixed at
$62$), $(\star)$ fails because the path becomes longer than the tree, so
slab A (along the bottleneck, with $p=62$ edges) becomes cheaper than
slab B (along the non-bottleneck, with $q$ edges). Sweep cut still finds
slab A — the *same* slab as before — but now slab A is the actual optimum,
so sweep ratio is $< 1$ relative to slab B, and 2D NJW k-means finds yet
another path-axis cut with even lower conductance.

The corollary is correct as stated — it's the regime, not the corollary,
that changes. But it's worth making this explicit: the framework's "sweep
fails" prediction lives in the joint hypothesis region (interleaving $\cap$
$(\star)$), and the empirical experiments at $q \in \{20, 25, 30, 40, 50\}$
all sit in this region, while $q \in \{70, 100\}$ exit it.

### 3.4.3 Pred4 path × path: predicted $\ell^* = 1$, empirical $\ell^* = 3$

For `path_x_path_unequal` ($q=10$, $p=50$), the predicted recovery dimension
is $\ell^* = 1$ (the slab template aligns with $f_2(G)$ directly, since the
bottleneck factor $H_2 = P_{50}$ has its Fiedler vector inherited as
$f_2(G) = \mathbf{1}_{P_{10}} \otimes f_2(P_{50})$). However, sweep cut on
$f_2(G)$ (a 1D method) achieves ratio $1.000$, recovering the optimum
exactly. Yet 2D NJW k-means at $d=2$ achieves only ratio $1.478$, and 3D
NJW k-means at $d=3$ achieves $1.042$.

This is internally inconsistent: a 1D method (sweep on $f_2$) recovers the
optimum, but the 2D extension (NJW with $d=2$) does not. The most likely
explanation is *not* the eigenvalue-multiplicity story originally floated
in Pred4. More plausible candidates:

1. **NJW row-normalization artifact.** NJW normalizes each row of the
   embedding matrix to unit length before $k$-means. With a 2D embedding
   $(f_1(u), f_2(u))$ where $f_1$ is constant (the trivial eigenvector
   $\propto D^{1/2}\mathbf{1}$), row-normalization rescales each point to
   the unit circle. This can disrupt the linear separability that sweep cut
   exploits.
2. **k-means local optima.** With $n=500$, $k=2$, k-means++ initialization,
   and a near-degenerate spectrum at the bottom, k-means may find suboptimal
   centroids that don't bisect along the Fiedler axis.
3. **Eigenvalue near-degeneracy.** The combinatorial Laplacian eigenvalues
   for $P_{10} \square P_{50}$ are pairwise sums $\mu_i(P_{10}) + \nu_j(P_{50})$
   with the path eigenvalues $4 \sin^2(\pi k/(2n))$. The bottom several
   eigenvalues of $G$ are clustered closely, and ARPACK may pick an arbitrary
   basis within the near-degenerate subspace.

We do not currently know which of these is the dominant cause. The writeup
should report the discrepancy and flag (1)-(3) as candidate explanations
without committing to one.

### 3.4.4 Pred5 spectral dimension diagnostic only works at strong SNR

Pred5 reports five datasets:
- Easy SBM: $d_{0.05} = 2$ ✓
- Hard SBM: $d_{0.05} \geq 30$ ✗
- HSBM-top: $d_{0.05} = 2$ ✓
- HSBM-leaf: $d_{0.05} \geq 30$ ✗
- Random 3-regular: $d_{0.05} \geq 30$ ✓ (correctly indicates no structure)

The diagnostic value of $d_\epsilon$ as a standalone tool is therefore
limited. It correctly identifies "no structure" (random 3-regular) and
"clean low-dim structure" (easy SBM, HSBM-top), but it does not distinguish
"hard but real cluster structure" (hard SBM, HSBM-leaf) from "no cluster
structure." This means $d_\epsilon$ is not a perfect substitute for
running the algorithms themselves — it requires a meaningful eigenvalue
gap to be informative.

Honest framing: the diagnostic is useful when the spectrum has clear
low-dim structure (which is exactly when spectral algorithms are guaranteed
to work), and uninformative otherwise. This limits its applicability for
the case where spectral algorithms are *most* needed — the borderline
regime.

### 3.4.5 Summary of discrepancies

The framework is empirically robust within the regime it explicitly covers
(regular-or-near-regular factors, interleaving condition, size condition
$(\star)$, strong SNR). It is empirically less robust in:
- non-regular factors (Pred1's L_sym failure mode);
- size regimes where $(\star)$ fails (Pred3 large-$q$);
- product graphs with eigenvalue near-degeneracies (Pred4 path×path);
- non-product graphs at borderline SNR (Pred5 hard SBM).

These are honest limitations, not framework failures: each one corresponds
to a hypothesis of the corollaries being violated. The writeup should
present each one explicitly and resist the temptation to gloss over them.

## 3.5 Summary verdict

The three corollaries hold up empirically within their stated regime
(regular-or-near-regular factors, interleaving condition, size condition
$(\star)$, strong SNR). Predictions 1, 2, 3, and 6 — which directly probe
the corollaries' mechanisms — pass cleanly. Prediction 4 passes 5/6, with
the single failure (path × path) flagged as an open puzzle in Section 3.4.3.
Prediction 5 passes qualitatively at strong SNR.

Since the corollaries derive from established results, the natural next
step toward an original theoretical contribution is to attempt a proof of
**Conjecture 1**, which would convert the empirical recovery-dimension
prediction (Pred3) into a rigorous guarantee. The clearest path is to
apply Macgregor-Sun's Theorem 4 to the meta-graph $M_{H_1} \square M_{H_2}$,
showing that distinguishability factorizes across Cartesian products. The
empirical evidence (predicted $\ell^*$ matches empirical $\ell^*$ exactly
across all six tested values of $q$) suggests the proof should go through.

## 3.6 Limitations

- Cartesian products are not real-world graphs. The framework is a clean
  special case where matching upper and lower bounds are tight; it does not
  directly apply to typical graph-clustering workloads.
- L_sym factorization breaks for non-regular factors. Corollaries are
  stated for combinatorial $L$; the L_sym versions inherit approximately.
- Corollary 1 requires both bottleneck condition and size condition $(\star)$.
  Outside this regime (e.g., large $q$ in tree-cross-path), sweep cut may
  succeed and the framework's "sweep fails" prediction does not apply.
- Eigenvalue near-degeneracies disrupt eigenvector-level predictions. The
  path × path failure (Pred4) is unresolved.
- Spectral dimension as a standalone diagnostic is informative only at
  strong SNR (Pred5). It cannot distinguish hard cluster structure from
  no cluster structure.

# Figures referenced

The user has these plots already generated:

- `experiments/plots/guattery_miller_conductance.png` (Section 3.2)
- `experiments/plots/spectral_dim_pred1.png` (Section 3.3, Prediction 1)
- `experiments/plots/spectral_dim_pred2.png` (Section 3.3, Prediction 2)
- `experiments/plots/spectral_dim_pred3.png` (Section 3.3, Prediction 3)
- `experiments/plots/spectral_dim_pred4.png` (Section 3.3, Prediction 4)
- `experiments/plots/spectral_dim_pred5.png` (Section 3.3, Prediction 5)

Use `\includegraphics{...}` placeholders. Do not generate new figures unless
explicitly asked.

# Suggested additional figures (offer to user, do not generate without ask)

1. **Schematic of the tree-cross-path graph** (cartoon showing path × double-tree
   structure with the slab cut highlighted).
2. **Eigenvalue interleaving diagram** (showing which factor contributes which
   eigenvalue for varying $q$, with the predicted recovery dimension annotated).
3. **Conceptual figure for spectral dimension** (cartoon showing $\bar g_{S^*}$
   as a vector and its projection onto increasing-dimensional subspaces, with
   the residual norm $r(d)$ illustrated).

# BibTeX entries to add

The current main.tex has only 3 references. Need to add ~20 more. List them
in a `\begin{thebibliography}` block (matching existing style) or in a
separate .bib file. The full list is:

cheeger1970lower, fiedler1973algebraic, fiedler1975property,
donath1973lower, alon1985isoperimetric, alon1986eigenvalues, mohar1989isoperimetric,
hagen1992new, pothen1990partitioning, guattery1995performance,
guattery1998quality, houdre1996isoperimetric, chung1998isoperimetric,
spielman1996spectral, spielman2007spectral, ng2001spectral,
louis2012many, lee2014multiway, kwok2013improved, peng2017partitioning,
kolev2016note, mizutani2021improved, macgregor2022tighter.

Plus the existing 3: luxburg2007tutorial, peng2015partitioning (=peng2017partitioning;
note the COLT/SICOMP year discrepancy), bravo2023fast.

# Settled technical points (resolved before any LaTeX writing)

This section was originally "Critical things to verify before writing LaTeX,"
but those items have now been resolved and the resolutions are baked into
Theorems 1, 2, 3 above. Recording the resolutions here so the reasoning
survives compactification.

## Settled point 1: slab direction in Corollary 1

When $\mu_2(H_1) < \nu_2(H_2)$ (so $H_1$ is the bottleneck factor):
- The **optimum slab** is $S^* = V(H_1) \times T^*$ where $T^* \subseteq V(H_2)$
  is the optimal cut of the **non-bottleneck factor** $H_2$.
- The **Guattery-Miller bad cut** (what sweep cut returns) is $S_{\text{bad}}
  = T_{\text{bad}} \times V(H_2)$ where $T_{\text{bad}} \subseteq V(H_1)$ is
  the bottleneck factor's optimal cut.

Why: the optimum cut size scales as $|V(H_1)| \cdot |\partial T^*|$, while
the sweep cut scales as $|V(H_2)| \cdot |\partial T_{\text{bad}}|$. The
non-bottleneck factor $H_2$ has a smaller optimal cut size (because its
$\nu_2$ is larger, so via Cheeger applied within $H_2$, the optimal cut
$|\partial T^*|$ is small relative to $|V(H_2)|$).

For tree-cross-path with $H_1 = P_q$ (bottleneck), $H_2 = T_p$: optimum cut
size = $q \cdot 1 = q$ (cut the root edge in each path-copy of the tree),
GM bad cut = $p \cdot 1 = p$ (cut the rungs between two path-positions).
Empirically confirmed: pred4.md reports "slab axis = 1" for path×doubletree,
i.e. axis-1 = $H_2$ axis = tree axis.

## Settled point 2: Cheeger-saturation expression in Corollary 3

The right expression for the sweep-cut/optimum ratio is
$$\frac{\Phi_{\text{sweep}}(G)}{\Phi(\text{opt})} = \Theta\left(\frac{\sqrt{\mu_2(H_1)}}{\nu_2(H_2)}\right),$$
NOT $\Theta(\sqrt{\nu_2/\mu_2})$ (wrong) or $\Theta(\sqrt{\mu_2 \nu_2})$ (also wrong).

Derivation (recapped):
- $\Phi_{\text{sweep}}(G) = \Theta(\sqrt{\mu_2(H_1)})$ from Cheeger applied
  within $H_1$ (the only factor sweep cut "sees").
- $\Phi(\text{opt}) = \Theta(\nu_2(H_2))$ from Chung-Tetali (slab structure)
  plus Cheeger lower bound applied within $H_2$.

Numerical verification: ratio $\Theta(\sqrt{\mu_2(P_q)}/\nu_2(T_p))
= \Theta((1/q)/(1/p)) = \Theta(p/q)$, matches empirical 1.50, 2.56, 2.52
across the three sizes in $q/p$ pairs (20/30, 25/62, 50/126).

## Settled point 3: framing of "Cheeger saturation"

The cleanest framing is:
- Cheeger upper bound applied to $\lambda_2(G) = \mu_2(H_1)$ gives
  $\Phi(G) \leq \sqrt{2\mu_2(H_1)}$. Sweep cut **saturates this upper bound**
  asymptotically.
- The actual optimum $\Phi(G) = \Theta(\nu_2(H_2))$ is **strictly larger**
  than the Cheeger lower bound $\lambda_2(G)/2 = \mu_2(H_1)/2$ on this graph,
  because the optimum cut lives in a different invariant subspace
  (span$(f_1, f_3)$, not span$(f_1, f_2)$).
- $\lambda_2(G)$ is the wrong eigenvalue for understanding the optimum;
  $\lambda_3(G) = \nu_2(H_2)$ is the right one.

This re-framing — that the optimum's spectral signature is $\lambda_3(G)$,
not $\lambda_2(G)$ — is the conceptual through-line of the whole paper.

## Settled point 4: Prediction 3's q-boundary

At $q = 40$ with $h = 4$ ($p = 62$): path's $\mu_3 \approx 4\pi^2/q^2 \approx 0.025$
crosses tree's $\nu_2 \approx 0.025$. So the boundary between "interleaving
holds" and "interleaving fails" is at $q^* \approx 40$. From pred3.md:
- $q = 30$ (interleaving holds): NJW d=2 succeeds.
- $q = 40$ (boundary): NJW d=2 fails (ratio 2.40), NJW d=3 succeeds.
- $q = 50$ and up: NJW needs progressively more eigenvectors.

Predicted recovery dimension matches empirical at every $q$. Confirmed.

# Style notes for the LaTeX writeup

- Use `\begin{corollary}`, `\begin{conjecture}`, `\begin{definition}` (no
  `\begin{theorem}` — we have no original theorems, only corollaries of
  prior work and one conjecture).
- Need to add `\usepackage{amsthm}` and `\newtheorem{corollary}{Corollary}`,
  `\newtheorem{conjecture}{Conjecture}`, `\newtheorem{definition}{Definition}`
  declarations to preamble.
- Include `\usepackage{graphicx}` for figures.
- Use `\Phi`, `\lambda`, `\mu`, `\nu` consistently.
- Tables: use `booktabs` (already loaded). Format with `\toprule`, `\midrule`, `\bottomrule`.
- Citations: numeric, `\cite{...}` (existing setup uses this).
- Section labels: `\label{sec:motivation}`, `\label{sec:theory}`, `\label{sec:empirical}`.
- Subsection labels for predictions: `\label{sec:pred1}`, etc.
- Corollary/Conjecture labels: `\label{cor:slab}`, `\label{cor:spectral-dim}`,
  `\label{cor:cheeger-tight}`, `\label{conj:recovery}`.

# Ordering of edits

1. Add packages to preamble (amsthm, graphicx, theorem environments).
2. Add 3 new sections after existing Section 7 ("Fallback Plan"), before the
   appendix.
3. Add bibliography entries (don't remove the existing 3, add ~20 more).
4. Sanity check: confirm new sections don't break existing structure.

# Things I should NOT do

- Do not generate new figures (user did not ask).
- Do not modify the existing project plan (Sections 1-7).
- Do not remove the existing References block; just extend it.
- Do not claim that Conjecture 1 is proven (it's a conjecture).
- Do not oversell the framework — it's a clean special case + diagnostic, not
  a new general theory.
- Do not skip the L_sym vs combinatorial Laplacian caveat.
- Do not replace Macgregor-Sun's meta-graph framework as the home of this work
  — we *specialize* it, we don't replace it.
