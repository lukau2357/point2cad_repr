# B-Rep Construction Pipeline

This document covers the complete pipeline from a pre-segmented point cloud to
a valid STEP file: surface fitting, adjacency detection, surface-surface
intersection, vertex finding, trimming, arc splitting, greedy oracle filtering,
wire assembly, face construction, and STEP export.

**Pipeline input:** a segmented point cloud — $n$ labelled clusters
$\{C_0, \ldots, C_{n-1}\}$, each an $(N_i \times 3)$ array of 3D points
belonging to one surface patch.  Segmentation is assumed given (e.g.\ from
Point2CAD).

**Pipeline output:** `TopoDS_Shape` (shell or solid) exportable to STEP.

---

## 1 — Surface fitting and OCC geometry objects

Each cluster $C_i$ is fitted with the surface type assigned by the upstream
segmentation.  The five supported types and their OCC representations are:

| Type | Fitted parameters | OCC class |
|---|---|---|
| Plane | unit normal $\mathbf{a}$, offset $d$ ($\mathbf{a}\cdot\mathbf{x}=d$) | `Geom_Plane` |
| Sphere | centre $\mathbf{c}$, radius $r$ | `Geom_SphericalSurface` |
| Cylinder | axis direction $\mathbf{a}$, axis point $\mathbf{c}$, radius $r$ | `Geom_CylindricalSurface` |
| Cone | apex, axis direction, half-angle $\alpha$ | `Geom_ConicalSurface` |
| INR (freeform) | trained MLP encoder–decoder | `Geom_BSplineSurface` (see below) |

Analytical surfaces (plane–cone) are converted by direct parameter marshalling
into the corresponding `gp_` geometry primitives.

**INR → BSpline conversion.**  The INR represents a surface implicitly as a
trained neural network $f_\theta : \mathbb{R}^2 \to \mathbb{R}^3$ mapping UV
parameters to 3D positions.  Because OCC requires an explicit parametric
representation, the INR is converted post-hoc:

1. **Sample.** Evaluate $f_\theta$ on a regular $G \times G$ grid in UV space
   (default $G = 100$), extended by a margin $\delta = 0.05$ beyond the
   cluster's UV bounding box to ensure the boundary circles do not get clipped
   by the BSpline knot span:
   $$Q_{kl} = f_\theta\!\left(\frac{k}{G-1},\, \frac{l}{G-1}\right), \quad k, l = 0, \ldots, G-1$$

2. **Fit.** Pass the $G \times G$ grid to
   `GeomAPI_PointsToBSplineSurface(points, deg_min=3, deg_max=8, C^2, tol=10^{-3})`.
   OCC selects the minimal degree in $[3, 8]$ needed to satisfy the $C^2$
   continuity and $10^{-3}$ approximation tolerance, then solves the
   tensor-product least-squares system (see `brep_reconstruction_plan.md`
   Section 4.9 for the mathematics).

3. **Result.** A `Geom_BSplineSurface` with knot vectors determined by OCC,
   parameter domain $[u_1, u_2] \times [v_1, v_2] \approx [-1.05, 1.05]^2$.
   The surface is **not periodic** even if the underlying shape is geometrically
   closed — this requires explicit handling at the face-construction stage
   (Section 10).

---

## 2 — Cluster spacing and proximity

Two spacing measures are used:

**Global reference spacing** $\sigma$ — the $p$-th percentile of nearest-neighbour
distances across all cluster points (default $p = 100$, i.e.\ the maximum):

$$
\begin{align*}
D_i &= \{\min_{y \in C_i, y \neq x}\|x - y\|_{2} \mid x \in C_i\} \\
\sigma &= \operatorname{percentile}_p\!\left(\bigcup_{i=1}^{N}D_i\right)
\end{align*}
$$

$\sigma$ is returned for backward compatibility but does not directly control
adjacency detection (see below).

**Per-cluster local spacing** $\sigma_i$ — the $p$-th percentile of
nearest-neighbour distances within cluster $C_i$ alone:

$$\sigma_i = \operatorname{percentile}_p(D_i)$$

Per-cluster spacings drive the per-pair adjacency threshold (Section 3) and
are also used as normalisation denominators in the greedy oracle filter's
scoring (Section 7).

Per-cluster KDTrees and NN-distance percentiles are built once by
`build_cluster_proximity`, before adjacency detection, and reused by both the
adjacency computation and the greedy oracle filter.  The same
`--spacing_percentile` parameter (default 100) controls both, avoiding a
redundant second parameter.

**Output.**
- `cluster_trees` — one KDTree per cluster.
- `cluster_nn_percentiles` — $\sigma_i$ per cluster.  Passed to
  `compute_adjacency_matrix` as `local_spacings` to avoid recomputing.

---

## 3 — Cluster adjacency matrix

**Goal.** Determine which surface pairs share a physical boundary and therefore
need to be intersected.  Intersecting all $\binom{n}{2}$ pairs is wasteful and
produces spurious curves for non-adjacent surfaces.

**Algorithm.**  For each unordered pair $(i, j)$ with $i < j$:

1. Let $C_\text{larger}$, $C_\text{smaller}$ be the two clusters sorted by size.
2. Build a KDTree on $C_\text{larger}$.
3. Query every point of $C_\text{smaller}$, obtaining nearest-neighbour
   distances $\{d_k\}$.
4. Compute the **per-pair adaptive threshold**:

$$\tau_{ij} = \tau \cdot \max(\sigma_i, \sigma_j)$$

where $\tau$ is the spacing factor (CLI: `--spacing_factor`, default $\tau = 2$) and $\sigma_i,
\sigma_j$ are the local spacings from Section 2.

5. Declare clusters adjacent if $\min_k d_k \le \tau_{ij}$.

Using the plain minimum rather than a percentile statistic makes the test
sensitive to even a single pair of very close points across the boundary,
which is the correct condition for adjacency.

**Boundary strips.**  For each adjacent pair $(i, j)$, the boundary strip
is constructed from the $C_\text{smaller} \to C_\text{larger}$ query:

$$B_{ij} = \{x \in C_\text{smaller} : d(x, C_\text{larger}) \le \tau_{ij}\} \cup \{x \in C_\text{larger} : d(x,C_\text{smaller}) \le \tau_{ij}\}$$

i.e.\ the union of the matching points from the smaller cluster and their
nearest-neighbour images in the larger cluster.  Note that the strip is
asymmetric: only one direction of NN query is performed.

**Full-adjacency mode.**  When `--full_adjacency` is set all $\binom{n}{2}$
pairs are intersected.  No adjacency matrix or boundary strips are computed.

**Output.**
- $A \in \{0,1\}^{n \times n}$ — symmetric Boolean adjacency matrix.
- `boundary_strips[(i,j)]` — $(|B_{ij}| \times 3)$ float32 array for each adjacent pair.
- `per_pair_thresholds[(i,j)]` — $\tau_{ij}$ value per adjacent pair.
---

## 4 — Vertex finding (`compute_vertices_intcs`)

### Which curve pairs can share a vertex

A vertex $v$ lies on $k \ge 3$ surfaces simultaneously.  For two edges
$C_{ij}$ and $C_{kl}$ to share a vertex, the codimension argument requires
exactly one shared surface index:

$$|\{i,j\} \cap \{k,l\}| = 1$$

**Dimension-counting justification.**  Each smooth surface in $\mathbb{R}^3$
is a codimension-1 submanifold.  The intersection of $n$ surfaces in general
position has codimension $n$:

| $n$ surfaces | Expected codimension | Expected dimension |
|---|---|---|
| 2 | 2 | 1 (a curve — an edge) |
| 3 | 3 | 0 (a point — a vertex) |
| 4 | 4 | $-1$ (generically empty) |

With $|\{i,j\} \cap \{k,l\}| = 1$, say the shared surface is $F_m$:

$$C_{ij} \cap C_{kl} = F_i \cap F_j \cap F_k \cap F_l / \text{shared} = F_m \cap F_a \cap F_b$$

Three surfaces $\Rightarrow$ vertex.  With $|\{i,j\} \cap \{k,l\}| = 0$, four
distinct surfaces meet, which is generically empty in $\mathbb{R}^3$.

Additionally, requiring both curves to lie on a common surface $F_m$ reduces
the effective ambient dimension from 3 to 2, so two curves on $F_m$ have
expected intersection dimension $1 + 1 - 2 = 0$ — a point.

**Consequence:** the triangle-in-adjacency-graph condition (all three pairs
adjacent) is unnecessarily strict and fails for non-adjacent curved surface
pairs (e.g.\ a cylinder and a side plane that are not directly adjacent but
share a boundary with a common cap).

### Curve–surface intersection (`GeomAPI_IntCS`)

The previous implementation used `GeomAPI_ExtremaCurveCurve` (closest approach
between two curves) — this is fundamentally wrong for finding intersection
points because it minimises the distance function between curves and can
produce phantom points (midpoints that do not lie on either curve's actual
intersection with the shared surface).

The current implementation uses `GeomAPI_IntCS` (curve–surface intersection).
For each pair of edges $e_a = (i, k)$ and $e_b = (j, k)$ sharing a surface
$F_k$, the vertex is found by intersecting the **third edge's curve**
$\mathbf{C}_{ij}$ with the shared surface $F_k$:

$$\mathbf{V} = \mathbf{C}_{ij}(t^*) \quad \text{where} \quad
\mathbf{n}_k \cdot \mathbf{C}_{ij}(t^*) = d_k$$

This avoids both the coplanar-curve degeneracy and the phantom-vertex problem.

### Why `GeomAPI_IntCS` instead of `GeomAPI_ExtremaCurveCurve`

**Coplanar curves.** When edges $e_a = (i, k)$ and $e_b = (j, k)$ share a
planar surface $F_k$, both curves lie exactly in the plane.  The 3D distance
function is zero everywhere on the intersection locus, making
`ExtremaCurveCurve`'s Jacobian rank-deficient — it returns no extrema.

**2D intersection also fails.** Projecting to $F_k$'s UV frame and using
`Geom2dAPI_InterCurveCurve` solves the rank-deficiency, but a line × conic in
2D yields up to two solutions.  In the plane∩cone case, both solutions lie on
the physical ellipse arc, so proximity filtering cannot distinguish real from
spurious.

**IntCS is well-posed.** $\mathbf{C}_{ij}$ is the intersection of $F_i$ and
$F_j$, neither of which is $F_k$.  Generically $\mathbf{C}_{ij}$ crosses
$F_k$ transversally, so the problem has a non-zero derivative and a unique
local solution.

### Algorithm (`compute_vertices_intcs`)

The algorithm has three phases: candidate generation, deduplication, and
triple-based edge attribution.

#### Phase 1 — Candidate generation

The outer loop iterates over **faces**, not edge pairs.  For each face $f$,
consider all unordered pairs of edges incident to $f$.

Let $\mathcal{E}$ denote the set of intersection edges (keyed by ordered surface
pairs $(i,j)$ with $i < j$).  Define the edges incident to face $f$:

$$\mathcal{E}_f = \{(i,j) \in \mathcal{E} \mid i = f \;\text{or}\; j = f\}$$

For each face $f \in \{0, \dots, n-1\}$, enumerate all unordered pairs of
distinct edges on $f$ by iterating over indices
$0 \le \alpha < \beta < |\mathcal{E}_f|$ into some fixed ordering of
$\mathcal{E}_f$.  Let $e_\alpha, e_\beta$ be the two selected edges.

For edge $e_\alpha = (i_\alpha, j_\alpha)$, let $g$ be the surface in
$e_\alpha$ other than $f$ (i.e. $g = j_\alpha$ if $i_\alpha = f$, else
$g = i_\alpha$).  Similarly, for $e_\beta = (i_\beta, j_\beta)$, let $h$ be
the surface other than $f$.  If $g = h$, both edges involve the same surface
pair — skip.  Otherwise the surface triple is $\{f, g, h\}$.

Generate candidates by intersecting each edge's curves with the third surface:

$$\forall\; \mathbf{C} \in \text{curves}(e_\alpha):\quad
  \text{IntCS}(\mathbf{C},\, S_h) \;\to\; \text{emit candidate}\;
  (\mathbf{p},\;\{f,g,h\}) \;\;\text{for each solution } \mathbf{p}$$

$$\forall\; \mathbf{C} \in \text{curves}(e_\beta):\quad
  \text{IntCS}(\mathbf{C},\, S_g) \;\to\; \text{emit candidate}\;
  (\mathbf{p},\;\{f,g,h\}) \;\;\text{for each solution } \mathbf{p}$$

**Why iterate over faces?** A vertex is the meeting point of three surfaces
$f, g, h$.  Those three surfaces define three edges: $(f,g)$, $(f,h)$, and
$(g,h)$.  Iterating over face $f$ and taking pairs of edges on $f$ naturally
enumerates all surface triples where $f$ participates.

**Why two IntCS calls per pair?** Given edges $e_a = (f,g)$ and $e_b = (f,h)$:

- `IntCS(C_{fg}, S_h)` finds where the curve on surfaces $f \cap g$ hits
  surface $h$ — i.e. the point on all three surfaces.
- `IntCS(C_{fh}, S_g)` finds the same point from the other direction.

Both should produce the same vertex (up to numerical noise), but either call
may fail or return different numbers of points depending on the curve and
surface types.  Running both provides redundancy: if one fails (e.g. a
BSpline curve that IntCS can't solve), the other may succeed.

**Why not intersect the "third edge" $(g,h)$ with surface $f$?**  The code
*does* effectively do this — when the outer loop reaches face $g$, it will
pair edge $(g,f)$ with edge $(g,h)$, generating `IntCS(C_{gf}, S_h)` and
`IntCS(C_{gh}, S_f)`.  The face-based iteration covers all three orientations
of each triple without explicitly computing the third edge.

**Note on `_as_safe_curve`:** IntCS is *not* wrapped with `_as_safe_curve`
here.  Unlike `ProjectPointOnCurve`, `GeomAPI_IntCS` is an analytical solver
that handles infinite curves natively.  Trimming the curve would restrict the
parameter domain and miss intersections outside the trim range.

#### Phase 2 — Greedy deduplication

Multiple IntCS calls for the same triple (and for different triples sharing
the same physical vertex) produce near-duplicate candidates.  These are merged
by greedy clustering.

Let $\mathbf{c}_1, \dots, \mathbf{c}_N$ be the candidate positions (in
arbitrary order) and $\tau$ the merge threshold (default $\tau = 10^{-3}$).
Maintain a boolean array $\text{used}[1..N]$, initially all false.

For $i = 1, \dots, N$: if $\text{used}[i]$ is true, skip.  Otherwise, collect
the cluster:

$$\mathcal{G}_i = \{j \mid j \ge i,\; \lVert \mathbf{c}_j - \mathbf{c}_i \rVert < \tau,\; \neg\,\text{used}[j]\}$$

Set $\text{used}[j] \leftarrow \text{true}$ for all $j \in \mathcal{G}_i$.
Emit a merged vertex with position $\bar{\mathbf{c}} = \frac{1}{|\mathcal{G}_i|}\sum_{j \in \mathcal{G}_i} \mathbf{c}_j$
and triple set $\mathcal{T} = \bigcup_{j \in \mathcal{G}_i} \text{triples}(j)$.

Two candidates at the same physical vertex typically differ by
$\sim 10^{-6}$ (numerical noise from IntCS), well within $\tau$.  Different
physical vertices are separated by at least the edge length of the model
($\sim 0.1$ in normalised coordinates), well outside $\tau$.

#### Phase 3 — Triple-based edge attribution

Each merged vertex $v$ carries a set of surface triples $\mathcal{T}_v$.
For each triple $\{a, b, c\} \in \mathcal{T}_v$ (with $a < b < c$), the
vertex should be attributed to all three edges $(a,b)$, $(a,c)$, $(b,c)$ —
but **only if all three edges exist** in $\mathcal{E}$:

$$\mathcal{E}_v = \bigcup_{\substack{\{a,b,c\} \in \mathcal{T}_v \\ (a,b) \in \mathcal{E},\; (a,c) \in \mathcal{E},\; (b,c) \in \mathcal{E}}}
\bigl\{(a,b),\; (a,c),\; (b,c)\bigr\}$$

Triples where any of the three edges is missing from $\mathcal{E}$ are
rejected.  If $\mathcal{E}_v = \emptyset$, vertex $v$ is dropped entirely.

**Why require all three edges?** A vertex at the junction of $f, g, h$ must
lie on all three intersection curves $(f,g)$, $(f,h)$, $(g,h)$.  If any of
these edges is missing from the intersection dictionary, the vertex cannot be
properly trimmed — `trim_by_vertices` would have no curve to project onto for
the missing edge.  More importantly, this acts as a **geometric filter**: when
an infinite curve (e.g. a `Geom_Line` from plane∩plane) is intersected with a
surface, IntCS may return points where the mathematical intersection exists
but far outside the physical model.  At such spurious locations, typically
only two of the three surface pairs are adjacent (the third pair is too far
apart to be detected by adjacency).  Requiring all three edges rejects these
without any distance-based tolerance.

**Why not attribute via projection?**  An earlier approach projected each
vertex onto all intersection curves and attributed it to any edge within a
distance tolerance.  This is fragile: it introduces a tunable threshold, and
for curves that pass near (but not through) the vertex, false attributions
produce extra split points that break wire assembly.  The triple-based approach
is threshold-free for attribution — the deduplication threshold is the only
tunable parameter, and it only affects which candidates merge together, not
which edges a vertex belongs to.

### Infinite-domain curves and parameter bounding

`GeomAPI_IntSS` returns `Geom_Curve` objects (the base pythonocc handle type).
For several intersection types the parameter domain is unbounded:

| Intersection | Returned type | Domain |
|---|---|---|
| Plane ∩ Cone (steep angle) | `Geom_Hyperbola` | $(-\infty, +\infty)$ |
| Plane ∩ Cone (generator angle) | `Geom_Parabola` | $(-\infty, +\infty)$ |
| Plane ∩ Plane | `Geom_Line` | $(-\infty, +\infty)$ |
| Plane ∩ Cylinder (secant) | `Geom_Line` × 2 | $(-\infty, +\infty)$ |
| Plane ∩ Cylinder / Cone (generic) | `Geom_Ellipse` | $[t_0, t_1]$ finite |
| Cone ∩ Cylinder, Cone ∩ Cone, … | `Geom_BSplineCurve` | $[0, 1]$ |

OCC internally parameterises a hyperbola as

$$\mathbf{P}(t) = \mathbf{C} + a\cosh(t)\,\hat{\mathbf{X}} + b\sinh(t)\,\hat{\mathbf{Y}}$$

When `GeomAPI_ProjectPointOnCurve` or `GeomAPI_IntCS` is called
without explicit parameter bounds, OCC's internal solver samples the curve at
extreme parameter values.  Because $\cosh(710) > \texttt{DBL\_MAX} \approx
1.8 \times 10^{308}$, this throws:

```
Standard_NumericError: Result of Cosh exceeds the maximum value Standard_Real
```

`Geom_Line` does not cause this error because its evaluation is linear
($\mathbf{P}(t) = \mathbf{P}_0 + t\,\hat{\mathbf{d}}$) and IEEE 754 handles
large-but-finite floating-point magnitudes gracefully.  `Geom_Parabola`
evaluates $t^2$ so in principle overflows at $|t| > 10^{154}$, but in practice
OCC's solver converges before reaching such values; however, it is bounded
conservatively along with the others.

#### Analytical parameter bounds

For a model normalised to extent $L$ (default $L = 2.0$, conservative for a
unit-cube model), the minimum parameter interval that covers all real vertices
is derived from each curve's own geometry:

**Hyperbola** — a flat conservative bound $|t| \le 10$ is used.
$\cosh(10) \approx 11013$, $\sinh(10) \approx 11013$ — completely safe from
overflow.  The analytical formula $|t| \le \cosh^{-1}(L/a)$ breaks when the
hyperbola's major radius $a > L$: since $L/a < 1$, $\cosh^{-1}$ is undefined
and the bound collapses to $\approx 0$, cutting off the actual vertex parameters.
Because $a$ can exceed $L$ for plane-cone intersections with wide cone angles,
the model-size-independent bound $|t| \le 10$ is used unconditionally.

**Parabola** — the parameterisation is $x(t) = t^2/(4f)$, $y(t) = t$ where
$f$ is the focal length.  Covering lateral extent $|y| \le L$ requires
$|t| \le L$; covering axial extent $|x| \le L$ requires $|t| \le 2\sqrt{fL}$.
Taking the maximum:

$$|t| \le \max\!\left(L,\; 2\sqrt{fL}\right), \qquad f = \texttt{Focal()}$$

**Line** — arc-length parameterisation; covering extent $L$ requires $|t| \le L$.

**Other / unknown infinite type** — conservative fallback $|t| \le 10$.

#### Why overloaded constructors are insufficient

`GeomAPI_ProjectPointOnCurve` and `GeomAPI_IntCS` have overloads that accept
explicit parameter bounds.  In practice these overloads do **not** prevent the
overflow, because OCC's internal solver infrastructure dispatches through a
`GeomAdaptor_Curve` whose `FirstParameter()` / `LastParameter()` still query
the underlying `Geom_Hyperbola` — returning ±`Precision::Infinite()` (≈ ±2×10¹⁰⁰)
regardless of the bounds passed at the API level.

#### Fix: pre-wrap in `Geom_TrimmedCurve`

`Geom_TrimmedCurve(base, u_min, u_max)` overrides the virtual `FirstParameter()` /
`LastParameter()` methods at the C++ level, so every OCC solver that queries
these on the handle gets the trim bounds directly.  The function
`_as_safe_curve(curve, model_extent=2.0)` in `surface_intersection.py`:

1. Returns the curve unchanged if its bounds are already finite.
2. Otherwise, classifies the curve via `GeomAdaptor_Curve.GetType()`, computes
   the analytical `t_bound` as above, and returns
   `Geom_TrimmedCurve(curve, -t_bound, +t_bound)`.

All calls to `GeomAPI_ProjectPointOnCurve` and `GeomAPI_IntCS` in
the pipeline go through `_as_safe_curve` first, using the standard
constructors on the resulting (always-finite) curve handle.

`Geom_TrimmedCurve` preserves the underlying parameterisation, so vertex
parameters obtained from the trimmed curve are directly usable with the
original curve's `Value(t)` method.

---

## 5 — Intersection curve trimming

### Why trimming is needed

`GeomAPI_IntSS` returns intersection curves over their full parameter domain.
A plane–plane intersection yields a `Geom_Line` with domain $(-\infty, +\infty)$.
All other pairs return a `Geom_TrimmedCurve` with finite bounds, but those bounds
cover the full analytic extent of the curve, not just the shared physical boundary.

The decision of whether to trim is made by checking the parameter bounds:

$$\text{trim if} \quad |t_0| \ge \texttt{Precision::Infinite()} \quad \text{or} \quad |t_1| \ge \texttt{Precision::Infinite()}$$

where $\texttt{Precision::Infinite()} = 10^{100}$.  Note that `math.isfinite`
returns `True` for $10^{100}$, so the comparison must use `abs(t) >= _OCC_INF`
rather than `not math.isfinite(t)`.  In practice only `Geom_Line`
(plane $\cap$ plane) triggers trimming.

### Vertex-based trimming (`trim_by_vertices`)

Trimming uses the B-Rep vertices computed in Section 4 (vertex finding runs on
the raw untrimmed curves first, then the results guide trimming).  Curves are
handled differently depending on their topology:

**Closed curves** are left unchanged.  `build_edge_arcs` (Section 6) will
later split them at their incident vertices.  Trimming here would discard the
wrap-around arc needed by the second adjacent face.

**Phantom filter** (open curves only).  When an edge has multiple open curves
(e.g.\ OCC returns both generators of a cylinder for a plane $\cap$ cylinder
pair), the real curve has the incident vertices lying on it (projection
distance $\approx 0$); the phantom generator is displaced by $\approx 2r$.
Any open curve whose minimum vertex projection distance exceeds
$\varepsilon_\text{phantom} = 10^{-3}$ is discarded.  The filter is skipped
when fewer than 2 vertices are available (fail-safe).

**Open curves with $\ge 2$ incident vertices.**  Project the incident vertex
positions $\{v_k\}$ onto the curve via `GeomAPI_ProjectPointOnCurve` to obtain
parameters $\{t_k^*\}$. The trim interval is

$$t_\text{min} = \min_k t_k^*, \qquad t_\text{max} = \max_k t_k^*$$

Extended by a relative margin $\alpha = 0.05$:

$$t_\text{min} \leftarrow t_\text{min} - \alpha(t_\text{max} - t_\text{min}), \qquad
  t_\text{max} \leftarrow t_\text{max} + \alpha(t_\text{max} - t_\text{min})$$

A `Geom_TrimmedCurve` is constructed from the original curve and the computed
interval.

**Open curves with $< 2$ incident vertices.**  If the curve already has finite
OCC bounds it is kept as-is; if it has infinite bounds (e.g.\ a `Geom_Line`)
it is discarded because there is no vertex support to guide trimming.

Curves that remain infinite after trimming are also discarded.

---

## 6 — Arc splitting

Each intersection curve $C$ may pass through multiple vertices.  The output
replaces curves with **arcs** — sub-intervals bounded by vertices.

### Closure test

`GeomAPI_IntSS` always returns `Geom_TrimmedCurve`, so `IsPeriodic()` is
always `False` even when the underlying geometry is a full circle.  Closure
is detected geometrically:

$$\text{closed} \iff \|C(t_\text{min}) - C(t_\text{max})\| < \varepsilon_\text{close}, \qquad \varepsilon_\text{close} = 10^{-4}$$

Full analytical circles give endpoint distance $\sim 10^{-17}$; OCC BSpline
curves (e.g. cone∩cylinder) may have small endpoint gaps $\sim 10^{-6}$;
trimmed open lines give $\sim 10^{-1}$.  The $10^{-4}$ threshold captures
nearly-closed BSplines while remaining well-separated from open curves.

### Splitting rules

Let $t_1^* \le \cdots \le t_k^*$ be the sorted vertex parameters on curve $C$:

| Curve | Vertices $k$ | B-Rep arcs produced |
|---|---|---|
| Closed | 0 | 1 closed-loop arc over $[t_\text{min}, t_\text{max}]$ |
| Closed | $k \ge 1$ | $k-1$ interior arcs $[t_1^*, t_2^*], \ldots$ + 1 wrap-around arc |
| Open | 0 | 1 arc with `v_start = v_end = None` |
| Open | $k \ge 1$ | $k-1$ interior arcs; boundary tails discarded |

Boundary tails of open curves are discarded because they lack a vertex at the
trim boundary and are not valid B-Rep edges in a closed solid.

**Wrap-around arc** for closed curves with $k \ge 1$: from $t_k^*$ back to
$t_1^*$ with parameter end $t_1^* + (t_\text{max} - t_\text{min})$.
Constructed by calling `BasisCurve()` on the `Geom_TrimmedCurve` to obtain the
underlying periodic curve (e.g.\ `Geom_Circle`), which accepts parameters
beyond $[t_\text{min}, t_\text{max}]$.  If `BasisCurve()` is unavailable or
OCC rejects the extended parameter range, the wrap-around arc is silently
skipped — no seam-split fallback is attempted.

Each arc dict carries:
- `curve` — `Geom_TrimmedCurve` for this arc's interval
- `v_start`, `v_end` — vertex indices (`None` for closed-loop arcs)
- `t_start`, `t_end` — parameter bounds
- `closed` — `True` for closed-loop arcs

---

## 7 — Greedy oracle filter

The greedy oracle filter replaces the earlier pipeline of proximity-based
filters (`filter_vertices_by_proximity`, `filter_curves_by_proximity`,
`filter_arcs_by_proximity`) and the progressive Euler filter
(`progressive_euler_filter`), all of which have been removed.

The oracle filter's strategy: rather than heuristically removing objects by
proximity thresholds, **try building the BRep and let OCC be the oracle**.
Arcs are scored by how far they lie from their incident clusters (same scoring
metric as the old filters), then greedily removed worst-first until OCC
produces a valid BRep.

### Scoring

**Vertex score.** For vertex $v$ associated with clusters $K_v$:

$$\text{score}(v) = \max_{k \in K_v} \frac{d(v, \text{cluster}_k)}{p_k}$$

where $d(v, \text{cluster}_k)$ is the NN distance from $v$ to cluster $k$,
and $p_k$ is the intra-cluster NN distance percentile.  The ratio normalises
distance by the cluster's own point spacing.

**Arc score.** For an arc on edge $(i, j)$: sample 10 points along the
interior (middle 50 % of the parameter range), compute
$\max(d_i / p_i,\; d_j / p_j)$ per sample, return the mean.  Closed arcs
(full circles/ellipses) get score 0 and are never removed.

### Algorithm

1. **Remove isolated vertices** — vertices with degree 0 in the arc graph.

2. **Try full model** — call `_try_build` with no arc removals.  If OCC
   reports the BRep as valid (all faces Eulerian + `BRepCheck_Analyzer` passes),
   return immediately.

3. **Greedy arc removal** — sort arc candidates by descending score
   (worst first).  For each candidate:
   - Add it to the removal set.
   - Call `_try_build` (which runs `_apply_removals` → `face_arc_incidence` →
     `assemble_wires` → `build_brep_shape`, all inside `redirect_stdout` to
     suppress noise).
   - If OCC reports valid: return.
   - If more faces than the previous best: save as best-so-far.

4. **Fallback** — if no removal combination produces a valid BRep, return the
   best-so-far result.

### `_try_build` internals

Each `_try_build` call:
1. Runs `_apply_removals` to compact vertices and arcs, dropping the marked
   arcs and re-indexing.  Open arcs with `v_start = None` and `v_end = None`
   (trimming failed → no vertex attribution) are silently dropped during
   vertex compaction.
2. Checks the Euler condition on all face wire graphs (`_non_eulerian_faces_direct`).
   If any face is non-Eulerian, short-circuits as invalid without attempting
   BRep construction.
3. Runs `assemble_wires` (Section 9) → `build_brep_shape` (Section 10) →
   `BRepCheck_Analyzer`.

### Dangling arc logging

After the initial `_try_build`, the oracle filter compares the input vs output
arc count.  If arcs were dropped beyond those explicitly removed, the count is
logged:

```
[oracle filter] N dangling arc(s) dropped (open, no vertex attribution)
```

This happens **outside** `redirect_stdout` so it is always visible.

---

## 8 — Face–arc incidence

Every arc stored under key $(i, j)$ lies simultaneously on face $i$ and
face $j$.  Face–arc incidence is therefore read directly from the dict keys:

$$\partial F_i = \bigl\{\, a : a \in \texttt{edge\_arcs}[(i,j)] \text{ for some } j \bigr\}$$

No adjacency lookup is required.  Output: `face_arcs: dict i → list[arc]`.

---

## 9 — Wire assembly with angular ordering

### Graph formulation

Define graph $G_i$ for face $i$:
- **Nodes** — vertex indices appearing in `face_arcs[i]`
- **Edges** — open arcs, connecting `v_start` to `v_end`

In a valid manifold B-Rep, every node of $G_i$ has **even degree $\ge 2$**
($\deg = 2k$ where $k$ is the number of wires passing through that vertex).
Isolated vertices ($\deg = 0$) cannot occur by construction.  Odd-degree
vertices imply an open boundary chain and are topologically invalid.

Because every node has even degree, $G_i$ admits an **Eulerian decomposition**
into edge-disjoint closed trails (classical theorem: a graph has such a
decomposition iff every vertex has even degree).  Each closed trail is one wire.
Closed-loop arcs form trivial one-arc trails.

The Eulerian decomposition is **not unique** when any vertex has degree $\ge 4$:
multiple pairings of arc-endpoints are possible, all valid as abstract graphs,
but only geometrically correct pairings produce a 2-manifold face locally at $v$.
This is why angular ordering is required — it selects the unique pairing that
respects the alternating inside/outside sector structure around $v$.

### Degree-2 case ($k = 1$)

The next arc is uniquely determined.  No angular computation needed.

### High-degree vertices (degree $2k$, $k \ge 2$)

At a vertex $v$ where $2k$ arc-endpoints meet on face $F$, the 2-manifold
constraint requires sectors around $v$ to alternate strictly between inside
and outside the face.  At degree 4 (the common case), the three possible
arc pairings reduce to two geometrically valid ones (adjacent-angle pairing)
and one invalid one (opposite-angle pairing), which produces crossing wires.
The arcs of each individual wire are angularly interleaved with arcs of every
other wire.

**Outgoing tangent** of arc $a$ at vertex $v$:

$$\mathbf{t}_a(v) = \begin{cases}
+\dfrac{d\mathbf{C}}{dt}\Big|_{t_\text{start}} & v = v_\text{start}(a) \\[6pt]
-\dfrac{d\mathbf{C}}{dt}\Big|_{t_\text{end}}   & v = v_\text{end}(a)
\end{cases}$$

The sign ensures $\mathbf{t}_a(v)$ always points away from $v$ along $a$.

The derivative is evaluated via OCC's `Geom_Curve.DN(t, N)`, which returns the
$N$-th derivative vector of the curve at parameter value $t$:

- **`t`** — the parameter at which to differentiate: `arc["t_start"]` or
  `arc["t_end"]`, which are the parameter values of the arc endpoints on the
  underlying `Geom_TrimmedCurve`.
- **`N = 1`** — first derivative order; returns $\frac{d\mathbf{C}}{dt}$, a
  `gp_Vec` in $\mathbb{R}^3$.

The result has units of (length / parameter unit) and its magnitude depends on
the curve's parameterisation — e.g.\ for a circle of radius $r$ parameterised
by arc length, $\|d\mathbf{C}/dt\| = r$.  Only the **direction** matters for
angular ordering, so the result is normalised before projection onto the
tangent plane.

**Surface normal** at $v$.  Angular ordering requires a consistent notion of
"counterclockwise" at $v$.  Since the face lies on a curved surface in
$\mathbb{R}^3$, angles between arcs must be measured in the **tangent plane**
of the surface at $v$, not in the ambient 3D space.  The tangent plane is the
2D subspace of $\mathbb{R}^3$ spanned by the two partial derivatives
$\mathbf{S}_u$ and $\mathbf{S}_v$ at the surface point corresponding to $v$.
Its normal — the vector perpendicular to both $\mathbf{S}_u$ and $\mathbf{S}_v$:

$$\mathbf{N}(u,v) = \mathbf{S}_u(u,v) \times \mathbf{S}_v(u,v)$$

— is the surface normal $\hat{N}$.  It defines the "up" direction at $v$:
angles are then measured as seen when looking down along $\hat{N}$.

To evaluate $\hat{N}$ at a 3D vertex position $\mathbf{p}$, we need the
parametric coordinates $(u^*, v^*)$ on the surface that correspond to
$\mathbf{p}$.  These are found by **point-to-surface projection**:

$$\min_{(u,v)} \|\mathbf{p} - \mathbf{S}(u,v)\|^2$$

This is a 2D nonlinear least-squares problem (vs.\ the 1D version for
point-to-curve projection).  The first-order optimality conditions are:

$$(\mathbf{p} - \mathbf{S}(u^*, v^*)) \cdot \mathbf{S}_u(u^*, v^*) = 0$$
$$(\mathbf{p} - \mathbf{S}(u^*, v^*)) \cdot \mathbf{S}_v(u^*, v^*) = 0$$

i.e., the residual vector $\mathbf{p} - \mathbf{S}(u^*, v^*)$ is perpendicular
to both tangent vectors — hence parallel to $\mathbf{N}(u^*, v^*)$.  These
two equations in two unknowns are solved iteratively (Newton's method on
$(u,v)$), which OCC implements in `GeomAPI_ProjectPointOnSurf`.  The surface
normal $\hat{N}$ is then:

$$\hat{N} = \frac{\mathbf{S}_u(u^*, v^*) \times \mathbf{S}_v(u^*, v^*)}
                 {\|\mathbf{S}_u(u^*, v^*) \times \mathbf{S}_v(u^*, v^*)\|}$$

evaluated via `GeomLProp_SLProps` at $(u^*, v^*)$.  Returns `None` on failure
(degenerate patch or projection divergence).

**CCW selection rule.**  Having arrived at $v$ via arc `prev`, define the
tangent-plane projection operator:

$$\Pi(\mathbf{x}) = \mathbf{x} - (\mathbf{x} \cdot \hat{N})\,\hat{N}$$

which strips the $\hat{N}$ component and retains only the part of $\mathbf{x}$
lying in the tangent plane $T_v S$.  Then:

$$\hat{\mathbf{e}}_1 = \frac{\Pi(\mathbf{t}_\text{prev}(v))}{\|\Pi(\mathbf{t}_\text{prev}(v))\|},
\qquad
\hat{\mathbf{e}}_2 = \hat{N} \times \hat{\mathbf{e}}_1$$

**Do $\hat{\mathbf{e}}_1, \hat{\mathbf{e}}_2$ span the tangent plane?**
Yes.  $\hat{\mathbf{e}}_1 = \Pi(\mathbf{t}_\text{prev}) / \|\cdots\|$ lies in
$T_v S$ by construction (projection removes all $\hat{N}$ content).
$\hat{\mathbf{e}}_2 = \hat{N} \times \hat{\mathbf{e}}_1$ is perpendicular to
$\hat{N}$ (so also in $T_v S$) and perpendicular to $\hat{\mathbf{e}}_1$.
The triple $(\hat{\mathbf{e}}_1, \hat{\mathbf{e}}_2, \hat{N})$ is
right-handed and orthonormal, and $\{\hat{\mathbf{e}}_1, \hat{\mathbf{e}}_2\}$
is an orthonormal basis for $T_v S$.

**Why project the arc tangents onto the tangent plane?**
Curve tangents $\mathbf{t}_a(v) = d\mathbf{C}/dt$ are 3D vectors lying in
$\mathbb{R}^3$, not necessarily in $T_v S$.  For a curve on a flat surface
(plane) they already lie in $T_v S$; for a curve on a curved surface
(cylinder, BSpline) they may have a small $\hat{N}$ component due to the
extrinsic curvature of the embedding.  Projecting removes this component so
that angular comparisons are made entirely within the tangent plane — i.e.\
"as seen from above the surface along $\hat{N}$" — which is the geometrically
correct notion of angle between curves on a surface.

**Role of the previously traversed arc.**
$\hat{\mathbf{e}}_1$ is set to the tangent-plane projection of
$\mathbf{t}_\text{prev}(v)$ — the outgoing tangent of the arc we just
traversed, evaluated at $v$ and pointing back toward the vertex we came from.
This is the **backward direction**.  All candidate arc angles are then measured
CCW relative to this backward direction: $\theta = 0$ corresponds to turning
back the way we came, $\theta = \pi$ corresponds to going straight ahead
(smooth continuation), and $\theta \in (0, \pi)$ is a left turn.  The
selection of the minimum strictly-positive angle picks the arc that is
**first encountered sweeping CCW from backward**, which is the arc bounding
the smallest angular sector to the left of the traversed arc.  This sector is
the "interior" between two consecutive boundary arcs at $v$ — exactly what
the 2-manifold alternating-sector constraint requires.

For each candidate arc $c$, project and normalise:

$$\hat{\mathbf{t}}_c = \frac{\Pi(\mathbf{t}_c(v))}{\|\Pi(\mathbf{t}_c(v))\|}$$

then measure:

$$\theta_c = \operatorname{atan2}(\hat{\mathbf{t}}_c \cdot \hat{\mathbf{e}}_2,\;
                                  \hat{\mathbf{t}}_c \cdot \hat{\mathbf{e}}_1), \qquad
\tilde\theta_c = \theta_c \bmod 2\pi \in [0, 2\pi)$$

Select:

$$c^* = \operatorname*{arg\,min}_{c:\,\tilde\theta_c > 0} \tilde\theta_c$$

This is the **first arc counterclockwise from the backward direction** in
the tangent plane at $v$.  Note that `DN(t, 1)` is called with exactly two
arguments: the parameter value $t$ (`t_start` or `t_end` of the arc) and
derivative order $1$, returning a `gp_Vec` $\in \mathbb{R}^3$.

**Why this is correct.**  The face interior occupies exactly one angular sector
at $v$ (Jordan curve theorem on the surface).  The sector is bounded on each
side by one arc.  The CCW rule selects the sector boundary immediately adjacent
to the face interior.  The outer wire traverses CCW when viewed from $\hat{N}$;
inner hole wires traverse CW.  Cycle extraction starts afresh from unvisited
arcs for each new wire, so holes are extracted as separate cycles automatically.

**Edge continuity heuristic.**  Before applying the general CCW rule, the
implementation checks whether any candidates belong to the same intersection
edge (same $(i,j)$ key) as the arc that was just traversed.  If such
same-edge candidates exist, the CCW selection is restricted to them.  This
keeps the wire travelling along a single intersection curve before switching
to a different curve at a shared vertex, which produces smoother wires in
practice.  The general CCW rule is used only when no same-edge candidates are
available.

**Failure modes:**
- Nearly-parallel tangents: `atan2` may select the wrong arc.
- Degenerate surface normal: falls back to `candidates[0]`.
- Non-manifold topology: odd-degree vertex; no arc-ordering can fix it.

---

## 10 — OCC topology assembly

### Analytical surface faces

For each face $i$ with an analytical surface:

1. Build `TopoDS_Vertex` for each position in `vertices`.
2. Build `TopoDS_Edge` from each arc.  Open arcs with vertex endpoints pass
   explicit `TopoDS_Vertex` objects so adjacent edges share the same instance.
   Closed-loop arcs and arcs with `v_start = None` are built from the curve alone.
   Each arc's `TopoDS_Edge` is cached by `_arc_key(arc)` so that the same arc
   referenced from two face lists (face $i$ and face $j$) shares a single
   `TopoDS_Edge`.  The key is `(edge_i, edge_j, v_start, v_end, id(curve))` —
   the `id(curve)` component distinguishes multiple closed arcs on the same
   edge that would otherwise collide at `(i, j, None, None)`.  Without this,
   two closed BSpline arcs on the same edge (e.g. from a cone∩cylinder
   intersection returning two nearly-closed BSplines) would share a single
   `TopoDS_Edge`, which is invalid BRep topology (two wires on the same face
   referencing the same edge).
3. Build `TopoDS_Wire` from the ordered arc sequence of each wire.  Arc
   orientation (forward / reversed) is encoded with `edge.Reversed()`.
4. Build `TopoDS_Face`:
   ```python
   face_maker = BRepBuilderAPI_MakeFace(surface, outer_wire)
   for inner_wire in hole_wires:
       face_maker.Add(inner_wire)
   ```
   OCC computes pcurves analytically (closed-form inverse map).

### BSpline (INR) faces

A tube-like BSpline face has **annular UV topology**: two boundary circles
(one per open end) that appear at different $v$-values in the parameter domain,
neither enclosing the other.  `MakeFace(surface, wire1)` followed by
`Add(wire2)` fails because OCC interprets the second wire as a hole, which
requires it to be enclosed by the first.  Without proper pcurves the resulting
face is also unbounded in UV space.

#### The seam problem

`GeomAPI_PointsToBSplineSurface` always produces a **non-periodic** BSpline,
even when the underlying shape is geometrically closed (e.g.\ a full revolution
surface such as a vase).  The fitted surface has a **seam**: the
$u = u_1$ and $u = u_2$ iso-parameter boundaries are geometrically identical
in 3D — they are the same curve in space.

Naïvely building `BRepBuilderAPI_MakeFace(S, u_1, u_2, v_\text{min}, v_\text{max}, \varepsilon)` on such a surface creates a face whose wire contains **two degenerate seam edges** (the iso-$u_1$ and iso-$u_2$ boundaries) that coincide in 3D.  `BRepBuilderAPI_Sewing`, running at tolerance $\varepsilon$, identifies these two edges as the same 3D curve and merges them, collapsing the face to a 1D line.

Attempting to avoid this by projecting arc sample points onto the surface to compute tighter bounds also fails: points near the seam project ambiguously — `GeomAPI_ProjectPointOnSurf` returns $u \approx u_1$ for all of them — so the computed $[u_\text{min}, u_\text{max}]$ is nearly the full range $[u_1, u_2]$ anyway, and the degenerate-seam problem persists.

#### Solution: geometric closure detection and `SetUPeriodic`

The correct OCC mechanism for a closed surface is to mark it as **periodic**,
which instructs OCC to represent the seam as a single internal seam edge (used
by the face with two different pcurves — one for each side) rather than two
separate degenerate boundary edges.

**Step 1 — Detect geometric closure.**  For each parametric direction, measure
the 3D distance between the two opposing surface boundaries at the domain
midpoint:

$$d_u = \bigl\|S(u_1,\, \tfrac{v_1+v_2}{2}) - S(u_2,\, \tfrac{v_1+v_2}{2})\bigr\|, \qquad
  d_v = \bigl\|S(\tfrac{u_1+u_2}{2},\, v_1) - S(\tfrac{u_1+u_2}{2},\, v_2)\bigr\|$$

If $d_u < \varepsilon_\text{geom}$ (default $0.05$ in unit-normalised
coordinates), the surface is geometrically closed in $u$.  Apply the same
test for $v$ independently.

**Step 2 — Make periodic.**  Call

$$\texttt{Geom\_BSplineSurface.SetUPeriodic()} \quad \text{if closed in } u$$
$$\texttt{Geom\_BSplineSurface.SetVPeriodic()} \quad \text{if closed in } v$$

This modifies the internal knot vector so that OCC treats the direction as
periodic with period $u_2 - u_1$.  After this call `MakeFace` automatically
creates a seam edge at the period boundary with the correct pair of pcurves.

**Step 3 — Constrain open direction(s) from arc projection.**  For the
non-closed direction (e.g.\ $v$, the axial direction of a vase), project
$n = 30$ sample points from each incident arc onto the surface via
`GeomAPI_ProjectPointOnSurf` and take:

$$v_\text{min} = \min_k v_k - 0.02\,\Delta v, \qquad
  v_\text{max} = \max_k v_k + 0.02\,\Delta v$$

clamped to the natural domain $[v_1, v_2]$.  The closed direction uses the
full natural domain $[u_1, u_2]$.

**Step 4 — Build the face.**

$$\texttt{BRepBuilderAPI\_MakeFace}(S,\; u_1,\; u_2,\; v_\text{min},\; v_\text{max},\; \varepsilon)$$

The resulting face has:
- A proper seam edge in the periodic $u$ direction with two pcurves (one for
  $u = u_1$, one for $u = u_2$), computed analytically by OCC.
- Two open iso-$v$ boundary edges (the $v = v_\text{min}$ and $v = v_\text{max}$
  circles) with straight-line pcurves in UV space.

Sewing then connects the two iso-$v$ circles to the boundary circles of the
adjacent analytical cap faces, producing a closed shell.

### Sewing

All faces (analytical and BSpline) are passed to `BRepBuilderAPI_Sewing`:

```python
sewing = BRepBuilderAPI_Sewing(tolerance)
for face in occ_faces:
    sewing.Add(face)
sewing.Perform()
shape = sewing.SewedShape()
```

Sewing identifies coincident 3D boundary edges between faces and merges them
topologically.  For BSpline faces this connects the iso-parameter boundary edges
to the boundary circles of adjacent planar caps.

---

## 11 — Healing and validation

After sewing, the following sequence of repair and normalisation steps is applied:

**Rebuild 3D curves.**  `breplib.BuildCurves3d(shape)` ensures all
edges have consistent 3D curves (some may be missing after sewing merges edges).

**Shape healing.**  `ShapeFix_Shape` repairs common defects:

| Defect | Fix |
|---|---|
| Gap between edge endpoint and vertex | `ShapeFix_Edge::FixVertexTolerance` |
| Wire not closed | `ShapeFix_Wire::FixClosed` |
| Face normal inconsistent | `ShapeFix_Face::FixOrientation` |
| Degenerate (zero-length) edge | `ShapeFix_Wire::FixDegenerated` |

```python
fixer = ShapeFix_Shape(shape)
fixer.SetPrecision(tolerance)
fixer.Perform()
shape = fixer.Shape()
```

**Re-parameterise edges.**  `breplib.SameParameter(shape, True)`
updates edge tolerances so they reflect the actual deviation between each edge's
3D curve and its two pcurves.  This fixes "Invalid curve on surface" errors
that appear in boolean-operation pre-checks when the stored edge tolerance is
inconsistent with the 3D/2D geometry.

**Orient closed solid.**  For topologically closed shapes (a shell or
solid returned by sewing), `breplib.OrientClosedSolid` ensures all face normals
point consistently outward.  If sewing returns a shell it is first wrapped into
a `TopoDS_Solid` via `BRepBuilderAPI_MakeSolid`.  This step fixes
"Self-intersection found" errors caused by inward-pointing face normals.

**Validity check.**  `BRepCheck_Analyzer(shape).IsValid()` runs the
standard OCC topology checker on the in-memory shape.  Note that the STEP
round-trip (write + re-read) can introduce coordinate rounding at
$\sim 10^{-7}$ that degrades the result; FreeCAD's Check Geometry panel runs
the same analyzer on the re-read shape and may report additional errors.

---

## 12 — STEP export

```python
writer = STEPControl_Writer()
writer.Transfer(shape, STEPControl_AsIs)
writer.Write(path)
```

STEP stores 3D curves, surfaces, and pcurves explicitly.  A face without valid
pcurves appears as an unbounded surface patch in the STEP file and is dropped
by most CAD viewers.

---

## 13 — Alternative assembly: `build_brep_shape_bop` (BOPAlgo_MakerVolume)

A second assembly path (`build_brep_shape_bop`) uses `BOPAlgo_MakerVolume`
to perform Boolean union of finite surface patches.  It avoids explicit wire
assembly and pcurve computation by delegating trimming to the Boolean engine.

**Face patch construction.**  For each face $i$, a finite bounded patch is
built by projecting available geometry onto the surface to determine UV bounds:

1. **Vertices available** for face $i$: project incident vertex positions onto
   the surface via `GeomAPI_ProjectPointOnSurf` → UV bounding box (cheapest,
   most precise).
2. **No vertices, but closed curves** on face $i$: sample points along incident
   closed curves → UV bounding box.
3. **BSpline surface**: use the natural UV domain `surface.Bounds()`.

UV bounds are expanded by `rel_margin` (default 50 %) and clipped to the
surface's natural domain to prevent wrapping past periodic boundaries.  The
resulting face is `BRepBuilderAPI_MakeFace(surface, u_min, u_max, v_min,
v_max, tolerance)`.

**BOPAlgo_MakerVolume** receives all face patches and internally computes their
intersections and Boolean union, producing a solid.  This approach is robust
against imprecise wire connectivity but relies on the Boolean engine handling
face overlaps — it may fail for highly irregular geometries.

**Output.**  `TopoDS_Shape` (solid) or `None` if `BOPAlgo_MakerVolume` fails.
The pipeline exports this shape as `{id}_bop.step` in addition to the primary
`{id}.step` from `build_brep_shape`.

---

## 14 — Parametric surfaces, pcurves, and the B-Rep data structure

### 14.1 — Parametric surfaces: UV → 3D

Every surface is an evaluation map $S : \mathbb{R}^2 \to \mathbb{R}^3$.
Each primitive is characterised by its intrinsic geometric parameters (a point,
an axis direction, a radius, etc.) together with a **local orthonormal frame**
$(\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3)$ fixed at construction time, where
$\mathbf{e}_3$ is the primary axis direction and $\{\mathbf{e}_1, \mathbf{e}_2\}$
are any two unit vectors spanning its orthogonal complement.  In OCC this frame
is stored as a `gp_Ax3` object.

**Plane** — anchor point $\mathbf{p}_0 \in \mathbb{R}^3$, unit normal $\mathbf{n}$;
frame $\mathbf{e}_3 = \mathbf{n}$, $(\mathbf{e}_1, \mathbf{e}_2)$ any orthonormal
basis of the plane; $u, v \in \mathbb{R}$:
$$S(u,v) \;=\; \mathbf{p}_0 + u\,\mathbf{e}_1 + v\,\mathbf{e}_2$$

**Cylinder** — axis point $\mathbf{c}$, unit axis direction $\mathbf{a}$, radius $r$;
frame $\mathbf{e}_3 = \mathbf{a}$, $(\mathbf{e}_1, \mathbf{e}_2) \perp \mathbf{a}$;
$u \in [0,2\pi)$, $v \in \mathbb{R}$:
$$S(u,v) \;=\; \mathbf{c} + r\cos u\;\mathbf{e}_1 + r\sin u\;\mathbf{e}_2 + v\,\mathbf{a}$$
$u$ is the angular coordinate around the axis; $v$ is the signed axial distance
from $\mathbf{c}$.

**Sphere** — centre $\mathbf{c}$, radius $r$; frame $\mathbf{e}_3$ toward the
north pole; $u \in [0,2\pi)$, $v \in [-\tfrac{\pi}{2}, \tfrac{\pi}{2}]$:
$$S(u,v) \;=\; \mathbf{c} + r\cos v\cos u\;\mathbf{e}_1
                           + r\cos v\sin u\;\mathbf{e}_2
                           + r\sin v\;\mathbf{e}_3$$
$u$ is longitude, $v$ is latitude ($v=0$ equator, $v=\pm\pi/2$ poles).

**Cone** — apex $\mathbf{v}$, unit axis direction $\mathbf{a}$ pointing from apex
into the cone body, semi-angle $\theta$;
frame $\mathbf{e}_3 = \mathbf{a}$, $(\mathbf{e}_1, \mathbf{e}_2) \perp \mathbf{a}$;
$u \in [0,2\pi)$, $v \ge 0$:
$$S(u,v) \;=\; \mathbf{v} + v\sin\theta\cos u\;\mathbf{e}_1
                           + v\sin\theta\sin u\;\mathbf{e}_2
                           + v\cos\theta\;\mathbf{a}$$
$v$ is the slant distance from the apex along a generator.

**BSpline** — $u \in [u_0,u_n]$, $v \in [v_0,v_m]$:
$$S(u,v) \;=\; \sum_{i=0}^{n}\sum_{j=0}^{m} N_i^p(u)\,N_j^q(v)\,\mathbf{P}_{ij}$$
Tensor product of B-spline bases of degrees $p,q$; no closed-form inverse.
Constructed via `GeomAPI_PointsToBSplineSurface`.

### 14.2 — Inverse maps: 3D → UV

For a point $\mathbf{p} \in \mathbb{R}^3$ lying on surface $S$, the closed-form
inverses use the same frame vectors and intrinsic parameters as Section 14.1:

**Plane** — let $\mathbf{q} = \mathbf{p} - \mathbf{p}_0$:
$$S^{-1}(\mathbf{p}) \;=\; \bigl(\mathbf{q}\cdot\mathbf{e}_1,\;\mathbf{q}\cdot\mathbf{e}_2\bigr)$$
Both coordinates are orthogonal projections; the inverse is a linear map.

**Cylinder** — let $\mathbf{q} = \mathbf{p} - \mathbf{c}$:
$$S^{-1}(\mathbf{p}) \;=\; \bigl(\operatorname{atan2}(\mathbf{q}\cdot\mathbf{e}_2,\;\mathbf{q}\cdot\mathbf{e}_1),\;\mathbf{q}\cdot\mathbf{a}\bigr)$$
$u$ recovers the angle by unwrapping the lateral projection; $v$ is the axial projection.

**Sphere** — let $\hat{\mathbf{q}} = (\mathbf{p} - \mathbf{c})/r$:
$$S^{-1}(\mathbf{p}) \;=\; \bigl(\operatorname{atan2}(\hat{\mathbf{q}}\cdot\mathbf{e}_2,\;\hat{\mathbf{q}}\cdot\mathbf{e}_1),\;\arcsin(\hat{\mathbf{q}}\cdot\mathbf{e}_3)\bigr)$$

**Cone** — let $\mathbf{q} = \mathbf{p} - \mathbf{v}$:
$$S^{-1}(\mathbf{p}) \;=\; \bigl(\operatorname{atan2}(\mathbf{q}\cdot\mathbf{e}_2,\;\mathbf{q}\cdot\mathbf{e}_1),\;\mathbf{q}\cdot\mathbf{a}/\cos\theta\bigr)$$
The axial projection is scaled by $1/\cos\theta$ to recover the slant distance $v$.

**BSpline:** no closed form; requires solving
$\min_{(u,v)}\|S(u,v)-\mathbf{p}\|^2$ by Newton iteration
(`GeomAPI_ProjectPointOnSurf`).

For analytical primitives OCC uses these closed-form expressions internally
(not Newton iteration).  The frame vectors $\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3$
correspond to the X, Y, Z directions of the `gp_Ax3` object stored in the surface.

### 14.3 — Pcurves: definition, analytical computation, and role

A **pcurve** of edge $e$ on face $f$ is a 2D curve $c : [t_0,t_1] \to \mathbb{R}^2$
in the UV domain of surface $S_f$ satisfying:

$$S_f(c(t)) = \mathbf{C}_e(t) \quad \forall\,t \in [t_0, t_1]$$

i.e.\ the pcurve is the composition $c = S_f^{-1} \circ \mathbf{C}_e$.  For
analytical surfaces this composition is evaluated in closed form by substituting
the 3D curve $\mathbf{C}_e(t)$ into the inverse map.

**Concrete examples** (all intersections arising in the Point2CAD pipeline):

*Line on a plane* — $\mathbf{C}(t) = \mathbf{A} + t\mathbf{d}$ (plane–plane intersection):
$$c(t) = \bigl((\mathbf{A}-\mathbf{O})\cdot\mathbf{X} + t\,(\mathbf{d}\cdot\mathbf{X}),\;
               (\mathbf{A}-\mathbf{O})\cdot\mathbf{Y} + t\,(\mathbf{d}\cdot\mathbf{Y})\bigr)$$
The inverse of the plane is linear, so a 3D line maps to a **2D line**
(`Geom2d_Line`).

*Circle on a cylinder, plane cut perpendicular to axis* — the circle lies at
constant height $h$, $\mathbf{C}(t) = \mathbf{O} + r\cos t\,\mathbf{X} + r\sin t\,\mathbf{Y} + h\mathbf{Z}$:
$$c(t) = \bigl(t,\; h\bigr)$$
The $\operatorname{atan2}$ applied to $(\cos t, \sin t)$ returns $t$ exactly;
the axial projection returns the constant $h$.  The pcurve is a **horizontal
line** $v = h$ in UV space (`Geom2d_Line`, direction $(1,0)$).

*Circle on a cylinder, oblique plane cut* — the intersection is still
parameterised by angle $t$ but the height now varies:
$v(t) = h_0 + A\cos t + B\sin t$ (from the plane equation).
$$c(t) = \bigl(t,\; h_0 + A\cos t + B\sin t\bigr)$$
This is a **sinusoidal (trigonometric) curve** in UV space, which OCC
represents as a `Geom2d_BSplineCurve` approximation.

*Circle on a plane* — the circle lies entirely in the plane, so the plane's
inverse (a linear map) maps it to a **2D circle** (`Geom2d_Circle`).

*Generator line on a cylinder* — at fixed angle $\phi_0$,
$\mathbf{C}(t) = \mathbf{O} + r\cos\phi_0\,\mathbf{X} + r\sin\phi_0\,\mathbf{Y} + t\mathbf{Z}$:
$$c(t) = \bigl(\phi_0,\; t\bigr)$$
A **vertical line** $u = \phi_0$ (`Geom2d_Line`, direction $(0,1)$).

The pattern is: the type of the pcurve is determined by the composition of the
3D curve type with the surface's inverse map.  Linear inverses (plane) preserve
curve type.  Trigonometric inverses (cylinder, sphere, cone) generally degrade
circles to sinusoids in UV space.

OCC's `BRepLib::BuildCurve2d` (called internally by `BRepBuilderAPI_MakeFace`)
handles the known (surface type, 3D curve type) pairs analytically and falls
back to numerical BSpline approximation for pairs it cannot resolve in closed form.

**Every edge in a valid B-Rep has exactly two pcurves** — one for each of
its two adjacent faces — even though geometrically they encode the same 3D
curve.  The two pcurves live in different UV spaces and generally look
different: for example the intersection circle between a plane and a cylinder
is a horizontal line $v = h$ in the cylinder's UV space, but a circle in the
plane's UV space (the plane's inverse is the identity up to a rigid frame
change, which maps a circle to a circle).

**Why pcurves are required:**

1. **Trimming.**  A face is a bounded region of an infinite surface.  OCC
   defines this region entirely in UV space as the area enclosed by the
   pcurves of the face's boundary wires.  Without pcurves the surface cannot
   be trimmed — OCC does not know where the face ends.

2. **Inside/outside determination.**  The orientation (direction of traversal)
   of the pcurve in UV space defines which side is the face interior.  By
   convention the interior is to the left of the pcurve when traversed in its
   natural direction.  This is what gives the face its outward normal
   orientation.

3. **STEP standard.**  ISO 10303-21 stores pcurves as explicit `PCURVE`
   entities.  A face without valid pcurves appears as an unbounded surface
   patch and is dropped or treated as invalid by importing applications.

4. **Downstream algorithms.**  Boolean operations, shelling, offsetting, and
   meshing all operate in UV space.  Pcurves are the bridge from 3D topology
   to 2D parameterisation.

### 14.4 — What is a B-Rep file?

A B-Rep solid is a hierarchy of topological entities, each paired with a
geometric object:

| Topological entity | Geometric object | Notes |
|---|---|---|
| **Vertex** | Point $\mathbf{p} \in \mathbb{R}^3$ | 0-dimensional |
| **Edge** | 3D curve $\mathbf{C}(t)$ + **two pcurves** $c_L(t), c_R(t)$ + two vertex refs | shared between exactly 2 faces |
| **Wire** | Ordered, oriented sequence of edges | closed loop |
| **Face** | Surface $S(u,v)$ + one or more wires | bounded region in UV |
| **Shell** | Connected collection of faces | open or closed |
| **Solid** | Closed shell(s) | encloses a volume |

The pcurves are the critical piece that connects the 1D topology (edges) to
the 2D topology (faces).  Each edge carries **two** pcurves because it borders
two faces, each with its own UV space.

In our pipeline the pcurves are **not computed explicitly**.  For analytical
faces `BRepBuilderAPI_MakeFace(surface, wire)` computes them internally using
the closed-form inverse maps.  For INR (BSpline) faces the UV-bounds path
`BRepBuilderAPI_MakeFace(surface, u_min, u_max, v_min, v_max, tol)` generates
iso-parameter boundary edges whose pcurves are trivial straight lines in UV
space (the iso-$u$ and iso-$v$ lines).  `BRepBuilderAPI_Sewing` then stitches
the faces together, computing and validating pcurves at shared edges.
`breplib.BuildCurves3d` rebuilds the 3D curves from pcurves where needed.
`ShapeFix_Shape` repairs any invalid pcurves during healing.  All of this
happens inside OCC; the STEP writer exports the final pcurves as explicit
`PCURVE` entities.

### 14.5 — BSpline faces: sidestepping explicit pcurve computation

For analytical faces, the closed-form inverse maps make pcurve computation
trivial — OCC evaluates them directly.  For BSpline faces there is no
closed-form inverse, so computing a pcurve explicitly requires projecting the
3D intersection arc onto the UV domain point-by-point via Newton iteration.
Our pipeline sidesteps this entirely by using a **rectangular UV-bounds face**:

$$\texttt{BRepBuilderAPI\_MakeFace}(S,\; u_\text{min},\; u_\text{max},\; v_\text{min},\; v_\text{max},\; \varepsilon)$$

The boundary of this face consists entirely of iso-parameter curves of $S$ —
lines in UV space by definition, requiring no inversion of the surface map.
The bounds for the open direction (e.g.\ $v_\text{min}, v_\text{max}$) are
estimated by projecting sample points from the neighboring intersection arcs
onto the BSpline surface and taking the parameter extremes with a small margin.

The iso-parameter boundary $S(u, v_\text{min})$ is not the exact 3D
intersection curve — it deviates from it by roughly the BSpline fitting
residual $\varepsilon_S$.  This gap is bridged by `BRepBuilderAPI_Sewing`:
it identifies the iso-parameter edge of the BSpline face and the intersection
curve edge of the neighboring analytical face as geometrically coincident
(within sewing tolerance) and merges them.  At the merged edge, sewing
computes the pcurve on the BSpline side numerically, delegating all pcurve
complexity to the sewing API.  If sewing succeeds the resulting BRep is
topologically valid and the pcurve error is bounded by the sewing tolerance —
consistent with the fitting residual already present throughout the pipeline.

**Manual pcurve computation (alternative).**  The exact approach is to project
each intersection arc $\mathbf{C}_e(t)$ onto the BSpline UV space: sample
$t_k$, solve $\min_{(u,v)}\|S(u,v) - \mathbf{C}_e(t_k)\|^2$ via Newton for
each sample to obtain $(u_k, v_k)$, then fit a `Geom2d_BSplineCurve` through
the sequence.  This pcurve is used directly when building the face with
explicit wires, making the boundary exactly the intersection curve.  However,
this approach is significantly more expensive (a nonlinear solve per sample
point per arc) and risks compounding errors: Newton iteration on a BSpline
surface can diverge near low-curvature regions or close to the seam; the
subsequent 2D BSpline fit introduces an additional approximation layer; and
any accumulated error exceeding the BRep tolerance causes `BRepCheck_Analyzer`
or sewing to reject the face.  The UV-bounds approach avoids all of these
failure modes at the cost of delegating the boundary approximation to the
sewing tolerance.

**The four iso-parameter edges.**  The rectangular UV-bounds face always has
exactly four boundary edges:

| Edge | 3D curve | UV pcurve |
|---|---|---|
| $S(u,\, v_\text{min})$, $u \in [u_1, u_2]$ | bottom cap boundary | horizontal line $v = v_\text{min}$ |
| $S(u,\, v_\text{max})$, $u \in [u_1, u_2]$ | top cap boundary | horizontal line $v = v_\text{max}$ |
| $S(u_min,\, v)$, $v \in [v_\text{min}, v_\text{max}]$ | left seam | vertical line $u = u_1$ |
| $S(u_max,\, v)$, $v \in [v_\text{min}, v_\text{max}]$ | right seam | vertical line $u = u_2$ |

For a surface periodic in $u$ (closed revolution), `SetUPeriodic` collapses
the left and right seam edges into a single internal seam edge with two
pcurves (one for each side), reducing the boundary to three curves.  Sewing
then attempts to merge each boundary edge with an edge from a neighboring face:
the $v_\text{min}$ and $v_\text{max}$ edges are candidates for merging with
the intersection curve edges of adjacent cap faces; the seam edge(s) have no
neighbor and remain internal.  In a typical closed tube with two planar caps
all mergeable edges are within sewing tolerance and the shell closes cleanly.

---

## 15 — Data-flow summary

```
clusters
        │
        │  Section 1: fit_surface + to_occ_surface
        ↓
Geom_Surface list
        │
        │  Section 2: build_cluster_proximity
        ↓
cluster_trees, cluster_nn_percentiles
        │
        │  Section 3: compute_adjacency_matrix (reuses local_spacings)
        ↓
adjacency matrix + boundary strips
        │
        │  intersect_surfaces (see notes/surface_intersection.md)
        ↓
raw (i,j) → [Geom_Curve]
        │
        │  Section 4: compute_vertices_intcs on raw curves
        │             GeomAPI_IntCS (curve ∩ surface), threshold 1e-3
        ↓
vertices (M,3),  vertex_edges
        │
        │  bbox pre-filter (vertex inside all incident cluster bboxes)
        ↓
vertices (M',3), vertex_edges  [filtered]
        │
        │  Section 5: trim_by_vertices (vertex-based trimming, ext=0.05)
        ↓
trimmed (i,j) → [Geom_TrimmedCurve]
        │
        │  Section 6: build_edge_arcs (arc splitting)
        ↓
edge_arcs: (i,j) → [arc_dict]
        │
        │  Section 7: greedy_oracle_filter
        │    scores arcs → tries building BRep →
        │    removes worst-scored arcs until OCC valid
        │    (internally runs Sections 8, 9, 10 per trial)
        ↓
edge_arcs, vertices, vertex_edges  [filtered]
+ TopoDS_Shape (shell/solid)
        │
        │  Section 11: breplib.BuildCurves3d
        │              ShapeFix_Shape
        │              breplib.SameParameter
        │              breplib.OrientClosedSolid
        │              BRepCheck_Analyzer
        ↓
        │  Section 12: STEPControl_Writer
        ↓
output.step  (and output_bop.step via Section 13)
```

---

## 16 — Known difficulties and alternatives

**Wire assembly at high-degree vertices.** The CCW rule works for exact
geometry but fails when two arcs at a degree-4 vertex are nearly tangent
($\theta \approx 0$) — `atan2` precision may select the wrong arc.

**BSpline fitting residual.** The BSpline approximation of the INR has a
residual error $\varepsilon_S$.  The UV-bounds face boundary may deviate from
the true physical boundary by $\varepsilon_S + \varepsilon_\text{proj}$
(projection tolerance).  If this exceeds the sewing tolerance the shared
boundary edge between the BSpline face and an adjacent planar cap will not be
merged by sewing, leaving a naked edge (Not Closed defect in FreeCAD).

**Alternative approaches if the assembled solid is invalid:**

A. *Pcurves via BRepLib::BuildCurve2d.*  Numerically project each 3D intersection
   curve onto the BSpline surface's UV space using `BRepLib::BuildCurve2d(edge, ref_face)`.
   Use the resulting `Geom2d_BSplineCurve` as the pcurve, then build the face
   with explicit wires.  Accurate but requires solving a nonlinear system per
   sample point, and does not resolve the annular-topology issue for tube-like faces.

B. *UV-space wire assembly.*  Work entirely in UV space: project all arc endpoints
   and sample points to UV, build planar wires in UV, use winding-number tests
   to classify outer vs.\ inner wires.  Numerically more stable.

C. *BRepAlgoAPI splitter.*  Build each surface as a large finite patch and each
   intersection curve as an edge; use the edges as splitting tools.  OCC handles
   trimming and wire assembly internally.

D. *Accept open shell.*  For applications requiring only visualization and
   measurement (not solid booleans), an open shell of individually correct face
   patches is sufficient.  Export each face as a separate STEP entity.

---

## 17 — Denormalisation: inverting the input normalisation on the output shape

The pipeline normalises the input point cloud before fitting.  The forward
transform is:

$$
P_\text{norm} = \frac{1}{s} \, R \, (P - \bar{P})
$$

where $\bar{P}$ is the point-cloud mean, $R$ is a rotation matrix (PCA-derived,
orthogonal), and $s = \max(\text{extents})$ is the largest bounding-box extent.
The inverse (denormalisation) is:

$$
P_\text{real} = s \, R^\top P_\text{norm} + \bar{P}
$$

This is a **similarity transform** (uniform scale + rotation + translation) and
is therefore exactly representable as a single `gp_Trsf`.  Once the assembled
`TopoDS_Shape` is available, a single `BRepBuilderAPI_Transform` call applies
the inverse to all faces, edges, and vertices at once — no per-surface parameter
manipulation is required.

### Implementation

```python
import numpy as np
from OCC.Core.gp               import gp_Trsf, gp_Vec, gp_Pnt
from OCC.Core.BRepBuilderAPI   import BRepBuilderAPI_Transform


def normalize_points(pts):
    """
    Returns (pts_norm, mean, R, scale).
    Store mean, R, scale to denormalise the output shape later.
    Uses eigh (not eig): the scatter matrix pts.T @ pts is symmetric PSD,
    so eigh guarantees real eigenvalues sorted in ascending order.
    """
    mean = pts.mean(axis=0)
    pts  = pts - mean
    _, U = np.linalg.eigh(pts.T @ pts)      # columns = eigenvectors, ascending order
    R    = rotation_matrix_a_to_b(U[:, 0], np.array([1.0, 0.0, 0.0]))
    pts  = (R @ pts.T).T
    extents = pts.max(axis=0) - pts.min(axis=0)
    scale   = float(np.max(extents)) + 1e-7
    return (pts / scale).astype(np.float32), mean, R, scale


def build_denorm_trsf(mean: np.ndarray, R: np.ndarray, scale: float) -> gp_Trsf:
    """
    Build the gp_Trsf that inverts the normalisation:
        P_real = scale * R^T * P_norm + mean

    Decomposed as three similarity transforms composed right-to-left
    (innermost applied first):
        T_denorm = T_translate(+mean) * T_rotate(R^T) * T_scale(scale)

    gp_Trsf.Multiply right-appends:  me = me * arg  (arg is applied first),
    so starting from the outermost (translation) and multiplying inward gives
    the correct composition order.
    """
    # Pure rotation by R^T  (scale = 1, translation = 0)
    Rt    = R.T
    t_rot = gp_Trsf()
    t_rot.SetValues(
        Rt[0, 0], Rt[0, 1], Rt[0, 2], 0.0,
        Rt[1, 0], Rt[1, 1], Rt[1, 2], 0.0,
        Rt[2, 0], Rt[2, 1], Rt[2, 2], 0.0,
    )

    # Uniform scaling by scale about the origin
    t_scale = gp_Trsf()
    t_scale.SetScale(gp_Pnt(0.0, 0.0, 0.0), scale)

    # Translation by +mean  (initialise t_denorm here — outermost operation)
    t_denorm = gp_Trsf()
    t_denorm.SetTranslation(gp_Vec(float(mean[0]), float(mean[1]), float(mean[2])))

    # Compose: T_translate * T_rotate * T_scale
    t_denorm.Multiply(t_rot)    # → T_translate * T_rotate
    t_denorm.Multiply(t_scale)  # → T_translate * T_rotate * T_scale

    return t_denorm


def apply_denorm_to_shape(shape, mean, R, scale):
    """Apply the denormalisation transform to an assembled TopoDS_Shape."""
    trsf = build_denorm_trsf(mean, R, scale)
    # copy=True: returns a new shape rather than modifying in place
    return BRepBuilderAPI_Transform(shape, trsf, True).Shape()
```

### Alternative: build the forward transform and invert it

If the forward `gp_Trsf` is already available (e.g. built during normalisation),
`gp_Trsf.Inverted()` avoids constructing the inverse manually:

```python
def build_fwd_trsf(mean, R, scale):
    t_trans = gp_Trsf()
    t_trans.SetTranslation(gp_Vec(-float(mean[0]), -float(mean[1]), -float(mean[2])))

    t_rot = gp_Trsf()
    t_rot.SetValues(
        R[0, 0], R[0, 1], R[0, 2], 0.0,
        R[1, 0], R[1, 1], R[1, 2], 0.0,
        R[2, 0], R[2, 1], R[2, 2], 0.0,
    )

    t_scale = gp_Trsf()
    t_scale.SetScale(gp_Pnt(0.0, 0.0, 0.0), 1.0 / scale)

    t_fwd = gp_Trsf()
    t_fwd.SetTranslation(gp_Vec(-float(mean[0]), -float(mean[1]), -float(mean[2])))
    t_fwd.Multiply(t_rot)
    t_fwd.Multiply(t_scale)
    return t_fwd

t_inv = build_fwd_trsf(mean, R, scale).Inverted()
shape_real = BRepBuilderAPI_Transform(shape, t_inv, True).Shape()
```
