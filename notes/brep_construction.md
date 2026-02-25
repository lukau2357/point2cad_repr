# B-Rep Construction Pipeline

This document covers the complete pipeline from a pre-segmented point cloud to
a valid STEP file: adjacency detection, surface fitting, surface-surface
intersection, trimming, vertex finding, wire assembly, face construction, and
STEP export.

**Pipeline input:** a segmented point cloud — $n$ labelled clusters
$\{C_0, \ldots, C_{n-1}\}$, each an $(N_i \times 3)$ array of 3D points
belonging to one surface patch.  Segmentation is assumed given (e.g.\ from
Point2CAD).

**Pipeline output:** `TopoDS_Shape` (shell or solid) exportable to STEP.

---

## 0a — Reference spacing

All distance thresholds in the pipeline are expressed as multiples of a single
**reference spacing** $\sigma$, defined as the median nearest-neighbour
distance across all cluster points:

$$
\begin{align*}
D_i &= \{\argmin_{y \in C_i, y \neq x}||x - y||_{2} \mid x \in C_i\} \\
\sigma &= \operatorname{median} \cup_{i=1}^{N}D_i\\
\end{align*}
$$

$N$ is the number of clusters, and $\sigma$ is the adapative threshold, used for the following step.

---

## 0b — Cluster adjacency matrix

**Goal.** Determine which surface pairs share a physical boundary and therefore
need to be intersected.  Intersecting all $\binom{n}{2}$ pairs is wasteful and
produces spurious curves for non-adjacent surfaces.

**Algorithm.**  For each unordered pair $(i, j)$ with $i < j$:

1. Build a KDTree on the larger cluster (say $C_i$).
2. Query every point of the smaller cluster $C_j$, obtaining nearest-neighbour
   distances $\{d_k\}_{k=1}^{|C_j|}$.
3. Compute the **robust minimum** — the $p$-th percentile of $\{d_k\}$ (default
   $p = 2$, i.e.\ the 2nd-percentile distance):

$$\hat{d}_{ij} = \operatorname{percentile}_p(\{d_k\})$$

4. Declare clusters adjacent if $\hat{d}_{ij} \le \tau \cdot \sigma$, where
   $\tau = 1.5$ is the threshold factor.

Using the $p$-th percentile rather than the minimum makes the test robust to
outliers and to slight cluster overlap near shared boundaries.  Two clusters
that touch along a narrow strip will have a few points very close to each
other; the 2nd percentile captures this even if the bulk of one cluster is far
from the other.

**Boundary strips.**  For each adjacent pair $(i, j)$, retain the subset of
points whose nearest-neighbour distance to the opposite cluster is at most
$\tau \cdot \sigma$:

$$B_{ij} = \{x \in C_i : d(x, C_j) \le \tau\sigma\}
  \cup \{x \in C_j : d(x, C_i) \le \tau\sigma\} $$

$B_{ij}$ is reused downstream in two places: trimming intersection curves
(Section 1) and computing UV bounds for BSpline faces (Section 7).

**Output.**
- $A \in \{0,1\}^{n \times n}$ — symmetric Boolean adjacency matrix.
- `boundary_strips[(i,j)]` — $(|B_{ij}| \times 3)$ float32 array for each adjacent pair.
---

## 0c — Surface fitting and OCC geometry objects

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
   (Section 7).

**Upstream inputs (produced before this document's scope)**

| Object | Type |
|---|---|
| Fitted OCC surfaces | `list[Geom_Surface]`, indexed by cluster $i$ |
| Adjacency matrix + boundary strips | `np.ndarray`, `dict` |
| Cluster point clouds | `list[np.ndarray (N,3)]` |
| Surface type IDs | `list[int]` (`SURFACE_PLANE` … `SURFACE_INR`)

---

## 1 — Intersection curve trimming

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

### Boundary-strip projection

For the adjacent pair $(i, j)$ the boundary strip of cluster $i$ is the subset
of its points whose nearest-neighbour distance to cluster $j$ is at most the
adjacency threshold $\varepsilon_\text{adj}$.  The union of both strips gives
boundary points $\{\mathbf{q}_k\}$ lying near the shared edge.

Each $\mathbf{q}_k$ is projected onto the intersection curve via
`GeomAPI_ProjectPointOnCurve`, which solves:

$$t^* = \operatorname*{arg\,min}_{t \in [t_0, t_1]} \|\mathbf{C}(t) - \mathbf{q}_k\|_2^2$$

using the OCC `Extrema_ExtPC` algorithm (convex-hull pruning of Bézier
segments, then Newton–Raphson).  Only projections with
$\|\mathbf{C}(t^*) - \mathbf{q}_k\| < c \cdot \text{spacing}$ are kept
(default $c = 3$).

The trim interval is:

$$t_\text{min} = \min_k t_k^*, \qquad t_\text{max} = \max_k t_k^*$$

Extended symmetrically by a relative margin $\alpha = 0.05$ to avoid endpoint
truncation:

$$t_\text{min} \leftarrow t_\text{min} - \alpha(t_\text{max} - t_\text{min}), \qquad
  t_\text{max} \leftarrow t_\text{max} + \alpha(t_\text{max} - t_\text{min})$$

A `Geom_TrimmedCurve` is constructed from the original curve and the computed
interval.

---

## 2 — Vertex finding (curve–curve closest approach)

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

### Minimum-distance formulation

Due to surface fitting residuals, two curves sharing a surface do not
intersect exactly.  The vertex is the closest approach:

$$\min_{t_1, t_2} D(t_1, t_2) = \|\mathbf{C}_1(t_1) - \mathbf{C}_2(t_2)\|_2^2$$

Interior minima satisfy the **common-perpendicular conditions**:

$$(\mathbf{C}_1(t_1) - \mathbf{C}_2(t_2)) \cdot \mathbf{C}_1'(t_1) = 0, \qquad
  (\mathbf{C}_1(t_1) - \mathbf{C}_2(t_2)) \cdot \mathbf{C}_2'(t_2) = 0$$

All local minima with $\sqrt{D} < \varepsilon_v$ are accepted as vertex
candidates (a circle meeting a line has two such minima).  The candidate
position is the midpoint $\tfrac{1}{2}(\mathbf{C}_1(t_1) + \mathbf{C}_2(t_2))$.
Candidates within $\varepsilon_v$ of each other are merged by averaging.

OCC implementation: `GeomAPI_ExtremaCurveCurve` reports all local extrema of
$D(t_1, t_2)$ via subdivision + Newton–Raphson.

---

## 3 — Arc splitting (Step 1)

Each intersection curve $C$ may pass through multiple vertices.  The output
of Step 1 replaces curves with **arcs** — sub-intervals bounded by vertices.

### Closure test

`GeomAPI_IntSS` always returns `Geom_TrimmedCurve`, so `IsPeriodic()` is
always `False` even when the underlying geometry is a full circle.  Closure
is detected geometrically:

$$\text{closed} \iff \|C(t_\text{min}) - C(t_\text{max})\| < \varepsilon_\text{close}, \qquad \varepsilon_\text{close} = 10^{-7}$$

Full cylinder–plane circles give endpoint distance $\sim 10^{-17}$; trimmed
lines give $\sim 10^{-1}$, so the threshold is unambiguous.

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
$t_1^*$.  Constructed by calling `BasisCurve()` on the `Geom_TrimmedCurve` to
obtain the underlying periodic curve (e.g.\ `Geom_Circle`), which accepts
parameters beyond $[t_\text{min}, t_\text{max}]$.  If `BasisCurve()` is
unavailable, a synthetic seam vertex is inserted at $C(t_\text{min})$ and the
wrap-around is split into two sub-arcs.

Each arc dict carries:
- `curve` — `Geom_TrimmedCurve` for this arc's interval
- `v_start`, `v_end` — vertex indices (`None` for closed-loop arcs)
- `t_start`, `t_end` — parameter bounds
- `closed` — `True` for closed-loop arcs

---

## 4 — Face–arc incidence (Step 2)

Every arc stored under key $(i, j)$ lies simultaneously on face $i$ and
face $j$.  Face–arc incidence is therefore read directly from the dict keys:

$$\partial F_i = \bigl\{\, a : a \in \texttt{edge\_arcs}[(i,j)] \text{ for some } j \bigr\}$$

No adjacency lookup is required.  Output: `face_arcs: dict i → list[arc]`.

---

## 5 — Wire assembly with angular ordering (Step 3)

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

**Failure modes:**
- Nearly-parallel tangents: `atan2` may select the wrong arc.
- Degenerate surface normal: falls back to `candidates[0]`.
- Non-manifold topology: odd-degree vertex; no arc-ordering can fix it.

---

## 6 — Parametric surfaces, pcurves, and the B-Rep data structure

### 6.1 — Parametric surfaces: UV → 3D

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

### 6.2 — Inverse maps: 3D → UV

For a point $\mathbf{p} \in \mathbb{R}^3$ lying on surface $S$, the closed-form
inverses use the same frame vectors and intrinsic parameters as Section 6.1:

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

### 6.3 — Pcurves: definition, analytical computation, and role

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

### 6.4 — What is a B-Rep file?

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

### 6.5 — BSpline faces: sidestepping explicit pcurve computation

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

## 7 — OCC topology assembly (Step 5)

### Analytical surface faces

For each face $i$ with an analytical surface:

1. Build `TopoDS_Vertex` for each position in `vertices`.
2. Build `TopoDS_Edge` from each arc.  Open arcs with vertex endpoints pass
   explicit `TopoDS_Vertex` objects so adjacent edges share the same instance.
   Closed-loop arcs and arcs with `v_start = None` are built from the curve alone.
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

## 8 — Healing and validation (Steps 6–7)

After sewing, `breplib.BuildCurves3d(shape)` ensures all edges have consistent
3D curves (some may be missing after sewing).  `ShapeFix_Shape` then repairs:

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

Validity is checked with `BRepCheck_Analyzer(shape).IsValid()`.  Note that
this runs on the in-memory OCC shape; the STEP round-trip (write + re-read)
can introduce coordinate rounding at $\sim 10^{-7}$ that degrades the result.
FreeCAD's Check Geometry panel runs the same analyzer on the re-read shape
and may report additional errors.

---

## 9 — STEP export (Step 8)

```python
writer = STEPControl_Writer()
writer.Transfer(shape, STEPControl_AsIs)
writer.Write(path)
```

STEP stores 3D curves, surfaces, and pcurves explicitly.  A face without valid
pcurves appears as an unbounded surface patch in the STEP file and is dropped
by most CAD viewers.

---

## 10 — Data-flow summary

```
Geom_Surface list   +   raw (i,j) → [Geom_Curve]
        │
        │  Section 1: boundary-strip projection
        ↓
trimmed (i,j) → [Geom_TrimmedCurve]
        │
        │  Section 2: GeomAPI_ExtremaCurveCurve, threshold ε_v
        ↓
vertices: np.ndarray (M,3),  vertex_edges: list[set]
        │
        │  Section 3: arc splitting (build_edge_arcs)
        ↓
edge_arcs: (i,j) → [arc_dict]
        │
        │  Section 4: key decomposition
        ↓
face_arcs: i → [arc_dict]
        │
        │  Section 5: cycle extraction + CCW angular ordering
        ↓
face_wires: i → [[(arc, forward)]]
        │
        │  Section 7: BRepBuilderAPI_MakeFace / UV-bounds / Sewing
        ↓
TopoDS_Shape (shell)
        │
        │  Sections 8–9: ShapeFix + BRepCheck + STEPControl_Writer
        ↓
output.step
```

---

## 11 — Known difficulties and alternatives

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

## 12 — Denormalisation: inverting the input normalisation on the output shape

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
