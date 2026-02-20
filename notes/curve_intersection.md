# Intersection Curve–Curve Intersection (Vertex Finding)

## Goal

Given a set of surfaces $\{F_i\}$ and their pairwise intersection curves
$\{C_{ij}\}$ for adjacent pairs $(i,j)$, find the **vertices** of the
B-Rep: the points where two or more edges simultaneously meet.

---

## Which curve pairs can share a vertex: a geometric argument

A vertex $v$ in a B-Rep lies on at least two edges.  Each edge $C_{ij}$
lies simultaneously on surface $F_i$ and on surface $F_j$.  If two edges
$C_{ij}$ and $C_{kl}$ share a vertex $v$, then $v$ lies on all surfaces
in $\{i,j\} \cup \{k,l\}$.  We claim the correct search space is pairs
with

$$|\{i,j\} \cap \{k,l\}| = 1,$$

i.e. exactly one shared surface index.  This is **not** merely a
topological convention — it is forced by a dimension-counting argument.

### Dimension counting / transversality

Each smooth surface $F_i \subset \mathbb{R}^3$ is a codimension-1
submanifold.  The intersection of $k$ surfaces in general position has
codimension $k$, so dimension $3 - k$.  Concretely:

| Number of surfaces | Expected dimension of intersection |
|---|---|
| 1 | 2 (a surface) |
| 2 | 1 (a curve) — this is an edge $C_{ij}$ |
| 3 | 0 (a point) — this is a vertex |
| 4 | $-1$ (generically **empty**) |

So in $\mathbb{R}^3$, **three** surfaces in general position meet in a
point, and **four** surfaces in general position do not meet at all.

Now consider the two cases for a pair of curves:

**Case $|\{i,j\} \cap \{k,l\}| = 1$**, say $i = k$.  Then

$$C_{ij} \cap C_{kl} = F_i \cap F_j \cap F_i \cap F_l = F_i \cap F_j \cap F_l,$$

the intersection of **three** surfaces — generically a point.  This is
the correct codimension and explains why vertices of this type generically
exist and are isolated.

**Case $|\{i,j\} \cap \{k,l\}| = 0$ (disjoint indices).**  Then

$$C_{ij} \cap C_{kl} = F_i \cap F_j \cap F_k \cap F_l,$$

the intersection of **four** distinct surfaces — generically empty in
$\mathbb{R}^3$.  Any geometric proximity between $C_{ij}$ and $C_{kl}$
when all four surface indices are distinct is a non-generic coincidence,
not a structural vertex.

An equivalent way to see this: two curves in $\mathbb{R}^3$ each have
dimension 1, so their expected intersection dimension is $1 + 1 - 3 = -1$
(empty).  For them to meet, one needs a constraint that reduces the
effective ambient dimension.  Sharing a common surface $F_m$ does exactly
this: both curves lie on the 2-dimensional $F_m$, so their intersection is
expected to have dimension $1 + 1 - 2 = 0$ — a point.  Without a shared
surface there is no such reduction and the intersection is generically empty.

### The B-Rep constraint encodes generic geometry

The B-Rep rule "a vertex lies on exactly 3 faces" is the topological
*encoding* of this generic geometric reality, not an independent assumption.
Four-surface vertices can exist (they are valid B-Rep configurations) but
they require a special, non-generic geometric coincidence — for instance,
four planes arranged to pass through a common line, then perturbed to meet
at a point.  In a model reconstructed from point cloud data they are
extremely unlikely to arise from the fitting process.

**Consequence:** the correct search space is all pairs of edges whose
index sets share exactly one index, regardless of whether the remaining
surfaces are mutually adjacent.

### Why adjacency-graph triangles are insufficient

An earlier formulation required the triple $(i, j, k)$ to form a
**triangle** in the adjacency graph (all three pairs adjacent).  This is
valid only for three mutually adjacent planar faces.  It fails for curved
surfaces:

**Example — cylinder capped by planes.**  Let surface $0$ be a cylinder,
surface $1$ be the top cap, and surface $2$ be a side face adjacent to the
cap.  The edges are:
- $C_{01}$: circle (cylinder $\cap$ cap)
- $C_{12}$: line (cap $\cap$ side face)

The vertex where the circular arc meets the straight edge lies on $F_0$,
$F_1$, and $F_2$.  However, the cylinder $F_0$ and the side face $F_2$ are
**not** adjacent (they share no boundary strip), so $(0, 1, 2)$ is not an
adjacency-graph triangle.  Yet $C_{01}$ and $C_{12}$ share surface $F_1$,
so $|\{0,1\} \cap \{1,2\}| = 1$ and the vertex must still be found.

The triangle requirement was discarded in favour of the single-shared-index
condition above.

---

## Minimum-distance formulation

With approximate surface fitting, two curves sharing a surface do not
intersect exactly.  The vertex is found as the **closest approach**:

Given two trimmed curves $C_1 : [a_1, b_1] \to \mathbb{R}^3$ and
$C_2 : [a_2, b_2] \to \mathbb{R}^3$, define the squared-distance function:

$$D(t_1, t_2) = \|\mathbf{C}_1(t_1) - \mathbf{C}_2(t_2)\|_2^2$$

The first-order optimality conditions at an interior minimum require the
connecting vector to be **perpendicular to both tangents** simultaneously
(the common perpendicular):

$$(\mathbf{C}_1(t_1) - \mathbf{C}_2(t_2)) \cdot \mathbf{C}_1'(t_1) = 0,
\qquad
(\mathbf{C}_1(t_1) - \mathbf{C}_2(t_2)) \cdot \mathbf{C}_2'(t_2) = 0.$$

### Multiple local minima

$D(t_1, t_2)$ can have several local minima — each is a candidate vertex.
This is the expected case for a circle meeting a line at two points: both
intersection locations appear as separate local minima at distance $\approx 0$.
**All** local minima below the acceptance threshold $\varepsilon_v$ are kept.

---

## Vertex candidate and acceptance

Each local minimum $({\hat t}_1, {\hat t}_2)$ with
$\sqrt{D(\hat t_1, \hat t_2)} < \varepsilon_v$ produces a candidate:

$$\hat{v} = \frac{\mathbf{C}_1(\hat{t}_1) + \mathbf{C}_2(\hat{t}_2)}{2}.$$

### Threshold $\varepsilon_v$

The fitting error that causes the gap between two curves at a true vertex
depends on the surface types involved.  For plane–plane intersections the
gap is of the same order as the point-cloud spacing, so
$\varepsilon_v \sim \text{spacing}$ works.  For curved surfaces (cylinder–plane)
the fitting error can be smaller, so $\varepsilon_v$ may need to be tuned
separately from the adjacency threshold.

### Deduplication

Multiple edge pairs can independently discover the same physical vertex.
Candidates within $\varepsilon_v$ of each other are merged by averaging.

---

## OCC implementation: `GeomAPI_ExtremaCurveCurve`

`GeomAPI_ExtremaCurveCurve` computes **all** local extrema of
$D(t_1, t_2)$ over the parameter domains of both curves, using a
subdivision + Newton approach.

```python
from OCC.Core.GeomAPI import GeomAPI_ExtremaCurveCurve

ext = GeomAPI_ExtremaCurveCurve(curve1, curve2)
for k in range(1, ext.NbExtrema() + 1):
    dist = ext.Distance(k)            # distance at k-th local extremum
    if dist < eps_v:
        t1, t2 = ext.Parameters(k)   # curve parameters at k-th extremum
        p1 = curve1.Value(t1)
        p2 = curve2.Value(t2)
        midpoint = (p1 + p2) / 2     # vertex candidate
```

Note: `NbExtrema()` returns **all** extrema (minima and maxima).  Maxima
have large distances and are filtered out by the threshold check.

* https://dev.opencascade.org/doc/refman/html/class_geom_a_p_i___extrema_curve_curve.html

---

## Summary of algorithm

1. Collect all edges: $\{(i,j) : \text{adj}[i,j] = \text{True}\}$.
2. For every pair of distinct edges $((i,j),\, (k,l))$ with
   $|\{i,j\} \cap \{k,l\}| = 1$, retrieve their trimmed curve lists.
3. For every combination of curves $(C_{ij}^{(m)},\, C_{kl}^{(n)})$,
   compute all local extrema via `GeomAPI_ExtremaCurveCurve`.
4. Accept every local extremum with distance $< \varepsilon_v$ as a vertex
   candidate; set $\hat{v} = \tfrac{1}{2}(P_1 + P_2)$.
5. Deduplicate: merge candidates within $\varepsilon_v$ of each other.
6. Return the merged set as the vertex positions.
