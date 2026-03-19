# Mesh-Based Surface Intersection Pipeline

This document describes the mesh intersection pathway for computing edge
curves and vertices between adjacent surfaces.  It replaces the analytical
+ OCC `GeomAPI_IntSS` approach with a purely geometric method: intersect
the triangulated surface meshes, extract polylines, compute vertices from
polyline crossings, and fit OCC BSpline curves to the trimmed polylines.

Implementation: `point2cad/mesh_intersections.py`.

---

## 1 — Mesh intersection via PyVista

### Input

Each fitted surface has a triangulated mesh (from the surface fitter).
The mesh uses double-sided triangles: 4 triangles per quad cell, where
indices 0–1 are the original orientation and indices 2–3 are reversed
duplicates (for rendering visibility from both sides).

### Pre-processing: single-sided filtering

Before intersection, the reversed duplicates are removed:

$$\text{mask}[k] = (k \bmod 4) < 2$$

This halves the triangle count and prevents degenerate self-intersections
that occur when a triangle intersects its own reversed copy at the same
spatial location.

### PyVista intersection

For each adjacent surface pair $(i, j)$:

```python
result = pv_mesh_i.intersection(pv_mesh_j,
                                split_first=False, split_second=False)
```

This calls VTK's `vtkIntersectionPolyDataFilter`, which:
1. Tests all triangle pairs between the two meshes
2. For each intersecting pair, computes the intersection line segment
3. Returns a `PolyData` containing all intersection segments as VTK cells

The `split_first=False, split_second=False` flags prevent VTK from
modifying the input meshes — we only need the intersection curve, not
the split meshes.

### VTK cell format

The result stores line segments in VTK's packed connectivity format:

$$[\underbrace{n_1, p_0^{(1)}, p_1^{(1)}, \ldots}_{\text{cell 1}},
  \underbrace{n_2, p_0^{(2)}, p_1^{(2)}, \ldots}_{\text{cell 2}}, \ldots]$$

where $n_k$ is the number of points in cell $k$ (typically 2 for a line
segment) and $p_i^{(k)}$ are point indices into `result.points`.

---

## 2 — Polyline extraction

### Graph construction

The line segments from VTK are parsed into an undirected adjacency graph
$G = (V, E)$ where:
- Vertices are point indices from the intersection result
- Edges connect the two endpoints of each line segment

### Connected component discovery

BFS identifies connected components.  Each component corresponds to one
continuous intersection polyline (or a closed loop).

### Polyline ordering

Within each component, a walk produces an ordered sequence of point indices:

1. **Find endpoints** — nodes with degree 1.  If endpoints exist, start
   from one of them (open polyline).  If no endpoints exist, the polyline
   is closed; start from an arbitrary node.

2. **Greedy walk** — at each node, move to the unvisited neighbour.
   Ties are broken by `min(neighbors)` for deterministic output.

3. **Closure detection** — if the walk returns to the start node with
   $|\text{path}| > 2$, mark the polyline as closed.

The result is an ordered $(N \times 3)$ array of 3D coordinates, one per
polyline.

---

## 3 — Vertex computation from polyline intersections

Vertices are points where three or more surfaces meet.  Geometrically,
these are points where two intersection polylines cross — if polyline
$(i, j)$ crosses polyline $(j, k)$, the crossing point is a vertex
incident to surfaces $i$, $j$, and $k$.

### Edge pair selection

Only polyline pairs sharing exactly one cluster index are tested:

$$|\{i, j\} \cap \{k, l\}| = 1$$

This ensures the intersection is between two edges meeting at a common
face, which is the geometric configuration that produces a B-Rep vertex.

### Segment–segment closest point

For two 3D line segments $\mathbf{P}(s) = \mathbf{p}_0 + s\,\mathbf{d}_1$
and $\mathbf{Q}(t) = \mathbf{q}_0 + t\,\mathbf{d}_2$ with
$s, t \in [0, 1]$, we seek the pair $(s^*, t^*)$ minimising:

$$f(s, t) = \|\mathbf{P}(s) - \mathbf{Q}(t)\|^2$$

**Unconstrained solution.**  Expanding:

$$f(s, t) = \|\mathbf{r} + s\,\mathbf{d}_1 - t\,\mathbf{d}_2\|^2$$

where $\mathbf{r} = \mathbf{p}_0 - \mathbf{q}_0$.  Setting
$\nabla f = \mathbf{0}$:

$$\frac{\partial f}{\partial s} = 2\,\mathbf{d}_1 \cdot (\mathbf{r} + s\,\mathbf{d}_1 - t\,\mathbf{d}_2) = 0$$
$$\frac{\partial f}{\partial t} = -2\,\mathbf{d}_2 \cdot (\mathbf{r} + s\,\mathbf{d}_1 - t\,\mathbf{d}_2) = 0$$

Let $a = \mathbf{d}_1 \cdot \mathbf{d}_1$,
$b = \mathbf{d}_1 \cdot \mathbf{d}_2$,
$c = \mathbf{d}_1 \cdot \mathbf{r}$,
$e = \mathbf{d}_2 \cdot \mathbf{d}_2$,
$f_0 = \mathbf{d}_2 \cdot \mathbf{r}$.  The system becomes:

$$\begin{pmatrix} a & -b \\ -b & e \end{pmatrix}
\begin{pmatrix} s \\ t \end{pmatrix}
= \begin{pmatrix} -c \\ f_0 \end{pmatrix}$$

The determinant is $ae - b^2 \ge 0$ by Cauchy–Schwarz, with equality
when $\mathbf{d}_1 \parallel \mathbf{d}_2$ (parallel segments).

**Convexity.**  The Hessian of $f$ is:

$$H = 2\begin{pmatrix} a & -b \\ -b & e \end{pmatrix}$$

which is positive semi-definite (PSD) since $ae - b^2 \ge 0$.  When the
segments are non-parallel, $H$ is positive definite (PD), so the
unconstrained stationary point is the unique global minimum.

**Constrained solution.**  The parameters are clamped to $[0, 1]$:

$$s^* = \operatorname{clamp}(s, 0, 1), \qquad t^* = \operatorname{clamp}(t, 0, 1)$$

After clamping one parameter, the other is re-solved from the
corresponding gradient equation with the clamped value substituted.
This is exact for the box-constrained convex quadratic.

**Parallel segments** ($ae - b^2 < \varepsilon$).  The system is
rank-deficient; $s = 0$ is fixed and $t$ is solved from $f_0 / e$.

**Degenerate segments** ($a < \varepsilon$ or $e < \varepsilon$).
One or both segments collapse to a point; handled as special cases.

**Return value:** the closest points $\mathbf{P}(s^*)$, $\mathbf{Q}(t^*)$,
and their distance $\|\mathbf{P}(s^*) - \mathbf{Q}(t^*)\|$.  The vertex
candidate position is the midpoint:

$$\mathbf{v} = \frac{\mathbf{P}(s^*) + \mathbf{Q}(t^*)}{2}$$

### KDTree pre-filtering

Testing all segment pairs between two polylines of lengths $M$ and $N$
is $O(MN)$.  For long polylines this is expensive.

**Observation:** two segments can only be within distance $\tau$ (the
vertex detection threshold) if at least one point on each segment is
within distance $\tau + \ell_{\max}^A + \ell_{\max}^B$, where
$\ell_{\max}^A$ and $\ell_{\max}^B$ are the maximum segment lengths
in polylines $A$ and $B$ respectively.

**Algorithm:**
1. Compute $\ell_{\max}^A$ and $\ell_{\max}^B$.
2. Build a KDTree on the points of polyline $B$.
3. Query all points of polyline $A$ with radius
   $R = \tau + \ell_{\max}^A + \ell_{\max}^B$.
4. For each point $a_i \in A$ with nearby points $\{b_j\} \subset B$,
   add the adjacent segment pairs to a candidate set:
   - Segments $(a_{i-1}, a_i)$ and $(a_i, a_{i+1})$ from polyline $A$
   - Segments $(b_{j-1}, b_j)$ and $(b_j, b_{j+1})$ from polyline $B$
5. Only test the segment pairs in the candidate set.

This reduces the number of tested pairs from $O(MN)$ to
$O(|\text{close pairs}| \cdot 4)$, which is typically orders of magnitude
smaller for well-separated polylines.

**Soundness.**  The search radius $R$ is a conservative upper bound.
If the closest points on segments $s_A$ and $s_B$ are within distance
$\tau$, then at least one vertex of $s_A$ is within distance
$\tau + \ell_{\max}^A$ of the closest point on $s_B$, which in turn
is within distance $\ell_{\max}^B$ of a vertex of $s_B$.  So the triangle
inequality gives $\|a_i - b_j\| \le \tau + \ell_{\max}^A + \ell_{\max}^B$
for some vertex pair, which the KDTree query captures.

**No false negatives:** the pre-filter never discards a segment pair
whose closest-point distance is below $\tau$.

### Greedy deduplication

Multiple segment pairs near the same geometric vertex produce multiple
candidates at slightly different positions.  These are merged:

1. Sort candidates by distance (best first).
2. For each unused candidate, find all candidates within a clustering
   radius (default: same as the detection threshold).
3. Require at least 2 distinct edges among clustered candidates — a
   vertex must be incident to at least 2 intersection curves.
4. Average the positions of all candidates in the cluster.

**Output:** $(M \times 3)$ vertex array and a list of edge incidence
sets `vertex_edges[v]` recording which edges are incident to vertex $v$.

---

## 4 — BSpline fitting to polylines

### Fitting algorithm

Given an ordered polyline (array of 3D points), fit an OCC BSpline:

```python
GeomAPI_PointsToBSpline(points, deg_min=3, deg_max=8,
                        continuity=GeomAbs_C2, tolerance=1e-4)
```

OCC solves a least-squares problem: find the $C^2$-continuous cubic
BSpline curve $\mathbf{C}(t)$ minimising the sum of squared distances
to the input points, subject to the continuity and tolerance constraints.
The knot vector is determined automatically.

### Closure snap

If the polyline's first and last points are within distance
$\varepsilon_\text{close} = 5 \times 10^{-3}$, the last point is snapped
to exactly match the first:

$$\mathbf{p}_{N-1} \leftarrow \mathbf{p}_0$$

This ensures the fitted BSpline has coincident endpoints, which is
necessary for closed-loop arcs.

---

## 5 — Vertex snapping and arc construction

### Problem statement

After fitting a BSpline to a polyline, the curve must pass through the
previously computed vertices — these are the points where the arc starts,
ends, or is split.  However, a least-squares BSpline fit does not
interpolate the input points exactly: there is a residual fitting error.
If the BSpline misses a vertex by more than the vertex tolerance, the
resulting `TopoDS_Edge` will be rejected by OCC.

### Solution: pre-fitting vertex injection

Before fitting the BSpline, all incident vertex positions are injected
directly into the polyline array at their corresponding indices:

1. **Find incident vertices.** For each vertex $v$ incident to the
   current edge (from `vertex_edges`), find the closest polyline point
   index:

   $$i^* = \arg\min_i \|\mathbf{p}_i - \mathbf{v}\|$$

   Accept if $\|\mathbf{p}_{i^*} - \mathbf{v}\| < \tau$ (threshold).

2. **Trim the polyline.** Extract the sub-array between the outermost
   incident vertex indices:

   $$\text{trimmed} = \mathbf{p}[i_\text{first} : i_\text{last} + 1]$$

   where $i_\text{first}$ and $i_\text{last}$ are the polyline indices
   of the first and last incident vertices.

3. **Snap vertex positions.** Replace the polyline points at each
   incident vertex index with the exact vertex position:

   $$\text{trimmed}[i_k - i_\text{first}] \leftarrow \mathbf{v}_k \qquad \forall k$$

4. **Fit BSpline.** The BSpline is fitted to the modified polyline.
   Since the vertex positions are now exact points in the input array,
   the fitted curve passes very close to them (within the BSpline
   fitting tolerance of $10^{-4}$).

### Arc splitting at interior vertices

When a polyline has $k \ge 3$ incident vertices, the single BSpline is
split into $k - 1$ arcs at the interior vertex positions:

1. **Project interior vertices onto the curve.** For each interior
   vertex $\mathbf{v}_m$, compute the parameter $t_m$ via:

   ```python
   proj = GeomAPI_ProjectPointOnCurve(gp_Pnt(v), curve, t_min, t_max)
   t_m = proj.LowerDistanceParameter()
   ```

   This finds the parameter $t_m$ where $\|\mathbf{C}(t_m) - \mathbf{v}_m\|$
   is minimised.  Since $\mathbf{v}_m$ was snapped into the polyline before
   fitting, the projection distance is small ($\sim 10^{-4}$).

2. **Create trimmed arcs.** Sort the parameters
   $t_\text{min} \le t_1 \le \cdots \le t_{k-2} \le t_\text{max}$
   and create one `Geom_TrimmedCurve` per consecutive pair:

   $$\text{arc}_m = \texttt{Geom\_TrimmedCurve}(\mathbf{C}, t_m, t_{m+1})$$

   Each arc has `v_start` and `v_end` set to the corresponding vertex
   indices.

### Edge cases

| Incident vertices | Behaviour |
|-------------------|-----------|
| $k = 0$ | Fit full polyline as a single arc, `v_start = v_end = None` |
| $k = 1$ | Fit full polyline as a closed loop, `v_start = v_end = v_0` |
| $k = 2$ | Trim polyline between the two vertices, fit one arc |
| $k \ge 3$ | Trim, fit, then split into $k - 1$ arcs at interior vertices |

---

## 6 — Drawbacks of this approach

### BSpline fitting error vs surface constraints

The BSpline is fitted to the mesh intersection polyline by minimising
point-to-curve distance.  It does **not** enforce that the curve lies on
either adjacent surface.  In exact CAD, an edge curve simultaneously
satisfies:

$$\mathbf{C}(t) \in S_i \qquad \text{and} \qquad \mathbf{C}(t) \in S_j$$

for all $t$.  The mesh-derived BSpline only approximately satisfies these
constraints:

$$\|\mathbf{C}(t) - S_i\| \sim \varepsilon_\text{mesh} + \varepsilon_\text{fit}$$

where $\varepsilon_\text{mesh}$ is the mesh tessellation error and
$\varepsilon_\text{fit}$ is the BSpline fitting residual.  This causes
`InvalidCurveOnSurface` errors in `BRepCheck_Analyzer` when the deviation
exceeds the edge tolerance.

The `SameParameter` healing step partially mitigates this by increasing
edge tolerances to reflect the actual 3D-to-2D deviation, but large
deviations can still cause boolean operation failures downstream.

### Mesh resolution dependence

The quality of the intersection polyline depends on the mesh tessellation
density.  Coarse meshes produce polylines with few points, leading to
BSplines that poorly approximate the true intersection curve.  Fine meshes
produce better polylines but increase computation time.

### Tangent geometries

When two surfaces are tangent (e.g., a plane tangent to a cylinder), their
meshes touch but do not intersect transversally.  The VTK intersection
filter produces either no output or a degenerate near-zero-area intersection.
This case requires a separate proximity-based approach (future work using
`vtkDistancePolyDataFilter` or similar).

### Vertex position accuracy

Vertices are computed as midpoints of closest-point pairs between polyline
segments.  The accuracy is limited by:
- Mesh resolution (segment length)
- Polyline ordering accuracy
- The greedy deduplication averaging

A vertex that is inaccurate by more than the tolerance sphere radius
will cause `MakeEdge` failures (error code 5: `DifferentsPointAndParameter`).

### Closed curve handling

Mesh intersection can produce closed polylines (e.g., a full circle where
a plane cuts a cylinder).  The closure detection threshold
$\varepsilon_\text{close} = 5 \times 10^{-3}$ may fail for curves that
are geometrically closed but have endpoint positions separated by more
than this threshold due to mesh resolution.  In such cases the curve is
treated as open, which can produce a small gap in the face boundary.
