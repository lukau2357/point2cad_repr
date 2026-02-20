# B-Rep Construction Pipeline

This document covers every stage of the pipeline that follows surface-surface
intersection: trimming raw curves to the physical boundary, finding vertices,
splitting curves into B-Rep arcs, assembling face wires, handling pcurves, and
assembling the final `TopoDS_Shape` for STEP export.

**Inputs (produced upstream)**

| Object | Type |
|---|---|
| Fitted OCC surfaces | `list[Geom_Surface]`, indexed by cluster $i$ |
| Raw intersection curves | `dict (i,j) → list[Geom_Curve]` |
| Adjacency matrix + boundary strips | `np.ndarray`, `list[list[np.ndarray]]` |
| Cluster point clouds | `list[np.ndarray (N,3)]` |

**Output:** `TopoDS_Shape` (shell or solid) exportable to STEP.

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

In a valid manifold B-Rep, every node of $G_i$ has degree exactly 2, so $G_i$
is a disjoint union of simple cycles.  Each cycle is one wire.  Closed-loop
arcs form trivial one-arc cycles.

### Degree-2 case

The next arc is uniquely determined.  No angular computation needed.

### High-degree vertices (degree $\ge 4$)

At a vertex $v$ where $2k$ arc-endpoints meet on face $F$, the 2-manifold
constraint requires alternating inside/outside sectors.  The arcs of each
individual wire are angularly interleaved with arcs of every other wire.

**Outgoing tangent** of arc $a$ at vertex $v$:

$$\mathbf{t}_a(v) = \begin{cases}
+\dfrac{d\mathbf{C}}{dt}\Big|_{t_\text{start}} & v = v_\text{start}(a) \\[6pt]
-\dfrac{d\mathbf{C}}{dt}\Big|_{t_\text{end}}   & v = v_\text{end}(a)
\end{cases}$$

The sign ensures $\mathbf{t}_a(v)$ always points away from $v$ along $a$.

**Surface normal** at $v$: project $v$ onto the surface via
`GeomAPI_ProjectPointOnSurf`, evaluate via `GeomLProp_SLProps`.  Returns
`None` on failure (degenerate patch).

**CCW selection rule.**  Having arrived at $v$ from arc `prev`, define:

$$\hat{\mathbf{e}}_1 = \frac{\mathbf{t}_\text{prev}(v)}{\|\mathbf{t}_\text{prev}(v)\|}, \qquad
  \hat{\mathbf{e}}_2 = \hat{N} \times \hat{\mathbf{e}}_1$$

so that $(\hat{\mathbf{e}}_1, \hat{\mathbf{e}}_2, \hat{N})$ is a right-handed
frame with $\hat{\mathbf{e}}_1$ pointing backward (away from $v$, toward where
we came from).  For each candidate arc $c$, project its outgoing tangent:

$$\theta_c = \operatorname{atan2}(\hat{\mathbf{t}}_c \cdot \hat{\mathbf{e}}_2,\;
                                  \hat{\mathbf{t}}_c \cdot \hat{\mathbf{e}}_1), \qquad
\tilde\theta_c = \theta_c \bmod 2\pi \in [0, 2\pi)$$

Select:

$$c^* = \operatorname*{arg\,min}_{c:\,\tilde\theta_c > 0} \tilde\theta_c$$

This is the **first arc counterclockwise from the backward direction**.

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

## 6 — pcurves (Step 4)

A **pcurve** (parameter-space curve) is a 2D curve $c : [t_0, t_1] \to \mathbb{R}^2$
in the UV domain of surface $S$ satisfying:

$$S(c(t)) = \mathbf{C}(t) \quad \forall\, t \in [t_0, t_1]$$

where $\mathbf{C}$ is the 3D edge curve.  Every edge has one pcurve per
adjacent face.  OCC requires them for:
1. Defining the trim region of a face in UV space (surface is trimmed to the
   region enclosed by the outer wire's pcurves).
2. STEP export (pcurves are stored explicitly as `PCURVE` entities).
3. Boolean operations, shelling, and offset (OCC's internal algorithms operate
   in UV space).

**Analytical surfaces** (plane, cylinder, cone, sphere) have closed-form
inverse maps from $\mathbb{R}^3$ to UV.  `BRepBuilderAPI_MakeFace(surface, wire)`
computes pcurves internally without any extra step.

**BSpline surfaces** have no closed-form inverse.  Finding UV from a 3D point
requires solving $\min_{(u,v)} \|S(u,v) - \mathbf{p}\|^2$ numerically.
`BRepBuilderAPI_MakeFace` does **not** perform this computation automatically
for BSpline surfaces, leaving the face without valid pcurves.

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
