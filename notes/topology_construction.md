# B-Rep Topology Construction

Picks up from the output of `surface_intersection.py` / `brep_pipeline.py`:

| Already available | Python object |
|---|---|
| Fitted surfaces | `list[Geom_Surface]` indexed by cluster $i$ |
| Trimmed intersection curves | `dict (i,j) → list[Geom_Curve]` |
| Vertices (positions + incident edges) | `np.ndarray (M,3)`, `list[set]` |

**From Step 1 onward, arcs are the fundamental unit** — not original curves.
Every subsequent step (face incidence, wire assembly, pcurves, OCC topology)
operates on the arc dict produced by Step 1.

The goal is to produce a valid `TopoDS_Shape` (shell or solid) that can be
exported to STEP.

---

## Step 1 — Vertex–edge attribution and curve splitting

**Input:** vertex positions $\{v_k\}$, the edge dict, and for each vertex
the set of edges it is incident to (tracked during `compute_vertices`).

**Goal:** for every intersection curve, find which vertices lie on it,
determine parameter values $t^*$ for each, and **split the curve into
B-Rep arcs** bounded by those vertices.  The result replaces the original
edge dict (one curve per entry) with an arc dict (possibly multiple arcs
per entry).

**Projection:** each vertex $v_k$ incident to edge $(i,j)$ is projected
onto each curve $C$ of that edge via `GeomAPI_ProjectPointOnCurve`.  The
projection with smallest distance (below threshold) gives the parameter
$t^* = \arg{min}_t \|C(t) - v_k\|$.

### Geometric closure test

OCC's `GeomAPI_IntSS` always returns `Geom_TrimmedCurve`, so
`IsPeriodic()` is always `False` even when the underlying geometry is a
full circle.  Periodicity is instead detected geometrically:

$$\text{closed} \;=\; \|C(t_{min}) - C(t_{max})\| < \varepsilon_\text{close}$$

with $\varepsilon_\text{close} = 10^{-10}$.  Empirically, full
cylinder–plane circles give endpoint distances $\sim 10^{-17}$ while
trimmed lines give distances $\sim 10^{-1}$, so the threshold is
unambiguous.

### Arc splitting rules

For a curve $C$ with domain $[t_{min}, t_{max}]$ and $k$ vertices at sorted
parameters $t_1^* \leq \cdots \leq t_k^*$:

| Curve type | $k$ | B-Rep arcs produced |
|---|---|---|
| Closed ($\|C(t_{min})-C(t_{max})\| < \varepsilon$) | 0 | 1 closed-loop arc over $[t_{min}, t_{max}]$ |
| Closed | $k \geq 1$ | $(k-1)$ interior arcs $[t_1^*,t_2^*],\ldots,[t_{k-1}^*,t_k^*]$ + 1 wrap-around arc |
| Open | 0 | 1 arc over $[t_{min}, t_{max}]$ with no vertex endpoints |
| Open | $k \geq 1$ | $k-1$ interior arcs $[t_1^*, t_2^*], \ldots, [t_{k-1}^*, t_k^*]$; tails $[t_{min}, t_1^*]$ and $[t_k^*, t_{max}]$ are **discarded** |

The boundary tails of open curves are artifacts of the conservative
boundary-strip trimming (extension\_factor = 0.15); in a complete closed
solid they have no vertex at the trim boundary and are not valid B-Rep edges.

**Closed arcs with $k=0$** (`v_start=None, v_end=None, closed=True`) occur
when a full circle has no other intersection curve meeting it — i.e., the two
adjacent surfaces share that circle as their complete common boundary, with no
corner vertex.  These form a one-arc, zero-vertex wire on their own and are
fully valid in B-Rep (e.g., the circular cap of a cylinder).

**Wrap-around arc** for closed curves: the arc from $t_k^*$ back to $t_1^*$
through the seam.

- **Primary path:** `Geom_TrimmedCurve.BasisCurve()` returns the underlying
  periodic `Geom_Curve` (e.g. `Geom_Circle`) via virtual dispatch — without
  downcasting.  The basis curve accepts parameters beyond $[t_{min},t_{max}]$
  because it is natively periodic, so
  $\text{Geom\_TrimmedCurve}(\text{basis},\; t_k^*,\; t_1^*+\text{span})$
  is constructed directly.

- **Seam-vertex fallback** (when `BasisCurve()` returns `None`): a synthetic
  seam vertex is inserted at $C(t_{min}) = C(t_{max})$ and appended to the
  vertex list.  The wrap-around is split into two sub-arcs:
  $[t_k^*, t_{max}]$ (to seam) and $[t_{min}, t_1^*]$ (from seam).
  The seam vertex participates only in this edge's wire.

**Output:** `edge_arcs: dict (i,j) → list[arc]`, where each arc carries:
- `curve` — `Geom_TrimmedCurve` for this arc
- `v_start`, `v_end` — vertex indices (`None` for closed-loop arcs)
- `t_start`, `t_end` — parameter bounds on the arc curve
- `closed` — `True` if the arc is a closed loop (no bounding vertices)

`build_edge_arcs` also returns an updated `(vertices, vertex_edges)` that may
include synthetic seam vertices appended during the fallback path.

The total number of arcs across all edges will generally exceed the number
of original intersection curves.

---

## Step 2 — Face–arc incidence

**Input:** `edge_arcs: dict (i,j) → list[arc]`.

**Goal:** for each surface/face $i$, collect all arcs on its boundary.

This is immediate from the arc dict key structure: every arc stored under
key $(i,j)$ lies simultaneously on face $i$ and face $j$.  No adjacency
matrix lookup is needed — the surface indices that generated each curve are
encoded in the dict key, and all arcs derived from that curve inherit the
same pair of faces.

$$
\partial F_i = \bigl\{\, \text{arc} : \text{arc} \in \texttt{edge\_arcs}[(i,j)]
               \text{ for some } j \bigr\}
\;\cup\;
\bigl\{\, \text{arc} : \text{arc} \in \texttt{edge\_arcs}[(j,i)]
               \text{ for some } j \bigr\}.
$$

In practice, keys are stored as $(i,j)$ with $i < j$, so only one of the
two unions applies depending on convention.

**Output:** `face_arcs: dict i → list[arc]`

---

## Step 3 — Wire assembly (boundary loop finding)

**Input:** `face_arcs: dict i → list[arc]` from Step 2.

**Goal:** partition the arc list for face $i$ into one or more **closed
loops** (wires).  Each loop bounds one connected component of the face
boundary — typically one outer loop, possibly inner loops for holes.

**Graph formulation:** define a graph $G_i$:
- **nodes** = all vertex indices appearing as `v_start` or `v_end` in the
  arcs of face $i$
- **edges** = all open arcs in `face_arcs[i]`, connecting `v_start` to
  `v_end`

In a valid manifold B-Rep, every node of $G_i$ has degree exactly 2
(exactly two arcs of face $i$ meet at each vertex).  Therefore $G_i$ is a
**disjoint union of simple cycles** — each cycle is one wire.

Closed-loop arcs (`closed=True`, no vertex endpoints) form a trivial
one-arc cycle by themselves and are added directly as single-arc wires.

**Algorithm (cycle extraction):**

```
for each face i:
    separate closed-loop arcs → each becomes its own single-arc wire
    build adjacency list of G_i from open arcs
    mark all arcs as unvisited
    while there are unvisited arcs:
        pick any unvisited arc a with (v_start, v_end)
        wire = [a],  v_cur = a.v_end
        while v_cur ≠ wire[0].v_start:
            a_next = unvisited arc in G_i incident to v_cur
            wire.append(a_next);  mark a_next visited
            v_cur = other endpoint of a_next
        wires_i.append(wire)
```

**Output:** `face_wires: dict i → list[list[arc]]`

Each inner list is an ordered sequence of arcs forming a closed loop.

### Arc orientation within a wire

Each arc has a natural direction (from `v_start` to `v_end` by increasing
$t$).  When traversing a wire, an arc may be used **forward** (same
direction) or **reversed**.  OCC encodes this with `TopAbs_FORWARD` /
`TopAbs_REVERSED` on `TopoDS_Edge`.

Consecutive arcs in a wire must share the same vertex at their junction
(one arc ends at the vertex where the next begins).

---

## Step 4 — pcurves (2D parameter-space curves)

**What they are:** for each edge $C_{ij}$ lying on face $i$, OCC requires a
2D curve $c_{ij}^{(i)} : [t_0, t_1] \to \mathbb{R}^2$ in the UV parameter
space of surface $F_i$ such that

$$F_i\bigl(c_{ij}^{(i)}(t)\bigr) = C_{ij}(t) \quad \forall t.$$

This is called the **pcurve** (or **2D curve on surface**) of the edge with
respect to face $i$.  Every edge has two pcurves — one for each of its two
adjacent faces.

**Why they are needed:**
1. OCC's `BRep_Builder` requires them for constructing `TopoDS_Face`.
2. STEP export stores pcurves explicitly.
3. OCC uses them internally for Boolean ops, shelling, offset, etc.

**How to compute them:**

| Surface type | Method |
|---|---|
| Plane | Closed-form: project 3D point onto plane basis $(\mathbf{e}_1, \mathbf{e}_2)$ |
| Cylinder / Cone / Sphere | Closed-form: invert the parametric map analytically |
| B-spline (INR-derived) | Numerical: `GeomProjLib::Project(curve3d, surface)` |

For analytical surfaces the pcurve is itself analytical (a line or conic in UV
space).  For B-spline surfaces OCC computes the pcurve numerically and returns
it as a `Geom2d_BSplineCurve`.

---

## Step 5 — OCC topology assembly

With vertices, edge curves + pcurves, and wires in hand, assemble the
`TopoDS_Shape` using `BRep_Builder`:

```
builder = BRep_Builder()

# 0-cells
for each vertex v_k:
    TopoDS_Vertex Vk  ←  builder.MakeVertex(gp_Pnt(v_k), tolerance)

# 1-cells
for each edge (i,j) with curve C and vertices (Va, Vb):
    TopoDS_Edge E  ←  BRepBuilderAPI_MakeEdge(C, Va, Vb)
    builder.UpdateEdge(E, pcurve_i, face_i, tolerance)
    builder.UpdateEdge(E, pcurve_j, face_j, tolerance)

# Closed edges (no vertices)
for each closed-loop edge (i,j):
    TopoDS_Edge E  ←  BRepBuilderAPI_MakeEdge(C)  # no vertices

# 1-chains
for each face i, each wire W in face_wires[i]:
    wire_maker = BRepBuilderAPI_MakeWire()
    for each edge E in W (with correct orientation):
        wire_maker.Add(E)
    TopoDS_Wire Wi  ←  wire_maker.Wire()

# 2-cells
for each face i:
    TopoDS_Face Fi  ←  BRepBuilderAPI_MakeFace(surface_i, outer_wire_i)
    for each inner wire W_hole:
        builder.Add(Fi, W_hole)  # cut holes

# Shell / Solid
shell_maker = BRepBuilderAPI_Sewing(tolerance)
for each face Fi:
    shell_maker.Add(Fi)
shell_maker.Perform()
TopoDS_Shell S  ←  shell_maker.SewedShape()

# Promote to solid if closed
solid  ←  BRepBuilderAPI_MakeSolid(S)
```

---

## Step 6 — Orientation fixing

A closed solid must have all face normals pointing **outward**.  OCC provides:

```python
BRepLib.OrientClosedSolid(solid)
```

For open shells, orientation must be set manually or by convention.

An incorrectly oriented face causes Boolean operations and mass-property
computations to fail.  The fix is to flip the face orientation:
`face.Complemented()`.

---

## Step 7 — Healing

Fitting errors leave small gaps between edges and faces that violate strict
topological consistency.  `ShapeFix_Shape` repairs common defects:

```python
fixer = ShapeFix_Shape(shape)
fixer.SetPrecision(tolerance)
fixer.Perform()
healed = fixer.Shape()
```

Common defects and their fixes:

| Defect | Fix |
|---|---|
| Gap between edge endpoint and vertex | `ShapeFix_Edge::FixVertexTolerance` |
| Wire not closed | `ShapeFix_Wire::FixClosed` |
| Face normal ambiguous | `ShapeFix_Face::FixOrientation` |
| Degenerate edges (zero-length) | `ShapeFix_Wire::FixDegenerated` |

The tolerance passed to `ShapeFix` should match the vertex threshold
$\varepsilon_v$ used during vertex finding.

---

## Step 8 — STEP export

```python
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs

writer = STEPControl_Writer()
writer.Transfer(healed, STEPControl_AsIs)
writer.Write("output.step")
```

---

## Summary of data flow

```
surfaces  {Geom_Surface}
    │
    │  adjacency matrix A
    ↓
edge dict  (i,j) → [Geom_Curve]          ← surface_intersection.py
    │
    │  compute_vertices  →  vertex positions + incident edges
    ↓
Step 1: vertex_edge_params  (i,j) → [(v_k, c_idx, t*)]    ← topology.py
    │
Step 2: face_edges  i → [(i,j)]
    │
Step 3: face_wires  i → [[ordered edges per loop]]
    │
Step 4: pcurves  (i,j), face i → Geom2d_Curve
    │
Step 5: TopoDS_Shell / TopoDS_Solid                        ← topology.py
    │
Step 6: orientation fix
    │
Step 7: ShapeFix healing
    │
Step 8: STEP export
```

---

## Known difficulties

**Wire assembly uniqueness.** The graph $G_i$ is a disjoint union of simple
cycles only in a manifold B-Rep.  Near degenerate geometry (two edges nearly
coincident, a vertex of degree > 2), the graph may not be a valid cycle cover.
This can happen when fitting errors cause spurious intersections.  Detection:
a vertex with degree $\neq 2$ in $G_i$ signals a topology error.

**pcurve accuracy.** For B-spline surfaces, the numerical pcurve from
`GeomProjLib::Project` may accumulate error.  If the pcurve deviates from the
3D curve by more than the tolerance, `ShapeFix` will struggle.  Mitigation:
compute the pcurve at higher precision or re-sample the 3D curve and
re-project.

**Closed edges on periodic surfaces.** A full circle on a cylindrical face
(the cap boundary) has no vertices.  It forms a one-edge wire, but OCC
requires correct seam-edge handling for the cylinder's $v$-periodic direction.
`BRepLib::BuildCurves3d` and `BRepLib::SameParameter` help ensure consistency.

**Hole classification.** When a face has multiple wires (outer + inner), OCC
needs to know which is the outer boundary.  For planar faces this is determined
by the signed area of each wire in UV space: positive = outer, negative = inner
(or vice versa depending on the normal orientation convention).  For curved
surfaces, the classification is done by ray-casting from the wire interior into
the surface.
