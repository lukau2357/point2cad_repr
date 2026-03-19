# Wire Assembly via BOPAlgo_BuilderFace

This document describes the current wire assembly pathway, which replaces the
manual angular ordering approach (Section 9 of `brep_construction.md`) with
OCC's `BOPAlgo_BuilderFace`.

---

## 1 — Motivation

The manual wire assembly algorithm (angular ordering at vertices, Eulerian
cycle extraction, inner/outer wire classification) is complex, unintuitive,
and has known bugs — notably an opposite-angle pairing failure at degree-4
vertices where two intersection curves cross.  It is also difficult to
justify in a written report since it has no supporting literature.

`BOPAlgo_BuilderFace` is the same algorithm OCC uses internally during
boolean operations to rebuild faces from edges.  It replaces the entire wire
assembly + face construction pipeline with a single OCC call per face:
given a surface and a set of edges, it internally builds wires, classifies
outer vs inner (hole) wires, and produces the bounded face patches.

---

## 2 — OCC topology objects involved

### `TopoDS_Vertex`

A vertex in OCC is not just a 3D point — it carries a **tolerance sphere**:

$$\text{tol}(V) = \max\bigl(r \mid \forall \text{ edge } E \text{ incident to } V,\; \|\mathbf{p}(V) - \mathbf{C}_E(t_V)\| \le r\bigr)$$

where $\mathbf{p}(V)$ is the vertex position and $\mathbf{C}_E(t_V)$ is the
curve of edge $E$ evaluated at the parameter corresponding to $V$.  Any
curve endpoint landing within this sphere is considered coincident with $V$.

Created via:

```python
vtx = BRepBuilderAPI_MakeVertex(gp_Pnt(x, y, z)).Vertex()
```

Default tolerance is `Precision::Confusion()` $= 10^{-7}$.  In our pipeline,
mesh-derived BSpline curves have fitting errors of $10^{-5}$ to $5 \times 10^{-5}$
between vertex positions and `curve.Value(t)`, so the default tolerance is
too tight.  We increase it:

```python
bb = BRep_Builder()
bb.UpdateVertex(vtx, tolerance)  # tolerance = 1e-3
```

This sets the tolerance sphere radius to $10^{-3}$, which comfortably
encompasses the BSpline fitting errors.  Without this, `BRepBuilderAPI_MakeEdge`
returns error code 5 (`DifferentsPointAndParameter`) for nearly all edges.

**Why the default tolerance is insufficient.**  `BRepBuilderAPI_MakeEdge(curve, v1, v2, t0, t1)`
internally checks:

$$\|\mathbf{p}(V_\text{start}) - \mathbf{C}(t_0)\| \le \text{tol}(V_\text{start})$$
$$\|\mathbf{p}(V_\text{end}) - \mathbf{C}(t_1)\| \le \text{tol}(V_\text{end})$$

With default $\text{tol} = 10^{-7}$ and actual mismatch $\sim 10^{-5}$,
the check fails.  After `UpdateVertex(vtx, 10^{-3})`, the check passes.

### `TopoDS_Edge`

An edge binds a `Geom_Curve` (or a `Geom_TrimmedCurve`) to a parameter
interval $[t_0, t_1]$ and two endpoint vertices.  Created via:

```python
edge = BRepBuilderAPI_MakeEdge(curve, vtx_start, vtx_end, t0, t1).Edge()
```

The 4-argument overload `(curve, t0, t1)` is used for arcs without
vertex endpoints.  The 1-argument overload `(curve)` is used for closed
curves (full circles, full ellipses).

**Edge orientation.**  A `TopoDS_Edge` has an intrinsic orientation
(FORWARD or REVERSED).  When two faces share an edge, they must reference
it with opposite orientations — the edge is traversed in opposite directions
by the two face wires.  This is a fundamental B-Rep consistency requirement.

### Shared vertex instances

Two edges are topologically connected at a vertex if and only if they
reference the **same `TopoDS_Vertex` object** (identity, not just equal
position).  This is why we build a `vtx_to_topo` dictionary mapping
vertex indices to `TopoDS_Vertex` instances:

```python
vtx_to_topo = {}  # vertex_index → TopoDS_Vertex
for arcs in face_arcs.values():
    for arc in arcs:
        for vi_key in ("v_start", "v_end"):
            vi = arc.get(vi_key)
            if vi is not None and vi not in vtx_to_topo:
                pos = vertices[vi]
                vtx = BRepBuilderAPI_MakeVertex(
                    gp_Pnt(float(pos[0]), float(pos[1]), float(pos[2]))
                ).Vertex()
                bb = BRep_Builder()
                bb.UpdateVertex(vtx, tolerance)
                vtx_to_topo[vi] = vtx
```

Every arc referencing vertex index $v$ gets the same `TopoDS_Vertex`
object.  This allows `BOPAlgo_BuilderFace` to recognise edge connectivity
and assemble wires, and allows `BRepBuilderAPI_Sewing` to identify shared
edges between faces.

### `TopoDS_Wire`

A wire is an ordered sequence of edges forming a connected path on a face.
In the manual pathway, wires are assembled explicitly by the angular
ordering algorithm.  In the BuilderFace pathway, wires are assembled
internally by OCC — the user never constructs `TopoDS_Wire` objects
directly.

### `TopoDS_Face`

A face is a bounded region on a surface, delimited by one or more wires.
The outer wire defines the face boundary; inner wires define holes.

---

## 3 — The BuilderFace algorithm

### Per-face processing

For each analytical face $i$ with surface $S_i$ and incident arcs
$\{a_1, \ldots, a_k\}$:

**Step 1 — Create edges.**  For each arc, create a fresh `TopoDS_Edge`
using the arc's curve, parameter bounds, and shared vertex instances.
Edges are created per-face (not shared across faces) to avoid orientation
conflicts — if face $i$ and face $j$ share an arc, they each get their own
`TopoDS_Edge` copy.  Sewing merges coincident edges afterward.

```python
edge_list = TopTools_ListOfShape()
for arc in arcs:
    v_start, v_end = arc.get("v_start"), arc.get("v_end")
    curve, t0, t1 = arc["curve"], arc["t_start"], arc["t_end"]
    if v_start is not None and v_end is not None:
        edge = BRepBuilderAPI_MakeEdge(
            curve, vtx_to_topo[v_start], vtx_to_topo[v_end], t0, t1
        ).Edge()
    elif arc.get("closed", False):
        edge = BRepBuilderAPI_MakeEdge(curve).Edge()
    else:
        edge = BRepBuilderAPI_MakeEdge(curve, t0, t1).Edge()
    edge_list.Append(edge)
```

**Step 2 — Create a reference face.**  An unbounded face on $S_i$,
used as the "canvas" on which BuilderFace will partition the surface:

```python
ref_face = BRepBuilderAPI_MakeFace(surface, tolerance).Face()
```

For analytical surfaces (plane, cylinder, sphere, cone), `MakeFace` with
just the surface creates a face covering the full natural domain — an
infinite plane, a full cylinder, etc.

**Step 3 — Run BuilderFace.**

```python
builder = BOPAlgo_BuilderFace()
builder.SetFace(ref_face)
builder.SetShapes(edge_list)
builder.SetFuzzyValue(tolerance)
builder.Perform()
```

Internally, `BOPAlgo_BuilderFace` does the following:

1. **Compute pcurves.**  Each 3D edge curve is projected onto the reference
   surface to obtain a 2D parametric curve (pcurve) in $(u, v)$ space.
   This is necessary because wire assembly and face bounding are 2D
   operations performed in the surface's parameter domain.

2. **Build 2D wire graph.**  The pcurves define a planar graph in $(u, v)$
   space.  Connected edges (sharing vertices) form paths.

3. **Assemble wires.**  Paths are closed into wires.  The 2D topology
   determines which edges connect — this is where shared `TopoDS_Vertex`
   instances are essential.  If two edges don't share a vertex object,
   BuilderFace cannot connect them even if their 3D endpoints coincide.

4. **Classify wires.**  Outer wires (bounding a finite region from outside)
   vs inner wires (holes) are determined by the 2D winding direction in
   $(u, v)$ space.

5. **Partition the surface.**  The reference face is split into bounded
   regions (the "areas") by the wire network.  Each area is a `TopoDS_Face`
   with its own outer wire and optional inner wires.

The result is accessed via `builder.Areas()`, which returns a
`TopTools_ListOfShape` of bounded faces.

**Step 4 — Collect faces.**  All bounded faces from all surfaces are added
to a sewing object for shell assembly.

### Why edges are created per-face

An edge shared between faces $i$ and $j$ must appear with **opposite
orientations** in the two face wires (FORWARD in one, REVERSED in the other).
If we create a single `TopoDS_Edge` and pass it to both faces' BuilderFace
instances, the first instance may set its orientation to FORWARD, and the
second needs REVERSED.  But OCC's orientation is stored on the `TopoDS_Edge`
itself (as a shape orientation flag), so sharing causes conflicts.

Instead, each face creates its own `TopoDS_Edge` from the same curve and
vertices.  After BuilderFace, sewing identifies coincident edges (same 3D
curve, same vertex endpoints within tolerance) and merges them with the
correct relative orientations.

---

## 4 — Sewing

After all faces are collected:

```python
sewing = BRepBuilderAPI_Sewing(tolerance)
# ... add all faces ...
sewing.Perform()
shape = sewing.SewedShape()
```

Sewing identifies **coincident free edges** between faces — edges that lie
on the boundary of exactly one face but whose 3D curves match (within
tolerance) an edge on another face.  It merges these into shared edges,
producing a topologically connected shell.

The sewing tolerance ($10^{-3}$) must be large enough to accommodate:
- BSpline fitting errors ($\sim 10^{-5}$)
- Vertex tolerance spheres ($10^{-3}$)
- Any small positional differences between per-face edge copies

If faces don't share enough edges (e.g., due to MakeEdge failures),
sewing produces a `TopoDS_Compound` (disconnected collection) instead of
a `TopoDS_Shell`.

---

## 5 — Post-sewing healing

### Face orientation fix (`ShapeFix_Shell`)

After sewing, if the result is a shell:

```python
shell_fix = ShapeFix_Shell(topods.Shell(shape))
shell_fix.FixFaceOrientation(topods.Shell(shape))
shell_fix.Perform()
shape = shell_fix.Shape()
```

This corrects face orientations so that adjacent faces have consistent
outward-pointing normals.  Without it, `BRepCheck_Analyzer` reports
`BadOrientationOfSubshape` errors.

If sewing produced a compound (not a shell), this step is skipped.

### General shape healing

```python
breplib.BuildCurves3d(shape)           # Rebuild missing 3D curves
fixer = ShapeFix_Shape(shape)          # Fix gaps, orientations, etc.
fixer.SetPrecision(tolerance)
fixer.Perform()
shape = fixer.Shape()
breplib.SameParameter(shape, True)     # Reconcile 3D/2D edge tolerances
```

### Solid orientation

For closed shells, wrap into a solid and orient normals outward:

```python
solid_maker = BRepBuilderAPI_MakeSolid(topods.Shell(shape))
solid = solid_maker.Solid()
breplib.OrientClosedSolid(solid)
```

---

## 6 — BSpline (INR) faces

INR faces bypass BuilderFace entirely.  They use the UV-bounds approach
from the manual pathway:

1. Detect geometric closure (`SetUPeriodic` / `SetVPeriodic`)
2. Project arc sample points onto the surface to compute UV bounds
3. Build a bounded face via `BRepBuilderAPI_MakeFace(surface, u_min, u_max, v_min, v_max, tol)`

This is because INR surfaces are `Geom_BSplineSurface` objects — their
intersection curves are also BSplines fitted to mesh polylines, and
the UV-bounds approach produces cleaner faces than trying to use
BuilderFace with potentially imprecise BSpline-on-BSpline edges.

---

## 7 — Comparison with manual wire assembly

| Aspect | Manual (angular ordering) | BuilderFace |
|--------|--------------------------|-------------|
| Wire assembly | Explicit Eulerian decomposition + CCW angle selection | Internal OCC algorithm in 2D parameter space |
| Inner/outer classification | Manual area-based check | Internal OCC winding analysis |
| Degree-4 vertex handling | Known bug (opposite-angle pairing) | Handled correctly by OCC |
| Edge sharing | Single `TopoDS_Edge` per arc, shared across faces | Fresh edge per face, sewing merges afterward |
| Face construction | `BRepBuilderAPI_MakeFace(surface, outer_wire)` + `Add(inner_wire)` | `BOPAlgo_BuilderFace` produces complete faces |
| Interpretability | Complex, no supporting literature | Standard OCC tool, well-documented in OCC sources |
| Failure mode | Wrong wire topology → self-intersecting wires | Missing edges → unbounded face patches |

---

## 8 — `BRepBuilderAPI_MakeEdge` error codes

| Code | Name | Meaning |
|------|------|---------|
| 0 | `EdgeDone` | Success |
| 1 | `PointProjectionFailed` | Vertex cannot be projected onto curve |
| 2 | `ParameterOutOfRange` | $t_0$ or $t_1$ outside curve domain |
| 3 | `DifferentPointsOnClosedCurve` | Endpoints on closed curve at wrong positions |
| 4 | `PointWithInfiniteParameter` | Vertex projects to infinite parameter |
| 5 | `DifferentsPointAndParameter` | $\|\mathbf{p}(V) - \mathbf{C}(t)\| > \text{tol}(V)$ |
| 6 | `LineThroughIdenticPoints` | Line with coincident start/end points |

Error 5 was the root cause of BuilderFace failures before the vertex
tolerance fix — mesh-derived BSplines have $10^{-5}$ to $5 \times 10^{-5}$
fitting errors, exceeding the default $10^{-7}$ tolerance.
