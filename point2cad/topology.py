"""
B-Rep topology construction — Step 1: vertex–edge attribution and arc splitting.

Given:
    intersections : dict (i,j) -> list[Geom_Curve]   (trimmed intersection curves)
    vertices      : np.ndarray (M, 3)                 (vertex positions)
    vertex_edges  : list[set]                         (vertex_edges[v] = set of (i,j) tuples)
    threshold     : float                             (projection distance tolerance)

Returns:
    edge_arcs    : dict (i,j) -> list[arc_dict]
    vertices     : np.ndarray (M', 3)   (may include added seam vertices)
    vertex_edges : list[set] of length M'

Each arc_dict has:
    curve   : Geom_TrimmedCurve   trimmed to this arc's parameter interval
    v_start : int or None         start vertex index (None for closed-loop arcs)
    v_end   : int or None         end vertex index   (None for closed-loop arcs)
    t_start : float               start parameter on `curve`
    t_end   : float               end parameter on `curve`
    closed  : bool                True if this arc forms a complete closed loop
"""

import math
import numpy as np

try:
    from OCC.Core.Geom    import Geom_TrimmedCurve
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve
    from OCC.Core.gp      import gp_Pnt
    OCC_AVAILABLE = True
except ImportError:
    OCC_AVAILABLE = False

# Geometric closure tolerance — endpoint distance below this means the curve
# is a closed loop (e.g. cylinder-plane circle gives ~1e-17, lines give ~1e-1).
CLOSURE_TOL = 1e-10


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _endpoint_dist(curve):
    """Euclidean distance between C(t_min) and C(t_max)."""
    t0 = curve.FirstParameter()
    t1 = curve.LastParameter()
    p0 = curve.Value(t0)
    p1 = curve.Value(t1)
    return math.sqrt(
        (p1.X() - p0.X()) ** 2 +
        (p1.Y() - p0.Y()) ** 2 +
        (p1.Z() - p0.Z()) ** 2
    )


def curve_is_closed(curve):
    """Return True if the curve forms a closed loop (geometric endpoint test)."""
    return _endpoint_dist(curve) < CLOSURE_TOL


def _basis_curve(curve):
    """
    Return the underlying basis curve of a Geom_TrimmedCurve via BasisCurve().
    The result is still typed as Geom_Curve, but OCC uses virtual dispatch so
    IsPeriodic() and Value(t) correctly delegate to the concrete type (e.g.
    Geom_Circle).  This allows Geom_TrimmedCurve(basis, t_k, t_1+span) to
    accept parameters beyond the original trim interval when the basis is periodic.

    Returns None if BasisCurve() is unavailable or returns None itself.
    """
    if not hasattr(curve, "BasisCurve"):
        return None
    try:
        basis = curve.BasisCurve()
        return basis  # may be None if the handle is null
    except Exception:
        return None


def _make_arc(source_curve, t_start, t_end):
    """Build a Geom_TrimmedCurve for [t_start, t_end] on `source_curve`."""
    return Geom_TrimmedCurve(source_curve, t_start, t_end)


def _project_vertex_on_curve(vertex_pos, curve, t_min, t_max):
    """
    Project `vertex_pos` (np.ndarray shape (3,)) onto `curve` restricted to
    [t_min, t_max].  Returns (t*, distance) or (None, None) on failure.
    """
    pnt  = gp_Pnt(float(vertex_pos[0]), float(vertex_pos[1]), float(vertex_pos[2]))
    proj = GeomAPI_ProjectPointOnCurve(pnt, curve, t_min, t_max)
    if proj.NbPoints() == 0:
        return None, None
    return proj.LowerDistanceParameter(), proj.LowerDistance()


def _pnt_to_np(pnt):
    return np.array([pnt.X(), pnt.Y(), pnt.Z()], dtype=np.float64)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def build_edge_arcs(intersections, vertices, vertex_edges, threshold=1e-3):
    """
    Step 1 of B-Rep topology: attribute vertices to curves and split each
    curve into B-Rep arcs.

    Parameters
    ----------
    intersections : dict (i,j) -> list[Geom_Curve]
        Trimmed intersection curves from surface_intersection.py.
    vertices : np.ndarray, shape (M, 3)
        Vertex positions from compute_vertices.
    vertex_edges : list[set]
        vertex_edges[v] = set of (i,j) tuples of edges incident to vertex v.
    threshold : float
        Maximum projection distance for a vertex to be considered incident to
        a curve.

    Returns
    -------
    edge_arcs    : dict (i,j) -> list[arc_dict]
    vertices     : np.ndarray (M', 3)   (may be extended with seam vertices)
    vertex_edges : list[set] of length M'
    """
    # Work on mutable copies so seam vertices can be appended when needed.
    verts      = list(vertices)
    vedge_sets = [s.copy() for s in vertex_edges]

    edge_arcs = {}

    for edge_key, curves in intersections.items():
        arcs_for_edge = []

        for curve in curves:
            t_min  = curve.FirstParameter()
            t_max  = curve.LastParameter()
            closed = curve_is_closed(curve)

            # ------------------------------------------------------------------
            # Find vertices incident to this edge that project onto this curve.
            # ------------------------------------------------------------------
            incident_params = []   # list of (t*, vertex_index)

            for v_idx, v_edge_set in enumerate(vedge_sets):
                if edge_key not in v_edge_set:
                    continue
                t_star, dist = _project_vertex_on_curve(
                    verts[v_idx], curve, t_min, t_max
                )
                if t_star is None or dist > threshold:
                    continue
                incident_params.append((t_star, v_idx))

            incident_params.sort(key=lambda x: x[0])
            k = len(incident_params)

            # ------------------------------------------------------------------
            # Arc splitting (see topology_construction.md Step 1)
            # ------------------------------------------------------------------
            if closed:
                if k == 0:
                    arcs_for_edge.append({
                        "curve":   _make_arc(curve, t_min, t_max),
                        "v_start": None,
                        "v_end":   None,
                        "t_start": t_min,
                        "t_end":   t_max,
                        "closed":  True,
                    })
                else:
                    span = t_max - t_min

                    # (k-1) interior arcs
                    for m in range(k - 1):
                        t_a, v_a = incident_params[m]
                        t_b, v_b = incident_params[m + 1]
                        arcs_for_edge.append({
                            "curve":   _make_arc(curve, t_a, t_b),
                            "v_start": v_a,
                            "v_end":   v_b,
                            "t_start": t_a,
                            "t_end":   t_b,
                            "closed":  False,
                        })

                    # Wrap-around arc: [t_k, t_1 + span] on the periodic basis.
                    t_last,  v_last  = incident_params[-1]
                    t_first, v_first = incident_params[0]
                    t_wrap_end = t_first + span

                    basis = _basis_curve(curve)
                    if basis is not None:
                        arcs_for_edge.append({
                            "curve":   _make_arc(basis, t_last, t_wrap_end),
                            "v_start": v_last,
                            "v_end":   v_first,
                            "t_start": t_last,
                            "t_end":   t_wrap_end,
                            "closed":  False,
                        })
                    else:
                        # Fallback: split wrap-around at the seam, insert a
                        # synthetic seam vertex at C(t_min) = C(t_max).
                        seam_idx = len(verts)
                        verts.append(_pnt_to_np(curve.Value(t_min)))
                        vedge_sets.append({edge_key})
                        print(
                            f"[topology] BasisCurve unavailable for edge {edge_key}: "
                            f"split wrap-around at seam, added vertex {seam_idx}"
                        )
                        arcs_for_edge.append({
                            "curve":   _make_arc(curve, t_last, t_max),
                            "v_start": v_last,
                            "v_end":   seam_idx,
                            "t_start": t_last,
                            "t_end":   t_max,
                            "closed":  False,
                        })
                        arcs_for_edge.append({
                            "curve":   _make_arc(curve, t_min, t_first),
                            "v_start": seam_idx,
                            "v_end":   v_first,
                            "t_start": t_min,
                            "t_end":   t_first,
                            "closed":  False,
                        })

            else:
                # Open curve.
                if k == 0:
                    arcs_for_edge.append({
                        "curve":   _make_arc(curve, t_min, t_max),
                        "v_start": None,
                        "v_end":   None,
                        "t_start": t_min,
                        "t_end":   t_max,
                        "closed":  False,
                    })
                else:
                    # k-1 interior arcs; boundary tails discarded.
                    for m in range(k - 1):
                        t_a, v_a = incident_params[m]
                        t_b, v_b = incident_params[m + 1]
                        arcs_for_edge.append({
                            "curve":   _make_arc(curve, t_a, t_b),
                            "v_start": v_a,
                            "v_end":   v_b,
                            "t_start": t_a,
                            "t_end":   t_b,
                            "closed":  False,
                        })

        edge_arcs[edge_key] = arcs_for_edge

    out_vertices = np.array(verts, dtype=np.float64) if verts else np.zeros((0, 3))
    return edge_arcs, out_vertices, vedge_sets


# ---------------------------------------------------------------------------
# Step 2 — Face–arc incidence
# ---------------------------------------------------------------------------

def face_arc_incidence(edge_arcs):
    """
    Step 2 of B-Rep topology: for each surface face i, collect all arcs
    whose edge key (i,j) contains i.

    Since every arc under key (i,j) lies simultaneously on face i and face j,
    attribution is a direct read of the dict keys — no adjacency matrix needed.

    Parameters
    ----------
    edge_arcs : dict (i,j) -> list[arc_dict]

    Returns
    -------
    face_arcs : dict i -> list[arc_dict]
        Each arc_dict also carries an extra key ``"edge_key"`` with the
        originating (i,j) tuple, so downstream steps can recover which pair
        of faces share an arc.
    """
    face_arcs = {}

    for edge_key, arcs in edge_arcs.items():
        fi, fj = edge_key
        for arc in arcs:
            tagged = dict(arc, edge_key=edge_key)
            face_arcs.setdefault(fi, []).append(tagged)
            face_arcs.setdefault(fj, []).append(tagged)

    return face_arcs


def print_face_arcs_summary(face_arcs):
    """Print a concise summary of the face–arc incidence."""
    print(f"Face arcs summary: {len(face_arcs)} faces")
    for face_idx in sorted(face_arcs):
        arcs   = face_arcs[face_idx]
        closed = sum(1 for a in arcs if a["closed"])
        open_  = len(arcs) - closed
        print(f"  face {face_idx:2d}  arcs={len(arcs)}  (open={open_}  closed={closed})")


# ---------------------------------------------------------------------------
# Step 3 — Wire assembly
# ---------------------------------------------------------------------------

def assemble_wires(face_arcs):
    """
    Step 3 of B-Rep topology: for each face, partition its arcs into closed
    wires (boundary loops).

    Closed arcs (closed=True) each form a trivial one-arc wire immediately.

    Open arcs are assembled via cycle extraction on the degree-2 boundary
    graph G_i:
      nodes  = vertex indices (v_start / v_end of open arcs)
      edges  = open arcs, connecting their two endpoint vertices

    Each wire is an ordered list of (arc, forward) pairs where `forward`
    indicates whether the arc is traversed in its natural direction
    (v_start → v_end) or reversed (v_end → v_start).

    Parameters
    ----------
    face_arcs : dict i -> list[arc_dict]

    Returns
    -------
    face_wires : dict i -> list[list[(arc_dict, bool)]]
        Outer list: one entry per wire on face i.
        Inner list: ordered (arc, forward) pairs forming a closed loop.

    Warnings are printed for faces where a vertex has degree != 2
    (topology error from fitting artefacts).
    """
    face_wires = {}

    for face_idx, arcs in face_arcs.items():
        wires = []

        # Trivial wires: each closed arc is its own single-arc wire.
        open_arcs = []
        for arc in arcs:
            if arc["closed"]:
                wires.append([(arc, True)])
            else:
                open_arcs.append(arc)

        if open_arcs:
            # Build adjacency list: vertex -> list of (arc, neighbour_vertex, forward)
            # forward=True  means the arc is stored as (v_start -> v_end)
            # forward=False means the arc is stored as (v_end -> v_start)
            adj = {}   # v -> list of (arc, exit_v, forward)
            for arc in open_arcs:
                vs, ve = arc["v_start"], arc["v_end"]
                adj.setdefault(vs, []).append((arc, ve,  True))
                adj.setdefault(ve, []).append((arc, vs, False))

            # Degree check
            for v, neighbours in adj.items():
                if len(neighbours) != 2:
                    print(
                        f"[topology] face {face_idx}: vertex {v} has degree "
                        f"{len(neighbours)} (expected 2) — topology may be degenerate"
                    )

            # Cycle extraction
            arc_index = {id(a): idx for idx, a in enumerate(open_arcs)}
            visited_arcs = set()   # indices into open_arcs already placed in a wire

            for start_idx, start_arc in enumerate(open_arcs):
                if start_idx in visited_arcs:
                    continue

                wire     = []
                v_target = start_arc["v_start"]   # must return here to close the loop
                arc      = start_arc
                forward  = True                    # enter start_arc from v_start side
                v_cur    = start_arc["v_end"]

                wire.append((arc, forward))
                visited_arcs.add(start_idx)

                broken = False
                while v_cur != v_target:
                    # Find the unvisited arc incident to v_cur
                    candidates = [
                        (a, exit_v, fwd)
                        for (a, exit_v, fwd) in adj.get(v_cur, [])
                        if arc_index[id(a)] not in visited_arcs
                    ]
                    if not candidates:
                        print(
                            f"[topology] face {face_idx}: open chain at vertex {v_cur}"
                            f" — wire left incomplete"
                        )
                        broken = True
                        break
                    arc, v_cur, forward = candidates[0]
                    wire.append((arc, forward))
                    visited_arcs.add(arc_index[id(arc)])

                if not broken:
                    wires.append(wire)

        face_wires[face_idx] = wires

    return face_wires


def print_face_wires_summary(face_wires):
    """Print a concise summary of the wire assembly result."""
    print(f"Wire assembly summary: {len(face_wires)} faces")
    for face_idx in sorted(face_wires):
        wires = face_wires[face_idx]
        print(f"  face {face_idx:2d}  wires={len(wires)}")
        for w_idx, wire in enumerate(wires):
            arc_descs = []
            for arc, fwd in wire:
                vs = arc["v_start"]
                ve = arc["v_end"]
                cl = "closed" if arc["closed"] else ("fwd" if fwd else "rev")
                arc_descs.append(f"({vs}→{ve},{cl})")
            print(f"    wire[{w_idx}]  arcs={len(wire)}  " + "  ".join(arc_descs))


# ---------------------------------------------------------------------------
# Step 5 — BRep assembly and STEP export
# ---------------------------------------------------------------------------

def _connected_components(open_arcs):
    """
    Group open arcs into connected components by shared vertex indices.
    Each component corresponds to one boundary wire on the face.
    """
    n = len(open_arcs)
    if n == 0:
        return []

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    vertex_to_first = {}
    for idx, arc in enumerate(open_arcs):
        for v in (arc["v_start"], arc["v_end"]):
            if v not in vertex_to_first:
                vertex_to_first[v] = idx
            else:
                union(vertex_to_first[v], idx)

    from collections import defaultdict
    components = defaultdict(list)
    for idx, arc in enumerate(open_arcs):
        components[find(idx)].append(arc)
    return list(components.values())


def build_brep_shape(face_arcs, occ_surfaces, vertices, tolerance=1e-3):
    """
    Build a TopoDS_Shape from face_arcs, delegating wire ordering and
    orientation entirely to ShapeFix_Wire / BRepBuilderAPI_Sewing.

    For each face:
      - Closed arcs each become a single-edge wire immediately.
      - Open arcs are grouped into connected components (one per boundary
        loop) and passed unordered to ShapeFix_Wire for reordering.
      - BRepBuilderAPI_MakeFace(surface, wires...) builds each face.
    All faces are sewn with BRepBuilderAPI_Sewing and healed with
    ShapeFix_Shape.

    Parameters
    ----------
    face_arcs    : dict i -> list[arc_dict]
    occ_surfaces : list[Geom_Surface], indexed by face index
    vertices     : np.ndarray (M, 3)
    tolerance    : sewing / ShapeFix tolerance

    Returns
    -------
    TopoDS_Shape
    """
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakeVertex,
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeWire,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_Sewing,
    )
    from OCC.Core.ShapeFix import ShapeFix_Wire, ShapeFix_Shape
    from OCC.Core.BRepLib  import breplib

    # 1. One TopoDS_Vertex per position.
    occ_verts = []
    for pos in vertices:
        occ_verts.append(
            BRepBuilderAPI_MakeVertex(
                gp_Pnt(float(pos[0]), float(pos[1]), float(pos[2]))
            ).Vertex()
        )

    # 2. One TopoDS_Edge per unique arc.
    #    Open arcs pass their endpoint TopoDS_Vertex objects so that adjacent
    #    edges share the same vertex object and are topologically connected.
    #    Closed arcs have no shared endpoints so only the curve is needed.
    arc_to_edge = {}   # id(arc) -> TopoDS_Edge
    for arcs in face_arcs.values():
        for arc in arcs:
            if id(arc) in arc_to_edge:
                continue
            try:
                if arc["closed"]:
                    edge = BRepBuilderAPI_MakeEdge(arc["curve"]).Edge()
                else:
                    edge = BRepBuilderAPI_MakeEdge(
                        arc["curve"],
                        occ_verts[arc["v_start"]],
                        occ_verts[arc["v_end"]],
                    ).Edge()
                arc_to_edge[id(arc)] = edge
            except Exception as exc:
                print(f"[brep] MakeEdge failed for arc on {arc.get('edge_key')}: {exc}")

    # 3. Build wires and faces, then sew.
    sewing = BRepBuilderAPI_Sewing(tolerance)

    for face_idx, arcs in face_arcs.items():
        if face_idx >= len(occ_surfaces) or occ_surfaces[face_idx] is None:
            continue
        surface = occ_surfaces[face_idx]

        closed_arcs = [a for a in arcs if     a["closed"]]
        open_arcs   = [a for a in arcs if not a["closed"]]

        occ_wires = []

        # Closed arcs: one edge, one wire, no reordering needed.
        for arc in closed_arcs:
            if id(arc) not in arc_to_edge:
                continue
            wm = BRepBuilderAPI_MakeWire(arc_to_edge[id(arc)])
            if wm.IsDone():
                occ_wires.append(wm.Wire())
            else:
                print(f"[brep] face {face_idx}: closed-arc wire failed")

        # Open arcs: group by connectivity, let ShapeFix reorder each group.
        for component in _connected_components(open_arcs):
            wm = BRepBuilderAPI_MakeWire()
            for arc in component:
                if id(arc) in arc_to_edge:
                    wm.Add(arc_to_edge[id(arc)])

            if not wm.IsDone():
                print(f"[brep] face {face_idx}: component wire maker failed")
                continue

            fix = ShapeFix_Wire()
            fix.Load(wm.Wire())
            fix.SetPrecision(tolerance)
            fix.FixReorder()
            fix.FixConnected()
            occ_wires.append(fix.Wire())

        if not occ_wires:
            print(f"[brep] face {face_idx}: no wires — skipping")
            continue

        try:
            face_maker = BRepBuilderAPI_MakeFace(surface, occ_wires[0])
            for inner in occ_wires[1:]:
                face_maker.Add(inner)
            if face_maker.IsDone():
                sewing.Add(face_maker.Face())
            else:
                print(f"[brep] face {face_idx}: MakeFace failed")
        except Exception as exc:
            print(f"[brep] face {face_idx}: MakeFace exception: {exc}")

    # 4. Sew all faces into a shell.
    print("Sewing faces ...")
    sewing.Perform()
    shape = sewing.SewedShape()

    # 5. Ensure consistent 3D curves, then heal.
    print("Fixing shape ...")
    breplib.BuildCurves3d(shape)
    fixer = ShapeFix_Shape(shape)
    fixer.SetPrecision(tolerance)
    fixer.Perform()

    return fixer.Shape()


def export_step(shape, path):
    """Export a TopoDS_Shape to a STEP file."""
    from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.IFSelect    import IFSelect_RetDone

    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    ok = writer.Write(path) == IFSelect_RetDone
    if ok:
        print(f"STEP written to {path}")
    else:
        print(f"STEP export failed")
    return ok


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def print_edge_arcs_summary(edge_arcs):
    """Print a concise summary of the arc splitting result."""
    total_arcs = sum(len(v) for v in edge_arcs.values())
    print(f"Edge arcs summary: {len(edge_arcs)} edges → {total_arcs} arcs total")
    for edge_key, arcs in sorted(edge_arcs.items()):
        for idx, arc in enumerate(arcs):
            vs = arc["v_start"]
            ve = arc["v_end"]
            cl = "closed" if arc["closed"] else "open"
            print(
                f"  edge {edge_key}  arc[{idx}]  [{arc['t_start']:.4f}, {arc['t_end']:.4f}]"
                f"  v_start={vs}  v_end={ve}  {cl}"
            )
