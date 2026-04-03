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

import io
import math
import contextlib
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree

try:
    from OCC.Core.Geom           import Geom_TrimmedCurve
    from OCC.Core.Geom2d         import Geom2d_TrimmedCurve
    from OCC.Core.GCE2d          import GCE2d_MakeSegment
    from OCC.Core.Geom2d         import Geom2d_BSplineCurve
    from OCC.Core.TColgp         import TColgp_Array1OfPnt2d
    from OCC.Core.TColStd        import TColStd_Array1OfReal, TColStd_Array1OfInteger
    from OCC.Core.GeomAPI        import GeomAPI_ProjectPointOnCurve, GeomAPI_ProjectPointOnSurf
    from OCC.Core.GeomLProp      import GeomLProp_SLProps
    from OCC.Core.gp             import gp_Pnt, gp_Pnt2d, gp_GTrsf, gp_Trsf, gp_Mat, gp_Vec, gp_Quaternion
    from OCC.Core.BRep           import BRep_Builder, BRep_Tool
    from OCC.Core.TopExp         import TopExp_Explorer
    from OCC.Core.TopAbs         import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX, TopAbs_WIRE, TopAbs_SHELL, TopAbs_SOLID
    from OCC.Core.TopoDS         import (topods, TopoDS_Edge, TopoDS_Wire,
                                         TopoDS_Face, TopoDS_Shell, TopoDS_Compound)
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakeVertex,
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeWire,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakeSolid,
        BRepBuilderAPI_Sewing,
        BRepBuilderAPI_GTransform,
        BRepBuilderAPI_Transform,
    )
    from OCC.Core.ShapeFix       import ShapeFix_Wire, ShapeFix_Shape, ShapeFix_Shell, ShapeFix_Face
    from OCC.Core.BRepLib        import breplib
    from OCC.Core.STEPControl    import (STEPControl_Writer, STEPControl_AsIs,
                                          STEPControl_Reader)
    from OCC.Core.IFSelect       import IFSelect_RetDone
    from OCC.Core.BOPAlgo        import BOPAlgo_MakerVolume, BOPAlgo_Builder, BOPAlgo_GlueFull, BOPAlgo_BuilderFace, BOPAlgo_CellsBuilder
    from OCC.Core.TopTools       import TopTools_ListOfShape
    from OCC.Core.BRepCheck      import BRepCheck_Analyzer, BRepCheck_NoError
    from OCC.Core.BRepGProp      import brepgprop
    from OCC.Core.GProp          import GProp_GProps
    from OCC.Core.Message        import Message_ProgressRange
    OCC_AVAILABLE = True
except ImportError:
    OCC_AVAILABLE = False

try:
    from point2cad.surface_types import SURFACE_INR
except ImportError:
    SURFACE_INR = None

# Geometric closure tolerance — endpoint distance below this means the curve
# is a closed loop (e.g. cylinder-plane circle gives ~1e-17, lines give ~1e-1).
CLOSURE_TOL = 1e-4


_BREP_CHECK_STATUS_NAMES = {
    0: "NoError",
    1: "InvalidPointOnCurve",
    2: "InvalidPointOnCurveOnSurface",
    3: "InvalidPointOnSurface",
    4: "No3DCurve",
    5: "Multiple3DCurve",
    6: "Invalid3DCurve",
    7: "NoCurveOnSurface",
    8: "InvalidCurveOnSurface",
    9: "InvalidCurveOnClosedSurface",
    10: "InvalidSameRangeFlag",
    11: "InvalidSameParameterFlag",
    12: "InvalidDegeneratedFlag",
    13: "FreeEdge",
    14: "InvalidMultiConnexity",
    15: "InvalidRange",
    16: "EmptyWire",
    17: "RedundantEdge",
    18: "SelfIntersectingWire",
    19: "NoSurface",
    20: "InvalidWire",
    21: "RedundantWire",
    22: "IntersectingWires",
    23: "InvalidImbricationOfWires",
    24: "EmptyShell",
    25: "RedundantFace",
    26: "InvalidToleranceValue",
    27: "UnorientableShape",
    28: "NotClosed",
    29: "NotConnected",
    30: "SubshapeNotInShape",
    31: "BadOrientation",
    32: "BadOrientationOfSubshape",
    33: "InvalidPolygonOnTriangulation",
    34: "InvalidToleranceValue",
    35: "EnclosedRegion",
    36: "CheckFail",
}


def _status_name(code):
    """Human-readable name for a BRepCheck status code."""
    if isinstance(code, int):
        return _BREP_CHECK_STATUS_NAMES.get(code, f"Unknown({code})")
    # pythonocc enum object — try .value or str
    try:
        val = int(code)
        return _BREP_CHECK_STATUS_NAMES.get(val, f"Unknown({val})")
    except (TypeError, ValueError):
        return str(code)


def _extract_status_errors(status_list):
    """Extract error names from a BRepCheck status list (multiple strategies)."""
    errors = []
    # Strategy 1: C++ iterator
    try:
        it = status_list.begin()
        end = status_list.end()
        while it != end:
            s = it.Value()
            if s != BRepCheck_NoError:
                errors.append(_status_name(s))
            it.Next()
        return errors
    except Exception:
        pass
    # Strategy 2: Python iteration
    try:
        for s in status_list:
            if s != BRepCheck_NoError:
                errors.append(_status_name(s))
        return errors
    except Exception:
        pass
    # Strategy 3: indexing
    try:
        for k in range(status_list.Length()):
            s = status_list.Value(k + 1)
            if s != BRepCheck_NoError:
                errors.append(_status_name(s))
        return errors
    except Exception:
        pass
    return [f"(could not iterate: {type(status_list).__name__})"]


def _print_brep_check_details(analyzer, shape):
    """Print per-sub-shape BRepCheck errors when the analyzer reports invalid."""
    _shape_type_names = {
        TopAbs_VERTEX: "Vertex", TopAbs_EDGE: "Edge", TopAbs_WIRE: "Wire",
        TopAbs_FACE: "Face", TopAbs_SHELL: "Shell", TopAbs_SOLID: "Solid",
    }
    any_errors = False
    for stype in (TopAbs_VERTEX, TopAbs_EDGE, TopAbs_WIRE,
                  TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID):
        exp = TopExp_Explorer(shape, stype)
        idx = 0
        while exp.More():
            sub = exp.Current()
            name = _shape_type_names.get(stype, str(stype))
            try:
                result = analyzer.Result(sub)
                if result is None:
                    idx += 1
                    exp.Next()
                    continue
                # Standalone status
                errors = _extract_status_errors(result.Status())
                # Context-dependent status (checks sub-shape within parent)
                ctx_errors = []
                try:
                    ctx_list = result.StatusOnShape(shape)
                    ctx_errors = _extract_status_errors(ctx_list)
                except Exception:
                    pass
                # Also check StatusOnShape for each parent face/shell
                for ptype in (TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID):
                    if ptype == stype:
                        continue
                    pexp = TopExp_Explorer(shape, ptype)
                    pidx = 0
                    while pexp.More():
                        try:
                            pctx = result.StatusOnShape(pexp.Current())
                            perrs = _extract_status_errors(pctx)
                            if perrs:
                                pname = _shape_type_names.get(ptype, str(ptype))
                                ctx_errors.extend(
                                    f"{e} (in {pname} {pidx})" for e in perrs)
                        except Exception:
                            pass
                        pidx += 1
                        pexp.Next()

                all_errors = errors + ctx_errors
                if all_errors:
                    any_errors = True
                    print(f"  [BRepCheck] {name} {idx}: {all_errors}")
            except Exception as exc:
                any_errors = True
                print(f"  [BRepCheck] {name} {idx}: error reading status: {exc}")
            idx += 1
            exp.Next()
    if not any_errors:
        print("  [BRepCheck] analyzer reported invalid but no specific errors found")


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


def _arc_key(arc):
    """Hashable key for an arc dict, unique per distinct curve.

    Uses id(curve) to distinguish multiple closed arcs on the same edge
    that would otherwise collide at (ei, ej, None, None).  The same arc
    object referenced from two face lists (face i and face j) keeps the
    same key, so the TopoDS_Edge is correctly shared between faces.
    """
    ei, ej = arc["edge_key"]
    return (ei, ej, arc["v_start"], arc["v_end"], id(arc["curve"]))


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
    t_star = proj.LowerDistanceParameter()
    return t_star, proj.LowerDistance()


def _pnt_to_np(pnt):
    return np.array([pnt.X(), pnt.Y(), pnt.Z()], dtype=np.float64)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def build_edge_arcs(intersections, vertices, vertex_edges, threshold=1e-4):
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
                if t_star is None:
                    continue
                if dist > threshold:
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
                        "edge_key": edge_key,
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
                            "edge_key": edge_key,
                        })

                    # Wrap-around arc: [t_k, t_1 + span] on the periodic basis.
                    t_last,  v_last  = incident_params[-1]
                    t_first, v_first = incident_params[0]
                    t_wrap_end = t_first + span

                    # For the wrap-around arc we need a curve that accepts
                    # parameters beyond [t_min, t_max].  Three cases:
                    #   1. Geom_TrimmedCurve → BasisCurve() gives the periodic basis
                    #   2. Raw closed curve (e.g. Geom_Circle) → use it directly
                    #   3. Non-periodic closed curve (e.g. BSpline) → split into
                    #      two sub-arcs [t_last, t_max] + [t_min, t_first] with a
                    #      seam vertex at the curve's start/end point.
                    basis = _basis_curve(curve)
                    if basis is None and curve_is_closed(curve):
                        basis = curve
                    wrap_ok = False
                    if basis is not None:
                        try:
                            arcs_for_edge.append({
                                "curve":   _make_arc(basis, t_last, t_wrap_end),
                                "v_start": v_last,
                                "v_end":   v_first,
                                "t_start": t_last,
                                "t_end":   t_wrap_end,
                                "closed":  False,
                                "edge_key": edge_key,
                            })
                            wrap_ok = True
                        except Exception:
                            pass  # OCC rejected wrap parameters (e.g. BSpline)
                    if not wrap_ok and curve_is_closed(curve):
                        # Seam-split fallback: emit two sub-arcs joined at a
                        # seam vertex placed at the curve start/end point.
                        seam_pos = _pnt_to_np(curve.Value(t_min))
                        seam_idx = len(verts)
                        verts.append(seam_pos)
                        vedge_sets.append({edge_key})
                        try:
                            arcs_for_edge.append({
                                "curve":   _make_arc(curve, t_last, t_max),
                                "v_start": v_last,
                                "v_end":   seam_idx,
                                "t_start": t_last,
                                "t_end":   t_max,
                                "closed":  False,
                                "edge_key": edge_key,
                            })
                            arcs_for_edge.append({
                                "curve":   _make_arc(curve, t_min, t_first),
                                "v_start": seam_idx,
                                "v_end":   v_first,
                                "t_start": t_min,
                                "t_end":   t_first,
                                "closed":  False,
                                "edge_key": edge_key,
                            })
                            wrap_ok = True
                        except Exception:
                            pass

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
                        "edge_key": edge_key,
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
                            "edge_key": edge_key,
                        })

        for arc_idx, arc in enumerate(arcs_for_edge):
            arc["arc_idx"] = arc_idx
        edge_arcs[edge_key] = arcs_for_edge

    out_vertices = np.array(verts, dtype=np.float64) if verts else np.zeros((0, 3))
    return edge_arcs, out_vertices, vedge_sets


# ---------------------------------------------------------------------------
# Step 1c — Scoring helpers (used by greedy oracle filter)
# ---------------------------------------------------------------------------

def _score_vertex(vpos, involved_clusters, cluster_trees, cluster_nn_percentiles):
    """
    Fitness score for a vertex.

    Computes d_k / p_k for each involved cluster, then takes the max of the
    3 lowest ratios.  This makes scoring robust to extra cluster associations
    from deduplication merging: a vertex genuinely at triangle (i,j,k) that
    picks up a 4th cluster m from a nearby merged candidate won't be penalised
    by cluster m's high distance.
    """
    ratios = []
    for k in involved_clusters:
        d, _ = cluster_trees[k].query(vpos, k=1)
        p = cluster_nn_percentiles[k]
        if p > 0:
            ratios.append(d / p)
        else:
            ratios.append(float('inf'))
    if not ratios:
        return float('inf')
    ratios.sort()
    # Use the 3 best (lowest) ratios — the genuine triangle's clusters.
    # For vertices with ≤3 clusters, this is just max over all.
    return ratios[min(2, len(ratios) - 1)]


def _score_arc(arc, cluster_i, cluster_j, cluster_trees, cluster_nn_percentiles,
               n_samples=10, sample_fraction=0.5):
    """
    Fitness score for an arc: mean over interior samples of max(d_i/p_i, d_j/p_j).
    Lower = closer to clusters = more likely real.
    """
    t0, t1 = arc["t_start"], arc["t_end"]
    t_mid = (t0 + t1) / 2
    half_span = (t1 - t0) * sample_fraction / 2
    t_start = t_mid - half_span
    t_end = t_mid + half_span

    tree_i = cluster_trees[cluster_i]
    tree_j = cluster_trees[cluster_j]
    p_i = cluster_nn_percentiles[cluster_i]
    p_j = cluster_nn_percentiles[cluster_j]

    ratios = []
    for k in range(n_samples):
        t = t_start + (t_end - t_start) * k / max(n_samples - 1, 1)
        try:
            p = arc["curve"].Value(t)
            pt = np.array([p.X(), p.Y(), p.Z()])
            d_i, _ = tree_i.query(pt, k=1)
            d_j, _ = tree_j.query(pt, k=1)
            r_i = d_i / p_i if p_i > 0 else float('inf')
            r_j = d_j / p_j if p_j > 0 else float('inf')
            ratios.append(max(r_i, r_j))
        except Exception:
            pass
    return np.mean(ratios) if ratios else float('inf')



def _vertex_degree(v_idx, work_arcs):
    """Count how many open arcs in work_arcs touch vertex v_idx."""
    deg = 0
    for arcs in work_arcs.values():
        for arc in arcs:
            if arc.get("closed"):
                continue
            if arc["v_start"] == v_idx or arc["v_end"] == v_idx:
                deg += 1
    return deg


def _non_eulerian_faces_direct(work_arcs):
    """Check Euler condition directly on work_arcs (no removed sets)."""
    face_degree = defaultdict(lambda: defaultdict(int))
    for (i, j), arcs in work_arcs.items():
        for arc in arcs:
            if arc.get("closed"):
                continue
            vs, ve = arc["v_start"], arc["v_end"]
            face_degree[i][vs] += 1
            face_degree[i][ve] += 1
            face_degree[j][vs] += 1
            face_degree[j][ve] += 1
    bad = set()
    for face_idx, vdeg in face_degree.items():
        for v, deg in vdeg.items():
            if deg % 2 != 0:
                bad.add(face_idx)
                break
    return bad




# ---------------------------------------------------------------------------
# Step 1d — Greedy oracle-guided filter
# ---------------------------------------------------------------------------

def _apply_removals(edge_arcs, vertices, vertex_edges, removed_vertices,
                    removed_arc_keys):
    """
    Apply a set of vertex and arc removals to edge_arcs/vertices/vertex_edges.

    removed_arc_keys : set of (edge_key, arc_idx) tuples

    Returns (new_edge_arcs, new_vertices, new_vertex_edges).
    """
    # Build work_arcs with arc removals applied
    work_arcs = {}
    for edge_key, arcs in edge_arcs.items():
        kept = []
        for arc_idx, arc in enumerate(arcs):
            if (edge_key, arc_idx) in removed_arc_keys:
                continue
            kept.append(dict(arc))
        work_arcs[edge_key] = kept

    # Remove arcs touching removed vertices
    for edge_key in list(work_arcs.keys()):
        work_arcs[edge_key] = [
            arc for arc in work_arcs[edge_key]
            if arc.get("closed") or
            (arc["v_start"] not in removed_vertices and
             arc["v_end"] not in removed_vertices)
        ]

    # Sweep for 0-degree vertices
    all_removed = set(removed_vertices)
    for v_idx in range(len(vertices)):
        if v_idx in all_removed:
            continue
        if _vertex_degree(v_idx, work_arcs) == 0:
            all_removed.add(v_idx)

    # Compact
    surviving_v = sorted(set(range(len(vertices))) - all_removed)
    v_remap = {old: new for new, old in enumerate(surviving_v)}

    new_vertices = vertices[surviving_v]
    new_vertex_edges = [vertex_edges[i] for i in surviving_v]

    new_edge_arcs = {}
    for edge_key, arcs in work_arcs.items():
        kept = []
        for arc in arcs:
            if arc.get("closed"):
                kept.append(arc)
                continue
            vs, ve = arc["v_start"], arc["v_end"]
            if vs in v_remap and ve in v_remap:
                arc = dict(arc)
                arc["v_start"] = v_remap[vs]
                arc["v_end"] = v_remap[ve]
                kept.append(arc)
        new_edge_arcs[edge_key] = kept

    return new_edge_arcs, new_vertices, new_vertex_edges


def greedy_oracle_filter(edge_arcs, vertices, vertex_edges,
                         clusters, cluster_trees, cluster_nn_percentiles,
                         occ_surfaces, surface_ids=None,
                         tolerance=1e-3,
                         cluster_bboxes=None, wire_method="manual"):
    """
    Greedy worst-first removal guided by OCC BRepCheck_Analyzer.

    Algorithm:
      1. Score all arcs by cluster proximity ratio.
      2. Try building the full BRep — if BRepCheck_Analyzer returns True, done.
      3. Remove arcs worst-first.  After each removal, rebuild and check.
         Stop when valid or no arc candidates remain.
      Vertex filtering is handled upstream (score_cap in the caller).

    The Euler condition is NOT enforced inside the loop — if wire assembly
    fails on a non-Eulerian graph, that counts as "invalid" and the loop
    continues removing objects.

    Parameters
    ----------
    edge_arcs, vertices, vertex_edges :
        Same as build_edge_arcs.
    clusters, cluster_trees, cluster_nn_percentiles :
        Per-cluster data for fitness scoring.
    occ_surfaces : list[Geom_Surface]
        OCC surfaces for BRep construction.
    surface_ids : list[int] or None
    tolerance : float
    cluster_bboxes : unused, kept for call-site compatibility

    Returns
    -------
    (edge_arcs, vertices, vertex_edges, shape, brep_info)
        Filtered topology + the final built shape and its info dict.
    """
    n_v = len(vertices)
    n_a = sum(len(a) for a in edge_arcs.values())
    print(f"[oracle filter] input: {n_v} vertices, {n_a} arcs")

    # ------------------------------------------------------------------
    # Phase 1: Score candidates
    # ------------------------------------------------------------------
    # Vertex scores (logged only — vertex filtering is upstream via score_cap)
    for v_idx, (vpos, edges) in enumerate(zip(vertices, vertex_edges)):
        involved = set()
        for edge in edges:
            involved.update(edge)
        score = _score_vertex(vpos, involved, cluster_trees,
                              cluster_nn_percentiles)
        print(f"  v{v_idx:>3d}  score={score:>10.4f}  "
              f"edges={sorted(edges)}")

    arc_scores = []  # (score, edge_key, arc_idx)
    for edge_key, arcs in edge_arcs.items():
        i, j = edge_key
        for arc_idx, arc in enumerate(arcs):
            score = _score_arc(arc, i, j, cluster_trees,
                               cluster_nn_percentiles)
            arc_scores.append((score, edge_key, arc_idx))

    # Build arc candidate list, sorted worst-first.
    arc_candidates = []
    for score, edge_key, arc_idx in arc_scores:
        arc = edge_arcs[edge_key][arc_idx]
        arc_candidates.append((score, (edge_key, arc_idx), arc_idx))
    arc_candidates.sort(key=lambda c: c[0], reverse=True)

    if arc_candidates:
        print(f"[oracle filter] {len(arc_candidates)} arc candidates "
              f"(score range: {arc_candidates[-1][0]:.4f} — "
              f"{arc_candidates[0][0]:.4f})")
    else:
        print("[oracle filter] no removable candidates")

    # ------------------------------------------------------------------
    # Phase 2: Remove isolated vertices unconditionally
    # ------------------------------------------------------------------
    removed_vertices = set()
    removed_arc_keys = set()  # (edge_key, arc_idx)

    for v_idx in range(len(vertices)):
        if _vertex_degree(v_idx, edge_arcs) == 0:
            removed_vertices.add(v_idx)
    if removed_vertices:
        print(f"[oracle filter] removing {len(removed_vertices)} isolated vertices")

    # ------------------------------------------------------------------
    # Phase 3: Greedy removal loop (arcs first, then vertices)
    # ------------------------------------------------------------------
    def _try_build(removed_v, removed_a):
        """Build BRep with given removals, return (valid, shape, info, ea, v, ve).

        Validity requires BOTH:
          - All face wire graphs are Eulerian (necessary for complete wires)
            (skipped when wire_method="builderface" — BuilderFace handles
            non-Eulerian graphs gracefully)
          - BRepCheck_Analyzer returns True (geometric correctness)
        """
        ea, verts, ve = _apply_removals(edge_arcs, vertices, vertex_edges,
                                        removed_v, removed_a)

        if wire_method == "builderface":
            fa = face_arc_incidence(ea)
            shape, info = build_brep_shape_builderface(
                fa, occ_surfaces, verts, surface_ids=surface_ids,
                tolerance=tolerance, clusters=clusters,
            )
        elif wire_method == "cells":
            fa = face_arc_incidence(ea)
            shape, info = build_brep_shape_cells(
                fa, occ_surfaces, verts, surface_ids=surface_ids,
                tolerance=tolerance, clusters=clusters,
            )
        elif wire_method == "direct":
            bad_faces = _non_eulerian_faces_direct(ea)
            if bad_faces:
                return False, None, {"valid": False, "n_faces": 0,
                                     "n_input_faces": 0,
                                     "non_eulerian": sorted(bad_faces)}, ea, verts, ve
            fa = face_arc_incidence(ea)
            fw = assemble_wires(fa, occ_surfaces, verts, surface_ids=surface_ids)
            shape, info = build_brep_shape_direct(
                fa, occ_surfaces, verts, surface_ids=surface_ids,
                face_wires=fw, tolerance=tolerance, clusters=clusters,
                cluster_trees=cluster_trees,
            )
        else:
            # Manual wire assembly (original method)
            # Fast Euler pre-check — if any face is non-Eulerian, wire assembly
            # will skip it, producing an incomplete (thus invalid) model.
            bad_faces = _non_eulerian_faces_direct(ea)
            if bad_faces:
                return False, None, {"valid": False, "n_faces": 0,
                                     "n_input_faces": 0,
                                     "non_eulerian": sorted(bad_faces)}, ea, verts, ve
            fa = face_arc_incidence(ea)
            fw = assemble_wires(fa, occ_surfaces, verts, surface_ids=surface_ids)
            shape, info = build_brep_shape(
                fa, occ_surfaces, verts, surface_ids=surface_ids,
                face_wires=fw, tolerance=tolerance,
            )

        valid = (info.get("valid", False) and
                 info.get("n_faces", 0) > 0)
        return valid, shape, info, ea, verts, ve

    # Try with everything first (minus isolated vertices)
    print("[oracle filter] trying full model ...")
    _captured = io.StringIO()
    with contextlib.redirect_stdout(_captured):
        valid, shape, info, final_ea, final_v, final_ve = _try_build(
            removed_vertices, removed_arc_keys)
    # Always print build details (previously suppressed on success).
    for _line in _captured.getvalue().splitlines():
        print(f"[oracle filter] {_line}")
    # Log arcs silently dropped by _apply_removals (open arcs with
    # v_start=None/v_end=None that can't survive vertex compaction).
    n_in = sum(len(a) for a in edge_arcs.values())
    n_out = sum(len(a) for a in final_ea.values())
    if n_out < n_in:
        dropped = n_in - n_out - len(removed_arc_keys)
        if dropped > 0:
            print(f"[oracle filter] {dropped} dangling arc(s) dropped "
                  f"(open, no vertex attribution)")
    if valid:
        print(f"[oracle filter] full model valid — "
              f"{info['n_faces']} faces, no removals needed")
        return final_ea, final_v, final_ve, shape, info

    non_euler = info.get("non_eulerian")
    if non_euler:
        print(f"[oracle filter] full model invalid — non-Eulerian faces: {non_euler}")
    else:
        print(f"[oracle filter] full model invalid — BRepCheck failed "
              f"({info.get('n_faces', 0)} faces)")
    print(f"[oracle filter] starting greedy removal")

    best_shape, best_info = shape, info
    best_ea, best_v, best_ve = final_ea, final_v, final_ve

    # Arc-only greedy removal, worst-first.
    for score, ident, arc_idx in arc_candidates:
        if ident in removed_arc_keys:
            continue
        removed_arc_keys.add(ident)
        edge_key, _ = ident
        arc = edge_arcs[edge_key][arc_idx]
        print(f"[oracle filter] removing arc {edge_key}[{arc_idx}] "
              f"t=[{arc['t_start']:.4f},{arc['t_end']:.4f}] score={score:.4f}")

        _captured = io.StringIO()
        with contextlib.redirect_stdout(_captured):
            valid, shape, info, ea, verts, ve = _try_build(
                removed_vertices, removed_arc_keys)
        for _line in _captured.getvalue().splitlines():
            print(f"[oracle filter] {_line}")

        n_faces = info.get("n_faces", 0)
        non_euler = info.get("non_eulerian")
        if non_euler:
            print(f"[oracle filter]   non-Eulerian faces: {non_euler}")
        if valid:
            print(f"[oracle filter] valid! {n_faces} faces after "
                  f"{len(removed_arc_keys)} arc removals")
            return ea, verts, ve, shape, info
        if n_faces > best_info.get("n_faces", 0):
            best_shape, best_info = shape, info
            best_ea, best_v, best_ve = ea, verts, ve

    # Exhausted all arc candidates without reaching validity
    print(f"[oracle filter] WARNING: could not achieve valid BRep "
          f"after removing all candidates. "
          f"Best: {best_info.get('n_faces', 0)} faces")
    return best_ea, best_v, best_ve, best_shape, best_info


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
            face_arcs.setdefault(fi, []).append(arc)
            face_arcs.setdefault(fj, []).append(arc)

    return face_arcs


def print_face_arcs_summary(face_arcs):
    """Print a concise summary of the face–arc incidence."""
    print(f"[face arcs] {len(face_arcs)} faces")
    for face_idx in sorted(face_arcs):
        arcs   = face_arcs[face_idx]
        closed = sum(1 for a in arcs if a["closed"])
        open_  = len(arcs) - closed
        print(f"  face {face_idx:2d}  arcs={len(arcs)}  (open={open_}  closed={closed})")


# ---------------------------------------------------------------------------
# Step 3 — Wire assembly
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Angular ordering helpers (used by assemble_wires)
# ---------------------------------------------------------------------------

def _arc_outgoing_tangent(arc, v_cur):
    """
    3D tangent of arc at vertex v_cur, pointing AWAY from v_cur along the arc.

    For an arc spanning [t_start, t_end]:
      - If v_cur == v_start: the arc leaves in the +t direction → +d/dt at t_start.
      - If v_cur == v_end:   the arc leaves in the -t direction → -d/dt at t_end.
    """
    # https://dev.opencascade.org/doc/refman/html/class_geom___curve.html
    if arc["v_start"] == v_cur:
        d = arc["curve"].DN(arc["t_start"], 1)
        return np.array([d.X(), d.Y(), d.Z()])
    else:
        d = arc["curve"].DN(arc["t_end"], 1)
        return -np.array([d.X(), d.Y(), d.Z()])


def _surface_normal_at_point(surface, pt):
    """
    Unit surface normal at 3D point pt, computed by projecting pt onto
    `surface` and evaluating surface properties there.
    Returns None on failure.
    """
    pnt = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
    try:
        proj = GeomAPI_ProjectPointOnSurf(pnt, surface)
        if proj.NbPoints() == 0:
            return None
        u, v = proj.LowerDistanceParameters()
        # https://dev.opencascade.org/doc/refman/html/class_geom_l_prop___s_l_props.html
        props = GeomLProp_SLProps(surface, u, v, 1, 1e-6)
        if not props.IsNormalDefined():
            return None
        n = props.Normal()
        return np.array([n.X(), n.Y(), n.Z()])
    except Exception:
        return None


def _select_next_arc_angular(v_cur, prev_arc, candidates, face_idx, occ_surfaces, vertices):
    """
    At vertex v_cur (arrived via prev_arc), choose the next arc from
    `candidates` using angular ordering in the tangent plane of face_idx.

    Rule: project all candidate outgoing tangents onto the tangent plane of
    the surface at v_cur, measure their angle relative to the backward
    direction (= outgoing tangent of prev_arc at v_cur), and return the
    candidate with the smallest strictly-positive CCW angle.

    Falls back to candidates[0] if the surface normal cannot be computed
    or all tangent projections are degenerate.

    Parameters
    ----------
    v_cur      : int             — current vertex index
    prev_arc   : arc_dict        — the arc we just traversed to reach v_cur
    candidates : list of (arc_dict, exit_v, forward)
    face_idx   : int
    occ_surfaces : list[Geom_Surface]
    vertices   : np.ndarray (M, 3)
    """
    pt = vertices[v_cur]

    if face_idx >= len(occ_surfaces) or occ_surfaces[face_idx] is None:
        print(
            f"[topology] face {face_idx}: vertex {v_cur}: angular ordering fallback "
            f"— surface missing or None, using candidates[0]"
        )
        return candidates[0]

    N = _surface_normal_at_point(occ_surfaces[face_idx], pt)
    if N is None:
        print(
            f"[topology] face {face_idx}: vertex {v_cur}: angular ordering fallback "
            f"— surface normal projection failed (point may be off surface), "
            f"using candidates[0]"
        )
        return candidates[0]
    N_norm = np.linalg.norm(N)
    if N_norm < 1e-10:
        print(
            f"[topology] face {face_idx}: vertex {v_cur}: angular ordering fallback "
            f"— surface normal degenerate (norm={N_norm:.2e}), using candidates[0]"
        )
        return candidates[0]
    N = N / N_norm

    # Backward direction: the direction we came FROM at v_cur.
    # _arc_outgoing_tangent(prev_arc, v_cur) points away from v_cur along
    # prev_arc, which is exactly toward the previous vertex — the backward dir.
    t_back = _arc_outgoing_tangent(prev_arc, v_cur)
    t_back_proj = t_back - np.dot(t_back, N) * N
    t_back_norm = np.linalg.norm(t_back_proj)
    if t_back_norm < 1e-10:
        print(
            f"[topology] face {face_idx}: vertex {v_cur}: angular ordering fallback "
            f"— backward arc tangent is parallel to surface normal "
            f"(proj norm={t_back_norm:.2e}), using candidates[0]"
        )
        return candidates[0]
    t_back_proj /= t_back_norm

    # Tangent plane basis: e1 = backward direction, e2 = N × e1 (CCW from e1).
    e1 = t_back_proj
    e2 = np.cross(N, e1)

    def _ccw_angle(arc, _exit_v, _fwd):
        t_out = _arc_outgoing_tangent(arc, v_cur)
        t_proj = t_out - np.dot(t_out, N) * N
        norm_t = np.linalg.norm(t_proj)
        if norm_t < 1e-10:
            print(
                f"[topology] face {face_idx}: vertex {v_cur}: candidate arc "
                f"{arc.get('edge_key')} tangent projects to zero in tangent plane "
                f"(norm={norm_t:.2e}) — pushed to last"
            )
            return 2 * math.pi   # degenerate: push to end
        t_proj /= norm_t
        a = math.atan2(np.dot(t_proj, e2), np.dot(t_proj, e1))
        a = a % (2 * math.pi)
        if a < 1e-8:             # avoid selecting the backward direction itself
            a += 2 * math.pi
        return a

    # Edge continuity: prefer candidates from the same edge as prev_arc.
    # This keeps the wire on a single intersection curve before switching
    # to a different curve at a shared vertex.
    # prev_edge = prev_arc.get("edge_key")
    # if prev_edge is not None:
    #     same_edge = [c for c in candidates if c[0].get("edge_key") == prev_edge]
    #     if same_edge:
    #         return min(same_edge, key=lambda c: _ccw_angle(*c))

    return min(candidates, key=lambda c: _ccw_angle(*c))


# ---------------------------------------------------------------------------
# Step 3 — Wire assembly
# ---------------------------------------------------------------------------

def assemble_wires(face_arcs, occ_surfaces=None, vertices=None, surface_ids=None):
    """
    Step 3 of B-Rep topology: for each face, partition its arcs into closed
    wires (boundary loops).

    All faces — including BSpline (INR) faces — are processed identically.

    Closed arcs (closed=True) each form a trivial one-arc wire immediately.

    Open arcs are assembled via cycle extraction on the boundary graph G_i.
    At vertices where degree == 2 the unique unvisited arc is taken directly.
    At vertices where degree > 2 (e.g. a planar face with a circular hole
    whose boundary shares vertices with the outer rectangular boundary),
    the next arc is chosen by angular ordering in the tangent plane of the
    face: the candidate whose outgoing tangent is the first CCW from the
    backward direction is selected.  Angular ordering requires `occ_surfaces`
    and `vertices` to be provided; otherwise a greedy fallback is used.

    Parameters
    ----------
    face_arcs    : dict i -> list[arc_dict]
    occ_surfaces : list[Geom_Surface] or None
        Required for angular ordering at high-degree vertices.
    vertices     : np.ndarray (M, 3) or None
        Vertex positions; required together with occ_surfaces.
    surface_ids  : list[int] or None
        Surface type ids from surface_fitter (reserved for future use).

    Returns
    -------
    face_wires : dict i -> list[list[(arc_dict, bool)]]
        Outer list: one entry per wire on face i.
        Inner list: ordered (arc, forward) pairs forming a closed loop.
    """
    use_angular = (occ_surfaces is not None) and (vertices is not None)
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
            adj = {}   # v -> list of (arc, exit_v, forward)
            for arc in open_arcs:
                vs, ve = arc["v_start"], arc["v_end"]
                adj.setdefault(vs, []).append((arc, ve,  True))
                adj.setdefault(ve, []).append((arc, vs, False))

            for v, neighbours in adj.items():
                deg = len(neighbours)
                if deg % 2 != 0:
                    print(
                        f"[topology] face {face_idx}: vertex {v} has odd degree {deg}"
                    )
                
            arc_index    = {_arc_key(a): idx for idx, a in enumerate(open_arcs)}
            visited_arcs = set()

            for start_idx, start_arc in enumerate(open_arcs):
                if start_idx in visited_arcs:
                    continue

                wire     = []
                v_target = start_arc["v_start"]
                arc      = start_arc
                forward  = True
                v_cur    = start_arc["v_end"]
                prev_arc = start_arc

                wire.append((arc, forward))
                visited_arcs.add(start_idx)

                broken = False
                while v_cur != v_target:
                    candidates = [
                        (a, exit_v, fwd)
                        for (a, exit_v, fwd) in adj.get(v_cur, [])
                        if arc_index[_arc_key(a)] not in visited_arcs
                    ]
                    if not candidates:
                        print(
                            f"[topology] face {face_idx}: open chain at vertex "
                            f"{v_cur} — wire left incomplete"
                        )
                        broken = True
                        break

                    if len(candidates) > 1:
                        # Edge continuity: prefer arcs from the same edge
                        prev_edge = prev_arc.get("edge_key")
                        same_edge = [c for c in candidates
                                     if c[0].get("edge_key") == prev_edge]
                        if same_edge:
                            arc, v_cur, forward = same_edge[0]
                        else:
                            arc, v_cur, forward = candidates[0]
                    else:
                        arc, v_cur, forward = candidates[0]

                    wire.append((arc, forward))
                    visited_arcs.add(arc_index[_arc_key(arc)])
                    prev_arc = arc

                if not broken:
                    wires.append(wire)

        face_wires[face_idx] = wires

    return face_wires


def print_face_wires_summary(face_wires):
    """Print a concise summary of the wire assembly result."""
    print(f"[wire assembly] {len(face_wires)} faces")
    for face_idx in sorted(face_wires):
        wires = face_wires[face_idx]
        print(f"  face {face_idx:2d}  wires={len(wires)}")
        for w_idx, wire in enumerate(wires):
            arc_descs = []
            for arc, fwd in wire:
                vs = arc["v_start"]
                ve = arc["v_end"]
                ek = arc.get("edge_key", None)
                ai = arc.get("arc_idx", "?")
                cl = "closed" if arc["closed"] else ("fwd" if fwd else "rev")
                edge_str = f"e({ek[0]}, {ek[1]})[{ai}]" if ek else "e?"
                arc_descs.append(f"({vs}→{ve}, {cl}, {edge_str})")
            print(f"    wire[{w_idx}]  arcs={len(wire)}  " + "  ".join(arc_descs))


# ---------------------------------------------------------------------------
# Step 5 — BRep assembly and STEP export
# ---------------------------------------------------------------------------

def build_brep_shape(face_arcs, occ_surfaces, vertices, surface_ids=None,
                     face_wires=None, tolerance=1e-3,
                     inr_geom_close_tol=0.05, inr_arc_samples=30,
                     inr_uv_margin=0.02,
                     same_parameter=True,
                     orient_solid=True):
    """
    Build a TopoDS_Shape from face_arcs.

    Wire assembly
    -------------
    `face_wires` is the output of assemble_wires (angular ordering).  Each
    wire is built directly from the ordered (arc, forward) sequence — arc
    orientation is encoded by edge.Reversed() when forward=False.

    BSpline (INR) surfaces
    ----------------------
    INR faces are built with explicit UV parameter bounds:
    `BRepBuilderAPI_MakeFace(surface, u_min, u_max, v_min, v_max, tolerance)`.
    The bounds are estimated by projecting sampled points from incident arcs
    onto the BSpline surface.  Geometric closure in each direction is
    detected and SetUPeriodic / SetVPeriodic called accordingly.

    All faces are sewn with BRepBuilderAPI_Sewing and healed with
    ShapeFix_Shape.

    Parameters
    ----------
    face_arcs         : dict i -> list[arc_dict]
    occ_surfaces      : list[Geom_Surface], indexed by face index
    vertices          : np.ndarray (M, 3)
    surface_ids       : list[int] or None  — SURFACE_INR flags BSpline faces
    face_wires        : dict i -> list[list[(arc_dict, bool)]] or None
    tolerance         : float
        Single tolerance passed to BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeFace
        (for primitive faces), and ShapeFix_Wire / ShapeFix_Shape.  In principle
        these could differ; a single value is a pragmatic simplification.
    inr_geom_close_tol : float
        Maximum 3D distance between opposite BSpline boundary curves for a
        parametric direction to be declared geometrically closed.
        Default 0.05 (5% of unit-normalised scale).
    inr_arc_samples   : int
        Number of points sampled per arc for UV bound estimation.  Default 30.
    inr_uv_margin     : float
        Relative margin added to projected UV bounds on each side.
        Default 0.02 (2%).

    Returns
    -------
    (TopoDS_Shape, dict)
        dict with keys:
          "valid"      : bool  — BRepCheck_Analyzer result
          "n_faces"    : int   — number of faces in the output shape
          "n_input_faces" : int — number of face indices in face_arcs
    """
    # 1. One TopoDS_Vertex per position.
    occ_verts = []
    for pos in vertices:
        occ_verts.append(
            BRepBuilderAPI_MakeVertex(
                gp_Pnt(float(pos[0]), float(pos[1]), float(pos[2]))
            ).Vertex()
        )

    # 2. One TopoDS_Edge per unique arc (1-arg form: floating edge from
    #    the trimmed curve alone).  Topological gaps are closed later by
    #    ShapeFix_Wire.FixConnected, so shared TopoDS_Vertex instances are
    #    not required here.
    arc_to_edge = {}   # _arc_key(arc) -> TopoDS_Edge
    for arcs in face_arcs.values():
        for arc in arcs:
            key = _arc_key(arc)
            if key in arc_to_edge:
                continue
            try:
                arc_to_edge[key] = BRepBuilderAPI_MakeEdge(arc["curve"]).Edge()
            except Exception as exc:
                print(f"[brep] MakeEdge failed for arc on {arc.get('edge_key')}: {exc}")

    # 3. Build wires and faces, then sew.
    sewing = BRepBuilderAPI_Sewing(tolerance)

    for face_idx, arcs in face_arcs.items():
        if face_idx >= len(occ_surfaces) or occ_surfaces[face_idx] is None:
            continue
        surface = occ_surfaces[face_idx]

        occ_wires = []

        for wire_arcs in face_wires.get(face_idx, []):
            # Build the wire with BRep_Builder (no topology-connectivity
            # checks) so that floating edges are accepted regardless of
            # gap size, then heal sub-millimetre gaps with ShapeFix_Wire.
            wire = TopoDS_Wire()
            bb = BRep_Builder()
            bb.MakeWire(wire)
            n_added = 0
            for arc, forward in wire_arcs:
                key = _arc_key(arc)
                if key not in arc_to_edge:
                    print(f"[brep] face {face_idx}: arc {key} missing from arc_to_edge — skipping")
                    continue
                edge = arc_to_edge[key]
                bb.Add(wire, edge if forward else edge.Reversed())
                n_added += 1
            if n_added == 0:
                print(f"[brep] face {face_idx}: no edges added to wire — skipping")
                continue
            fix = ShapeFix_Wire()
            fix.Load(wire)
            fix.SetPrecision(tolerance)
            fix.FixConnected()
            healed = fix.Wire()
            if healed.IsNull():
                print(f"[brep] face {face_idx}: ShapeFix_Wire produced null wire — skipping")
                continue
            occ_wires.append(healed)

        is_inr = (surface_ids is not None and
                  face_idx < len(surface_ids) and
                  surface_ids[face_idx] == SURFACE_INR)

        # INR setup: geometric closure detection and periodisation.
        closed_u = closed_v = False
        nu1 = nu2 = nv1 = nv2 = None
        if is_inr:
            nu1, nu2, nv1, nv2 = surface.Bounds()
            v_mid = (nv1 + nv2) / 2.0
            u_mid = (nu1 + nu2) / 2.0
            try:
                pu1 = surface.Value(nu1, v_mid)
                pu2 = surface.Value(nu2, v_mid)
                d_u = math.sqrt((pu1.X()-pu2.X())**2 +
                                (pu1.Y()-pu2.Y())**2 +
                                (pu1.Z()-pu2.Z())**2)
                pv1 = surface.Value(u_mid, nv1)
                pv2 = surface.Value(u_mid, nv2)
                d_v = math.sqrt((pv1.X()-pv2.X())**2 +
                                (pv1.Y()-pv2.Y())**2 +
                                (pv1.Z()-pv2.Z())**2)
            except Exception:
                d_u, d_v = 1.0, 1.0
            closed_u = d_u < inr_geom_close_tol
            closed_v = d_v < inr_geom_close_tol
            print(f"[brep] face {face_idx}: BSpline seam check "
                  f"d_u={d_u:.4f} d_v={d_v:.4f} "
                  f"closed_u={closed_u} closed_v={closed_v}")
            try:
                if closed_u:
                    surface.SetUPeriodic()
                if closed_v:
                    surface.SetVPeriodic()
                nu1, nu2, nv1, nv2 = surface.Bounds()
            except Exception as exc:
                print(f"[brep] face {face_idx}: SetPeriodic failed: {exc}")
                closed_u = closed_v = False

        # BSpline (INR) face — UV-bounds method (DISABLED: using wire-based
        # construction for all faces, same as planes/cylinders/etc.)
        if False and is_inr:
            # Start from full natural domain; constrain OPEN direction(s) by
            # projecting arc sample points onto the surface.  Sampling and
            # projection are skipped entirely when both directions are closed.
            u_min, u_max = nu1, nu2
            v_min, v_max = nv1, nv2

            if not closed_u or not closed_v:
                sample_pts = []
                for arc in face_arcs.get(face_idx, []):
                    t0, t1 = arc["t_start"], arc["t_end"]
                    for k in range(inr_arc_samples):
                        t = t0 + (t1 - t0) * k / max(inr_arc_samples - 1, 1)
                        try:
                            p = arc["curve"].Value(t)
                            sample_pts.append([p.X(), p.Y(), p.Z()])
                        except Exception:
                            pass

                if sample_pts:
                    sample_arr = np.array(sample_pts, dtype=np.float32)
                    bounds = _cluster_uv_bounds(surface, sample_arr, rel_margin=inr_uv_margin)
                    if bounds is not None:
                        bu_min, bu_max, bv_min, bv_max = bounds
                        if not closed_u:
                            u_min = max(bu_min, nu1)
                            u_max = min(bu_max, nu2)
                        if not closed_v:
                            v_min = max(bv_min, nv1)
                            v_max = min(bv_max, nv2)

            face_added = False
            try:
                face_maker = BRepBuilderAPI_MakeFace(
                    surface, u_min, u_max, v_min, v_max, tolerance
                )
                if face_maker.IsDone():
                    print(f"[brep] face {face_idx}: BSpline UV "
                          f"[{u_min:.3f},{u_max:.3f}]×[{v_min:.3f},{v_max:.3f}]")
                    sewing.Add(face_maker.Face())
                    face_added = True
                else:
                    print(f"[brep] face {face_idx}: BSpline MakeFace failed")
            except Exception as exc:
                print(f"[brep] face {face_idx}: BSpline MakeFace exception: {exc}")

            if not face_added:
                # Last resort: full natural domain without periodisation.
                try:
                    face_maker = BRepBuilderAPI_MakeFace(surface, tolerance)
                    if face_maker.IsDone():
                        print(f"[brep] face {face_idx}: BSpline natural domain fallback")
                        sewing.Add(face_maker.Face())
                    else:
                        print(f"[brep] face {face_idx}: BSpline natural domain failed")
                except Exception as exc:
                    print(f"[brep] face {face_idx}: BSpline fallback exception: {exc}")
            continue

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
    print("[brep] Sewing faces ...")
    sewing.Perform()
    shape = sewing.SewedShape()

    n_input_faces = len(face_arcs)
    if shape is None or shape.IsNull():
        print("[brep] sewing produced no shape")
        return shape, {"valid": False, "n_faces": 0, "n_input_faces": n_input_faces}

    # 5. Ensure consistent 3D curves, then heal.
    print("[brep] Fixing shape ...")
    try:
        breplib.BuildCurves3d(shape)
    except Exception as exc:
        print(f"[brep] BuildCurves3d failed: {exc}")
    fixer = ShapeFix_Shape(shape)
    fixer.SetPrecision(tolerance)
    fixer.Perform()
    shape = fixer.Shape()

    # 6. Re-parameterise edges so stored tolerances reflect the actual
    #    3D-curve / pcurve deviation.  Fixes "Invalid curve on surface"
    #    errors in boolean operation pre-checks.
    #    Disable with same_parameter=False if this causes regressions.
    if same_parameter:
        try:
            breplib.SameParameter(shape, True)
            print("[brep] SameParameter done")
        except Exception as exc:
            print(f"[brep] SameParameter failed: {exc}")

    # 7. Ensure consistent face-normal orientation for a closed shell/solid.
    #    Fixes "Self-intersection found" errors caused by inward-pointing
    #    face normals.  Requires a TopoDS_Solid; shells are wrapped first.
    #    Disable with orient_solid=False if this causes regressions.
    if orient_solid:
        try:
            stype = shape.ShapeType()
            if stype == TopAbs_SOLID:
                solid = topods.Solid(shape)
                breplib.OrientClosedSolid(solid)
                shape = solid
                print("[brep] OrientClosedSolid done (solid)")
            elif stype == TopAbs_SHELL:
                solid_maker = BRepBuilderAPI_MakeSolid(topods.Shell(shape))
                if solid_maker.IsDone():
                    solid = solid_maker.Solid()
                    breplib.OrientClosedSolid(solid)
                    shape = solid
                    print("[brep] OrientClosedSolid done (shell → solid)")
                else:
                    print("[brep] OrientClosedSolid: MakeSolid from shell failed")
            else:
                print(f"[brep] OrientClosedSolid: skipped "
                      f"(shape type {stype}, expected shell or solid)")
        except Exception as exc:
            print(f"[brep] OrientClosedSolid failed: {exc}")

    analyzer = BRepCheck_Analyzer(shape)
    eval_results = analyzer.IsValid()

    # Count faces in the output shape
    n_output_faces = 0
    face_exp = TopExp_Explorer(shape, TopAbs_FACE)
    while face_exp.More():
        n_output_faces += 1
        face_exp.Next()

    print(f"[brep] Results of BRep correctness analyzer: {eval_results}")
    if not eval_results:
        _print_brep_check_details(analyzer, shape)
    print(f"[brep] Output faces: {n_output_faces}/{n_input_faces}")

    result_info = {
        "valid": eval_results,
        "n_faces": n_output_faces,
        "n_input_faces": n_input_faces,
    }
    return shape, result_info


def _wire_uv_signed_area(wire_arcs, surface, n_samples=20):
    """Signed area of a wire projected into surface UV space (shoelace).

    Parameters
    ----------
    wire_arcs : list[(arc_dict, bool)]
        Ordered arc sequence with forward/reverse flags.
    surface : Geom_Surface
    n_samples : int
        Points sampled per arc.

    Returns
    -------
    float   Signed area.  |area| is the enclosed UV region; the wire with the
            largest |area| is the outer boundary.
    """
    uv_pts = []
    for arc, forward in wire_arcs:
        curve = arc["curve"]
        t0 = curve.FirstParameter()
        t1 = curve.LastParameter()
        if not forward:
            t0, t1 = t1, t0
        for k in range(n_samples):
            t = t0 + (t1 - t0) * k / max(n_samples - 1, 1)
            try:
                p = curve.Value(t)
            except Exception:
                continue
            proj = GeomAPI_ProjectPointOnSurf(p, surface)
            if proj.NbPoints() == 0:
                continue
            u, v = proj.LowerDistanceParameters()
            uv_pts.append((u, v))
    if len(uv_pts) < 3:
        return 0.0
    # Shoelace formula
    area = 0.0
    n = len(uv_pts)
    for i in range(n):
        j = (i + 1) % n
        area += uv_pts[i][0] * uv_pts[j][1]
        area -= uv_pts[j][0] * uv_pts[i][1]
    return area / 2.0



def _build_face_with_uv_classification(face_idx, surface, wire_items,
                                        tolerance):
    """Build a face with correct outer/inner wire classification.

    Uses signed UV area to identify the outer wire, then constructs the
    face with BRepBuilderAPI_MakeFace(surface, outer_wire) + Add(inner).

    Parameters
    ----------
    wire_items   : list[(TopoDS_Wire, wire_arcs_list)]
    Returns the constructed TopoDS_Face, or None on failure.
    """
    occ_wires = [w for w, _ in wire_items]

    if len(occ_wires) == 1:
        outer_wire = occ_wires[0]
        inner_wires = []
        outer_idx = 0
    else:
        areas = []
        for _, wire_arcs in wire_items:
            a = _wire_uv_signed_area(wire_arcs, surface)
            areas.append(a)
        abs_areas = [abs(a) for a in areas]
        area_log = "  ".join(f"w{i}={areas[i]:.4f}" for i in range(len(areas)))
        print(f"[brep-direct] face {face_idx}: UV areas: {area_log}")
        outer_idx = max(range(len(abs_areas)), key=lambda i: abs_areas[i])
        outer_wire = occ_wires[outer_idx]
        inner_wires = [w for i, w in enumerate(occ_wires) if i != outer_idx]

    print(f"[brep-direct] face {face_idx}: {len(occ_wires)} wire(s)")
    face_maker = BRepBuilderAPI_MakeFace(surface, outer_wire)
    for iw in inner_wires:
        face_maker.Add(iw)
    if not face_maker.IsDone():
        print(f"[brep-direct] face {face_idx}: MakeFace not done")
        return None

    face = face_maker.Face()
    n_wires = sum(1 for _ in _iter_explorer(face, TopAbs_WIRE))
    print(f"[brep-direct] face {face_idx}: {len(occ_wires)} wires "
          f"(outer=w{outer_idx}) → {n_wires} after MakeFace")
    return face


def _iter_explorer(shape, shape_type):
    """Yield sub-shapes from a TopExp_Explorer."""
    exp = TopExp_Explorer(shape, shape_type)
    while exp.More():
        yield exp.Current()
        exp.Next()


def build_brep_shape_direct(face_arcs, occ_surfaces, vertices, surface_ids=None,
                            face_wires=None, tolerance=1e-3, clusters=None,
                            cluster_trees=None):
    """
    Build a TopoDS_Shape with signed-UV-area wire classification.

    For each face, wires are classified as outer/inner by projecting
    sample points into UV space and computing the signed polygon area
    (shoelace formula).  The wire with the largest |area| is the outer
    boundary; the rest are holes.

    When clusters are provided, both wire orientations are tried and
    the face with lower mean cluster projection error is selected.

    Parameters / return value match build_brep_shape for drop-in use.
    """
    bb = BRep_Builder()

    # 1. Build one TopoDS_Edge per unique arc.
    arc_to_edge = {}
    for arcs in face_arcs.values():
        for arc in arcs:
            key = _arc_key(arc)
            if key in arc_to_edge:
                continue
            try:
                arc_to_edge[key] = BRepBuilderAPI_MakeEdge(
                    arc["curve"]).Edge()
            except Exception as exc:
                print(f"[brep-direct] MakeEdge failed for arc on "
                      f"{arc.get('edge_key')}: {exc}")

    # 2. Build wires and faces, then sew.
    sewing = BRepBuilderAPI_Sewing(tolerance)

    for face_idx, arcs in face_arcs.items():
        if face_idx >= len(occ_surfaces) or occ_surfaces[face_idx] is None:
            continue
        surface = occ_surfaces[face_idx]

        # Build OCC wires, keeping the arc-level description alongside
        # each wire so we can compute UV areas for classification.
        wire_items = []   # list of (TopoDS_Wire, wire_arcs_list)
        for wire_arcs in face_wires.get(face_idx, []):
            wire = TopoDS_Wire()
            bb.MakeWire(wire)
            n_added = 0
            for arc, forward in wire_arcs:
                key = _arc_key(arc)
                if key not in arc_to_edge:
                    continue
                edge = arc_to_edge[key]
                bb.Add(wire, edge if forward else edge.Reversed())
                n_added += 1
            if n_added == 0:
                continue
            fix_w = ShapeFix_Wire()
            fix_w.Load(wire)
            fix_w.SetPrecision(tolerance)
            fix_w.FixConnected()
            healed = fix_w.Wire()
            if healed.IsNull():
                print(f"[brep-direct] face {face_idx}: ShapeFix_Wire "
                      f"produced null wire — skipping")
                continue
            wire_items.append((healed, wire_arcs))

        if not wire_items:
            print(f"[brep-direct] face {face_idx}: no wires — skipping")
            continue

        try:
            built_face = _build_face_with_uv_classification(
                face_idx, surface, wire_items, tolerance)
        except Exception as exc:
            print(f"[brep-direct] face {face_idx}: exception: {exc}")
            built_face = None
        if built_face is None:
            print(f"[brep-direct] face {face_idx}: face construction failed")
        else:
            sewing.Add(built_face)

    # 3. Sew all faces into a shell.
    print("[brep-direct] Sewing faces ...")
    sewing.Perform()
    shape = sewing.SewedShape()

    n_input_faces = len(face_arcs)
    if shape is None or shape.IsNull():
        print("[brep-direct] sewing produced no shape")
        return shape, {"valid": False, "n_faces": 0,
                       "n_input_faces": n_input_faces}

    # 4. Ensure consistent 3D curves, then heal.
    print("[brep-direct] Fixing shape ...")
    try:
        breplib.BuildCurves3d(shape)
    except Exception as exc:
        print(f"[brep-direct] BuildCurves3d failed: {exc}")

    fixer = ShapeFix_Shape(shape)
    fixer.SetPrecision(tolerance)
    fixer.Perform()
    shape = fixer.Shape()

    try:
        breplib.SameParameter(shape, True)
        print("[brep-direct] SameParameter done")
    except Exception as exc:
        print(f"[brep-direct] SameParameter failed: {exc}")

    # 5. Check validity.
    analyzer = BRepCheck_Analyzer(shape)
    eval_results = analyzer.IsValid()

    n_output_faces = 0
    face_exp = TopExp_Explorer(shape, TopAbs_FACE)
    while face_exp.More():
        n_output_faces += 1
        face_exp.Next()

    print(f"[brep-direct] BRepCheck valid: {eval_results}")
    if not eval_results:
        _print_brep_check_details(analyzer, shape)
    print(f"[brep-direct] Output faces: {n_output_faces}/{n_input_faces}")

    return shape, {
        "valid": eval_results,
        "n_faces": n_output_faces,
        "n_input_faces": n_input_faces,
    }


def build_brep_shape_builderface(face_arcs, occ_surfaces, vertices,
                                  surface_ids=None, tolerance=1e-3,
                                  inr_geom_close_tol=0.05, inr_arc_samples=30,
                                  inr_uv_margin=0.02,
                                  same_parameter=True, orient_solid=True,
                                  clusters=None):
    """
    Build a TopoDS_Shape using BOPAlgo_BuilderFace for wire assembly.

    Drop-in replacement for build_brep_shape.  Instead of manual angular
    ordering (assemble_wires), each face's edges are passed to OCC's
    BOPAlgo_BuilderFace, which internally:
      - connects edges into wires
      - classifies outer vs inner (hole) wires
      - builds the bounded face(s)

    Parameters and return value match build_brep_shape exactly.
    """
    # 1. Create shared TopoDS_Vertex instances so edges can be connected.
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

    # 2. Build faces using BOPAlgo_BuilderFace.
    sewing = BRepBuilderAPI_Sewing(tolerance)
    n_input_faces = len(face_arcs)

    for face_idx, arcs in face_arcs.items():
        if face_idx >= len(occ_surfaces) or occ_surfaces[face_idx] is None:
            continue
        surface = occ_surfaces[face_idx]

        is_inr = (surface_ids is not None and
                  face_idx < len(surface_ids) and
                  surface_ids[face_idx] == SURFACE_INR)

        # --- BSpline (INR) face: same UV-bounds / explicit-pcurve logic
        # as build_brep_shape (INR surfaces don't go through BuilderFace) ---
        if is_inr:
            # Geometric closure detection and periodisation
            nu1, nu2, nv1, nv2 = surface.Bounds()
            v_mid = (nv1 + nv2) / 2.0
            u_mid = (nu1 + nu2) / 2.0
            closed_u = closed_v = False
            try:
                pu1 = surface.Value(nu1, v_mid)
                pu2 = surface.Value(nu2, v_mid)
                d_u = math.sqrt((pu1.X()-pu2.X())**2 +
                                (pu1.Y()-pu2.Y())**2 +
                                (pu1.Z()-pu2.Z())**2)
                pv1 = surface.Value(u_mid, nv1)
                pv2 = surface.Value(u_mid, nv2)
                d_v = math.sqrt((pv1.X()-pv2.X())**2 +
                                (pv1.Y()-pv2.Y())**2 +
                                (pv1.Z()-pv2.Z())**2)
            except Exception:
                d_u, d_v = 1.0, 1.0
            closed_u = d_u < inr_geom_close_tol
            closed_v = d_v < inr_geom_close_tol
            print(f"[builderface] face {face_idx}: BSpline seam check "
                  f"d_u={d_u:.4f} d_v={d_v:.4f} "
                  f"closed_u={closed_u} closed_v={closed_v}")
            try:
                if closed_u:
                    surface.SetUPeriodic()
                if closed_v:
                    surface.SetVPeriodic()
                nu1, nu2, nv1, nv2 = surface.Bounds()
            except Exception as exc:
                print(f"[builderface] face {face_idx}: SetPeriodic failed: {exc}")
                closed_u = closed_v = False

            # UV-bounds method for INR
            u_min, u_max = nu1, nu2
            v_min, v_max = nv1, nv2
            if not closed_u or not closed_v:
                sample_pts = []
                for arc in arcs:
                    t0, t1 = arc["t_start"], arc["t_end"]
                    for k in range(inr_arc_samples):
                        t = t0 + (t1 - t0) * k / max(inr_arc_samples - 1, 1)
                        try:
                            p = arc["curve"].Value(t)
                            sample_pts.append([p.X(), p.Y(), p.Z()])
                        except Exception:
                            pass
                if sample_pts:
                    sample_arr = np.array(sample_pts, dtype=np.float32)
                    bounds = _cluster_uv_bounds(surface, sample_arr,
                                                rel_margin=inr_uv_margin)
                    if bounds is not None:
                        bu_min, bu_max, bv_min, bv_max = bounds
                        if not closed_u:
                            u_min = max(bu_min, nu1)
                            u_max = min(bu_max, nu2)
                        if not closed_v:
                            v_min = max(bv_min, nv1)
                            v_max = min(bv_max, nv2)

            face_added = False
            try:
                face_maker = BRepBuilderAPI_MakeFace(
                    surface, u_min, u_max, v_min, v_max, tolerance)
                if face_maker.IsDone():
                    print(f"[builderface] face {face_idx}: BSpline UV "
                          f"[{u_min:.3f},{u_max:.3f}]×[{v_min:.3f},{v_max:.3f}]")
                    sewing.Add(face_maker.Face())
                    face_added = True
                else:
                    print(f"[builderface] face {face_idx}: BSpline MakeFace failed")
            except Exception as exc:
                print(f"[builderface] face {face_idx}: BSpline MakeFace exception: {exc}")
            if not face_added:
                try:
                    face_maker = BRepBuilderAPI_MakeFace(surface, tolerance)
                    if face_maker.IsDone():
                        print(f"[builderface] face {face_idx}: BSpline natural domain fallback")
                        sewing.Add(face_maker.Face())
                    else:
                        print(f"[builderface] face {face_idx}: BSpline natural domain failed")
                except Exception as exc:
                    print(f"[builderface] face {face_idx}: BSpline fallback exception: {exc}")
            continue

        # --- Analytical faces: use BOPAlgo_BuilderFace ---
        # Create fresh edges per face to avoid orientation conflicts
        # when the same arc is shared between two faces.  Sewing will
        # merge coincident edges afterward.
        edge_list = TopTools_ListOfShape()
        n_edges = 0
        for arc in arcs:
            try:
                v_start = arc.get("v_start")
                v_end = arc.get("v_end")
                curve = arc["curve"]
                t0 = arc["t_start"]
                t1 = arc["t_end"]

                if v_start is not None and v_end is not None:
                    edge_maker = BRepBuilderAPI_MakeEdge(
                        curve, vtx_to_topo[v_start], vtx_to_topo[v_end],
                        t0, t1)
                elif arc.get("closed", False):
                    edge_maker = BRepBuilderAPI_MakeEdge(curve)
                else:
                    edge_maker = BRepBuilderAPI_MakeEdge(curve, t0, t1)

                if not edge_maker.IsDone():
                    err = edge_maker.Error()
                    # Compute diagnostic distances
                    diag = ""
                    if v_start is not None and v_end is not None:
                        try:
                            p0 = curve.Value(t0)
                            p1 = curve.Value(t1)
                            vs = vertices[v_start]
                            ve = vertices[v_end]
                            d0 = ((p0.X()-vs[0])**2 + (p0.Y()-vs[1])**2 + (p0.Z()-vs[2])**2)**0.5
                            d1 = ((p1.X()-ve[0])**2 + (p1.Y()-ve[1])**2 + (p1.Z()-ve[2])**2)**0.5
                            diag = f" d(v_start,C(t0))={d0:.6f} d(v_end,C(t1))={d1:.6f}"
                        except Exception:
                            pass
                    print(f"[builderface] face {face_idx}: MakeEdge error={err} "
                          f"for arc {arc.get('edge_key')} t=[{t0:.4f},{t1:.4f}]{diag}")
                    continue

                edge = edge_maker.Edge()
                # Improve pcurve/3D-curve consistency before BuilderFace
                # uses pcurves for region detection.
                try:
                    breplib.SameParameter(edge, tolerance)
                except Exception as sp_exc:
                    print(f"[builderface] face {face_idx}: SameParameter failed "
                          f"for arc {arc.get('edge_key')}: {sp_exc}")
                edge_list.Append(edge)
                n_edges += 1
            except Exception as exc:
                print(f"[builderface] face {face_idx}: MakeEdge exception "
                      f"for arc {arc.get('edge_key')}: {exc}")

        if n_edges == 0:
            print(f"[builderface] face {face_idx}: no edges — skipping")
            continue

        # Create a reference face for BuilderFace (unbounded face on the surface)
        try:
            ref_face_maker = BRepBuilderAPI_MakeFace(surface, tolerance)
            if not ref_face_maker.IsDone():
                print(f"[builderface] face {face_idx}: MakeFace for reference failed")
                continue
            ref_face = ref_face_maker.Face()
        except Exception as exc:
            print(f"[builderface] face {face_idx}: reference face exception: {exc}")
            continue

        try:
            builder = BOPAlgo_BuilderFace()
            builder.SetFace(ref_face)
            builder.SetShapes(edge_list)
            builder.SetFuzzyValue(tolerance)
            builder.Perform()

            if builder.HasErrors():
                print(f"[builderface] face {face_idx}: BuilderFace failed with errors")
                continue

            # BuilderFace partitions the surface into bounded regions.
            # When multiple regions exist, hole fillings (single-wire
            # regions) overlap with adjacent surfaces.  The actual
            # BRep face is the region with the most wires (outer +
            # inner hole wires).  When only one region exists, use it
            # directly.
            from OCC.Core.TopTools import TopTools_ListIteratorOfListOfShape
            areas = builder.Areas()
            n_areas = areas.Size()

            if n_areas == 0:
                print(f"[builderface] face {face_idx}: {n_edges} edges → "
                      f"0 regions")
                continue

            # Collect all candidate regions.
            candidate_regions = []
            it = TopTools_ListIteratorOfListOfShape(areas)
            while it.More():
                candidate_regions.append(it.Value())
                it.Next()

            if n_areas == 1:
                best_face = candidate_regions[0]
                n_wires = 0
                wexp = TopExp_Explorer(best_face, TopAbs_WIRE)
                while wexp.More():
                    n_wires += 1
                    wexp.Next()
                print(f"[builderface] face {face_idx}: {n_edges} edges → "
                      f"1 region, {n_wires} wire(s)")
            elif clusters is not None and face_idx < len(clusters):
                # Score each region by meshing it and computing mean NN
                # distance from cluster points to the mesh vertices.
                from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
                from OCC.Core.BRep import BRep_Tool
                from OCC.Core.TopLoc import TopLoc_Location

                cluster_pts = clusters[face_idx]
                cluster_tree = cKDTree(cluster_pts)

                best_face = None
                best_score = float("inf")
                best_ri = -1
                for ri, region in enumerate(candidate_regions):
                    # Mesh the candidate region
                    mesh = BRepMesh_IncrementalMesh(region, tolerance * 10)
                    mesh.Perform()

                    # Extract mesh vertices
                    face_shape = topods.Face(region)
                    loc = TopLoc_Location()
                    tri = BRep_Tool.Triangulation(face_shape, loc)
                    if tri is None:
                        print(f"[builderface] face {face_idx}: region {ri} "
                              f"meshing failed, skipping")
                        continue

                    n_nodes = tri.NbNodes()
                    mesh_pts = np.array(
                        [[tri.Node(k).X(), tri.Node(k).Y(), tri.Node(k).Z()]
                         for k in range(1, n_nodes + 1)],
                        dtype=np.float64,
                    )

                    # Mean NN distance from cluster points to mesh vertices
                    mesh_tree = cKDTree(mesh_pts)
                    dists, _ = mesh_tree.query(cluster_pts, k=1)
                    score = float(np.mean(dists))

                    n_w = 0
                    wexp = TopExp_Explorer(region, TopAbs_WIRE)
                    while wexp.More():
                        n_w += 1
                        wexp.Next()
                    print(f"[builderface] face {face_idx}: region {ri} — "
                          f"{n_nodes} mesh pts, {n_w} wire(s), "
                          f"mean NN dist = {score:.6f}")

                    if score < best_score:
                        best_score = score
                        best_face = region
                        best_ri = ri

                if best_face is None:
                    print(f"[builderface] face {face_idx}: all regions failed "
                          f"meshing, skipping")
                    continue

                print(f"[builderface] face {face_idx}: {n_edges} edges → "
                      f"{n_areas} region(s), selected region {best_ri} "
                      f"(score {best_score:.6f})")
            else:
                # Fallback: pick region with most wires (no cluster data).
                best_face = None
                best_n_wires = -1
                for region in candidate_regions:
                    n_wires = 0
                    wexp = TopExp_Explorer(region, TopAbs_WIRE)
                    while wexp.More():
                        n_wires += 1
                        wexp.Next()
                    if n_wires > best_n_wires:
                        best_n_wires = n_wires
                        best_face = region
                print(f"[builderface] face {face_idx}: {n_edges} edges → "
                      f"{n_areas} region(s), keeping region with "
                      f"{best_n_wires} wire(s) (no cluster data, fallback)")

            # Heal wire orientation and pcurve issues before sewing.
            # FixOrientation corrects outer/inner wire classification
            # but can crash with "Bnd_Box is void" on some surfaces,
            # so it's wrapped in try/except.
            try:
                face_shape = best_face
                ff = ShapeFix_Face(topods.Face(face_shape))
                ff.SetPrecision(tolerance)
                try:
                    result = ff.FixOrientation()
                    print(f"[builderface] face {face_idx}: FixOrientation "
                          f"returned {result}")
                except Exception as fo_exc:
                    print(f"[builderface] face {face_idx}: FixOrientation "
                          f"crashed: {fo_exc}")
                ff.FixIntersectingWires()
                ff.Perform()
                face_shape = ff.Face()
                sewing.Add(face_shape)
            except Exception as fix_exc:
                print(f"[builderface] face {face_idx}: ShapeFix_Face "
                      f"failed: {fix_exc}")
                sewing.Add(best_face)

        except Exception as exc:
            print(f"[builderface] face {face_idx}: BuilderFace exception: {exc}")

    # 3. Sew all faces into a shell.
    print("[builderface] Sewing faces ...")
    sewing.Perform()
    shape = sewing.SewedShape()

    if shape is None or shape.IsNull():
        print("[builderface] sewing produced no shape")
        return shape, {"valid": False, "n_faces": 0,
                       "n_input_faces": n_input_faces}

    # 4. Fix face orientations within the shell.
    try:
        stype = shape.ShapeType()
        if stype == TopAbs_SHELL:
            shell_fix = ShapeFix_Shell(topods.Shell(shape))
            shell_fix.SetPrecision(tolerance)
            shell_fix.FixFaceOrientation(topods.Shell(shape))
            shell_fix.Perform()
            shape = shell_fix.Shape()
            n_err = shell_fix.NbShells()
            print(f"[builderface] ShapeFix_Shell: FixFaceOrientation done "
                  f"({n_err} shell(s))")
        else:
            print(f"[builderface] ShapeFix_Shell skipped "
                  f"(shape type {stype}, not shell)")
    except Exception as exc:
        print(f"[builderface] ShapeFix_Shell failed: {exc}")

    # 5. Heal.
    print("[builderface] Fixing shape ...")
    try:
        breplib.BuildCurves3d(shape)
    except Exception as exc:
        print(f"[builderface] BuildCurves3d failed: {exc}")
    fixer = ShapeFix_Shape(shape)
    fixer.SetPrecision(tolerance)
    fixer.Perform()
    shape = fixer.Shape()

    if same_parameter:
        try:
            breplib.SameParameter(shape, True)
            print("[builderface] SameParameter done")
        except Exception as exc:
            print(f"[builderface] SameParameter failed: {exc}")

    if orient_solid:
        try:
            stype = shape.ShapeType()
            if stype == TopAbs_SOLID:
                solid = topods.Solid(shape)
                breplib.OrientClosedSolid(solid)
                shape = solid
                print("[builderface] OrientClosedSolid done (solid)")
            elif stype == TopAbs_SHELL:
                solid_maker = BRepBuilderAPI_MakeSolid(topods.Shell(shape))
                if solid_maker.IsDone():
                    solid = solid_maker.Solid()
                    breplib.OrientClosedSolid(solid)
                    shape = solid
                    print("[builderface] OrientClosedSolid done (shell → solid)")
                else:
                    print("[builderface] OrientClosedSolid: MakeSolid from shell failed")
            else:
                print(f"[builderface] OrientClosedSolid: skipped "
                      f"(shape type {stype}, expected shell or solid)")
        except Exception as exc:
            print(f"[builderface] OrientClosedSolid failed: {exc}")

    analyzer = BRepCheck_Analyzer(shape)
    eval_results = analyzer.IsValid()

    n_output_faces = 0
    face_exp = TopExp_Explorer(shape, TopAbs_FACE)
    while face_exp.More():
        n_output_faces += 1
        face_exp.Next()

    print(f"[builderface] Results of BRep correctness analyzer: {eval_results}")
    if not eval_results:
        _print_brep_check_details(analyzer, shape)
    print(f"[builderface] Output faces: {n_output_faces}/{n_input_faces}")

    return shape, {"valid": eval_results, "n_faces": n_output_faces,
                   "n_input_faces": n_input_faces}


def build_brep_shape_cells(face_arcs, occ_surfaces, vertices,
                           surface_ids=None, tolerance=1e-3,
                           clusters=None,
                           same_parameter=True, orient_solid=True):
    """
    Build a TopoDS_Shape using BOPAlgo_CellsBuilder.

    Creates unbounded faces for each surface, edges from all arcs, and
    feeds them to CellsBuilder which partitions surfaces at edges into
    cells.  Cells are selected by scoring against cluster point clouds
    (mean NN distance from cluster points to meshed cell).

    Parameters match build_brep_shape; clusters is a list of numpy arrays
    (per-face point clouds) used for cell selection.
    """
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.TopTools import TopTools_ListIteratorOfListOfShape

    n_input_faces = len(face_arcs)

    # 1. Create unbounded reference faces for each surface.
    ref_faces = {}
    for face_idx in face_arcs:
        if face_idx >= len(occ_surfaces) or occ_surfaces[face_idx] is None:
            continue
        surface = occ_surfaces[face_idx]
        is_inr = (surface_ids is not None and
                  face_idx < len(surface_ids) and
                  surface_ids[face_idx] == SURFACE_INR)
        if is_inr:
            continue  # skip INR surfaces for now
        try:
            fm = BRepBuilderAPI_MakeFace(surface, tolerance)
            if fm.IsDone():
                ref_faces[face_idx] = fm.Face()
            else:
                print(f"[cells] face {face_idx}: MakeFace for reference failed")
        except Exception as exc:
            print(f"[cells] face {face_idx}: MakeFace exception: {exc}")

    print(f"[cells] created {len(ref_faces)} reference faces "
          f"from {n_input_faces} input faces")

    # 2. Create edges from all arcs.
    arc_edges = []
    arc_to_edge = {}
    for arcs in face_arcs.values():
        for arc in arcs:
            key = _arc_key(arc)
            if key in arc_to_edge:
                continue
            try:
                edge = BRepBuilderAPI_MakeEdge(arc["curve"]).Edge()
                arc_to_edge[key] = edge
                arc_edges.append(edge)
            except Exception as exc:
                print(f"[cells] MakeEdge failed for arc on "
                      f"{arc.get('edge_key')}: {exc}")

    print(f"[cells] created {len(arc_edges)} edges from arcs")

    # 3. Run CellsBuilder.
    cb = BOPAlgo_CellsBuilder()
    cb.SetFuzzyValue(tolerance)

    # Add reference faces as arguments
    for face_idx, face in ref_faces.items():
        cb.AddArgument(face)

    # Add edges as tools (splitting elements)
    for edge in arc_edges:
        cb.AddArgument(edge)

    print("[cells] performing split ...")
    cb.Perform()

    if cb.HasErrors():
        print("[cells] CellsBuilder failed with errors")
        return None, {"valid": False, "n_faces": 0,
                      "n_input_faces": n_input_faces}

    # 4. Get all cells (split faces).
    cb.MakeContainers()
    result_shape = cb.Shape()

    # Extract face-type cells from the result compound.
    cells = []
    cell_exp = TopExp_Explorer(result_shape, TopAbs_FACE)
    while cell_exp.More():
        cells.append(cell_exp.Current())
        cell_exp.Next()

    n_cells = len(cells)
    print(f"[cells] split produced {n_cells} face cells")

    if n_cells == 0:
        return None, {"valid": False, "n_faces": 0,
                      "n_input_faces": n_input_faces}

    # For each cell, mesh it and find which cluster it belongs to.
    selected_cells = []

    if clusters is not None:
        # Build KD-trees for all clusters
        cluster_trees = [cKDTree(c) for c in clusters]

        for ci, cell in enumerate(cells):
            face_shape = topods.Face(cell)

            # Mesh the cell
            mesh = BRepMesh_IncrementalMesh(face_shape, tolerance * 10)
            mesh.Perform()

            loc = TopLoc_Location()
            tri = BRep_Tool.Triangulation(face_shape, loc)
            if tri is None:
                print(f"[cells] cell {ci}: meshing failed, skipping")
                continue

            n_nodes = tri.NbNodes()
            if n_nodes == 0:
                continue

            mesh_pts = np.array(
                [[tri.Node(k).X(), tri.Node(k).Y(), tri.Node(k).Z()]
                 for k in range(1, n_nodes + 1)],
                dtype=np.float64,
            )

            # Find the best-matching cluster for this cell
            best_cluster = -1
            best_score = float("inf")
            for cj, ct in enumerate(cluster_trees):
                dists, _ = ct.query(mesh_pts, k=1)
                score = float(np.mean(dists))
                if score < best_score:
                    best_score = score
                    best_cluster = cj

            print(f"[cells] cell {ci}: {n_nodes} mesh pts, "
                  f"best cluster={best_cluster}, score={best_score:.6f}")

            # Accept cell if it matches a cluster well enough
            selected_cells.append((ci, cell, best_cluster, best_score))
    else:
        # No clusters — select all face cells
        for ci, cell in enumerate(cells):
            selected_cells.append((ci, cell, -1, 0.0))

    print(f"[cells] {len(selected_cells)} face cells found")

    # 6. For each cluster, pick the best cell (lowest score).
    #    A cluster may have multiple cells (e.g. a plane split by edges).
    cluster_to_cells = defaultdict(list)
    for ci, cell, cj, score in selected_cells:
        if cj >= 0:
            cluster_to_cells[cj].append((ci, cell, score))

    # Select: for each cluster, take all cells that match it
    # (a face may be split into multiple valid sub-faces).
    sewing = BRepBuilderAPI_Sewing(tolerance)
    n_selected = 0
    for cj in sorted(cluster_to_cells.keys()):
        cells_for_cluster = cluster_to_cells[cj]
        # Sort by score and take cells with reasonable scores
        cells_for_cluster.sort(key=lambda x: x[2])
        best = cells_for_cluster[0][2]
        for ci, cell, score in cells_for_cluster:
            # Accept cells within 2x of the best score for this cluster
            if score <= best * 2.0 or score < tolerance * 10:
                sewing.Add(cell)
                n_selected += 1
                print(f"[cells] selecting cell {ci} for cluster {cj} "
                      f"(score={score:.6f})")

    print(f"[cells] selected {n_selected} cells for {len(cluster_to_cells)} "
          f"clusters")

    # 7. Sew selected cells.
    print("[cells] Sewing ...")
    sewing.Perform()
    shape = sewing.SewedShape()

    if shape is None or shape.IsNull():
        print("[cells] sewing produced no shape")
        return shape, {"valid": False, "n_faces": 0,
                       "n_input_faces": n_input_faces}

    # 8. Fix.
    try:
        stype = shape.ShapeType()
        if stype == TopAbs_SHELL:
            shell_fix = ShapeFix_Shell(topods.Shell(shape))
            shell_fix.SetPrecision(tolerance)
            shell_fix.FixFaceOrientation(topods.Shell(shape))
            shell_fix.Perform()
            shape = shell_fix.Shape()
            print("[cells] ShapeFix_Shell done")
        else:
            print(f"[cells] ShapeFix_Shell skipped (shape type {stype})")
    except Exception as exc:
        print(f"[cells] ShapeFix_Shell failed: {exc}")

    print("[cells] Fixing shape ...")
    try:
        breplib.BuildCurves3d(shape)
    except Exception as exc:
        print(f"[cells] BuildCurves3d failed: {exc}")

    fixer = ShapeFix_Shape(shape)
    fixer.SetPrecision(tolerance)
    fixer.Perform()
    shape = fixer.Shape()

    if same_parameter:
        try:
            breplib.SameParameter(shape, True)
            print("[cells] SameParameter done")
        except Exception as exc:
            print(f"[cells] SameParameter failed: {exc}")

    if orient_solid:
        try:
            stype = shape.ShapeType()
            if stype == TopAbs_SOLID:
                solid = topods.Solid(shape)
                breplib.OrientClosedSolid(solid)
                shape = solid
                print("[cells] OrientClosedSolid done (solid)")
            elif stype == TopAbs_SHELL:
                solid_maker = BRepBuilderAPI_MakeSolid(topods.Shell(shape))
                if solid_maker.IsDone():
                    solid = solid_maker.Solid()
                    breplib.OrientClosedSolid(solid)
                    shape = solid
                    print("[cells] OrientClosedSolid done (shell → solid)")
                else:
                    print("[cells] MakeSolid from shell failed")
            else:
                print(f"[cells] OrientClosedSolid skipped (shape type {stype})")
        except Exception as exc:
            print(f"[cells] OrientClosedSolid failed: {exc}")

    # 9. Validate.
    analyzer = BRepCheck_Analyzer(shape, True)
    eval_results = analyzer.IsValid()

    n_output_faces = 0
    face_exp = TopExp_Explorer(shape, TopAbs_FACE)
    while face_exp.More():
        n_output_faces += 1
        face_exp.Next()

    print(f"[cells] Results of BRep correctness analyzer: {eval_results}")
    if not eval_results:
        _print_brep_check_details(analyzer, shape)
    print(f"[cells] Output faces: {n_output_faces}/{n_input_faces}")

    return shape, {"valid": eval_results, "n_faces": n_output_faces,
                   "n_input_faces": n_input_faces}


def export_step(shape, path):
    """Export a TopoDS_Shape to a STEP file."""
    if shape is None or shape.IsNull():
        print(f"[step] export skipped — shape is null")
        return False
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs, True, Message_ProgressRange())
    ok = writer.Write(path) == IFSelect_RetDone
    if ok:
        print(f"STEP written to {path}")
    else:
        print(f"STEP export failed")
    return ok


def merge_step_files(step_paths, output_path):
    """
    Load each STEP file in step_paths and combine their shapes into a single
    Compound, then export to output_path.  Files that fail to load or produce
    a null shape are silently skipped.  Returns the count of shapes added.
    """
    builder  = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    n_added = 0
    for path in step_paths:
        reader = STEPControl_Reader()
        if reader.ReadFile(path) != 1:
            print(f"[merge] read error: {path}")
            continue
        reader.TransferRoots()
        sub = reader.OneShape()
        if sub is None or sub.IsNull():
            print(f"[merge] null shape: {path}")
            continue
        builder.Add(compound, sub)
        n_added += 1
    if n_added == 0:
        print("[merge] no valid shapes — export skipped")
        return 0
    writer = STEPControl_Writer()
    writer.Transfer(compound, STEPControl_AsIs, True, Message_ProgressRange())
    writer.Write(output_path)
    print(f"[merge] {n_added} part(s) → {output_path}")
    return n_added


def apply_inverse_normalization(shape, mean, R, scale):
    """
    Undo the per-part normalize_points transform on an OCC shape.

    Forward normalization (brep_pipeline.py):
        pts_norm = (R @ (pts_orig - mean)) / scale

    Inverse:
        pts_orig = scale * R^T @ pts_norm + mean

    The linear part of the affine map is (scale * R^T), a rotation times a
    uniform scalar — so all analytical surfaces (planes, cylinders, …) remain
    analytical after the transform.  Uses gp_GTrsf + BRepBuilderAPI_GTransform.

    Returns the original shape unchanged if it is None or null.
    """
    if shape is None or shape.IsNull():
        return shape

    Rt = np.asarray(R, dtype=np.float64).T          # 3x3 rotation
    mean = np.asarray(mean, dtype=np.float64)
    s = float(scale)

    # Build gp_Trsf (uniform scale + rotation + translation).
    # gp_Trsf represents  p' = ScaleFactor * RotationMatrix * p + Translation
    # Our inverse normalization is  p' = scale * R^T * p + mean
    rot_mat = gp_Mat(
        Rt[0, 0], Rt[0, 1], Rt[0, 2],
        Rt[1, 0], Rt[1, 1], Rt[1, 2],
        Rt[2, 0], Rt[2, 1], Rt[2, 2],
    )
    quat = gp_Quaternion(rot_mat)

    trsf = gp_Trsf()
    trsf.SetRotation(quat)
    trsf.SetScaleFactor(s)
    trsf.SetTranslationPart(gp_Vec(float(mean[0]), float(mean[1]), float(mean[2])))

    # BRepBuilderAPI_Transform (not GTransform) handles BSpline curves
    # by transforming control points only — knot vectors stay unchanged.
    result = BRepBuilderAPI_Transform(shape, trsf, True)
    if not result.IsDone():
        print("[brep] apply_inverse_normalization: transform failed, returning shape as-is")
        return shape
    return result.Shape()


# ---------------------------------------------------------------------------
# Alternative Step 5 — BOPAlgo_MakerVolume approach
# ---------------------------------------------------------------------------

def _cluster_uv_bounds(surface, cluster_pts, rel_margin=0.1):
    """
    Project cluster_pts onto `surface` using GeomAPI_ProjectPointOnSurf and
    return (umin, umax, vmin, vmax) expanded by rel_margin * span on each side.
    Returns None if projection fails for all points.
    """
    u_vals, v_vals = [], []
    step = max(1, len(cluster_pts) // 300)
    for pt in cluster_pts[::step]:
        pnt = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
        try:
            proj = GeomAPI_ProjectPointOnSurf(pnt, surface)
            if proj.NbPoints() > 0:
                u, v = proj.LowerDistanceParameters()
                u_vals.append(u)
                v_vals.append(v)
        except Exception:
            pass

    if not u_vals:
        return None

    umin_r, umax_r = min(u_vals), max(u_vals)
    vmin_r, vmax_r = min(v_vals), max(v_vals)
    mu = rel_margin * max(umax_r - umin_r, 1e-6)
    mv = rel_margin * max(vmax_r - vmin_r, 1e-6)
    return umin_r - mu, umax_r + mu, vmin_r - mv, vmax_r + mv


def _uv_bounds_from_3d_points(surface, points_3d, rel_margin=1.0):
    """
    Project 3D points onto `surface` and return (umin, umax, vmin, vmax)
    expanded by rel_margin * span on each side, clipped to the surface's
    natural UV domain.

    A large rel_margin (default 1.0 = 100%) is appropriate for
    BOPAlgo_MakerVolume which needs face patches to overlap generously;
    the boolean operation trims them to the correct extent.

    Returns None if projection fails for all points.
    """
    u_vals, v_vals = [], []
    for pt in points_3d:
        pnt = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
        try:
            proj = GeomAPI_ProjectPointOnSurf(pnt, surface)
            if proj.NbPoints() > 0:
                u, v = proj.LowerDistanceParameters()
                u_vals.append(u)
                v_vals.append(v)
        except Exception:
            pass

    if not u_vals:
        return None

    umin_r, umax_r = min(u_vals), max(u_vals)
    vmin_r, vmax_r = min(v_vals), max(v_vals)
    mu = rel_margin * max(umax_r - umin_r, 1e-6)
    mv = rel_margin * max(vmax_r - vmin_r, 1e-6)

    umin_out = umin_r - mu
    umax_out = umax_r + mu
    vmin_out = vmin_r - mv
    vmax_out = vmax_r + mv

    # Clip to the surface's natural UV domain to avoid wrapping past
    # periodic boundaries.
    su1, su2, sv1, sv2 = surface.Bounds()
    umin_out = max(umin_out, su1)
    umax_out = min(umax_out, su2)
    vmin_out = max(vmin_out, sv1)
    vmax_out = min(vmax_out, sv2)

    return umin_out, umax_out, vmin_out, vmax_out


def _project_to_uv(surface, pts_3d):
    """Project 3D points onto surface, return (N,2) UV array and mask of successful projections."""
    uv = []
    ok = []
    # step = max(1, len(pts_3d) // 500)  # subsample for speed
    for pt in pts_3d:
        pnt = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
        try:
            proj = GeomAPI_ProjectPointOnSurf(pnt, surface)
            if proj.NbPoints() > 0:
                u, v = proj.LowerDistanceParameters()
                uv.append([u, v])
                ok.append(True)
            else:
                ok.append(False)
        except Exception:
            ok.append(False)
    if not uv:
        return None
    return np.array(uv, dtype=np.float64)


def _douglas_peucker(pts, epsilon):
    """Simplify a closed polygon in-place using Douglas-Peucker.

    pts: (N,2) ordered polygon vertices (not closed — first != last).
    Returns simplified (M,2) array with M <= N.
    """
    if len(pts) <= 3:
        return pts

    # For a closed polygon, find the two farthest points to split
    n = len(pts)
    dists = np.linalg.norm(pts - pts[0], axis=1)
    split = int(np.argmax(dists))
    if split == 0:
        return pts

    def _simplify(points, eps):
        if len(points) <= 2:
            return points
        # Find point farthest from line (points[0] -> points[-1])
        start, end = points[0], points[-1]
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-15:
            return points[[0]]
        line_dir = line_vec / line_len
        vecs = points - start
        proj = vecs @ line_dir
        perp = vecs - np.outer(proj, line_dir)
        d = np.linalg.norm(perp, axis=1)
        idx = int(np.argmax(d))
        if d[idx] > eps:
            left = _simplify(points[:idx + 1], eps)
            right = _simplify(points[idx:], eps)
            return np.vstack([left[:-1], right])
        else:
            return points[[0, -1]]

    # Split closed polygon into two chains and simplify each
    chain1 = np.vstack([pts[0:split + 1]])
    chain2 = np.vstack([pts[split:], pts[:1]])
    s1 = _simplify(chain1, epsilon)
    s2 = _simplify(chain2, epsilon)
    # Merge (drop duplicate junction points)
    result = np.vstack([s1[:-1], s2[:-1]])
    if len(result) < 3:
        return pts
    return result


def _alpha_shape_boundary(uv_pts, alpha=0.0, simplify_epsilon=0.0):
    """
    Compute the boundary polygon of a 2D point set using alpha shapes.

    alpha=0 uses the convex hull.  alpha>0 uses Delaunay triangulation
    filtered by circumradius < 1/alpha, giving a concave boundary.

    If simplify_epsilon > 0, Douglas-Peucker simplification is applied.

    Returns an ordered (M,2) array of boundary UV points, or None on failure.
    """
    from scipy.spatial import Delaunay, ConvexHull

    if len(uv_pts) < 3:
        return None

    if alpha <= 0:
        try:
            hull = ConvexHull(uv_pts)
            boundary = uv_pts[hull.vertices]
            if simplify_epsilon > 0:
                boundary = _douglas_peucker(boundary, simplify_epsilon)
            return boundary
        except Exception:
            return None

    try:
        tri = Delaunay(uv_pts)
    except Exception:
        return None

    # Filter triangles by circumradius
    from collections import defaultdict
    edge_count = defaultdict(int)

    for simplex in tri.simplices:
        a, b, c = uv_pts[simplex[0]], uv_pts[simplex[1]], uv_pts[simplex[2]]
        # Circumradius
        ab = np.linalg.norm(b - a)
        bc = np.linalg.norm(c - b)
        ca = np.linalg.norm(a - c)
        s = (ab + bc + ca) / 2.0
        area = max(s * (s - ab) * (s - bc) * (s - ca), 0.0) ** 0.5
        if area < 1e-15:
            continue
        circum_r = (ab * bc * ca) / (4.0 * area)
        if circum_r < 1.0 / alpha:
            for e in [(simplex[0], simplex[1]),
                      (simplex[1], simplex[2]),
                      (simplex[2], simplex[0])]:
                edge = tuple(sorted(e))
                edge_count[edge] += 1

    # Boundary edges appear exactly once
    boundary_edges = [e for e, cnt in edge_count.items() if cnt == 1]
    if not boundary_edges:
        return None

    # Order boundary edges into a polygon
    from collections import deque
    adj = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)

    # Walk the boundary
    start = boundary_edges[0][0]
    ordered = [start]
    visited = {start}
    current = start
    while True:
        nexts = [n for n in adj[current] if n not in visited]
        if not nexts:
            break
        current = nexts[0]
        ordered.append(current)
        visited.add(current)

    if len(ordered) < 3:
        return None

    boundary = uv_pts[np.array(ordered)]
    if simplify_epsilon > 0:
        boundary = _douglas_peucker(boundary, simplify_epsilon)
    return boundary


def build_brep_shape_bop(occ_surfaces, clusters, surface_ids=None,
                          tolerance=1e-3, alpha=0.0, simplify_epsilon=0.0,
                          inflate=0.0):
    """
    Build a TopoDS_Shape by creating faces bounded by cluster UV boundaries.

    For each fitted surface, project cluster points into UV space, compute
    the boundary polygon (convex hull or alpha shape), map boundary UV points
    back to 3D, and build an OCC wire + face.

    Parameters
    ----------
    occ_surfaces   : list[Geom_Surface or None]
    clusters       : list[np.ndarray (N_i, 3)]  — cluster point clouds
    surface_ids    : list[int] or None
    tolerance      : float
    alpha          : float — alpha shape parameter (0 = convex hull)
    simplify_epsilon : float — Douglas-Peucker tolerance (0 = no simplification)
    inflate        : float — fractional inflation of UV boundary (0 = none)
    """
    n = len(occ_surfaces)

    occ_faces = []
    for i in range(n):
        surface = occ_surfaces[i]
        if surface is None:
            continue
        if i >= len(clusters) or len(clusters[i]) == 0:
            print(f"[bop] face {i}: no cluster points — skipping")
            continue

        dtype = surface.DynamicType().Name()
        try:
            # Project cluster points to UV
            uv = _project_to_uv(surface, clusters[i])
            if uv is None or len(uv) < 3:
                print(f"[bop] face {i} ({dtype}): UV projection failed — skipping")
                continue

            # Compute boundary polygon in UV
            boundary_uv = _alpha_shape_boundary(uv, alpha=alpha,
                                                    simplify_epsilon=simplify_epsilon)
            if boundary_uv is None or len(boundary_uv) < 3:
                print(f"[bop] face {i} ({dtype}): boundary extraction failed — skipping")
                continue

            # Inflate boundary outward from centroid
            if inflate > 0:
                centroid = boundary_uv.mean(axis=0)
                boundary_uv = centroid + (1.0 + inflate) * (boundary_uv - centroid)

            # Degree-3 B-spline with alpha-shape vertices as control points.
            # Rounds corners naturally without oscillation or closure gaps.
            # Clamped: last pole = first pole ensures exact geometric closure.
            n_pts = len(boundary_uv)
            deg = min(3, n_pts)
            n_poles = n_pts + 1
            n_interior = n_pts - deg
            n_knots = n_interior + 2
            poles = TColgp_Array1OfPnt2d(1, n_poles)
            for k, uv in enumerate(boundary_uv):
                poles.SetValue(k + 1, gp_Pnt2d(float(uv[0]), float(uv[1])))
            poles.SetValue(n_poles, gp_Pnt2d(float(boundary_uv[0][0]),
                                             float(boundary_uv[0][1])))
            knots_arr = TColStd_Array1OfReal(1, n_knots)
            for k in range(n_knots):
                knots_arr.SetValue(k + 1, float(k))
            mults_arr = TColStd_Array1OfInteger(1, n_knots)
            for k in range(n_knots):
                mults_arr.SetValue(k + 1, 1)
            mults_arr.SetValue(1, deg + 1)
            mults_arr.SetValue(n_knots, deg + 1)
            curve2d = Geom2d_BSplineCurve(poles, knots_arr, mults_arr, deg)
            edge = BRepBuilderAPI_MakeEdge(curve2d, surface).Edge()
            breplib.BuildCurves3d(edge)

            wire_builder = BRepBuilderAPI_MakeWire()
            wire_builder.Add(edge)
            if not wire_builder.IsDone():
                print(f"[bop] face {i} ({dtype}): wire construction failed")
                continue
            wire = wire_builder.Wire()

            # Build face: analytical surface bounded by wire
            face_maker = BRepBuilderAPI_MakeFace(surface, wire)
            if not face_maker.IsDone():
                print(f"[bop] face {i} ({dtype}): MakeFace failed "
                      f"(error {face_maker.Error()})")
                continue

            occ_faces.append(face_maker.Face())
            print(f"[bop] face {i} ({dtype}): {len(boundary_uv)} boundary pts "
                  f"({len(clusters[i])} cluster pts)")

        except Exception as exc:
            print(f"[bop] face {i} ({dtype}): exception: {exc}")

    if not occ_faces:
        print("[bop] no faces built — aborting")
        return None

    # Sew faces together
    print(f"[bop] sewing {len(occ_faces)} faces ...")
    sewing = BRepBuilderAPI_Sewing(tolerance)
    for face in occ_faces:
        sewing.Add(face)
    sewing.Perform()
    shape = sewing.SewedShape()

    if shape is None or shape.IsNull():
        print("[bop] sewing produced no shape")
        return None

    # Heal
    try:
        breplib.BuildCurves3d(shape)
    except Exception:
        pass
    fixer = ShapeFix_Shape(shape)
    fixer.SetPrecision(tolerance)
    fixer.Perform()
    shape = fixer.Shape()

    # Count output faces
    n_output = 0
    fexp = TopExp_Explorer(shape, TopAbs_FACE)
    while fexp.More():
        n_output += 1
        fexp.Next()

    analyzer = BRepCheck_Analyzer(shape)
    valid = analyzer.IsValid()
    print(f"[bop] BRep correctness: {valid}")
    if not valid:
        _print_brep_check_details(analyzer, shape)
    print(f"[bop] done — {n_output} faces, "
          f"shape type: {shape.ShapeType()}")
    return shape


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def print_edge_arcs_summary(edge_arcs):
    """Print a concise summary of the arc splitting result."""
    total_arcs = sum(len(v) for v in edge_arcs.values())
    print(f"[edge arcs] {len(edge_arcs)} edges → {total_arcs} arcs total")
    for edge_key, arcs in sorted(edge_arcs.items()):
        for idx, arc in enumerate(arcs):
            vs = arc["v_start"]
            ve = arc["v_end"]
            cl = "closed" if arc["closed"] else "open"
            print(
                f"  edge {edge_key}  arc[{idx}]  [{arc['t_start']:.4f}, {arc['t_end']:.4f}]"
                f"  v_start={vs}  v_end={ve}  {cl}"
            )
