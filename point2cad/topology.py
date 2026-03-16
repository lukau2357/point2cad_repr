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

try:
    from OCC.Core.Geom           import Geom_TrimmedCurve
    from OCC.Core.Geom2dAPI      import Geom2dAPI_PointsToBSpline
    from OCC.Core.GeomAPI        import GeomAPI_ProjectPointOnCurve, GeomAPI_ProjectPointOnSurf
    from OCC.Core.GeomLProp      import GeomLProp_SLProps
    from OCC.Core.gp             import gp_Pnt, gp_Pnt2d, gp_GTrsf
    from OCC.Core.TColgp         import TColgp_Array1OfPnt2d
    from OCC.Core.BRep           import BRep_Builder, BRep_Tool
    from OCC.Core.TopExp         import TopExp_Explorer
    from OCC.Core.TopAbs         import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX, TopAbs_SHELL, TopAbs_SOLID
    from OCC.Core.TopoDS         import topods, TopoDS_Wire, TopoDS_Compound
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakeVertex,
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeWire,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakeSolid,
        BRepBuilderAPI_Sewing,
        BRepBuilderAPI_GTransform,
    )
    from OCC.Core.ShapeFix       import ShapeFix_Wire, ShapeFix_Shape
    from OCC.Core.BRepLib        import breplib
    from OCC.Core.STEPControl    import (STEPControl_Writer, STEPControl_AsIs,
                                          STEPControl_Reader)
    from OCC.Core.IFSelect       import IFSelect_RetDone
    from OCC.Core.BOPAlgo        import BOPAlgo_MakerVolume
    from OCC.Core.BRepCheck      import BRepCheck_Analyzer
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
                         bspline_method="uv_bounds", tolerance=1e-3,
                         cluster_bboxes=None):
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
    bspline_method : str
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
          - BRepCheck_Analyzer returns True (geometric correctness)
        """
        ea, verts, ve = _apply_removals(edge_arcs, vertices, vertex_edges,
                                        removed_v, removed_a)
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
            bspline_method=bspline_method,
        )
        valid = (info.get("valid", False) and
                 info.get("n_faces", 0) > 0)
        return valid, shape, info, ea, verts, ve

    # Try with everything first (minus isolated vertices)
    print("[oracle filter] trying full model ...")
    with contextlib.redirect_stdout(io.StringIO()):
        valid, shape, info, final_ea, final_v, final_ve = _try_build(
            removed_vertices, removed_arc_keys)
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

        with contextlib.redirect_stdout(io.StringIO()):
            valid, shape, info, ea, verts, ve = _try_build(
                removed_vertices, removed_arc_keys)

        n_faces = info.get("n_faces", 0)
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
    Whether the resulting wires are used for face construction depends on the
    `bspline_method` passed to `build_brep_shape`.

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

                    if len(candidates) > 1 and use_angular:
                        arc, v_cur, forward = _select_next_arc_angular(
                            v_cur, prev_arc, candidates,
                            face_idx, occ_surfaces, vertices,
                        )
                    else:
                        if len(candidates) > 1:
                            print(
                                f"[topology] face {face_idx}: vertex {v_cur}: "
                                f"{len(candidates)} candidates but angular ordering "
                                f"unavailable — using greedy first candidate"
                            )
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

def _build_pcurve_on_bspline(curve_3d, t0, t1, surface, n_samples=50, tolerance=1e-3):
    """
    Compute a Geom2d pcurve for the 3D curve segment [t0, t1] on a BSpline surface.

    Steps:
      1. Sample n_samples points evenly on curve_3d between t0 and t1.
      2. Project each 3D point onto the surface via GeomAPI_ProjectPointOnSurf
         (Newton iteration) to obtain (u_k, v_k).
      3. Fit a Geom2d_BSplineCurve through the (u_k, v_k) sequence via
         Geom2dAPI_PointsToBSpline (degree 3–8, C2 continuity).

    Returns the Geom2d_BSplineCurve on success, or None if fewer than 2
    points project successfully or if the 2D BSpline fit fails.
    """
    uv_pts = []
    for k in range(n_samples):
        t = t0 + (t1 - t0) * k / max(n_samples - 1, 1)
        try:
            p3d = curve_3d.Value(t)
            proj = GeomAPI_ProjectPointOnSurf(p3d, surface)
            if proj.NbPoints() > 0:
                u, v = proj.LowerDistanceParameters()
                uv_pts.append((float(u), float(v)))
        except Exception:
            pass

    if len(uv_pts) < 2:
        return None

    try:
        pts_arr = TColgp_Array1OfPnt2d(1, len(uv_pts))
        for i, (u, v) in enumerate(uv_pts):
            pts_arr.SetValue(i + 1, gp_Pnt2d(u, v))
        approx = Geom2dAPI_PointsToBSpline(pts_arr)
        if not approx.IsDone():
            return None
        return approx.Curve()
    except Exception:
        return None


def build_brep_shape(face_arcs, occ_surfaces, vertices, surface_ids=None,
                     face_wires=None, tolerance=1e-3,
                     inr_geom_close_tol=0.05, inr_arc_samples=30,
                     inr_uv_margin=0.02,
                     bspline_method="uv_bounds",
                     same_parameter=True,
                     orient_solid=True):
    """
    Build a TopoDS_Shape from face_arcs.

    Wire assembly
    -------------
    `face_wires` is the output of assemble_wires (angular ordering).  Each
    wire is built directly from the ordered (arc, forward) sequence — arc
    orientation is encoded by edge.Reversed() when forward=False.

    BSpline (INR) surfaces — two methods
    -------------------------------------
    bspline_method="uv_bounds"  (default)
        The face is built with explicit UV parameter bounds:
        `BRepBuilderAPI_MakeFace(surface, u_min, u_max, v_min, v_max, tolerance)`.
        The bounds are estimated by projecting sampled points from incident arcs
        onto the BSpline surface.  Geometric closure in each direction is
        detected and SetUPeriodic / SetVPeriodic called accordingly.
        OCC computes trivial iso-parameter pcurves automatically.
        Sewing bridges the gap between these iso-parameter edges and the exact
        intersection curve edges of adjacent analytical faces.
        Restriction: the face must have rectangular UV topology (at most two
        open boundary loops); fails for non-tubular or multi-boundary shapes.

    bspline_method="explicit_pcurve"
        The face is built from the intersection curve wires assembled by
        assemble_wires, exactly as for analytical faces.  After MakeFace,
        `breplib.BuildCurve2d` is called for every edge of the BSpline face to
        project the 3D intersection curves onto the UV domain and compute
        explicit pcurves via Newton iteration.  Topologically general — handles
        any number of boundary loops — but more expensive and can fail if Newton
        diverges near surface seams or low-curvature regions.

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
        (uv_bounds only) Maximum 3D distance between opposite BSpline boundary
        curves for a parametric direction to be declared geometrically closed.
        Default 0.05 (5% of unit-normalised scale).
    inr_arc_samples   : int
        (uv_bounds only) Number of points sampled per arc for UV bound estimation.
        Default 30.
    inr_uv_margin     : float
        (uv_bounds only) Relative margin added to projected UV bounds on each side.
        Default 0.02 (2%).
    bspline_method    : str
        How BSpline (INR) faces are constructed.  "uv_bounds" (default) or
        "explicit_pcurve".

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

        # Shared INR setup: geometric closure detection and periodisation.
        # Runs before both bspline_method branches so that the surface is
        # already periodic (if applicable) when either branch builds its face.
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

        # BSpline (INR) face — explicit pcurve method:
        # Build the face from assembled intersection-curve wires (same as
        # analytical faces) on the already-periodic surface.  Then for every
        # edge in the resulting face, extract its 3D curve via BRep_Tool.Curve,
        # compute a pcurve by sampling + Newton projection onto the BSpline UV
        # domain + Geom2d BSpline fit (_build_pcurve_on_bspline), and attach
        # it via BRep_Builder.UpdateEdge.
        if is_inr and bspline_method == "explicit_pcurve":
            if not occ_wires:
                print(f"[brep] face {face_idx}: BSpline explicit-pcurve "
                      f"— no wires assembled, skipping")
                continue
            try:
                face_maker = BRepBuilderAPI_MakeFace(surface, occ_wires[0])
                for inner in occ_wires[1:]:
                    face_maker.Add(inner)
                if not face_maker.IsDone():
                    print(f"[brep] face {face_idx}: BSpline explicit-pcurve "
                          f"MakeFace failed (error {face_maker.Error()})")
                    continue
                face = face_maker.Face()
                brep_builder = BRep_Builder()
                n_ok, n_fail = 0, 0
                seen_edges = set()
                explorer = TopExp_Explorer(face, TopAbs_EDGE)
                while explorer.More():
                    edge = topods.Edge(explorer.Current())
                    eid = edge.__hash__()
                    if eid not in seen_edges:
                        seen_edges.add(eid)
                        try:
                            crv, t0, t1 = BRep_Tool.Curve(edge)
                            if crv is None:
                                n_fail += 1
                            else:
                                pcurve = _build_pcurve_on_bspline(
                                    crv, t0, t1, surface,
                                    n_samples=50, tolerance=tolerance,
                                )
                                if pcurve is not None:
                                    brep_builder.UpdateEdge(
                                        edge, pcurve, face, tolerance
                                    )
                                    n_ok += 1
                                else:
                                    print(
                                        f"[brep] face {face_idx}: "
                                        f"_build_pcurve_on_bspline returned "
                                        f"None for an edge"
                                    )
                                    n_fail += 1
                        except Exception as exc:
                            print(f"[brep] face {face_idx}: pcurve "
                                  f"computation failed: {exc}")
                            n_fail += 1
                    explorer.Next()
                print(f"[brep] face {face_idx}: BSpline explicit-pcurve "
                      f"pcurves ok={n_ok} fail={n_fail}")
                sewing.Add(face)
            except Exception as exc:
                print(f"[brep] face {face_idx}: BSpline explicit-pcurve "
                      f"exception: {exc}")
            continue

        # BSpline (INR) face — UV-bounds method:
        # Build a parameter-bounded face from the BSpline's (now possibly
        # periodic) UV domain, clipped to the extent of incident arcs.
        if is_inr:
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
    print(f"[brep] Output faces: {n_output_faces}/{n_input_faces}")

    result_info = {
        "valid": eval_results,
        "n_faces": n_output_faces,
        "n_input_faces": n_input_faces,
    }
    return shape, result_info

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

    mat = float(scale) * np.asarray(R, dtype=np.float64).T  # 3x3: scale * R^T
    mean = np.asarray(mean, dtype=np.float64)

    trsf = gp_GTrsf()
    for i in range(3):
        for j in range(3):
            trsf.SetValue(i + 1, j + 1, mat[i, j])
        trsf.SetValue(i + 1, 4, mean[i])

    result = BRepBuilderAPI_GTransform(shape, trsf, True)
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


def build_brep_shape_bop(occ_surfaces, vertices, vertex_edges, face_arcs,
                          surface_ids=None,
                          tolerance=1e-3, rel_margin=0.05,
                          curve_samples=100):
    """
    Build a TopoDS_Shape using BOPAlgo_MakerVolume.

    Face patch UV bounds are determined from the intersection geometry:

      1. **Vertices exist** for the face: project incident vertices onto the
         surface → UV bounding box (cheapest, most precise).
      2. **No vertices, but closed curves** on the face: sample points along
         incident closed curves → UV bounding box.
      3. **Geom_BSplineSurface**: uses the natural UV domain.

    UV bounds are clipped to surface.Bounds() to prevent wrapping past
    periodic boundaries.

    Parameters
    ----------
    occ_surfaces   : list[Geom_Surface or None]
    vertices       : np.ndarray (M, 3)
    vertex_edges   : list[set]
    face_arcs      : dict i -> list[arc_dict]
    surface_ids    : list[int] or None
    tolerance      : float
    rel_margin     : float — relative UV margin (default 5%)
    curve_samples  : int — samples per closed curve for fallback UV bounds
    """
    n = len(occ_surfaces)

    # Per-face incident vertices from surviving arcs.
    face_vertex_positions = {}
    for face_idx, arcs in face_arcs.items():
        vset = set()
        for arc in arcs:
            if arc.get("v_start") is not None:
                vset.add(arc["v_start"])
            if arc.get("v_end") is not None:
                vset.add(arc["v_end"])
        if vset:
            face_vertex_positions[face_idx] = [vertices[v] for v in vset]

    occ_faces = []
    for i in range(n):
        surface = occ_surfaces[i]
        if surface is None:
            continue

        dtype = surface.DynamicType().Name()
        face = None
        try:
            # Tier 1: incident vertices.
            vpos_list = face_vertex_positions.get(i)
            if vpos_list:
                bounds = _uv_bounds_from_3d_points(
                    surface, np.array(vpos_list), rel_margin=rel_margin
                )
                if bounds is not None:
                    umin, umax, vmin, vmax = bounds
                    fm = BRepBuilderAPI_MakeFace(surface, umin, umax,
                                                 vmin, vmax, tolerance)
                    if fm.IsDone():
                        face = fm.Face()
                        print(f"[bop] face {i}: UV from {len(vpos_list)} "
                              f"vertices [{umin:.3f},{umax:.3f}]×"
                              f"[{vmin:.3f},{vmax:.3f}]")

            # Tier 2: sample closed curves.
            if face is None:
                arcs = face_arcs.get(i, [])
                sample_pts = []
                for arc in arcs:
                    if not arc.get("closed", False):
                        continue
                    t0, t1 = arc["t_start"], arc["t_end"]
                    for k in range(curve_samples):
                        t = t0 + (t1 - t0) * k / max(curve_samples - 1, 1)
                        try:
                            p = arc["curve"].Value(t)
                            sample_pts.append([p.X(), p.Y(), p.Z()])
                        except Exception:
                            pass
                if sample_pts:
                    bounds = _uv_bounds_from_3d_points(
                        surface, np.array(sample_pts),
                        rel_margin=rel_margin
                    )
                    if bounds is not None:
                        umin, umax, vmin, vmax = bounds
                        fm = BRepBuilderAPI_MakeFace(
                            surface, umin, umax, vmin, vmax, tolerance
                        )
                        if fm.IsDone():
                            face = fm.Face()
                            print(f"[bop] face {i}: UV from "
                                  f"{len(sample_pts)} curve samples "
                                  f"[{umin:.3f},{umax:.3f}]×"
                                  f"[{vmin:.3f},{vmax:.3f}]")

            if face is not None:
                occ_faces.append(face)
            else:
                print(f"[bop] face {i}: could not build patch — skipping")
        except Exception as exc:
            print(f"[bop] face {i} ({dtype}): exception: {exc}")

    if not occ_faces:
        print("[bop] no faces built — aborting")
        return None

    print(f"[bop] built {len(occ_faces)} face patches, "
          f"running BOPAlgo_MakerVolume ...")
    mv = BOPAlgo_MakerVolume()
    mv.SetIntersect(True)
    for face in occ_faces:
        mv.AddArgument(face)
    mv.Perform()

    if mv.HasErrors():
        print("[bop] BOPAlgo_MakerVolume reported errors")
        return None

    shape = mv.Shape()
    analyzer = BRepCheck_Analyzer(shape)
    eval_result = analyzer.IsValid()
    print(f"[bop] BRep correctness: {eval_result}")
    print(f"[bop] MakerVolume done — shape type: {shape.ShapeType()}")
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
