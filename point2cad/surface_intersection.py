"""
Surface-surface intersection for the Point2CAD B-Rep pipeline.

Analytical formulas are used for:
    plane  ∩ plane  -> Geom_Line
    plane  ∩ sphere -> Geom_Circle  (or isolated gp_Pnt for tangency)
    sphere ∩ sphere -> Geom_Circle  (via the radical plane)

All other surface-type pairs fall back to GeomAPI_IntSS.
"""

import math
import numpy as np
from scipy.spatial import KDTree

try:
    from .surface_fitter import (
        SURFACE_PLANE, SURFACE_SPHERE, SURFACE_CYLINDER, SURFACE_CONE, SURFACE_INR,
        SURFACE_NAMES,
    )
    from .cluster_adjacency import adjacency_pairs, adjacency_triangles
except ImportError:
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from point2cad.surface_fitter import (
        SURFACE_PLANE, SURFACE_SPHERE, SURFACE_CYLINDER, SURFACE_CONE, SURFACE_INR,
        SURFACE_NAMES,
    )
    from point2cad.cluster_adjacency import adjacency_pairs, adjacency_triangles

try:
    from OCC.Core.gp          import gp_Pnt, gp_Dir, gp_Lin, gp_Circ, gp_Ax2
    from OCC.Core.Geom        import Geom_Line, Geom_Circle, Geom_TrimmedCurve
    from OCC.Core.GeomAPI     import GeomAPI_IntSS, GeomAPI_ProjectPointOnCurve, GeomAPI_ExtremaCurveCurve
    from OCC.Core.GeomAdaptor import GeomAdaptor_Curve
    from OCC.Core.GeomAbs     import (GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse,
                                       GeomAbs_Hyperbola, GeomAbs_Parabola,
                                       GeomAbs_BezierCurve, GeomAbs_BSplineCurve,
                                       GeomAbs_OtherCurve)
    from OCC.Core.Precision   import precision
    _OCC_INF = precision.Infinite()
    _GEOMABS_NAMES = {
        GeomAbs_Line: "Line", GeomAbs_Circle: "Circle", GeomAbs_Ellipse: "Ellipse",
        GeomAbs_Hyperbola: "Hyperbola", GeomAbs_Parabola: "Parabola",
        GeomAbs_BezierCurve: "Bezier", GeomAbs_BSplineCurve: "BSpline",
        GeomAbs_OtherCurve: "Other",
    }
    OCC_AVAILABLE = True
except ImportError as err:
    _OCC_INF       = float("inf")
    _GEOMABS_NAMES = {}
    OCC_AVAILABLE  = False

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unit(v):
    v = np.asarray(v, dtype = np.float64).ravel()
    return v / np.linalg.norm(v)

def _perp_to(n):
    """Return an arbitrary unit vector perpendicular to n."""
    n = _unit(n)
    t = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v = np.cross(n, t)
    return v / np.linalg.norm(v)

def _gp_pnt(p):
    return gp_Pnt(float(p[0]), float(p[1]), float(p[2]))

def _gp_dir(v):
    return gp_Dir(float(v[0]), float(v[1]), float(v[2]))

def _result(curves, points, curve_type, method):
    return {"curves": curves, "points": points, "type": curve_type, "method": method}


def _as_safe_curve(curve, model_extent=2.0):
    """
    Return the curve unchanged if it has finite parameter bounds, or wrapped in
    Geom_TrimmedCurve with analytically-derived finite bounds otherwise.

    Motivation: GeomAPI_ExtremaCurveCurve and GeomAPI_ProjectPointOnCurve
    internally call curve.FirstParameter() / LastParameter() (virtual dispatch)
    to initialise their numerical solvers, ignoring any explicit bounds passed
    in overloaded constructors.  For Geom_Hyperbola the native bounds are
    ±Precision::Infinite() (≈ ±2e100), causing cosh(2e100) → Standard_
    NumericError.  Wrapping in Geom_TrimmedCurve overrides those virtual calls
    at the C++ level so the solvers see only the safe interval.

    The trim bounds are derived analytically so the trimmed portion covers all
    geometry within a model of extent `model_extent` (default 2.0, generous for
    a unit-cube model):

      Geom_Hyperbola : t ∈ [−10, +10]   (cosh(10)≈11013, safe and model-size-independent)
      Geom_Parabola  : t ∈ [−max(L, 2√(fL)), +max(L, 2√(fL))]  f = Focal()
      Geom_Line      : t ∈ [−L, +L]  (arc-length parameterisation)
      other/unknown  : t ∈ [−10, +10]  (conservative fallback)

    See notes/brep_construction.md §2 for the derivations.
    """
    t0 = curve.FirstParameter()
    t1 = curve.LastParameter()
    if abs(t0) < _OCC_INF and abs(t1) < _OCC_INF:
        return curve   # already finite — no wrapping needed

    L       = model_extent
    adaptor = GeomAdaptor_Curve(curve)
    ctype   = adaptor.GetType()

    if ctype == GeomAbs_Hyperbola:
        # Use a flat conservative bound: cosh(10) ≈ 11013, sinh(10) ≈ 11013 —
        # completely safe from overflow and large enough to cover any vertex in a
        # unit-cube model regardless of the hyperbola's major radius.
        # The analytical arccosh(L/a) formula breaks when a > L because then
        # L/a < 1 and arccosh is undefined, collapsing the bound to ≈ 0.
        t_bound = 10.0
    elif ctype == GeomAbs_Parabola:
        f       = max(adaptor.Parabola().Focal(), 1e-10)
        t_bound = max(L, 2.0 * math.sqrt(f * L))
    elif ctype == GeomAbs_Line:
        t_bound = L
    else:
        t_bound = 10.0   # conservative fallback for unknown infinite-domain types

    return Geom_TrimmedCurve(curve, -t_bound, t_bound)


# ---------------------------------------------------------------------------
# Analytical intersections
# ---------------------------------------------------------------------------

def _intersect_plane_plane(pi, pj):
    n1 = _unit(pi["a"]);  d1 = float(pi["d"])
    n2 = _unit(pj["a"]);  d2 = float(pj["d"])

    dir_vec = np.cross(n1, n2)
    dir_norm = np.linalg.norm(dir_vec)
    if dir_norm < 1e-10:
        return _result([], [], "empty", "analytical")   # parallel planes
    dir_vec /= dir_norm

    # Minimum-norm point on the intersection line.
    # The line is the solution set of the 2x3 system  A p = b  (A = [n1; n2]).
    # The unique minimum-norm solution is  p0 = A^T (A A^T)^{-1} b.
    A  = np.stack([n1, n2])          # (2, 3)
    b  = np.array([d1, d2])
    p0, _, _, _ = np.linalg.lstsq(A, b, rcond = None)
    
    line = Geom_Line(gp_Lin(_gp_pnt(p0), _gp_dir(dir_vec)))
    return _result([line], [], "line", "analytical")


def _intersect_plane_sphere(pi, pj):
    n    = _unit(pi["a"]);  d = float(pi["d"])
    c    = np.asarray(pj["center"], dtype = np.float64)
    r    = float(pj["radius"])

    dist = float(n @ c) - d          # signed distance from sphere centre to plane

    if abs(dist) > r + 1e-10:
        return _result([], [], "empty", "analytical")

    if abs(abs(dist) - r) < 1e-10:
        pt = c - dist * n            # tangency point
        return _result([], [_gp_pnt(pt)], "tangent", "analytical")

    center   = c - dist * n          # foot of perpendicular from c onto the plane
    r_circle = math.sqrt(max(r * r - dist * dist, 0.0))
    x_dir    = _perp_to(n)
    ax2      = gp_Ax2(_gp_pnt(center), _gp_dir(n), _gp_dir(x_dir))
    circle   = Geom_Circle(gp_Circ(ax2, r_circle))
    return _result([circle], [], "circle", "analytical")


def _intersect_sphere_sphere(pi, pj):
    c1 = np.asarray(pi["center"], dtype = np.float64);  r1 = float(pi["radius"])
    c2 = np.asarray(pj["center"], dtype = np.float64);  r2 = float(pj["radius"])

    axis = c2 - c1
    d    = float(np.linalg.norm(axis))
    if d < 1e-10:
        return _result([], [], "empty", "analytical")

    # Spheres completely separated or one inside the other
    if d > r1 + r2 + 1e-10 or d < abs(r1 - r2) - 1e-10:
        return _result([], [], "empty", "analytical")

    n  = axis / d
    # Signed distance from c1 to the radical plane along n:
    #   h = (d^2 + r1^2 - r2^2) / (2d)
    h  = (d * d + r1 * r1 - r2 * r2) / (2.0 * d)
    p0 = c1 + h * n                  # centre of the intersection circle
    d_p = float(n @ p0)              # radical-plane offset (n . x = d_p)

    # Reuse plane-sphere: the radical plane cuts sphere 1 in the intersection circle
    return _intersect_plane_sphere({"a": n, "d": d_p}, pi)

# ---------------------------------------------------------------------------
# Plane ∩ Cylinder tangent fallback
# ---------------------------------------------------------------------------

def _intersect_plane_cylinder_tangent(pi_plane, pi_cyl,
                                       tol_parallel=0.001, tol_tangent=1e-3):
    """
    Analytical fallback for plane ∩ cylinder when GeomAPI_IntSS returns empty.

    Detects the degenerate tangent case: the plane is (nearly) parallel to the
    cylinder axis and tangent to its surface, so the intersection is a single
    generator (ruling) line.

    The intersection curve exists iff the perpendicular distance δ from the
    cylinder axis to the plane equals r:

        δ = |D| / |n_⊥|   where D = d − n·c,  n_⊥ = n − (n·a) a

    When δ = r the tangent foot on the cylinder surface is:

        q = c + (D / |n_⊥|²) · n_⊥

    and the result is Geom_Line(q, a).

    tol_parallel : |n·a| threshold below which the plane is declared parallel
    tol_tangent  : absolute tolerance on |δ − r|
    """
    n = _unit(pi_plane["a"])
    d = float(pi_plane["d"])
    a = _unit(pi_cyl["a"])
    c = np.asarray(pi_cyl["center"], dtype=np.float64)
    r = float(pi_cyl["radius"])

    # Plane must be (nearly) parallel to the cylinder axis
    alpha = float(np.dot(n, a))
    if abs(alpha) >= tol_parallel:
        return _result([], [], "empty", "analytical")

    # Component of n perpendicular to a
    n_perp    = n - alpha * a
    n_perp_sq = float(np.dot(n_perp, n_perp))
    if n_perp_sq < 1e-12:
        return _result([], [], "empty", "analytical")

    # Perpendicular distance from the cylinder axis to the plane
    D    = d - float(np.dot(n, c))          # d − n·c
    dist = abs(D) / math.sqrt(n_perp_sq)    # δ = |D| / |n_⊥|

    # Tangency: δ ≈ r  (secant δ < r is non-degenerate and handled by OCC)
    if abs(dist - r) > tol_tangent:
        return _result([], [], "empty", "analytical")

    # Tangent foot on the cylinder surface (lies on both the plane and cylinder)
    q = c + (D / n_perp_sq) * n_perp

    line = Geom_Line(gp_Lin(_gp_pnt(q), _gp_dir(a)))
    return _result([line], [], "line", "analytical")


# ---------------------------------------------------------------------------
# OCC generic fallback
# ---------------------------------------------------------------------------

def _intersect_occ(occ_surf_i, occ_surf_j, tol, label=""):
    # https://dev.opencascade.org/doc/refman/html/class_geom_a_p_i___int_s_s.html#details
    try:
        inter = GeomAPI_IntSS(occ_surf_i, occ_surf_j, tol)
    except Exception as e:
        return _result([], [], "failed", "occ")

    if not inter.IsDone():
        return _result([], [], "failed", "occ")

    curves = [inter.Line(k) for k in range(1, inter.NbLines() + 1)]

    if not curves:
        # GeomAPI_IntSS exposes no point query; NbLines()==0 after IsDone()
        # means either no intersection or point tangency (indistinguishable
        # at this API level).
        return _result([], [], "empty", "occ")

    prefix = f"[intersect] {label}: " if label else "[intersect] "
    for c in curves:
        t0, t1 = c.FirstParameter(), c.LastParameter()
        gtype  = _GEOMABS_NAMES.get(GeomAdaptor_Curve(c).GetType(), "?")
        t0_str = f"{t0:.4g}" if abs(t0) < _OCC_INF else "-inf"
        t1_str = f"{t1:.4g}" if abs(t1) < _OCC_INF else "+inf"
        print(f"  {prefix}{gtype}  t=[{t0_str}, {t1_str}]")

    return _result(curves, [], "curve", "occ")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def intersect_surfaces(surface_id_i, result_i, occ_surf_i,
                       surface_id_j, result_j, occ_surf_j,
                       tol = 1e-6, label = ""):
    """
    Compute the intersection between two adjacent surfaces.

    Uses analytical formulas for plane/plane, plane/sphere, sphere/sphere, and
    the plane/cylinder tangent degenerate case.
    All other pairs call GeomAPI_IntSS.

    Returns a dict:
        curves  : list[Geom_Curve]
        points  : list[gp_Pnt]    -- isolated tangency points
        type    : str             -- "line" | "circle" | "ellipse" | "conic" |
                                     "bspline" | "curve" | "tangent" |
                                     "empty" | "failed"
        method  : str             -- "analytical" | "occ"
    """
    si, sj   = surface_id_i, surface_id_j
    pi, pj   = result_i["params"], result_j["params"]
    oi, oj   = occ_surf_i, occ_surf_j

    # Normalise so that si <= sj (surfaces are type-indexed 0..4)
    if si > sj:
        si, sj = sj, si
        pi, pj = pj, pi
        oi, oj = oj, oi

    if si == SURFACE_PLANE  and sj == SURFACE_PLANE:
        return _intersect_plane_plane(pi, pj)
    if si == SURFACE_PLANE  and sj == SURFACE_SPHERE:
        return _intersect_plane_sphere(pi, pj)
    if si == SURFACE_SPHERE and sj == SURFACE_SPHERE:
        return _intersect_sphere_sphere(pi, pj)

    # Plane ∩ Cylinder: try OCC first (handles circle / ellipse correctly);
    # fall back to analytical tangent-line if OCC returns empty.
    # TODO: add plane ∩ cone tangent fallback when needed.
    if si == SURFACE_PLANE and sj == SURFACE_CYLINDER:
        occ = _intersect_occ(oi, oj, tol, label=label)
        if occ["type"] == "empty":
            return _intersect_plane_cylinder_tangent(pi, pj)
        return occ

    return _intersect_occ(oi, oj, tol, label=label)


def compute_all_intersections(adj, surface_ids, results, occ_surfaces, tol = 1e-6):
    """
    Compute intersections for every adjacent pair in adj.

    Args:
        adj          : (n, n) bool adjacency matrix
        surface_ids  : list[int]              -- SURFACE_* constant per cluster
        results      : list[dict]             -- fit result dict per cluster
        occ_surfaces : list[Geom_Surface]     -- OCC surface per cluster

    Returns:
        dict mapping (i, j) with i < j to an intersection result dict.
    """
    return {
        (i, j): intersect_surfaces(
            surface_ids[i], results[i], occ_surfaces[i],
            surface_ids[j], results[j], occ_surfaces[j],
            tol   = tol,
            label = f"({i},{j}) {SURFACE_NAMES[surface_ids[i]]}∩{SURFACE_NAMES[surface_ids[j]]}",
        )
        for i, j in adjacency_pairs(adj)
    }


def sample_curve(curve, boundary_pts=None, threshold=None,
                 n_points=200, extension_factor=0.15, line_extent=1.0):
    """
    Sample a Geom_Curve into an (n_points, 3) numpy array.

    When *boundary_pts* and *threshold* are provided the sample range is
    determined by projecting boundary_pts onto the curve and taking the span
    of the resulting parameters (extended by extension_factor on each side).
    This is used for **visualisation only** and does not affect B-Rep
    construction — it ensures that closed curves (e.g. full 2π ellipses from
    cylinder∩plane intersections) are displayed only over the arc that covers
    the actual shared boundary.

    Without boundary_pts the full parameter domain is sampled.  Infinite
    parameter bounds (Geom_Line) are clamped to ±line_extent so that
    np.linspace produces a usable range.
    """
    t0 = curve.FirstParameter()
    t1 = curve.LastParameter()
    is_inf = abs(t0) >= _OCC_INF or abs(t1) >= _OCC_INF

    if boundary_pts is not None and threshold is not None and len(boundary_pts) >= 2:
        params = []
        for pt in boundary_pts:
            proj = GeomAPI_ProjectPointOnCurve(
                gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2])), curve
            )
            # No distance filter: visualization-only path — just need the
            # parameter span, not geometric accuracy.
            if proj.NbPoints() > 0:
                params.append(proj.LowerDistanceParameter())

        if len(params) >= 2:
            ta   = float(min(params))
            tb   = float(max(params))
            span = tb - ta
            ta  -= extension_factor * span
            tb  += extension_factor * span
            if not is_inf:
                ta = max(ta, t0)
                tb = min(tb, t1)
            pts = np.zeros((n_points, 3))
            for i, t in enumerate(np.linspace(ta, tb, n_points)):
                p = curve.Value(t)
                pts[i] = (p.X(), p.Y(), p.Z())
            return pts

    # Fallback: full parameter domain (boundary projection unavailable or
    # yielded fewer than 2 projections within threshold).
    if is_inf:
        t0, t1 = -line_extent, line_extent
    pts = np.zeros((n_points, 3))
    for i, t in enumerate(np.linspace(t0, t1, n_points)):
        p = curve.Value(t)
        pts[i] = (p.X(), p.Y(), p.Z())
    return pts


# ---------------------------------------------------------------------------
# Curve trimming (vertex-based)
# ---------------------------------------------------------------------------


_CLOSURE_TOL = 1e-7   # mirrors topology.CLOSURE_TOL


def _curve_is_closed(curve):
    """True when C(t_min) and C(t_max) coincide — mirrors topology.curve_is_closed."""
    t0 = curve.FirstParameter()
    t1 = curve.LastParameter()
    if abs(t0) >= _OCC_INF or abs(t1) >= _OCC_INF:
        return False
    p0 = curve.Value(t0)
    p1 = curve.Value(t1)
    return math.sqrt(
        (p1.X() - p0.X()) ** 2 +
        (p1.Y() - p0.Y()) ** 2 +
        (p1.Z() - p0.Z()) ** 2
    ) < _CLOSURE_TOL


def trim_by_vertices(raw_intersections, vertices, vertex_edges,
                     extension_factor=0.05,
                     phantom_vertex_dist=1e-3):
    """
    Trim intersection curves using vertex parameters.

    * **Closed curves**: left unchanged — ``build_edge_arcs`` splits them at
      their vertices; trimming here would discard the wrap-around arc needed
      by the second adjacent face.

    * **Phantom filter**: when an edge has multiple open curves (e.g. OCC
      returns both generators of a cylinder for a plane∩cylinder pair),
      vertex attribution is edge-level so both curves see the same incident
      vertices.  The real curve has those vertices ON it (projection distance
      ≈ 0); the phantom generator is ~2·radius away.  Any open curve whose
      minimum vertex projection distance exceeds ``phantom_vertex_dist`` is
      discarded.  The filter is skipped when fewer than 2 vertices are
      available (fail-safe).

    * **Open curves with ≥ 2 incident vertices**: trimmed to
      ``[t_min − ext, t_max + ext]`` where ``t_min``/``t_max`` are the
      parameters obtained by projecting the incident vertices onto the curve
      and ``ext = extension_factor * span``.

    * **Open curves with < 2 incident vertices**:
        - Already compact (finite OCC bounds): kept as-is.
        - Infinite bounds: discarded (no vertex support to guide trimming).

    * Curves still infinite after trimming: discarded.

    Parameters
    ----------
    raw_intersections  : dict (i,j) -> result dict
    vertices           : (M, 3) float64 array
    vertex_edges       : list[set] of length M
    extension_factor   : relative extension applied to the trim interval
    phantom_vertex_dist: curves whose closest incident-vertex projection
                         distance exceeds this are discarded as phantoms
                         (default 1e-3; real curves have distance ≈ 0,
                         phantom generators have distance ≈ 2·radius)
    """
    trimmed = {}

    edge_to_vpos = {}
    for v_idx in range(len(vertices)):
        for edge in vertex_edges[v_idx]:
            edge_to_vpos.setdefault(edge, []).append(vertices[v_idx])

    for (i, j), inter in raw_intersections.items():
        vpositions = edge_to_vpos.get((i, j), [])

        # Split into closed and open curves
        closed_curves = []
        open_curves   = []
        for c in inter["curves"]:
            (closed_curves if _curve_is_closed(c) else open_curves).append(c)

        # ------------------------------------------------------------------
        # Phantom filter: for edges with multiple open curves, discard those
        # whose incident vertices are all far from the curve geometry.
        # ------------------------------------------------------------------
        if len(open_curves) > 1 and len(vpositions) >= 2:
            def _min_vertex_dist(c):
                best  = float("inf")
                c_s   = _as_safe_curve(c)
                for vpos in vpositions:
                    proj = GeomAPI_ProjectPointOnCurve(
                        gp_Pnt(float(vpos[0]), float(vpos[1]), float(vpos[2])), c_s
                    )
                    if proj.NbPoints() > 0:
                        best = min(best, float(proj.LowerDistance()))
                return best

            dists = [_min_vertex_dist(c) for c in open_curves]
            kept  = [c for c, d in zip(open_curves, dists) if d <= phantom_vertex_dist]
            open_curves = kept if kept else open_curves  # fail-safe

        # ------------------------------------------------------------------
        # Trim surviving open curves by vertex parameters
        # ------------------------------------------------------------------
        new_curves = list(closed_curves)

        for c in open_curves:
            t0_orig     = c.FirstParameter()
            t1_orig     = c.LastParameter()
            is_infinite = abs(t0_orig) >= _OCC_INF or abs(t1_orig) >= _OCC_INF

            trimmed_c = None
            if len(vpositions) >= 2:
                params = []
                c_safe = _as_safe_curve(c)
                for vpos in vpositions:
                    proj = GeomAPI_ProjectPointOnCurve(
                        gp_Pnt(float(vpos[0]), float(vpos[1]), float(vpos[2])), c_safe
                    )
                    if proj.NbPoints() > 0:
                        params.append(float(proj.LowerDistanceParameter()))

                if len(params) >= 2:
                    t_min = min(params)
                    t_max = max(params)
                    if t_max > t_min:
                        span   = t_max - t_min
                        t_min -= extension_factor * span
                        t_max += extension_factor * span
                        try:
                            trimmed_c = Geom_TrimmedCurve(c, t_min, t_max)
                        except Exception:
                            pass

            if trimmed_c is None:
                if is_infinite:
                    continue   # no vertex support, can't trim — discard
                trimmed_c = c  # compact OCC arc — keep as-is

            if (abs(trimmed_c.FirstParameter()) >= _OCC_INF or
                    abs(trimmed_c.LastParameter()) >= _OCC_INF):
                continue

            new_curves.append(trimmed_c)

        trimmed[(i, j)] = {
            **inter,
            "curves":       new_curves,
            "boundary_pts": np.empty((0, 3), dtype=np.float32),
        }

    return trimmed


def compute_vertices(adj, intersections, threshold = 1e-4):
    """
    Find B-Rep vertices: points where three or more surfaces simultaneously meet.

    For each triangle (i, j, k) in the adjacency graph, the three intersection
    curves C_ij, C_ik, C_jk all pass through a common triple-junction point.
    The vertex is found as the closest approach between pairs of curves that
    share a common surface: C_ij & C_ik share surface i, C_ij & C_jk share
    surface j, and C_ik & C_jk share surface k.

    All (ca, cb) combinations whose closest-approach distance is below
    `threshold` contribute a vertex candidate (midpoint of the closest points).
    Candidates within `threshold` of each other are merged by averaging.

    Parameters
    ----------
    adj           : (n, n) bool adjacency matrix
    intersections : dict (i, j) i<j -> result dict with "curves" list
    threshold     : distance threshold for both acceptance and deduplication
                    (use the adjacency threshold from compute_adjacency_matrix)

    Returns
    -------
    vertices     : (M, 3) float64 array of vertex positions, or (0, 3) if none
    vertex_edges : list[set] of length M; vertex_edges[v] is the set of edge
                   tuples (i, j) that vertex v is incident to
    """
    candidates      = []   # list of (M, 3) positions
    candidate_edges = []   # list of {ea, eb} edge sets, parallel to candidates

    edges = list(intersections.keys())   # each is (i, j) with i < j

    for idx_a in range(len(edges)):
        ea = edges[idx_a]
        curves_a = intersections[ea].get("curves", [])
        if not curves_a:
            continue

        for idx_b in range(idx_a + 1, len(edges)):
            eb = edges[idx_b]
            # Only consider pairs that share exactly one surface index
            if len(set(ea) & set(eb)) != 1:
                continue

            curves_b = intersections[eb].get("curves", [])
            if not curves_b:
                continue

            for ca in curves_a:
                for cb in curves_b:
                    try:
                        ca_s = _as_safe_curve(ca)
                        cb_s = _as_safe_curve(cb)
                        ext = GeomAPI_ExtremaCurveCurve(ca_s, cb_s)
                    except Exception as err:
                        print(f"[compute_vertices] ExtremaCurveCurve exception ({ea},{eb}): {err}")
                        continue
                    # First pass: find the global minimum distance.
                    best_dist = float("inf")
                    for m in range(1, ext.NbExtrema() + 1):
                        d = ext.Distance(m)
                        if d < best_dist:
                            best_dist = d
                    # Second pass: accept every extremum within a small
                    # absolute tolerance of the global minimum.  This
                    # rejects spurious vertices from phantom curves (where
                    # best_dist >> threshold) while correctly recovering all
                    # genuine intersection points (each at distance ≈ 0).
                    if best_dist < threshold:
                        for m in range(1, ext.NbExtrema() + 1):
                            if ext.Distance(m) <= best_dist + 1e-10:
                                try:
                                    t1, t2 = ext.Parameters(m)
                                except Exception:
                                    continue
                                p1 = ca.Value(t1)
                                p2 = cb.Value(t2)
                                midpoint = np.array([
                                    (p1.X() + p2.X()) / 2.0,
                                    (p1.Y() + p2.Y()) / 2.0,
                                    (p1.Z() + p2.Z()) / 2.0,
                                ])
                                candidates.append(midpoint)
                                candidate_edges.append({ea, eb})

    if not candidates:
        return np.empty((0, 3), dtype=np.float64), []

    # Greedy deduplication: merge candidates within threshold of each other
    candidates = np.array(candidates, dtype=np.float64)
    used = np.zeros(len(candidates), dtype=bool)
    merged_positions  = []
    merged_edge_sets  = []
    for idx in range(len(candidates)):
        if used[idx]:
            continue
        dists = np.linalg.norm(candidates - candidates[idx], axis=1)
        close = dists < threshold
        merged_positions.append(candidates[close].mean(axis=0))
        # Union of edge sets from all merged candidates
        merged_set = set()
        for ci in np.where(close)[0]:
            merged_set |= candidate_edges[ci]
        merged_edge_sets.append(merged_set)
        used[close] = True

    return np.array(merged_positions, dtype=np.float64), merged_edge_sets
