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
    from OCC.Core.gp        import gp_Pnt, gp_Dir, gp_Lin, gp_Circ, gp_Ax2
    from OCC.Core.Geom      import Geom_Line, Geom_Circle, Geom_TrimmedCurve
    from OCC.Core.GeomAPI   import GeomAPI_IntSS, GeomAPI_ProjectPointOnCurve, GeomAPI_ExtremaCurveCurve
    from OCC.Core.Precision import precision
    _OCC_INF     = precision.Infinite()
    OCC_AVAILABLE = True
except ImportError as err:
    _OCC_INF      = float("inf")
    OCC_AVAILABLE = False

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

def _curve_sample_range(curve, cluster_pts):
    """Return a finite (t0, t1) sampling range for a curve.

    For compact curves the existing parameter bounds are used.
    For infinite curves (Geom_Line from plane∩plane) the range is clamped to
    a bounding-box diagonal centred on the projection of the cluster centroid
    onto the curve.
    """
    t0 = curve.FirstParameter()
    t1 = curve.LastParameter()
    if abs(t0) < _OCC_INF and abs(t1) < _OCC_INF:
        return float(t0), float(t1)
    # Infinite bounds: project the cluster centroid onto the line to find a
    # sensible centre parameter, then span ± bbox diagonal from there.
    bbox_min = cluster_pts.min(axis=0)
    bbox_max = cluster_pts.max(axis=0)
    diag     = float(np.linalg.norm(bbox_max - bbox_min))
    centroid = cluster_pts.mean(axis=0)
    proj = GeomAPI_ProjectPointOnCurve(
        gp_Pnt(float(centroid[0]), float(centroid[1]), float(centroid[2])), curve
    )
    t_center = float(proj.LowerDistanceParameter()) if proj.NbPoints() > 0 else 0.0
    return t_center - diag, t_center + diag

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
# OCC generic fallback
# ---------------------------------------------------------------------------

def _intersect_occ(occ_surf_i, occ_surf_j, tol):
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

    return _result(curves, [], "curve", "occ")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def intersect_surfaces(surface_id_i, result_i, occ_surf_i,
                       surface_id_j, result_j, occ_surf_j,
                       tol = 1e-6):
    """
    Compute the intersection between two adjacent surfaces.

    Uses analytical formulas for plane/plane, plane/sphere and sphere/sphere.
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

    return _intersect_occ(oi, oj, tol)


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
            tol = tol,
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
# Curve trimming
# ---------------------------------------------------------------------------

def trim_curve(curve, boundary_pts, threshold, extension_factor = 0.05):
    """
    Trim a Geom_Curve to the portion covering the shared boundary.

    Only curves with at least one infinite parameter bound are trimmed.
    Curves with a compact (finite) parameter domain — such as Geom_TrimmedCurve
    returned by GeomAPI_IntSS or Geom_Circle — are returned unchanged; they are
    continuous functions on a compact set and their existing bounds are
    already well-defined.

    For infinite curves (e.g. Geom_Line from plane∩plane), the trim interval
    [t_min, t_max] is computed by projecting boundary_pts onto the curve and
    taking the span of the projected parameters.  The interval is then extended
    symmetrically by extension_factor * span on each side to avoid gaps at the
    trimmed endpoints.

    Parameters
    ----------
    curve            : Geom_Curve
    boundary_pts     : (N, 3) float array — boundary strip (from compute_adjacency_matrix)
    threshold        : adjacency threshold used as projection tolerance
    extension_factor : relative extension applied symmetrically to the trim
                       interval (default 0.05 → 5 % of span on each side)
    """
    t0 = curve.FirstParameter()
    t1 = curve.LastParameter()

    # Compact parameter space → leave the curve unchanged
    if abs(t0) < _OCC_INF and abs(t1) < _OCC_INF:
        return curve

    if len(boundary_pts) == 0:
        return curve

    params = []
    for pt in boundary_pts:
        # https://dev.opencascade.org/doc/refman/html/class_geom2d_a_p_i___project_point_on_curve.html
        proj = GeomAPI_ProjectPointOnCurve(
            gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2])), curve
        )
        if proj.NbPoints() > 0 and proj.LowerDistance() <= threshold:
            params.append(proj.LowerDistanceParameter())

    if len(params) < 2:
        return curve

    t_min = float(min(params))
    t_max = float(max(params))

    if t_max <= t_min:
        return curve

    span   = t_max - t_min
    t_min -= extension_factor * span
    t_max += extension_factor * span

    try:
        return Geom_TrimmedCurve(curve, t_min, t_max)
    except Exception:
        return curve


def trim_intersections(intersections, clusters,
                       boundary_strips=None, threshold=None,
                       per_pair_thresholds=None,
                       k_real=4.0, k_strip=1.8, n_samples=80,
                       extension_factor=0.05):
    """
    Trim intersection curves to their shared boundary portions.

    Two modes are supported, selected by whether *boundary_strips* is provided:

    **Constrained adjacency** (``boundary_strips`` is not None)
        Original flow, unchanged.  Pre-computed boundary strips from
        ``compute_adjacency_matrix`` are used directly with ``trim_curve``.
        All raw curves are kept; no phantom filtering is performed.

    **Full adjacency** (``boundary_strips`` is None)
        New flow.  For each pair (i, j) a per-pair reference spacing is
        computed as the median NN distance from the smaller cluster to the
        larger cluster.  Each curve is sampled at *n_samples* points and
        queried against ``clusters[i] ∪ clusters[j]``:

        - ``d.min() > k_real * pair_spacing`` → phantom curve, discard.
        - Otherwise cluster points within ``k_strip * pair_spacing`` of the
          curve become the boundary strip for trimming.

        Any curve still infinite after trimming is also discarded.

    In both modes every result dict in the returned mapping gains a
    ``"boundary_pts"`` key: the (M, 3) float32 array of boundary points for
    that pair (empty when none exist).

    Parameters
    ----------
    intersections    : dict (i, j) -> result dict (from compute_all_intersections)
    clusters         : list of (Ni, 3) float arrays — one per surface cluster
    boundary_strips  : dict (i, j) -> (N, 3) float32, or None for full adjacency
    threshold        : adjacency threshold used by the constrained flow
    k_real           : phantom detection multiplier  [full adjacency only]
    k_strip          : boundary-strip multiplier      [full adjacency only]
    n_samples        : sample points per curve        [full adjacency only]
    extension_factor : relative extension passed to trim_curve
    """
    trimmed = {}

    # ------------------------------------------------------------------
    # Constrained adjacency: original flow
    # ------------------------------------------------------------------
    if boundary_strips is not None:
        for (i, j), inter in intersections.items():
            bpts = boundary_strips.get((i, j), np.empty((0, 3), dtype=np.float32))
            # Use per-pair threshold for projection check if available,
            # otherwise fall back to the global threshold.
            thr = (per_pair_thresholds.get((i, j), threshold)
                   if per_pair_thresholds is not None else threshold)
            new_curves = []
            for c in inter["curves"]:
                tc = trim_curve(c, bpts, thr, extension_factor)
                if abs(tc.FirstParameter()) >= _OCC_INF or abs(tc.LastParameter()) >= _OCC_INF:
                    continue  # still infinite after trimming — phantom, discard
                new_curves.append(tc)
            trimmed[(i, j)] = {
                **inter,
                "curves":       new_curves,
                "boundary_pts": np.asarray(bpts, dtype=np.float32),
            }
        return trimmed

    # ------------------------------------------------------------------
    # Full adjacency: phantom filtering + proximity-based boundary strips
    # ------------------------------------------------------------------
    for (i, j), inter in intersections.items():
        if not inter["curves"]:
            trimmed[(i, j)] = {**inter, "boundary_pts": np.empty((0, 3), dtype=np.float32)}
            continue

        ci, cj          = clusters[i], clusters[j]
        larger, smaller = (ci, cj) if len(ci) >= len(cj) else (cj, ci)
        inter_tree      = KDTree(larger)
        inter_dists, _  = inter_tree.query(smaller, k=1)
        pair_spacing    = float(np.median(inter_dists))

        threshold_real  = k_real  * pair_spacing
        threshold_strip = k_strip * pair_spacing

        cluster_pts = np.vstack([ci, cj]).astype(np.float64)
        tree        = KDTree(cluster_pts)

        keep_curves = []
        all_bnd_idx = set()

        for curve in inter["curves"]:
            t0, t1 = _curve_sample_range(curve, cluster_pts)
            sample_pts = np.zeros((n_samples, 3), dtype=np.float64)
            for k, t in enumerate(np.linspace(t0, t1, n_samples)):
                p = curve.Value(t)
                sample_pts[k] = (p.X(), p.Y(), p.Z())

            dists, idx = tree.query(sample_pts, k=1)

            if float(dists.min()) > threshold_real:
                continue   # phantom — discard

            close_mask    = dists <= threshold_strip
            curve_bnd_idx = np.unique(idx[close_mask])
            all_bnd_idx.update(curve_bnd_idx.tolist())

            curve_bpts = cluster_pts[curve_bnd_idx].astype(np.float32)
            trimmed_c  = trim_curve(curve, curve_bpts, threshold_strip, extension_factor)

            tc0 = trimmed_c.FirstParameter()
            tc1 = trimmed_c.LastParameter()
            if abs(tc0) >= _OCC_INF or abs(tc1) >= _OCC_INF:
                continue   # still infinite after trimming — discard

            keep_curves.append(trimmed_c)

        bnd_arr = (cluster_pts[sorted(all_bnd_idx)].astype(np.float32)
                   if all_bnd_idx else np.empty((0, 3), dtype=np.float32))

        trimmed[(i, j)] = {**inter, "curves": keep_curves, "boundary_pts": bnd_arr}

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
                        ext = GeomAPI_ExtremaCurveCurve(ca, cb)
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
                                t1, t2 = ext.Parameters(m)
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
