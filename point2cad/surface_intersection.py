"""
Surface-surface intersection for the Point2CAD B-Rep pipeline.

Analytical formulas are used for:
    plane    ∩ plane    -> Geom_Line
    plane    ∩ sphere   -> Geom_Circle  (or isolated gp_Pnt for tangency)
    sphere   ∩ sphere   -> Geom_Circle  (via the radical plane)
    plane    ∩ cylinder -> Geom_Line    (tangent to axis, OCC returns empty)
    cylinder ∩ cylinder -> Geom_Line(s) (parallel axes, OCC returns IsDone=False)

All other surface-type pairs fall back to GeomAPI_IntSS.
"""

import math
import numpy as np
from scipy.spatial import KDTree

try:
    from .surface_types import (
        SURFACE_PLANE, SURFACE_SPHERE, SURFACE_CYLINDER, SURFACE_CONE, SURFACE_INR,
        SURFACE_NAMES,
    )
    from .cluster_adjacency import adjacency_pairs, adjacency_triangles
except ImportError:
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from point2cad.surface_types import (
        SURFACE_PLANE, SURFACE_SPHERE, SURFACE_CYLINDER, SURFACE_CONE, SURFACE_INR,
        SURFACE_NAMES,
    )
    from point2cad.cluster_adjacency import adjacency_pairs, adjacency_triangles

try:
    from OCC.Core.gp          import gp_Pnt, gp_Dir, gp_Lin, gp_Circ, gp_Ax2
    from OCC.Core.Geom        import Geom_Line, Geom_Circle, Geom_TrimmedCurve
    from OCC.Core.GeomAPI     import (GeomAPI_IntSS, GeomAPI_IntCS,
                                      GeomAPI_ProjectPointOnCurve,
                                      GeomAPI_ProjectPointOnSurf)
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


def _intersect_cylinder_cylinder_parallel(pi, pj,
                                           tol_parallel=0.001, tol_tangent=1e-3):
    """
    Analytical fallback for cylinder ∩ cylinder when axes are parallel.

    Two infinite cylinders with parallel axes intersect in 0, 1, or 2
    generator lines.  Project both axes onto the plane perpendicular to
    the common direction **a**: the problem reduces to intersecting two
    circles (radii r₁, r₂, centres q₁, q₂) in 2D.

    Let d = |q₂ − q₁|.  Then:
        d > r₁ + r₂         → no intersection
        d = r₁ + r₂         → 1 line (external tangent)
        |r₁ − r₂| < d < r₁ + r₂  → 2 lines (secant)
        d = |r₁ − r₂|       → 1 line (internal tangent)
        d < |r₁ − r₂|       → no intersection (one inside the other)

    Each 2D intersection point p₂d lifts to Geom_Line(c₁ + proj + p₂d, a).
    """
    a1 = _unit(pi["a"])
    a2 = _unit(pj["a"])
    c1 = np.asarray(pi["center"], dtype=np.float64)
    c2 = np.asarray(pj["center"], dtype=np.float64)
    r1 = float(pi["radius"])
    r2 = float(pj["radius"])

    # Check axes are parallel
    dot = abs(float(np.dot(a1, a2)))
    if dot < 1.0 - tol_parallel:
        return None   # not parallel — caller should use OCC

    # Use a1 as the common axis direction
    a = a1

    # Project centres onto plane perpendicular to a
    # q_i = c_i − (c_i · a) a   (perpendicular component)
    q1 = c1 - float(np.dot(c1, a)) * a
    q2 = c2 - float(np.dot(c2, a)) * a

    diff = q2 - q1
    d = float(np.linalg.norm(diff))

    if d < 1e-12:
        # Coaxial cylinders — intersection is empty (different r) or
        # degenerate (same r, handled by equiv).
        return _result([], [], "empty", "analytical")

    e = diff / d   # unit vector from q1 to q2 in the perp plane

    # Circle–circle intersection:
    #   x = (d² + r₁² − r₂²) / (2d)     (distance from q1 along e)
    #   h² = r₁² − x²                     (perpendicular offset²)
    x = (d * d + r1 * r1 - r2 * r2) / (2.0 * d)
    h_sq = r1 * r1 - x * x

    if h_sq < -tol_tangent * max(r1, r2):
        return _result([], [], "empty", "analytical")

    # Build the perpendicular direction in the plane ⊥ a
    # e is in the perp plane; need a vector ⊥ both a and e
    f = np.cross(a, e)
    f = f / np.linalg.norm(f)

    # Base point in the perp plane (relative to c1)
    base = q1 + x * e

    if h_sq <= tol_tangent * max(r1, r2):
        # Tangent case — single generator line
        # Reconstruct 3D point: base is already in perp plane,
        # add the a-component from c1
        p3d = base + float(np.dot(c1, a)) * a
        line = Geom_Line(gp_Lin(_gp_pnt(p3d), _gp_dir(a)))
        return _result([line], [], "line", "analytical")

    # Secant case — two generator lines
    h = math.sqrt(h_sq)
    a_comp = float(np.dot(c1, a)) * a

    p1_3d = base + h * f + a_comp
    p2_3d = base - h * f + a_comp

    line1 = Geom_Line(gp_Lin(_gp_pnt(p1_3d), _gp_dir(a)))
    line2 = Geom_Line(gp_Lin(_gp_pnt(p2_3d), _gp_dir(a)))
    return _result([line1, line2], [], "line", "analytical")


# ---------------------------------------------------------------------------
# OCC generic fallback
# ---------------------------------------------------------------------------

def _intersect_occ(occ_surf_i, occ_surf_j, tol, label=""):
    # https://dev.opencascade.org/doc/refman/html/class_geom_a_p_i___int_s_s.html#details
    try:
        inter = GeomAPI_IntSS(occ_surf_i, occ_surf_j, tol)
    except Exception as e:
        if label:
            print(f"  [intersect] {label}: FAILED (exception: {e})")
        return _result([], [], "failed", "occ")

    if not inter.IsDone():
        if label:
            print(f"  [intersect] {label}: FAILED (IsDone=False)")
        return _result([], [], "failed", "occ")

    curves = [inter.Line(k) for k in range(1, inter.NbLines() + 1)]

    if not curves:
        # GeomAPI_IntSS exposes no point query; NbLines()==0 after IsDone()
        # means either no intersection or point tangency (indistinguishable
        # at this API level).
        if label:
            print(f"  [intersect] {label}: empty (NbLines=0)")
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
                       tol=1e-6, label=""):
    """
    Compute the intersection between two adjacent surfaces.

    Uses analytical formulas for plane/plane, plane/sphere, sphere/sphere,
    the plane/cylinder tangent degenerate case, and cylinder/cylinder with
    parallel axes (where OCC often fails with IsDone=False).
    All other pairs call GeomAPI_IntSS.

    Returns a dict:
        curves  : list[Geom_Curve]
        points  : list[gp_Pnt]    -- isolated tangency points
        type    : str             -- "line" | "circle" | "ellipse" | "conic" |
                                     "bspline" | "curve" | "tangent" |
                                     "empty" | "failed"
        method  : str             -- "analytical" | "occ"
    """
    si, sj = surface_id_i, surface_id_j
    pi, pj = result_i["params"], result_j["params"]
    oi, oj = occ_surf_i, occ_surf_j

    # Normalise so that si <= sj (surfaces are type-indexed 0..4)
    if si > sj:
        si, sj = sj, si
        pi, pj = pj, pi
        oi, oj = oj, oi

    if si == SURFACE_PLANE and sj == SURFACE_PLANE:
        return _intersect_plane_plane(pi, pj)
    if si == SURFACE_PLANE and sj == SURFACE_SPHERE:
        return _intersect_plane_sphere(pi, pj)
    if si == SURFACE_SPHERE and sj == SURFACE_SPHERE:
        return _intersect_sphere_sphere(pi, pj)

    # Plane ∩ Cylinder: try OCC first (handles circle / ellipse correctly);
    # fall back to analytical tangent-line if OCC returns empty.
    # TODO: add plane ∩ cone tangent fallback when needed.
    if si == SURFACE_PLANE and sj == SURFACE_CYLINDER:
        # Always check the analytical tangent condition first.
        # OCC sometimes returns a degenerate ellipse for near-tangent cases
        # (plane barely secant to cylinder, δ ≈ r) instead of returning empty.
        tangent = _intersect_plane_cylinder_tangent(pi, pj)
        if tangent["type"] != "empty":
            return tangent
        occ = _intersect_occ(oi, oj, tol, label=label)
        if occ["type"] == "empty":
            return _intersect_plane_cylinder_tangent(pi, pj)
        return occ

    # Cylinder ∩ Cylinder: try OCC first; fall back to analytical parallel-
    # axis formula if OCC fails (IsDone=False) or returns empty.
    if si == SURFACE_CYLINDER and sj == SURFACE_CYLINDER:
        occ = _intersect_occ(oi, oj, tol, label=label)
        if occ["type"] in ("failed", "empty"):
            analytical = _intersect_cylinder_cylinder_parallel(pi, pj)
            if analytical is not None and analytical["type"] != "empty":
                if label:
                    for c in analytical["curves"]:
                        t0, t1 = c.FirstParameter(), c.LastParameter()
                        t0s = f"{t0:.4g}" if abs(t0) < _OCC_INF else "-inf"
                        t1s = f"{t1:.4g}" if abs(t1) < _OCC_INF else "+inf"
                        print(f"  [intersect] {label}: Line (analytical)  t=[{t0s}, {t1s}]")
                return analytical
        return occ

    return _intersect_occ(oi, oj, tol, label=label)


def find_equivalent_surfaces(adj, surface_ids, results,
                             angle_tol=1e-2, dist_tol=1e-2, radius_tol=1e-2,
                             theta_tol=1e-2):
    """
    Find adjacent clusters that share the same underlying surface.

    Two adjacent clusters are considered equivalent if they have the same
    surface type and their fitted parameters are close enough:
      - Plane:    normals nearly parallel AND offsets close
      - Cylinder: axes nearly parallel, radii close, AND axis lines close
      - Sphere:   centres close AND radii close
      - Cone:     axes nearly parallel, apexes close, AND half-angles close

    Returns a canonical-index map: canon[i] = j means cluster i is equivalent
    to cluster j (j <= i).  Non-equivalent clusters map to themselves.
    """
    n = adj.shape[0]
    canon = list(range(n))

    def _find(x):
        while canon[x] != x:
            canon[x] = canon[canon[x]]
            x = canon[x]
        return x

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            lo, hi = (ra, rb) if ra < rb else (rb, ra)
            canon[hi] = lo

    for i, j in adjacency_pairs(adj):
        if surface_ids[i] != surface_ids[j]:
            continue
        sid = surface_ids[i]
        pi, pj = results[i]["params"], results[j]["params"]

        # Planes are non-periodic — never seam-split by OCC, so no
        # face merging needed.  Skip to avoid false-positive merges.
        if sid == SURFACE_PLANE:
            continue

        elif sid == SURFACE_CYLINDER:
            ai, aj = _unit(pi["a"]), _unit(pj["a"])
            dot = abs(float(np.dot(ai, aj)))
            ri, rj = float(pi["radius"]), float(pj["radius"])
            ci = np.asarray(pi["center"], dtype=np.float64)
            cj = np.asarray(pj["center"], dtype=np.float64)
            diff = cj - ci
            perp = diff - float(np.dot(diff, ai)) * ai
            d_angle = 1.0 - dot
            d_radius = abs(ri - rj)
            d_axis = float(np.linalg.norm(perp))
            print(f"[surface equiv] ({i},{j}) cylinder: "
                  f"1-|dot|={d_angle:.6e}  Δr={d_radius:.6e}  "
                  f"Δaxis={d_axis:.6e}")
            if d_angle >= angle_tol or d_radius > radius_tol or d_axis > dist_tol:
                continue
            _union(i, j)

        elif sid == SURFACE_SPHERE:
            ci = np.asarray(pi["center"], dtype=np.float64)
            cj = np.asarray(pj["center"], dtype=np.float64)
            d_center = float(np.linalg.norm(ci - cj))
            ri, rj = float(pi["radius"]), float(pj["radius"])
            d_radius = abs(ri - rj)
            print(f"[surface equiv] ({i},{j}) sphere: "
                  f"Δcenter={d_center:.6e}  Δr={d_radius:.6e}")
            if d_center > dist_tol or d_radius > radius_tol:
                continue
            _union(i, j)

        elif sid == SURFACE_CONE:
            ai, aj = _unit(pi["a"]), _unit(pj["a"])
            dot = abs(float(np.dot(ai, aj)))
            vi = np.asarray(pi["v"], dtype=np.float64)
            vj = np.asarray(pj["v"], dtype=np.float64)
            ti, tj = float(pi["theta"]), float(pj["theta"])
            d_angle = 1.0 - dot
            d_apex = float(np.linalg.norm(vi - vj))
            d_theta = abs(ti - tj)
            print(f"[surface equiv] ({i},{j}) cone: "
                  f"1-|dot|={d_angle:.6e}  Δapex={d_apex:.6e}  "
                  f"Δθ={d_theta:.6e}")
            if d_angle >= angle_tol or d_apex > dist_tol or d_theta > theta_tol:
                continue
            _union(i, j)

    # Flatten
    for i in range(n):
        canon[i] = _find(i)

    # Report
    groups = {}
    for i in range(n):
        groups.setdefault(canon[i], []).append(i)
    for rep, members in groups.items():
        if len(members) > 1:
            print(f"[surface equiv] clusters {members} share the same "
                  f"{SURFACE_NAMES[surface_ids[rep]]} surface "
                  f"(canonical={rep})")

    return canon


def compute_all_intersections(adj, surface_ids, results, occ_surfaces,
                              tol=1e-6):
    """
    Compute intersections for every adjacent pair in adj.

    Returns:
        dict mapping (i, j) with i < j to an intersection result dict.
    """
    out = {}
    for i, j in adjacency_pairs(adj):
        out[(i, j)] = intersect_surfaces(
            surface_ids[i], results[i], occ_surfaces[i],
            surface_ids[j], results[j], occ_surfaces[j],
            tol   = tol,
            label = f"({i},{j}) {SURFACE_NAMES[surface_ids[i]]}∩{SURFACE_NAMES[surface_ids[j]]}",
        )
    return out


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


def compute_vertices_intcs(adj, intersections, occ_surfaces, threshold=1e-3):
    """
    Find B-Rep vertices using curve–surface intersection (GeomAPI_IntCS).

    For each face f, every pair of incident edges (C_fg, C_fh) defines a
    potential vertex where surfaces f, g, h meet.  Instead of curve–curve
    closest-approach, we intersect curve C_fg with surface S_h (and vice
    versa).  This finds actual intersection points — no phantom candidates.

    Each IntCS result is attributed to the edge whose curve was intersected.
    After deduplication (merging candidates within `threshold`), each merged
    vertex inherits the union of edge attributions from all its constituents,
    giving complete vertex–edge attribution without a projection step.

    Parameters
    ----------
    adj           : (n, n) bool adjacency matrix
    intersections : dict (i, j) i<j -> result dict with "curves" list
    occ_surfaces  : list of OCC Geom_Surface, indexed by cluster id
    threshold     : deduplication radius for merging nearby candidates

    Returns
    -------
    vertices     : (M, 3) float64 array of vertex positions, or (0, 3) if none
    vertex_edges : list[set] of length M; vertex_edges[v] is the set of edge
                   tuples (i, j) that vertex v is incident to
    """
    n = adj.shape[0]

    # Build per-face incidence: face f -> list of edge keys (i, j) with i < j
    face_edges = {f: [] for f in range(n)}
    for (i, j), inter in intersections.items():
        if not inter.get("curves"):
            continue
        face_edges[i].append((i, j))
        face_edges[j].append((i, j))

    candidates       = []   # list of (3,) positions
    candidate_triples = []  # list of frozenset({f, g, h}), parallel to candidates

    all_edge_keys = set(k for k, v in intersections.items() if v.get("curves"))

    for f in range(n):
        edges_on_f = face_edges[f]
        if len(edges_on_f) < 2:
            continue

        for idx_a in range(len(edges_on_f)):
            edge_a = edges_on_f[idx_a]
            g = edge_a[0] if edge_a[1] == f else edge_a[1]
            curves_a = intersections[edge_a].get("curves", [])
            if not curves_a:
                continue

            for idx_b in range(idx_a + 1, len(edges_on_f)):
                edge_b = edges_on_f[idx_b]
                h = edge_b[0] if edge_b[1] == f else edge_b[1]

                if g == h:
                    continue

                curves_b = intersections[edge_b].get("curves", [])
                if not curves_b:
                    continue

                triple = frozenset({f, g, h})
                surf_h = occ_surfaces[h]
                surf_g = occ_surfaces[g]

                # IntCS(C_fg, S_h): point lies on surfaces f, g, h
                # NOTE: do NOT wrap with _as_safe_curve — IntCS is an
                # analytical solver that handles infinite curves natively.
                # Trimming would restrict the parameter domain and miss
                # intersections (e.g. line∩cylinder outside [-2,+2]).
                for ca in curves_a:
                    try:
                        intcs = GeomAPI_IntCS(ca, surf_h)
                    except Exception as err:
                        print(f"[compute_vertices_intcs] IntCS({edge_a}, S_{h}) "
                              f"exception: {err}")
                        continue
                    for k in range(1, intcs.NbPoints() + 1):
                        p = intcs.Point(k)
                        candidates.append(np.array([p.X(), p.Y(), p.Z()]))
                        candidate_triples.append(triple)

                # IntCS(C_fh, S_g): point lies on surfaces f, h, g
                for cb in curves_b:
                    try:
                        intcs = GeomAPI_IntCS(cb, surf_g)
                    except Exception as err:
                        print(f"[compute_vertices_intcs] IntCS({edge_b}, S_{g}) "
                              f"exception: {err}")
                        continue
                    for k in range(1, intcs.NbPoints() + 1):
                        p = intcs.Point(k)
                        candidates.append(np.array([p.X(), p.Y(), p.Z()]))
                        candidate_triples.append(triple)

    print(f"[compute_vertices_intcs] {len(candidates)} raw candidates "
          f"before dedup (threshold={threshold})")
    for i, (pos, tri) in enumerate(zip(candidates, candidate_triples)):
        print(f"  cand {i}: ({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f})  "
              f"triple={sorted(tri)}")

    if not candidates:
        return np.empty((0, 3), dtype=np.float64), []

    # Greedy deduplication: merge candidates within threshold of each other.
    # Each candidate carries a surface triple; merging collects all triples.
    candidates = np.array(candidates, dtype=np.float64)
    used = np.zeros(len(candidates), dtype=bool)
    merged_positions = []
    merged_triple_sets = []  # list of set-of-frozensets
    for idx in range(len(candidates)):
        if used[idx]:
            continue
        dists = np.linalg.norm(candidates - candidates[idx], axis=1)
        close = (dists < threshold) & ~used
        merged_pos = candidates[close].mean(axis=0)
        triple_set = set()
        merged_indices = np.where(close)[0]
        for ci in merged_indices:
            triple_set.add(candidate_triples[ci])
        v_idx = len(merged_positions)
        n_merged = int(close.sum())
        print(f"  v{v_idx}: merged {n_merged} candidates "
              f"[{', '.join(str(int(i)) for i in merged_indices)}] → "
              f"({merged_pos[0]:.6f}, {merged_pos[1]:.6f}, {merged_pos[2]:.6f})  "
              f"triples={[sorted(t) for t in triple_set]}")
        merged_positions.append(merged_pos)
        merged_triple_sets.append(triple_set)
        used[close] = True

    print(f"[compute_vertices_intcs] {len(merged_positions)} vertices "
          f"after dedup")

    # ------------------------------------------------------------------
    # Triple-based attribution: for each vertex, derive edge set from its
    # surface triples.  A triple (a, b, c) contributes edges {(a,b),
    # (a,c), (b,c)} ONLY IF all three edges exist in the adjacency graph.
    # This filters spurious points (where infinite surfaces meet outside
    # the model boundary) without any distance-based tolerances.
    # ------------------------------------------------------------------
    merged_edge_sets = []
    for v_idx, (pos, triple_set) in enumerate(
            zip(merged_positions, merged_triple_sets)):
        edge_set = set()
        for triple in triple_set:
            surfs = sorted(triple)
            a, b, c = surfs[0], surfs[1], surfs[2]
            e_ab = (min(a, b), max(a, b))
            e_ac = (min(a, c), max(a, c))
            e_bc = (min(b, c), max(b, c))
            if (e_ab in all_edge_keys and e_ac in all_edge_keys
                    and e_bc in all_edge_keys):
                edge_set.update({e_ab, e_ac, e_bc})
            else:
                missing = []
                if e_ab not in all_edge_keys:
                    missing.append(str(e_ab))
                if e_ac not in all_edge_keys:
                    missing.append(str(e_ac))
                if e_bc not in all_edge_keys:
                    missing.append(str(e_bc))
                print(f"  v{v_idx}: triple {surfs} rejected — "
                      f"missing edges: {', '.join(missing)}")
        if edge_set:
            print(f"  v{v_idx}: edges={sorted(edge_set)}  "
                  f"(from {len(triple_set)} triple(s))")
        else:
            print(f"  v{v_idx}: NO valid triples — vertex will be dropped")
        merged_edge_sets.append(edge_set)

    # Drop vertices with no valid edges
    keep = [i for i, es in enumerate(merged_edge_sets) if es]
    if len(keep) < len(merged_positions):
        n_drop = len(merged_positions) - len(keep)
        print(f"[compute_vertices_intcs] dropping {n_drop} vertices "
              f"with no valid triples")
        merged_positions = [merged_positions[i] for i in keep]
        merged_edge_sets = [merged_edge_sets[i] for i in keep]

    print(f"[compute_vertices_intcs] {len(merged_positions)} vertices "
          f"after triple attribution")
    return (np.array(merged_positions, dtype=np.float64) if merged_positions
            else np.empty((0, 3), dtype=np.float64)), merged_edge_sets
