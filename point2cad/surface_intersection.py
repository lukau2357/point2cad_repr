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

try:
    from .surface_fitter import (
        SURFACE_PLANE, SURFACE_SPHERE, SURFACE_CYLINDER, SURFACE_CONE, SURFACE_INR,
        SURFACE_NAMES,
    )
except ImportError:
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from point2cad.surface_fitter import (
        SURFACE_PLANE, SURFACE_SPHERE, SURFACE_CYLINDER, SURFACE_CONE, SURFACE_INR,
        SURFACE_NAMES,
    )

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
    from OCC.Core.gp import gp_Pnt
    return gp_Pnt(float(p[0]), float(p[1]), float(p[2]))

def _gp_dir(v):
    from OCC.Core.gp import gp_Dir
    return gp_Dir(float(v[0]), float(v[1]), float(v[2]))

def _result(curves, points, curve_type, method):
    return {"curves": curves, "points": points, "type": curve_type, "method": method}

# ---------------------------------------------------------------------------
# Analytical intersections
# ---------------------------------------------------------------------------

def _intersect_plane_plane(pi, pj):
    from OCC.Core.gp import gp_Lin
    from OCC.Core.Geom import Geom_Line

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
    from OCC.Core.gp import gp_Circ, gp_Ax2
    from OCC.Core.Geom import Geom_Circle

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

def _classify_occ_curve(curve):
    from OCC.Core.Geom import Geom_Line, Geom_Circle, Geom_Ellipse, Geom_BSplineCurve, Geom_Conic
    print(curve)
    if isinstance(curve, Geom_Line):        return "line"
    if isinstance(curve, Geom_Circle):      return "circle"
    if isinstance(curve, Geom_Ellipse):     return "ellipse"
    if isinstance(curve, Geom_Conic):       return "conic"
    if isinstance(curve, Geom_BSplineCurve): return "bspline"
    return "curve"


def _intersect_occ(occ_surf_i, occ_surf_j, tol):
    from OCC.Core.GeomAPI import GeomAPI_IntSS

    # https://dev.opencascade.org/doc/refman/html/class_geom_a_p_i___int_s_s.html#details
    inter = GeomAPI_IntSS(occ_surf_i, occ_surf_j, tol)
    if not inter.IsDone():
        return _result([], [], "failed", "occ")

    curves = [inter.Line(k) for k in range(1, inter.NbLines() + 1)]

    if not curves:
        # GeomAPI_IntSS exposes no point query; NbLines()==0 after IsDone()
        # means either no intersection or point tangency (indistinguishable
        # at this API level).
        return _result([], [], "empty", "occ")

    curve_type = _classify_occ_curve(curves[0])
    return _result(curves, [], curve_type, "occ")

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
    try:
        from .cluster_adjacency import adjacency_pairs
    except ImportError:
        from point2cad.cluster_adjacency import adjacency_pairs

    return {
        (i, j): intersect_surfaces(
            surface_ids[i], results[i], occ_surfaces[i],
            surface_ids[j], results[j], occ_surfaces[j],
            tol = tol,
        )
        for i, j in adjacency_pairs(adj)
    }


def sample_curve(curve, n_points = 200, line_extent = 1.0):
    """
    Sample a Geom_Curve into an (n_points, 3) numpy array.

    Lines have an infinite parameter domain; they are clipped to
    [-line_extent, +line_extent].  Set line_extent to a value comparable
    to the diameter of your normalised point cloud (typically ~1).
    """
    t0 = curve.FirstParameter()
    t1 = curve.LastParameter()
    if not math.isfinite(t0): t0 = -line_extent
    if not math.isfinite(t1): t1 =  line_extent

    pts = np.zeros((n_points, 3))
    for i, t in enumerate(np.linspace(t0, t1, n_points)):
        p = curve.Value(t)
        pts[i] = (p.X(), p.Y(), p.Z())
    return pts


# ---------------------------------------------------------------------------
# Open3D helpers
# ---------------------------------------------------------------------------

CURVE_TYPE_COLORS = {
    "line":    [1.0, 0.2, 0.2],
    "circle":  [0.2, 0.85, 0.2],
    "ellipse": [0.2, 0.4,  1.0],
    "conic":   [0.8, 0.2,  0.8],
    "bspline": [1.0, 0.6,  0.0],
    "curve":   [0.8, 0.8,  0.0],
    "tangent": [1.0, 1.0,  1.0],
}


def make_curve_lineset(curve, color, n_points = 200, line_extent = 1.0):
    """
    Build an Open3D LineSet by sampling a Geom_Curve.

    Parameters
    ----------
    curve      : Geom_Curve
    color      : RGB list/array, e.g. [1.0, 0.2, 0.2]
    n_points   : number of sample points
    line_extent: clipping half-length for infinite lines (Geom_Line)
    """
    import open3d as o3d
    pts   = sample_curve(curve, n_points = n_points, line_extent = line_extent)
    lines = [[i, i + 1] for i in range(len(pts) - 1)]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


# ---------------------------------------------------------------------------
# __main__ smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os, sys, time, torch
    import open3d as o3d
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import point2cad.primitive_fitting_utils as pfu
    from point2cad.surface_fitter    import fit_surface, SURFACE_NAMES
    from point2cad.occ_surfaces      import to_occ_surface
    from point2cad.cluster_adjacency import compute_adjacency_matrix, adjacency_pairs
    from point2cad.color_config      import get_surface_color

    SAMPLE = os.path.join(os.path.dirname(__file__), "..", "sample_clouds", "abc_00949.xyzc")
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    def normalize_points(pts):
        pts = pts - np.mean(pts, axis = 0, keepdims = True)
        S, U = np.linalg.eig(pts.T @ pts)
        R = pfu.rotation_matrix_a_to_b(U[:, np.argmin(S)], np.array([1, 0, 0]))
        pts = (R @ pts.T).T
        extents = np.max(pts, axis = 0) - np.min(pts, axis = 0)
        return (pts / (np.max(extents) + 1e-7)).astype(np.float32)

    data = np.loadtxt(SAMPLE)
    data[:, :3] = normalize_points(data[:, :3])
    unique_clusters = np.unique(data[:, -1].astype(int))
    np_rng = np.random.default_rng(41)

    clusters, surface_ids, fit_results, occ_surfs = [], [], [], []
    for cid in unique_clusters:
        cluster = data[data[:, -1].astype(int) == cid, :3].astype(np.float32)
        clusters.append(cluster)
        res = fit_surface(cluster, {"hidden_dim": 64, "use_shortcut": True, "fraction_siren": 0.5},
                          np_rng, DEVICE, inr_fit_kwargs = {"max_steps": 1500})
        sid  = res["surface_id"]
        surface_ids.append(sid)
        fit_results.append(res["result"])
        occ_surfs.append(to_occ_surface(sid, res["result"], cluster = cluster))
        print(f"Cluster {cid}: {SURFACE_NAMES[sid]}")

    adj, threshold, spacing = compute_adjacency_matrix(clusters)
    print(f"\nSpacing={spacing:.5f}  threshold={threshold:.5f}")
    print(f"Adjacent pairs: {adjacency_pairs(adj)}\n")

    intersections = compute_all_intersections(adj, surface_ids, fit_results, occ_surfs)

    curve_linesets = []
    for (i, j), inter in intersections.items():
        si, sj = SURFACE_NAMES[surface_ids[i]], SURFACE_NAMES[surface_ids[j]]
        print(f"({i},{j})  {si} ∩ {sj}  ->  type={inter['type']}  method={inter['method']}"
              f"  curves={len(inter['curves'])}  points={len(inter['points'])}")
        color = CURVE_TYPE_COLORS.get(inter["type"], [0.8, 0.8, 0.8])
        for k, curve in enumerate(inter["curves"]):
            pts = sample_curve(curve, n_points = 5, line_extent = 1.0)
            print(f"  curve[{k}] sampled range: {pts[0]} .. {pts[-1]}")
            curve_linesets.append(
                make_curve_lineset(curve, color, n_points = 200, line_extent = 1.0)
            )

    # Build cluster point clouds coloured by surface type
    # cluster_pcds = []
    # for cluster, sid in zip(clusters, surface_ids):
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(cluster)
    #     pcd.paint_uniform_color(get_surface_color(SURFACE_NAMES[sid]).tolist())
    #     cluster_pcds.append(pcd)

    # # Window 1: intersection curves only
    # vis1 = o3d.visualization.Visualizer()
    # vis1.create_window(window_name = "Intersection curves", width = 960, height = 720, left = 0, top = 50)
    # for ls in curve_linesets:
    #     vis1.add_geometry(ls)

    # # Window 2: cluster point clouds + curves
    # vis2 = o3d.visualization.Visualizer()
    # vis2.create_window(window_name = "Point clouds + curves", width = 960, height = 720, left = 960, top = 50)
    # for pcd in cluster_pcds:
    #     vis2.add_geometry(pcd)
    # for ls in curve_linesets:
    #     vis2.add_geometry(ls)
    # vis2.get_render_option().point_size = 2.0

    # running1, running2 = True, True
    # while running1 and running2:
    #     if running1:
    #         running1 = vis1.poll_events()
    #         vis1.update_renderer()
    #     if running2:
    #         running2 = vis2.poll_events()
    #         vis2.update_renderer()
    #     time.sleep(0.01)

    # vis1.destroy_window()
    # vis2.destroy_window()
