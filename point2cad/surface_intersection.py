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
    print(err)
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


def sample_curve(curve, n_points = 200, line_extent = 1.0):
    """
    Sample a Geom_Curve into an (n_points, 3) numpy array.

    Lines have an infinite parameter domain; they are clipped to
    [-line_extent, +line_extent].  Set line_extent to a value comparable
    to the diameter of your normalised point cloud (typically ~1).
    """
    t0 = curve.FirstParameter()
    t1 = curve.LastParameter()
    if not math.isfinite(t0) or abs(t0) >= _OCC_INF: t0 = -line_extent
    if not math.isfinite(t1) or abs(t1) >= _OCC_INF: t1 =  line_extent

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


def trim_intersections(intersections, boundary_strips, threshold, extension_factor = 0.05):
    """
    Trim all intersection curves to their shared boundary portions.

    Parameters
    ----------
    intersections    : dict (i, j) -> result dict (from compute_all_intersections)
    boundary_strips  : dict (i, j) -> (N, 3) float32 array
                       (from compute_adjacency_matrix)
    threshold        : adjacency threshold (from compute_adjacency_matrix)
    extension_factor : passed through to trim_curve

    Returns
    -------
    New intersections dict with infinite curves replaced by Geom_TrimmedCurve
    where trimming succeeded; compact curves are passed through unchanged.
    """
    trimmed = {}
    for (i, j), inter in intersections.items():
        if not inter["curves"]:
            trimmed[(i, j)] = inter
            continue

        boundary_pts = boundary_strips.get((i, j), np.empty((0, 3)))
        new_curves   = [trim_curve(c, boundary_pts, threshold, extension_factor)
                        for c in inter["curves"]]
        trimmed[(i, j)] = {**inter, "curves": new_curves}

    return trimmed


def compute_vertices(adj, intersections, threshold):
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
    vertices : (M, 3) float64 array of vertex positions, or (0, 3) if none
    """
    candidates = []
    edges = list(intersections.keys())   # each is (i, j) with i < j
    threshold = 1e-3
    
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
                    for m in range(1, ext.NbExtrema() + 1):
                        if ext.Distance(m) < threshold:
                            t1, t2 = ext.Parameters(m)
                            p1 = ca.Value(t1)
                            p2 = cb.Value(t2)
                            midpoint = np.array([
                                (p1.X() + p2.X()) / 2.0,
                                (p1.Y() + p2.Y()) / 2.0,
                                (p1.Z() + p2.Z()) / 2.0,
                            ])
                            candidates.append(midpoint)

    if not candidates:
        return np.empty((0, 3), dtype=np.float64)

    # Greedy deduplication: merge candidates within threshold of each other
    candidates = np.array(candidates, dtype = np.float64)
    used = np.zeros(len(candidates), dtype=bool)
    merged = []
    for idx in range(len(candidates)):
        if used[idx]:
            continue
        dists = np.linalg.norm(candidates - candidates[idx], axis=1)
        close = dists < threshold
        merged.append(candidates[close].mean(axis=0))
        used[close] = True

    return np.array(merged, dtype=np.float64)


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
    import argparse, os, sys, time, glob as _glob, torch
    parser = argparse.ArgumentParser(
        description = "Surface intersection smoke-test",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--visualize", action = "store_true",
                        help = "Load saved results and visualize (host mode, no OCC needed)")
    parser.add_argument("--input", type = str, default = None,
                        help = "Path to .xyzc file (compute mode only)")
    parser.add_argument("--visualize_id", type = str, help = "ID of the point cloud to visualize the results for")
    parser.add_argument("--output_dir", type = str, default = "output_surfaceinter",
                        help = "Directory for saved results")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Host mode: load saved results and open Open3D windows
    # ------------------------------------------------------------------
    if args.visualize:
        import open3d as o3d

        # Accept either output_dir/pc_id/ or output_dir/ directly.
        # If metadata.npz is not at the top level, look for a single subdirectory.
        out_dir = args.output_dir
        pc_id = args.visualize_id
        out_dir = os.path.join(out_dir, pc_id)
        
        meta = np.load(os.path.join(out_dir, "metadata.npz"), allow_pickle = True)
        n_clusters   = int(meta["n_clusters"])
        surface_ids  = meta["surface_ids"]
        surf_names   = meta["surface_names"]
        clust_colors = meta["cluster_colors"]

        cluster_pcds = []
        for i in range(n_clusters):
            pts = np.load(os.path.join(out_dir, f"cluster_{i}.npy"))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.paint_uniform_color(clust_colors[i].tolist())
            cluster_pcds.append(pcd)

        def _pts_to_lineset(pts, color):
            lines = [[m, m + 1] for m in range(len(pts) - 1)]
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(pts)
            ls.lines  = o3d.utility.Vector2iVector(lines)
            ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
            return ls

        trimmed_linesets   = []
        untrimmed_linesets = []
        for inter_path in sorted(_glob.glob(os.path.join(out_dir, "inter_*.npz"))):
            d          = np.load(inter_path, allow_pickle = True)
            curve_type = str(d["curve_type"])
            n_curves   = int(d["n_curves"])
            n_raw      = int(d["n_untrimmed_curves"])
            ci, cj     = int(d["cluster_i"]), int(d["cluster_j"])
            si_name    = str(d["surface_i_name"])
            sj_name    = str(d["surface_j_name"])
            print(f"({ci},{cj})  {si_name} ∩ {sj_name}  ->  type={curve_type}"
                  f"  trimmed={n_curves}  raw={n_raw}")
            color = CURVE_TYPE_COLORS.get(curve_type, [0.8, 0.8, 0.8])
            for k in range(n_curves):
                trimmed_linesets.append(_pts_to_lineset(d[f"curve_points_{k}"], color))
            for k in range(n_raw):
                untrimmed_linesets.append(_pts_to_lineset(d[f"untrimmed_curve_points_{k}"], color))

        inr_meshes = []
        for mesh_path in sorted(_glob.glob(os.path.join(out_dir, "inr_mesh_*.npz"))):
            ci    = int(os.path.basename(mesh_path).replace("inr_mesh_", "").replace(".npz", ""))
            d     = np.load(mesh_path)
            mesh  = o3d.geometry.TriangleMesh()
            mesh.vertices  = o3d.utility.Vector3dVector(d["vertices"])
            mesh.triangles = o3d.utility.Vector3iVector(d["triangles"])
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color(clust_colors[ci].tolist())
            inr_meshes.append(mesh)

        # 2×2 window layout (fits 1920×1080)
        W, H = 960, 490

        vis1 = o3d.visualization.Visualizer()
        vis1.create_window(window_name = "Untrimmed curves", width = W, height = H, left = 0, top = 50)
        for ls in untrimmed_linesets:
            vis1.add_geometry(ls)

        vertex_pcd = None
        vertex_path = os.path.join(out_dir, "vertices.npz")
        if os.path.exists(vertex_path):
            vdata = np.load(vertex_path)
            verts = vdata["vertices"]
            if len(verts) > 0:
                vertex_pcd = o3d.geometry.PointCloud()
                vertex_pcd.points = o3d.utility.Vector3dVector(verts)
                vertex_pcd.paint_uniform_color([1.0, 1.0, 0.0])
                print(f"Loaded {len(verts)} vertices")

        vis2 = o3d.visualization.Visualizer()
        vis2.create_window(window_name = "Trimmed curves + vertices", width = W, height = H, left = W, top = 50)
        for ls in trimmed_linesets:
            vis2.add_geometry(ls)
        if vertex_pcd is not None:
            vis2.add_geometry(vertex_pcd)
        vis2.get_render_option().point_size = 8.0

        vis3 = o3d.visualization.Visualizer()
        vis3.create_window(window_name = "Point clouds + trimmed curves", width = W, height = H, left = 0, top = 50 + H + 40)
        for pcd in cluster_pcds:
            vis3.add_geometry(pcd)
        for ls in trimmed_linesets:
            vis3.add_geometry(ls)
        vis3.get_render_option().point_size = 2.0

        vis4 = o3d.visualization.Visualizer()
        vis4.create_window(window_name = "INR surfaces", width = W, height = H, left = W, top = 50 + H + 40)
        for mesh in inr_meshes:
            vis4.add_geometry(mesh)

        running1 = running2 = running3 = running4 = True
        while running1 and running2 and running3 and running4:
            if running1:
                running1 = vis1.poll_events()
                vis1.update_renderer()
            if running2:
                running2 = vis2.poll_events()
                vis2.update_renderer()
            if running3:
                running3 = vis3.poll_events()
                vis3.update_renderer()
            if running4:
                running4 = vis4.poll_events()
                vis4.update_renderer()
            time.sleep(0.01)

        vis1.destroy_window()
        vis2.destroy_window()
        vis3.destroy_window()
        vis4.destroy_window()
        sys.exit(0)

    # ------------------------------------------------------------------
    # Compute mode (Docker): fit surfaces, intersect, save results
    # ------------------------------------------------------------------
    if not OCC_AVAILABLE:
        print("OCC bindings not available — cannot run intersection computation.")
        sys.exit(1)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import point2cad.primitive_fitting_utils as pfu
    from point2cad.surface_fitter    import fit_surface, SURFACE_NAMES
    from point2cad.occ_surfaces      import to_occ_surface
    from point2cad.cluster_adjacency import compute_adjacency_matrix, adjacency_pairs, adjacency_triangles
    from point2cad.color_config      import get_surface_color

    SAMPLE = args.input or os.path.join(os.path.dirname(__file__), "..", "sample_clouds", "abc_00949.xyzc")
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    pc_id   = os.path.basename(SAMPLE).split("_")[-1].split(".")[0]
    out_dir = os.path.join(args.output_dir, pc_id)
    os.makedirs(out_dir, exist_ok = True)

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
    torch.manual_seed(41)
    torch.cuda.manual_seed(41)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    
    clusters, surface_ids, fit_results, fit_meshes, occ_surfs = [], [], [], [], []
    for cid in unique_clusters:
        cluster = data[data[:, -1].astype(int) == cid, :3].astype(np.float32)
        clusters.append(cluster)
        res = fit_surface(cluster, {"hidden_dim": 64, "use_shortcut": True, "fraction_siren": 0.5},
                          np_rng, DEVICE, 
                          inr_fit_kwargs = {"max_steps": 1500, "noise_magnitude_3d": 0.05, "noise_magnitude_uv": 0.05, "initial_lr": 1e-1},
                          inr_mesh_kwargs = {"mesh_dim": 200, "uv_margin": 0.2, "threshold_multiplier": 1.5})
        
        sid = res["surface_id"]
        surface_ids.append(sid)
        fit_results.append(res["result"])
        fit_meshes.append(res["mesh"])
        occ_surfs.append(to_occ_surface(sid, res["result"], cluster = cluster, uv_margin = 0.05))
        print(f"Cluster {cid}: {SURFACE_NAMES[sid]}")

    adj, threshold, spacing, boundary_strips = compute_adjacency_matrix(clusters)
    print(f"\nSpacing={spacing:.5f}  threshold={threshold:.5f}")
    print(f"Adjacent pairs: {adjacency_pairs(adj)}\n")

    # Save metadata and cluster point clouds BEFORE intersection so the
    # visualizer always has something to display even if OCC fails later.
    np.savez(
        os.path.join(out_dir, "metadata.npz"),
        n_clusters     = len(clusters),
        surface_ids    = np.array(surface_ids),
        surface_names  = np.array([SURFACE_NAMES[s] for s in surface_ids]),
        cluster_colors = np.array([get_surface_color(SURFACE_NAMES[s]).tolist() for s in surface_ids]),
    )
    for i, cluster in enumerate(clusters):
        np.save(os.path.join(out_dir, f"cluster_{i}.npy"), cluster)

    for i, (sid, mesh) in enumerate(zip(surface_ids, fit_meshes)):
        if sid == SURFACE_INR:
            np.savez(
                os.path.join(out_dir, f"inr_mesh_{i}.npz"),
                vertices  = np.asarray(mesh.vertices),
                triangles = np.asarray(mesh.triangles),
            )
    print(f"Cluster files saved to {out_dir}/")

    raw_intersections   = compute_all_intersections(adj, surface_ids, fit_results, occ_surfs)
    trim_intersections_ = trim_intersections(raw_intersections, boundary_strips, threshold, extension_factor = 0.15)

    vertices = compute_vertices(adj, trim_intersections_, threshold)
    print(f"Found {len(vertices)} vertices")
    np.savez(os.path.join(out_dir, "vertices.npz"), vertices=vertices)

    # Save intersections: sample curves to numpy so the host needs no OCC.
    # Both untrimmed (raw) and trimmed samples are stored in the same npz.
    for (i, j) in raw_intersections:
        inter_raw  = raw_intersections[(i, j)]
        inter_trim = trim_intersections_[(i, j)]
        si, sj = SURFACE_NAMES[surface_ids[i]], SURFACE_NAMES[surface_ids[j]]
        print(f"({i},{j})  {si} ∩ {sj}  ->  type={inter_raw['type']}  method={inter_raw['method']}"
              f"  raw_curves={len(inter_raw['curves'])}  trimmed_curves={len(inter_trim['curves'])}")
        kw = dict(
            cluster_i           = i,
            cluster_j           = j,
            surface_i_name      = si,
            surface_j_name      = sj,
            curve_type          = inter_raw["type"],
            method              = inter_raw["method"],
            n_curves            = len(inter_trim["curves"]),
            n_untrimmed_curves  = len(inter_raw["curves"]),
        )
        for k, curve in enumerate(inter_trim["curves"]):
            print(f"  trimmed  curve[{k}] params: [{curve.FirstParameter():.6f}, {curve.LastParameter():.6f}]")
            kw[f"curve_points_{k}"] = sample_curve(curve, n_points = 200, line_extent = 1.0)
        for k, curve in enumerate(inter_raw["curves"]):
            print(f"  raw      curve[{k}] params: [{curve.FirstParameter():.6f}, {curve.LastParameter():.6f}]")
            kw[f"untrimmed_curve_points_{k}"] = sample_curve(curve, n_points = 200, line_extent = 1.0)
        np.savez(os.path.join(out_dir, f"inter_{i}_{j}.npz"), **kw)

    print(f"\nAll results saved to {out_dir}/")
