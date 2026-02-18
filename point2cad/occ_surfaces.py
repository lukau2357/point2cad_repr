import time
import numpy as np
import torch
try:
    from .surface_fitter import SURFACE_PLANE, SURFACE_SPHERE, SURFACE_CYLINDER, SURFACE_CONE, SURFACE_INR
except ImportError:
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from point2cad.surface_fitter import SURFACE_PLANE, SURFACE_SPHERE, SURFACE_CYLINDER, SURFACE_CONE, SURFACE_INR

def _make_ax3(origin, main_dir):
    from OCC.Core.gp import gp_Ax3, gp_Pnt, gp_Dir
    main_dir = np.asarray(main_dir, dtype = np.float64)
    main_dir = main_dir / np.linalg.norm(main_dir)
    return gp_Ax3(
        gp_Pnt(float(origin[0]), float(origin[1]), float(origin[2])),
        gp_Dir(float(main_dir[0]), float(main_dir[1]), float(main_dir[2]))
    )

def fit_bspline_surface(xyz_grid, degree_min = 3, degree_max = 8, continuity = 2, tol3d = 1e-3):
    from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface
    from OCC.Core.TColgp import TColgp_Array2OfPnt
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.GeomAbs import GeomAbs_C0, GeomAbs_C1, GeomAbs_C2, GeomAbs_C3

    continuity_map = {0: GeomAbs_C0, 1: GeomAbs_C1, 2: GeomAbs_C2, 3: GeomAbs_C3}

    M, N, _ = xyz_grid.shape
    points = TColgp_Array2OfPnt(1, M, 1, N)
    for i in range(M):
        for j in range(N):
            x, y, z = xyz_grid[i, j]
            points.SetValue(i + 1, j + 1, gp_Pnt(float(x), float(y), float(z)))

    t0 = time.time()
    approx = GeomAPI_PointsToBSplineSurface(points, degree_min, degree_max, continuity_map[continuity], tol3d)
    fitting_time = time.time() - t0

    if not approx.IsDone():
        raise RuntimeError(f"GeomAPI_PointsToBSplineSurface failed (grid={M}x{N}, deg=[{degree_min},{degree_max}], tol={tol3d})")

    return approx.Surface(), fitting_time

def plane_to_occ(params):
    from OCC.Core.Geom import Geom_Plane
    from OCC.Core.gp import gp_Pln, gp_Pnt, gp_Dir

    a = np.asarray(params["a"], dtype = np.float64)
    a = a / np.linalg.norm(a)
    d = float(params["d"])

    # Point on plane: p = d * a (from a . x = d with unit normal a)
    p = d * a
    return Geom_Plane(gp_Pln(
        gp_Pnt(float(p[0]), float(p[1]), float(p[2])),
        gp_Dir(float(a[0]), float(a[1]), float(a[2]))
    ))


def sphere_to_occ(params):
    from OCC.Core.Geom import Geom_SphericalSurface
    from OCC.Core.gp import gp_Sphere

    center = np.asarray(params["center"], dtype = np.float64)
    # Ax3 orientation is arbitrary for a sphere
    ax3 = _make_ax3(center, np.array([0.0, 0.0, 1.0]))
    return Geom_SphericalSurface(gp_Sphere(ax3, float(params["radius"])))


def cylinder_to_occ(params):
    from OCC.Core.Geom import Geom_CylindricalSurface
    from OCC.Core.gp import gp_Cylinder

    ax3 = _make_ax3(np.asarray(params["center"], dtype = np.float64),
                    np.asarray(params["a"], dtype = np.float64))
    return Geom_CylindricalSurface(gp_Cylinder(ax3, float(params["radius"])))


def cone_to_occ(params, cluster = None):
    from OCC.Core.Geom import Geom_ConicalSurface
    from OCC.Core.gp import gp_Cone

    axis   = np.asarray(params["a"], dtype = np.float64)
    vertex = np.asarray(params["v"], dtype = np.float64)
    theta  = float(params["theta"])

    axis = axis / np.linalg.norm(axis)

    # OCC Geom_ConicalSurface only covers the nappe in the positive Ax3 Z direction.
    # fit_cone solves a double-cone equation so the axis may point toward either nappe;
    # orient it toward the majority of cluster points (same logic as sample_cone).
    if cluster is not None:
        proj = (cluster - vertex) @ axis
        if np.sum(proj < 0) > np.sum(proj > 0):
            axis = -axis

    ax3 = _make_ax3(vertex, axis)
    return Geom_ConicalSurface(gp_Cone(ax3, theta, 0.0))

def inr_to_occ(params, grid_resolution = 50, degree_min = 3, degree_max = 8, continuity = 2, tol3d = 1e-3):
    model = params["model"]
    xyz_grid = model.sample_points(
        grid_resolution,
        params["uv_bb_min"].copy(),
        params["uv_bb_max"].copy(),
        params["cluster_mean"],
        params["cluster_scale"],
    ).cpu().numpy().reshape(grid_resolution, grid_resolution, 3)

    surface, _ = fit_bspline_surface(xyz_grid, degree_min = degree_min, degree_max = degree_max,
                                     continuity = continuity, tol3d = tol3d)
    return surface

def to_occ_surface(surface_id, result, cluster = None, **kwargs):
    params = result["params"]

    if surface_id == SURFACE_PLANE:
        return plane_to_occ(params)
    
    elif surface_id == SURFACE_SPHERE:
        return sphere_to_occ(params)
    
    elif surface_id == SURFACE_CYLINDER:
        return cylinder_to_occ(params)
    
    elif surface_id == SURFACE_CONE:
        return cone_to_occ(params, cluster = cluster)
    
    elif surface_id == SURFACE_INR:
        return inr_to_occ(params, **kwargs)
    else:
        raise ValueError(f"Unknown surface_id: {surface_id}")

def validate_occ_surface(occ_surface, reference_points):
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCC.Core.gp import gp_Pnt

    dists = []
    n_failed = 0

    for pt in reference_points:
        proj = GeomAPI_ProjectPointOnSurf(gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2])), occ_surface)
        if proj.IsDone() and proj.NbPoints() > 0:
            dists.append(proj.LowerDistance())
        else:
            n_failed += 1

    dists = np.array(dists)
    return {
        "mean_dist":   float(np.mean(dists))   if len(dists) else float("nan"),
        "max_dist":    float(np.max(dists))    if len(dists) else float("nan"),
        "median_dist": float(np.median(dists)) if len(dists) else float("nan"),
        "n_failed":    n_failed,
    }

if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from point2cad.surface_fitter import fit_surface, SURFACE_NAMES
    import point2cad.primitive_fitting_utils as pfu

    SAMPLE = os.path.join(os.path.dirname(__file__), "..", "sample_clouds", "abc_00470.xyzc")
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    def normalize_points(points):
        points = points - np.mean(points, axis = 0, keepdims = True)
        S, U = np.linalg.eig(points.T @ points)
        smallest_ev = U[:, np.argmin(S)]
        R = pfu.rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
        points = (R @ points.T).T
        extents = np.max(points, axis = 0) - np.min(points, axis = 0)
        return (points / (np.max(extents) + 1e-7)).astype(np.float32)

    data = np.loadtxt(SAMPLE)
    data[:, :3] = normalize_points(data[:, :3])
    unique_clusters = np.unique(data[:, -1].astype(int))
    np_rng = np.random.default_rng(41)

    for cid in unique_clusters:
        cluster = data[data[:, -1].astype(int) == cid, :3].astype(np.float32)
        print(f"\nCluster {cid}: {cluster.shape[0]} points")

        res = fit_surface(cluster, {"hidden_dim": 64, "use_shortcut": True, "fraction_siren": 0.5}, np_rng, DEVICE)
        sid = res["surface_id"]
        print(f"  Surface type: {SURFACE_NAMES[sid]}")

        occ_surf = to_occ_surface(sid, res["result"], cluster = cluster)
        print(f"  OCC surface: {type(occ_surf).__name__}")

        stats = validate_occ_surface(occ_surf, cluster)
        print(f"  Validation: mean={stats['mean_dist']:.6f}, max={stats['max_dist']:.6f}, "
              f"median={stats['median_dist']:.6f}, failed={stats['n_failed']}/{cluster.shape[0]}")
