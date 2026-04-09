import numpy as np
import open3d as o3d
import trimesh
import igl

from .inr_fitting import fit_inr
from .primitive_fitting import fit_plane_numpy, fit_sphere_numpy, fit_cylinder_optimized, fit_cone
from .primitive_fitting_utils import generate_plane_mesh, generate_sphere_mesh, generate_cylinder_mesh, generate_cone_mesh
from .color_config import get_surface_color
from .surface_types import (
    SURFACE_PLANE, SURFACE_SPHERE, SURFACE_CYLINDER, SURFACE_CONE, SURFACE_INR,
    SURFACE_NAMES,
)

# Registry: surface_id → fitter callable.  Keys must match the constants in
# surface_types.py and be contiguous starting from 0 so that
# errors[sid] = primitive_results[sid]["error"] is a valid 1-D array.
# When adding a new primitive, add its fitter here with the correct key.
PRIMITIVE_FITTERS = {
    SURFACE_PLANE:    fit_plane_numpy,
    SURFACE_SPHERE:   fit_sphere_numpy,
    SURFACE_CYLINDER: fit_cylinder_optimized,
    SURFACE_CONE:     fit_cone,
}

def ratio(x, y, eps = 1e-8):
    return (x + eps) / (y + eps)

def _inflate_mesh(o3d_mesh, trimesh_mesh, surface_id, params,
                  radius_inflation, angle_inflation_deg):
    """Push mesh vertices outward after trimming. Modifies meshes in-place."""
    if surface_id == SURFACE_SPHERE and radius_inflation != 0.0:
        center = params["center"].reshape(1, 3)
        verts = np.asarray(o3d_mesh.vertices)
        dirs = verts - center
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        verts += dirs / norms * radius_inflation
        o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
        trimesh_mesh.vertices += (dirs / norms * radius_inflation)[:len(trimesh_mesh.vertices)]

    elif surface_id == SURFACE_CYLINDER and radius_inflation != 0.0:
        center = params["center"].reshape(1, 3)
        axis = params["a"].reshape(3)
        verts = np.asarray(o3d_mesh.vertices)
        shifted = verts - center
        along = (shifted @ axis).reshape(-1, 1) * axis.reshape(1, 3)
        radial = shifted - along
        radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
        radial_norm = np.maximum(radial_norm, 1e-10)
        offset = radial / radial_norm * radius_inflation
        verts += offset
        o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
        trimesh_mesh.vertices += offset[:len(trimesh_mesh.vertices)]

    elif surface_id == SURFACE_CONE and angle_inflation_deg != 0.0:
        vertex = params["v"].reshape(1, 3)
        axis = params["a"].reshape(3)
        theta = params["theta"]
        theta_new = theta + np.radians(angle_inflation_deg)
        scale = np.tan(theta_new) / (np.tan(theta) + 1e-10)

        verts = np.asarray(o3d_mesh.vertices)
        shifted = verts - vertex
        along = (shifted @ axis).reshape(-1, 1) * axis.reshape(1, 3)
        radial = shifted - along
        new_verts = vertex + along + radial * scale
        o3d_mesh.vertices = o3d.utility.Vector3dVector(new_verts)
        tv = trimesh_mesh.vertices
        t_shifted = tv - vertex
        t_along = (t_shifted @ axis).reshape(-1, 1) * axis.reshape(1, 3)
        t_radial = t_shifted - t_along
        trimesh_mesh.vertices = vertex + t_along + t_radial * scale


def resolve_mesh(surface_id,
                 result,
                 cluster,
                 np_rng,
                 device,
                 plane_mesh_kwargs,
                 sphere_mesh_kwargs,
                 cylinder_mesh_kwargs,
                 cone_mesh_kwargs,
                 inr_mesh_kwargs,
                 radius_inflation=0.0,
                 angle_inflation_deg=0.0):

    params = result["params"]

    if surface_id == SURFACE_PLANE:
        return generate_plane_mesh(
            a = params["a"],
            d = params["d"],
            cluster = cluster,
            np_rng = np_rng,
            device = device,
**plane_mesh_kwargs
        )

    if surface_id == SURFACE_SPHERE:
        mesh = generate_sphere_mesh(
            radius = params["radius"],
            center = params["center"],
            cluster = cluster,
            device = device,
**sphere_mesh_kwargs
        )
        if radius_inflation != 0.0:
            _inflate_mesh(mesh[0], mesh[1], surface_id, params, radius_inflation, angle_inflation_deg)
        return mesh

    if surface_id == SURFACE_CYLINDER:
        mesh = generate_cylinder_mesh(
            radius = params["radius"],
            center = params["center"],
            axis = params["a"],
            cluster = cluster,
            device = device,
**cylinder_mesh_kwargs
        )
        if radius_inflation != 0.0:
            _inflate_mesh(mesh[0], mesh[1], surface_id, params, radius_inflation, angle_inflation_deg)
        return mesh

    if surface_id == SURFACE_CONE:
        mesh = generate_cone_mesh(
            vertex = params["v"],
            axis = params["a"],
            theta = params["theta"],
            cluster_points = cluster,
            device = device,
**cone_mesh_kwargs
        )
        if angle_inflation_deg != 0.0:
            _inflate_mesh(mesh[0], mesh[1], surface_id, params, radius_inflation, angle_inflation_deg)
        return mesh

    if surface_id == SURFACE_INR:
        model = params["model"]
        return model.sample_mesh(
            uv_bb_min = params["uv_bb_min"],
            uv_bb_max = params["uv_bb_max"],
            cluster = cluster,
            cluster_mean = params["cluster_mean"],
            cluster_scale = params["cluster_scale"],
            **inr_mesh_kwargs
        )

def cone_special_handling(results, errors, simple_error_threshold, plane_cone_ratio_threshold, cone_theta_tolerance_degrees):
    """
    If cone is selected as a best simple surface, special handling for it. Returns the ID of the best matching surface after
    processing, or -1 if not simple surface meets the criteria.
    """
    # If plane is satisfiable, return the plane
    if ratio(errors[SURFACE_PLANE], errors[SURFACE_CONE]) < plane_cone_ratio_threshold:
        return SURFACE_PLANE

    cone_angle = results[SURFACE_CONE]["params"]["theta"]
    cone_theta_tolerance_rad = cone_theta_tolerance_degrees * (np.pi / 180)

    # If cone is near a degenerate cone, try the next best simple surface
    cone_diff_pi2 = abs(cone_angle - np.pi / 2)
    cone_diff_0 = cone_angle
    
    if cone_diff_pi2 < cone_theta_tolerance_rad or cone_diff_0 < cone_theta_tolerance_rad:
        simple_min = np.argmin(errors[:-1])

        return simple_min if errors[simple_min] < simple_error_threshold else -1
    
    return SURFACE_CONE
        

def plane_sphere_arbitration(errors, plane_sphere_ratio_threshold):
    """
    Symmetric arbiter between plane and sphere. Returns SURFACE_SPHERE iff

        plane_error / sphere_error  >=  plane_sphere_ratio_threshold

    (i.e. sphere is substantially better than plane), otherwise SURFACE_PLANE.

    Used from both branches of the dispatch (`simple_min == SURFACE_PLANE`
    and `simple_min == SURFACE_SPHERE`) so that the plane-vs-sphere decision
    is identical regardless of which one wins the raw argmin. Without this,
    a noisy planar patch can be silently described as a giant-radius sphere
    just because the extra DoF undercut plane error by a hair.
    """
    if ratio(errors[SURFACE_PLANE], errors[SURFACE_SPHERE]) >= plane_sphere_ratio_threshold:
        return SURFACE_SPHERE
    return SURFACE_PLANE


def generate_bpa_mesh(cluster, spacing):
    """Generate a mesh from a point cloud using Ball Pivoting Algorithm."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cluster)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn = 30))
    pcd.orient_normals_consistent_tangent_plane(k = 30)

    radii = [spacing, 2 * spacing, 4 * spacing]
    print(f"  [bpa] spacing={spacing:.6f}  radii=[{', '.join(f'{r:.6f}' for r in radii)}]")

    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    bpa_mesh.compute_vertex_normals()
    color = get_surface_color("inr")
    bpa_mesh.paint_uniform_color(color)

    verts = np.asarray(bpa_mesh.vertices)
    faces = np.asarray(bpa_mesh.triangles)
    trimesh_mesh = trimesh.Trimesh(verts, faces)
    return bpa_mesh, trimesh_mesh


def generate_bspline_mesh(cluster, spacing, mesh_dim=200, uv_margin=0.1):
    """BPA → LSCM → B-spline fitting → oversized mesh generation."""
    from scipy.spatial import cKDTree

    # Step 1: BPA mesh
    print(f"  [bspline] step 1/4: BPA mesh ...")
    bpa_mesh, _ = generate_bpa_mesh(cluster, spacing)
    V = np.asarray(bpa_mesh.vertices)
    F = np.asarray(bpa_mesh.triangles)
    print(f"  [bspline] BPA: {len(V)} vertices, {len(F)} faces")

    if len(F) == 0:
        raise RuntimeError("BPA produced empty mesh, cannot fit B-spline")

    # Step 2: LSCM parameterization
    print(f"  [bspline] step 2/4: LSCM parameterization ...")
    boundary = igl.boundary_loop(F)
    if len(boundary) < 2:
        raise RuntimeError("Mesh has no boundary loop, cannot run LSCM")
    b = np.array([boundary[0], boundary[len(boundary) // 2]])
    bc = np.array([[0.0, 0.0], [1.0, 0.0]])
    uv, _ = igl.lscm(V, F, b, bc)
    if hasattr(uv, 'toarray'):
        uv = uv.toarray()
    uv = np.ascontiguousarray(uv, dtype=np.float64)
    if not np.all(np.isfinite(uv)):
        n_nan = np.sum(np.isnan(uv))
        n_inf = np.sum(np.isinf(uv))
        raise RuntimeError(f"LSCM produced non-finite UV (NaN={n_nan}, Inf={n_inf})")
    print(f"  [bspline] UV shape={uv.shape}", flush=True)
    print(f"  [bspline] UV range: u=[{uv[:, 0].min():.4f}, {uv[:, 0].max():.4f}]  "
          f"v=[{uv[:, 1].min():.4f}, {uv[:, 1].max():.4f}]")

    # Step 3: Resample on regular UV grid via barycentric interpolation, then fit B-spline
    print(f"  [bspline] step 3/4: B-spline fitting ...", flush=True)
    from scipy.interpolate import LinearNDInterpolator, RectBivariateSpline
    uv_min = uv.min(axis=0)
    uv_max = uv.max(axis=0)
    uv_range = uv_max - uv_min

    # Barycentric interpolation: UV triangulation → regular grid
    interp = LinearNDInterpolator(uv, V)
    n_fit = min(mesh_dim, 100)
    u_fit = np.linspace(uv_min[0], uv_max[0], n_fit)
    v_fit = np.linspace(uv_min[1], uv_max[1], n_fit)
    uu, vv = np.meshgrid(u_fit, v_fit, indexing='ij')
    grid_3d = interp(uu, vv)  # (n_fit, n_fit, 3)

    # Grid points outside the UV triangulation get NaN — fill with nearest
    for dim in range(3):
        layer = grid_3d[:, :, dim]
        nan_mask = np.isnan(layer)
        if nan_mask.any():
            from scipy.interpolate import NearestNDInterpolator
            nn_interp = NearestNDInterpolator(uv, V[:, dim])
            layer[nan_mask] = nn_interp(uu[nan_mask], vv[nan_mask])

    # Fit smooth B-spline per coordinate on the regular grid
    splines = []
    for dim in range(3):
        spline = RectBivariateSpline(u_fit, v_fit, grid_3d[:, :, dim], kx=3, ky=3)
        splines.append(spline)
    print(f"  [bspline] splines fitted", flush=True)

    # Step 4: Evaluate on (optionally extended) grid for mesh
    print(f"  [bspline] step 4/4: generating mesh (dim={mesh_dim}, margin={uv_margin}) ...", flush=True)
    u_eval = np.linspace(uv_min[0] - uv_range[0] * uv_margin,
                         uv_max[0] + uv_range[0] * uv_margin, mesh_dim)
    v_eval = np.linspace(uv_min[1] - uv_range[1] * uv_margin,
                         uv_max[1] + uv_range[1] * uv_margin, mesh_dim)

    vertices = np.zeros((mesh_dim * mesh_dim, 3))
    for dim in range(3):
        vertices[:, dim] = splines[dim](u_eval, v_eval).ravel()

    # Triangulate the grid
    triangles = []
    for i in range(mesh_dim - 1):
        for j in range(mesh_dim - 1):
            v0 = i * mesh_dim + j
            v1 = (i + 1) * mesh_dim + j
            v2 = (i + 1) * mesh_dim + (j + 1)
            v3 = i * mesh_dim + (j + 1)
            triangles.append([v0, v1, v2])
            triangles.append([v0, v2, v3])
    triangles = np.array(triangles)

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    o3d_mesh.remove_unreferenced_vertices()
    o3d_mesh.compute_vertex_normals()
    color = get_surface_color("inr")
    o3d_mesh.paint_uniform_color(color)

    trimesh_mesh = trimesh.Trimesh(vertices, triangles)

    # Error: cluster → mesh
    mesh_tree = cKDTree(vertices)
    dists, _ = mesh_tree.query(cluster)
    error = float(np.mean(dists))
    print(f"  [bspline] error={error:.6f}")

    return o3d_mesh, trimesh_mesh, error


def fit_surface(cluster,
                inr_network_parameters,
                np_rng,
                device,
                simple_error_threshold = 8e-3,
                simple_inr_ratio_threshold = 1.5,
                plane_cone_ratio_threshold = 2.5,
                plane_sphere_ratio_threshold = 1.4,
                cone_theta_tolerance_degrees = 5,
                freeform_method = "inr",
                spacing = None,
                cluster_tree = None,
                sphere_fit_kwargs = None,
                cylinder_fit_kwargs = None,
                cone_fit_kwargs = None,
                inr_fit_kwargs = None,
                plane_mesh_kwargs = None,
                sphere_mesh_kwargs = None,
                cylinder_mesh_kwargs = None,
                cone_mesh_kwargs = None,
                inr_mesh_kwargs = None,
                radius_inflation = 0.0,
                angle_inflation_deg = 0.0):

    sphere_fit_kwargs = sphere_fit_kwargs or {}
    cylinder_fit_kwargs = cylinder_fit_kwargs or {}
    cone_fit_kwargs = cone_fit_kwargs or {}
    inr_fit_kwargs = inr_fit_kwargs or {}
    plane_mesh_kwargs = plane_mesh_kwargs or {"mesh_dim": 100}
    sphere_mesh_kwargs = sphere_mesh_kwargs or {"dim_theta": 100, "dim_lambda": 100}
    cylinder_mesh_kwargs = cylinder_mesh_kwargs or {"dim_theta": 100, "dim_height": 50}
    cone_mesh_kwargs = cone_mesh_kwargs or {"dim_theta": 100, "dim_height": 100}
    inr_mesh_kwargs = inr_mesh_kwargs or {"mesh_dim": 100}

    fitter_kwargs = {
        SURFACE_PLANE:    {},
        SURFACE_SPHERE:   sphere_fit_kwargs,
        SURFACE_CYLINDER: cylinder_fit_kwargs,
        SURFACE_CONE:     cone_fit_kwargs,
    }

    # Fit all registered primitives in surface-ID order.
    # results[sid] holds the fit dict; errors[sid] = reconstruction error.
    results = {
        sid: PRIMITIVE_FITTERS[sid](cluster, **fitter_kwargs[sid])
        for sid in sorted(PRIMITIVE_FITTERS)
    }
    errors = np.array([results[sid]["error"] for sid in range(len(PRIMITIVE_FITTERS))])
    # Collect all primitive errors for diagnostics (INR added later if fitted)
    _all_errors = {SURFACE_NAMES[sid]: float(results[sid]["error"])
                   for sid in sorted(PRIMITIVE_FITTERS)}
    # Turn off the cone?
    # errors[SURFACE_CONE] = float("inf")
    simple_min = np.argmin(errors)

    if errors[simple_min] < simple_error_threshold:
        # Spetial treatment for cone
        # print(f"Plane error: {plane_result['error']:.4f} Sphere error: {sphere_result['error']:.4f} Cylinder error: {cylinder_result['error']:.4f} Cone error: {cone_result['error']:.4f}")

        if simple_min == SURFACE_CONE:
            cone_results = cone_special_handling(results, errors, simple_error_threshold, plane_cone_ratio_threshold, cone_theta_tolerance_degrees)
            if cone_results == SURFACE_PLANE:
                cone_results = plane_sphere_arbitration(errors, plane_sphere_ratio_threshold)

            if cone_results != -1:
                mesh = resolve_mesh(cone_results, results[cone_results], cluster, np_rng, device,
                            plane_mesh_kwargs, sphere_mesh_kwargs, cylinder_mesh_kwargs, cone_mesh_kwargs, inr_mesh_kwargs,
                            radius_inflation=radius_inflation, angle_inflation_deg=angle_inflation_deg)

                return {"surface_id": cone_results, "result": results[cone_results], "mesh": mesh[0], "trimesh_mesh": mesh[1], "all_errors": _all_errors}

        else:
            surface_to_use = simple_min
            if simple_min == SURFACE_PLANE or simple_min == SURFACE_SPHERE:
                surface_to_use = plane_sphere_arbitration(errors, plane_sphere_ratio_threshold)
            mesh = resolve_mesh(surface_to_use, results[surface_to_use], cluster, np_rng, device,
            plane_mesh_kwargs, sphere_mesh_kwargs, cylinder_mesh_kwargs, cone_mesh_kwargs, inr_mesh_kwargs,
            radius_inflation=radius_inflation, angle_inflation_deg=angle_inflation_deg)
            return {"surface_id": surface_to_use, "result": results[surface_to_use], "mesh": mesh[0], "trimesh_mesh": mesh[1], "all_errors": _all_errors}
        
    errors_str = "  ".join(f"{SURFACE_NAMES[sid]}={errors[sid]:.6f}" for sid in range(len(PRIMITIVE_FITTERS)))
    print(f"  [surface fitter] no primitive below threshold ({simple_error_threshold:.4f}), "
          f"best={SURFACE_NAMES[simple_min]} ({errors[simple_min]:.6f})")
    print(f"  [surface fitter] primitive errors: {errors_str}")

    if freeform_method == "bpa":
        assert spacing is not None, "spacing must be provided for BPA freeform method"
        assert cluster_tree is not None, "cluster_tree must be provided for BPA freeform method"
        print(f"  [surface fitter] generating BPA mesh ...")
        bpa_mesh, bpa_trimesh = generate_bpa_mesh(cluster, spacing)
        mesh_verts = np.asarray(bpa_mesh.vertices)
        from scipy.spatial import cKDTree
        mesh_tree = cKDTree(mesh_verts)
        dists, _ = mesh_tree.query(cluster)
        bpa_error = float(np.mean(dists))
        print(f"  [bpa] error={bpa_error:.6f}  ({len(mesh_verts)} mesh vertices, {len(cluster)} cluster points)")
        bpa_result = {
            "surface_type": "bpa",
            "error": bpa_error,
            "params": {},
        }
        _all_errors["bpa"] = bpa_error
        return {"surface_id": SURFACE_INR, "result": bpa_result, "mesh": bpa_mesh, "trimesh_mesh": bpa_trimesh, "all_errors": _all_errors}

    if freeform_method == "bpa_bspline":
        assert spacing is not None, "spacing must be provided for bpa_bspline freeform method"
        print(f"  [surface fitter] fitting B-spline (BPA → LSCM → B-spline) ...")
        bspline_mesh, bspline_trimesh, bspline_error = generate_bspline_mesh(cluster, spacing, uv_margin=0)
        bspline_result = {
            "surface_type": "bpa_bspline",
            "error": bspline_error,
            "params": {},
        }
        _all_errors["bpa_bspline"] = bspline_error
        return {"surface_id": SURFACE_INR, "result": bspline_result, "mesh": bspline_mesh, "trimesh_mesh": bspline_trimesh, "all_errors": _all_errors}

    print(f"  [surface fitter] fitting INR ...")
    inr_result = fit_inr(cluster, inr_network_parameters, device = device, **inr_fit_kwargs)
    results[SURFACE_INR] = inr_result
    errors = np.append(errors, inr_result["error"])
    _all_errors[SURFACE_NAMES[SURFACE_INR]] = float(inr_result["error"])

    global_min = np.argmin(errors)
    resulting_min = global_min

    if global_min == SURFACE_INR and ratio(errors[simple_min] , errors[SURFACE_INR]) < simple_inr_ratio_threshold:
        resulting_min = simple_min
        if resulting_min == SURFACE_CONE:
            resulting_min = cone_special_handling(results, errors[:-1], simple_error_threshold, plane_cone_ratio_threshold, cone_theta_tolerance_degrees)
            # It could happen that the cone handling function returned -1: second best simple surface is not within the given error threshold
            # In that case, use INR.
            if resulting_min == -1:
                resulting_min = global_min
            
            elif resulting_min == SURFACE_PLANE:
                resulting_min = plane_sphere_arbitration(errors[:-1], plane_sphere_ratio_threshold)
        elif resulting_min == SURFACE_PLANE or resulting_min == SURFACE_SPHERE:
            resulting_min = plane_sphere_arbitration(errors[:-1], plane_sphere_ratio_threshold)

    mesh = resolve_mesh(resulting_min, results[resulting_min], cluster, np_rng, device,
                        plane_mesh_kwargs, sphere_mesh_kwargs, cylinder_mesh_kwargs, cone_mesh_kwargs, inr_mesh_kwargs,
                        radius_inflation=radius_inflation, angle_inflation_deg=angle_inflation_deg)

    return {"surface_id": resulting_min, "result": results[resulting_min], "mesh": mesh[0], "trimesh_mesh": mesh[1], "all_errors": _all_errors}