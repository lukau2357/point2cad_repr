import numpy as np

from inr_fitting import fit_inr
from primitive_fitting import fit_plane_numpy, fit_sphere_numpy, fit_cylinder_optimized, fit_cone
from primitive_fitting_utils import generate_plane_mesh, generate_sphere_mesh, generate_cylinder_mesh, generate_cone_mesh

SURFACE_PLANE = 0
SURFACE_SPHERE = 1
SURFACE_CYLINDER = 2
SURFACE_CONE = 3
SURFACE_INR = 4

def ratio(x, y, eps = 1e-8):
    return (x + eps) / (y + eps)

def resolve_mesh(surface_id,
                 result,
                 cluster,
                 np_rng,
                 device,
                 plane_mesh_kwargs,
                 sphere_mesh_kwargs,
                 cylinder_mesh_kwargs,
                 cone_mesh_kwargs,
                 inr_mesh_kwargs):

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
        return generate_sphere_mesh(
            radius = params["radius"],
            center = params["center"],
            cluster = cluster,
            device = device,
            **sphere_mesh_kwargs
        )

    if surface_id == SURFACE_CYLINDER:
        return generate_cylinder_mesh(
            radius = params["radius"],
            center = params["center"],
            axis = params["a"],
            cluster = cluster,
            device = device,
            **cylinder_mesh_kwargs
        )

    if surface_id == SURFACE_CONE:
        return generate_cone_mesh(
            vertex = params["v"],
            axis = params["a"],
            theta = params["theta"],
            cluster_points = cluster,
            device = device,
            **cone_mesh_kwargs
        )

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
    print(f"0 diff: {cone_diff_0} PI/2 diff: {cone_diff_pi2}")
    if abs(cone_angle - np.pi / 2) < cone_theta_tolerance_rad or cone_angle < cone_theta_tolerance_rad:
        simple_min = np.argmin(errors[:-1])

        return simple_min if errors[simple_min] < simple_error_threshold else -1
    
    return SURFACE_CONE
        

def fit_surface(cluster,
                inr_network_parameters,
                np_rng,
                device,
                simple_error_threshold = 8e-3,
                simple_inr_ratio_threshold = 1.5,
                plane_cone_ratio_threshold = 2.5,
                cone_theta_tolerance_degrees = 5,
                sphere_fit_kwargs = None,
                cylinder_fit_kwargs = None,
                cone_fit_kwargs = None,
                inr_fit_kwargs = None,
                plane_mesh_kwargs = None,
                sphere_mesh_kwargs = None,
                cylinder_mesh_kwargs = None,
                cone_mesh_kwargs = None,
                inr_mesh_kwargs = None):

    sphere_fit_kwargs = sphere_fit_kwargs or {}
    cylinder_fit_kwargs = cylinder_fit_kwargs or {}
    cone_fit_kwargs = cone_fit_kwargs or {}
    inr_fit_kwargs = inr_fit_kwargs or {}
    plane_mesh_kwargs = plane_mesh_kwargs or {"mesh_dim": 100}
    sphere_mesh_kwargs = sphere_mesh_kwargs or {"dim_theta": 100, "dim_lambda": 100}
    cylinder_mesh_kwargs = cylinder_mesh_kwargs or {"dim_theta": 100, "dim_height": 50}
    cone_mesh_kwargs = cone_mesh_kwargs or {"dim_theta": 100, "dim_height": 100}
    inr_mesh_kwargs = inr_mesh_kwargs or {"mesh_dim": 100}

    plane_result = fit_plane_numpy(cluster)
    sphere_result = fit_sphere_numpy(cluster, **sphere_fit_kwargs)
    cylinder_result = fit_cylinder_optimized(cluster, **cylinder_fit_kwargs)
    cone_result = fit_cone(cluster, **cone_fit_kwargs)

    results = [plane_result, sphere_result, cylinder_result, cone_result]
    errors = np.array([r["error"] for r in results])
    # Turn off the cone?
    # errors[SURFACE_CONE] = float("inf")
    simple_min = np.argmin(errors)

    if errors[simple_min] < simple_error_threshold:
        # Spetial treatment for cone
        if simple_min == SURFACE_CONE:
            cone_results = cone_special_handling(results, errors, simple_error_threshold, plane_cone_ratio_threshold, cone_theta_tolerance_degrees)
        
            if cone_results != -1:
                mesh = resolve_mesh(cone_results, results[cone_results], cluster, np_rng, device,
                            plane_mesh_kwargs, sphere_mesh_kwargs, cylinder_mesh_kwargs, cone_mesh_kwargs, inr_mesh_kwargs)

                return {"surface_id": cone_results, "result": results[cone_results], "mesh": mesh}
        
        else:
            mesh = resolve_mesh(simple_min, results[simple_min], cluster, np_rng, device,
            plane_mesh_kwargs, sphere_mesh_kwargs, cylinder_mesh_kwargs, cone_mesh_kwargs, inr_mesh_kwargs)
            return {"surface_id": simple_min, "result": results[simple_min], "mesh": mesh}
        
    inr_result = fit_inr(cluster, inr_network_parameters, device = device, **inr_fit_kwargs)
    results.append(inr_result)
    errors = np.append(errors, inr_result["error"])

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

    mesh = resolve_mesh(resulting_min, results[resulting_min], cluster, np_rng, device,
                        plane_mesh_kwargs, sphere_mesh_kwargs, cylinder_mesh_kwargs, cone_mesh_kwargs, inr_mesh_kwargs)

    if resulting_min == SURFACE_INR:
        return {"surface_id": resulting_min, "result": results[resulting_min], "mesh": mesh[0], "trimesh_mesh": mesh[1]}

    return {"surface_id": resulting_min, "result": results[resulting_min], "mesh": mesh[0], "trimesh_mesh": mesh[1]}