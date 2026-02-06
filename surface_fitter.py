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
            cluster_mean = params["cluster_mean"],
            cluster_scale = params["cluster_scale"],
            **inr_mesh_kwargs
        )

def fit_surface(cluster,
                inr_network_parameters,
                np_rng,
                device,
                simple_error_threshold = 8e-3,
                simple_inr_ratio_threshold = 1.5,
                plane_cone_ratio_threshold = 2.5,
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

    cluster_mean = cluster.mean(axis = 0)
    cluster_scale = cluster.std(axis = 0).max()
    cluster_normalized = (cluster - cluster_mean) / (cluster_scale + 1e-6)

    plane_result = fit_plane_numpy(cluster_normalized)
    sphere_result = fit_sphere_numpy(cluster_normalized, **sphere_fit_kwargs)
    cylinder_result = fit_cylinder_optimized(cluster_normalized, **cylinder_fit_kwargs)
    cone_result = fit_cone(cluster_normalized, **cone_fit_kwargs)

    results = [plane_result, sphere_result, cylinder_result, cone_result]
    errors = np.array([r["error"] for r in results])
    global_min = np.argmin(errors)

    if errors[global_min] < simple_error_threshold:
        if global_min == SURFACE_CONE and ratio(errors[SURFACE_PLANE], errors[SURFACE_CONE]) < plane_cone_ratio_threshold:
            global_min = SURFACE_PLANE

        mesh = resolve_mesh(global_min, results[global_min], cluster_normalized, np_rng, device,
                            plane_mesh_kwargs, sphere_mesh_kwargs, cylinder_mesh_kwargs, cone_mesh_kwargs, inr_mesh_kwargs)
        return {"surface_id": global_min, "result": results[global_min], "mesh": mesh}

    inr_result = fit_inr(cluster, inr_network_parameters, device = device, **inr_fit_kwargs)
    results.append(inr_result)
    errors = np.append(errors, inr_result["error"])

    global_min = np.argmin(errors)
    simple_min = np.argmin(errors[:-1])
    resulting_min = global_min

    if global_min == SURFACE_INR and ratio(errors[simple_min], errors[global_min]) < simple_inr_ratio_threshold:
        resulting_min = simple_min

    if resulting_min == SURFACE_CONE and ratio(errors[SURFACE_PLANE], errors[SURFACE_CONE]) < plane_cone_ratio_threshold:
        resulting_min = SURFACE_PLANE

    mesh_cluster = cluster if resulting_min == SURFACE_INR else cluster_normalized
    mesh = resolve_mesh(resulting_min, results[resulting_min], mesh_cluster, np_rng, device,
                        plane_mesh_kwargs, sphere_mesh_kwargs, cylinder_mesh_kwargs, cone_mesh_kwargs, inr_mesh_kwargs)

    return {"surface_id": resulting_min, "result": results[resulting_min], "mesh": mesh}