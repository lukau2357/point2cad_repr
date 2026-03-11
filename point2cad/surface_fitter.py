import numpy as np

from .inr_fitting import fit_inr
from .primitive_fitting import fit_plane_numpy, fit_sphere_numpy, fit_cylinder_optimized, fit_cone
from .primitive_fitting_utils import generate_plane_mesh, generate_sphere_mesh, generate_cylinder_mesh, generate_cone_mesh
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
    
    if cone_diff_pi2 < cone_theta_tolerance_rad or cone_diff_0 < cone_theta_tolerance_rad:
        simple_min = np.argmin(errors[:-1])

        return simple_min if errors[simple_min] < simple_error_threshold else -1
    
    return SURFACE_CONE
        

def plane_special_handling(results, errors, plane_sphere_ratio_threshold):
    """
    If plane is selected as the best simple surface, check whether a sphere
    fits significantly better.  Returns SURFACE_SPHERE when

        plane_error / sphere_error  >=  plane_sphere_ratio_threshold

    (i.e. the sphere must be much better to justify replacing the plane),
    otherwise returns SURFACE_PLANE.
    """
    if ratio(errors[SURFACE_PLANE], errors[SURFACE_SPHERE]) >= plane_sphere_ratio_threshold:
        return SURFACE_SPHERE
    return SURFACE_PLANE


def fit_surface(cluster,
                inr_network_parameters,
                np_rng,
                device,
                simple_error_threshold = 8e-3,
                simple_inr_ratio_threshold = 1.5,
                plane_cone_ratio_threshold = 2.5,
                plane_sphere_ratio_threshold = 2.5,
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
        
            if cone_results != -1:
                mesh = resolve_mesh(cone_results, results[cone_results], cluster, np_rng, device,
                            plane_mesh_kwargs, sphere_mesh_kwargs, cylinder_mesh_kwargs, cone_mesh_kwargs, inr_mesh_kwargs)

                return {"surface_id": cone_results, "result": results[cone_results], "mesh": mesh[0], "trimesh_mesh": mesh[1], "all_errors": _all_errors}

        else:
            surface_to_use = simple_min
            if simple_min == SURFACE_PLANE:
                surface_to_use = plane_special_handling(results, errors, plane_sphere_ratio_threshold)
            mesh = resolve_mesh(surface_to_use, results[surface_to_use], cluster, np_rng, device,
            plane_mesh_kwargs, sphere_mesh_kwargs, cylinder_mesh_kwargs, cone_mesh_kwargs, inr_mesh_kwargs)
            return {"surface_id": surface_to_use, "result": results[surface_to_use], "mesh": mesh[0], "trimesh_mesh": mesh[1], "all_errors": _all_errors}
        
    errors_str = "  ".join(f"{SURFACE_NAMES[sid]}={errors[sid]:.6f}" for sid in range(len(PRIMITIVE_FITTERS)))
    print(f"  [surface fitter] no primitive below threshold ({simple_error_threshold:.4f}), "
          f"best={SURFACE_NAMES[simple_min]} ({errors[simple_min]:.6f})")
    print(f"  [surface fitter] primitive errors: {errors_str}")
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
            resulting_min = plane_special_handling(results, errors[:-1], plane_sphere_ratio_threshold)

    mesh = resolve_mesh(resulting_min, results[resulting_min], cluster, np_rng, device,
                        plane_mesh_kwargs, sphere_mesh_kwargs, cylinder_mesh_kwargs, cone_mesh_kwargs, inr_mesh_kwargs)

    return {"surface_id": resulting_min, "result": results[resulting_min], "mesh": mesh[0], "trimesh_mesh": mesh[1], "all_errors": _all_errors}