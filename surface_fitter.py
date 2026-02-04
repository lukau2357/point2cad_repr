import numpy as np

from inr_fitting import fit_inr
from primitive_fitting import fit_plane_numpy, fit_sphere_numpy, fit_cylinder_optimized, fit_cone
from inr_fitting import fit_inr

def ratio(x, y, eps = 1e-8):
    return (x + eps) / (y + eps)

def fit_surface(cluster, 
                inr_network_parameters,
                simple_error_threshold = 8e-3,
                simple_inr_ratio_threshold = 1.5,
                plane_cone_ratio_threshold = 2.5,
                sphere_kwargs = None, 
                cylinder_kwargs = None, 
                cone_kwargs = None, 
                inr_kwargs = None):
    
    plane_result = fit_plane_numpy(cluster)
    sphere_result = fit_sphere_numpy(cluster, **sphere_kwargs) if sphere_kwargs is not None else fit_sphere_numpy(cluster)
    cylinder_result = fit_cylinder_optimized(cluster, **cylinder_kwargs) if cylinder_kwargs is not None else fit_cylinder_optimized(cluster)
    cone_result = fit_cone(cluster, **cone_kwargs) if cone_kwargs is not None else fit_cone(cluster)

    results = [plane_result, sphere_result, cylinder_result, cone_result]
    errors = np.array([plane_result["error"], sphere_result["error"], cylinder_result["error"], cone_result["error"]])

    global_min = np.argmin(errors)
    print(errors, ratio(errors[0], errors[3]))

    if errors[global_min] < simple_error_threshold:
        # SPECIAL CASE: Cone is the best fit, and errors for cone and plane are within the given margin => always prefer 
        # the plane in that case!
        if global_min == 3 and (errors[0] / errors[3]) < plane_cone_ratio_threshold:
            global_min = 0

        print(f"Best surface type for the given cluster: {results[global_min]['surface_type']}")
        print(f"Resulting error: {errors[global_min]}") 
        return results[global_min]

    inr_result = fit_inr(cluster, inr_network_parameters, **inr_kwargs) if inr_kwargs is not None else fit_inr(cluster, inr_network_parameters)
    results.append(inr_result)
    errors = np.append(errors, [inr_result["error"]])

    global_min = np.argmin(errors)
    simple_min = np.argmin(errors[:-1])
    resulting_min = global_min

    # INR is worse than a simple surface within a given margin => rollback to simple surface.
    if global_min == 4 and ratio(errors[simple_min], errors[global_min]) < simple_inr_ratio_threshold:
        resulting_min = simple_min
    
    # Same edge case as before.
    if resulting_min == 3 and (errors[0] / errors[3]) < plane_cone_ratio_threshold:
        resulting_min = 0

    print(f"Best surface type for the given cluster: {results[resulting_min]['surface_type']}")
    print(f"Resulting error: {errors[resulting_min]}") 
    return results[resulting_min]