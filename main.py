import numpy as np
import os
import open3d as o3d
import matplotlib.pyplot as plt
import time
import torch
import primitive_fitting_utils

from primitive_fitting import fit_plane_numpy, fit_sphere_numpy, fit_cylinder, fit_cylinder_optimized, fit_cone
from surface_fitter import fit_surface

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

if __name__ == "__main__":
    path = os.path.join("sample_clouds", "abc_00470.xyzc")
    data = np.loadtxt(path)
    points = data[:, :3]
    clusters = data[:, -1].astype(int)

    unique_clusters = np.unique(clusters)
    num_clusters = len(unique_clusters)

    # Map cluster IDs to [0, 1]
    cluster_id_to_color_idx = {
        cid: i / max(num_clusters - 1, 1)
        for i, cid in enumerate(unique_clusters)
    }

    colors = np.array([
        plt.cm.tab20(cluster_id_to_color_idx[cid])[:3]
        for cid in clusters
    ])

    for cluster_id in unique_clusters:
        cluster = data[data[:, 3] == cluster_id][:, :3].astype(np.float32)
        print(f"Processing cluster with id {cluster_id}.")
        print(f"Number of points in the current cluster: {cluster.shape[0]}")

        fit_surface(cluster, inr_network_parameters = {
            "hidden_dim": 64,
            "use_shortcut": True,
            "fraction_siren": 0.5
        })

        # res_plane = fit_plane_numpy(cluster)
        # print(f"Time taken for plane fitting: {res_plane['metadata']['fitting_time_seconds']} seconds.")
        # print(f"Plane error: {res_plane['error']}")

        # res_sphere = fit_sphere_numpy(cluster)
        # print(f"Time taken for sphere fitting: {res_sphere['metadata']['fitting_time_seconds']} seconds.")
        # print(f"Sphere error: {res_sphere['error']}")

        # res_native = fit_cylinder(cluster)
        # end_c = time.time()
        # print(f"Cylinder fitting time with native code: {res_native[-1]} seconds.")
        # print(f"Native cylinder error: {primitive_fitting_utils.cylinder_error(cluster, res_native[1], res_native[0], res_native[2])}")

        # res_optimized = fit_cylinder_optimized(cluster)
        # print(f"Cylinder fitting time with optimized code: {res_optimized['metadata']['fitting_time_seconds']} seconds.")
        # print(f"Optimized cylinder error: {res_optimized['error']}")
        # axis_cylinder_native, center_cylinder_native, radius_cylinder_native, _, _ = res_native

        # print(f"L2 difference between axis vectors: {np.linalg.norm(axis_cylinder_native - res_optimized['params']['a'])}")
        # print(f"L2 difference between center points: {np.linalg.norm(center_cylinder_native - res_optimized['params']['center'])}")
        # print(f"Difference in fitted radii: {abs(radius_cylinder_native - res_optimized['params']['radius'])}")

        # speedup = (res_native[-1]) / res_optimized["metadata"]["fitting_time_seconds"]
        # print(f"Speedup factor: {speedup}")
        # print(f"Cylinder convergence status: {res_optimized['metadata']['optimizer_converged']}")

        # cone_res = fit_cone(cluster)
        # print(f"Time taken for cone fitting: {cone_res["metadata"]["fitting_time_seconds"]} seconds.")
        # print(f"Cone error: {cone_res['error']}")
        # print(f"Obtained cone angle: {cone_res['params']['theta']}")
        # print(f"Cone convergence status: {cone_res['metadata']['optimizer_converged']}\n")

    # O3D point cloud:
    # https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html#open3d.geometry.PointCloud.points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # O3D utilities:
    # https://www.open3d.org/docs/release/python_api/open3d.utility.Vector3dVector.html
    
    # Draw geometries:
    # https://www.open3d.org/docs/release/python_api/open3d.visualization.draw_geometries.html
    o3d.visualization.draw_geometries([pcd])