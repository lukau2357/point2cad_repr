import numpy as np
import os
import open3d as o3d
import matplotlib.pyplot as plt
import time
import torch

from primitive_fitting import fit_plane, fit_plane_numpy, fit_sphere_numpy, fit_cylinder, fit_cylinder_optimized, fit_cone

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
        cluster = data[data[:, 3] == cluster_id][:, :3]
        start_t = time.time()
        print(f"Number of points in the current cluster: {cluster.shape[0]}")
        a_t, d_t = fit_plane(torch.tensor(cluster, device = DEVICE))
        end_t = time.time()

        start_np = time.time()
        a_np, d_np = fit_plane_numpy(cluster)
        end_np = time.time()

        time_t = end_t - start_t
        time_np = end_np - start_np

        # print(f"Normal vectors norm difference: {np.linalg.norm(a_t.cpu().numpy() - a_np)}")
        # print(f"Distance parameter absolute difference: {abs(d_t - d_np)}")
        # print(f"Numpy vs Torch fitting time difference: {time_np - time_t}")

        start_c = time.time()
        res = fit_cylinder(cluster)
        print(res)
        end_c = time.time()
        print(f"Cylinder fitting time: {end_c - start_c} seconds.")

        start_cc = time.time()
        res = fit_cylinder_optimized(cluster)
        print(res)
        end_cc = time.time()
        print(f"Cylinder fitting time with optimized code: {end_cc - start_cc} seconds.")

        speedup = (end_c - start_c) / (end_cc - start_cc)
        print(f"Speedup factor: {speedup}")

        cone_start = time.time()
        cone_res = fit_cone(points)
        print(cone_res)
        cone_end = time.time()
        print(f"Time taken for cone fitting: {cone_end - cone_start} seconds.")

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