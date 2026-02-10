import numpy as np
import os
import open3d as o3d
import matplotlib.pyplot as plt
import time
import torch
import primitive_fitting_utils

from primitive_fitting import fit_plane_numpy, fit_sphere_numpy, fit_cylinder, fit_cylinder_optimized, fit_cone
from surface_fitter import fit_surface, SURFACE_NAMES

try:
    import mesh_postprocessing
    HAS_PYMESH = True
except ImportError:
    HAS_PYMESH = False
    print("PyMesh not available. Post-processing (clipping/topology) will be skipped.")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

NORMALIZE_POINTS = False

def normalize_points(points):
    points = points - np.mean(points, axis = 0, keepdims = True)

    S, U = np.linalg.eig(points.T @ points)
    smallest_ev = U[:, np.argmin(S)]
    R = primitive_fitting_utils.rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
    points = (R @ points.T).T

    extents = np.max(points, axis = 0) - np.min(points, axis = 0)
    points = points / (np.max(extents) + 1e-7)

    return points.astype(np.float32)

def cylinder_benchmark(cluster):
    res_native = fit_cylinder(cluster)
    print(f"Cylinder fitting time with native code: {res_native[-1]} seconds.")
    print(f"Native cylinder error: {primitive_fitting_utils.cylinder_error(cluster, res_native[1], res_native[0], res_native[2])}")

    res_optimized = fit_cylinder_optimized(cluster)
    print(f"Cylinder fitting time with optimized code: {res_optimized['metadata']['fitting_time_seconds']} seconds.")
    print(f"Optimized cylinder error: {res_optimized['error']}")
    axis_cylinder_native, center_cylinder_native, radius_cylinder_native, _, _ = res_native

    print(f"L2 difference between axis vectors: {np.linalg.norm(axis_cylinder_native - res_optimized['params']['a'])}")
    print(f"L2 difference between center points: {np.linalg.norm(center_cylinder_native - res_optimized['params']['center'])}")
    print(f"Difference in fitted radii: {abs(radius_cylinder_native - res_optimized['params']['radius'])}")

    speedup = (res_native[-1]) / res_optimized["metadata"]["fitting_time_seconds"]
    print(f"Speedup factor: {speedup}")
    print(f"Cylinder convergence status: {res_optimized['metadata']['optimizer_converged']}\n")

if __name__ == "__main__":
    out_dir = "output"

    # Docker mode: run full pipeline and save all output files
    if HAS_PYMESH:
        path = os.path.join("sample_clouds", "abc_00470.xyzc")
        data = np.loadtxt(path)
        points = data[:, :3]
        if NORMALIZE_POINTS:
            points = normalize_points(points)
            data[:, :3] = points
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

        meshes = []
        trimesh_meshes = []
        surface_types = []
        cluster_points_list = []

        device = "cuda:0"
        np_rng = np.random.default_rng(41)
        torch.manual_seed(41)
        torch.cuda.manual_seed(41)

        for cluster_id in unique_clusters:
            cluster = data[data[:, 3] == cluster_id][:, :3].astype(np.float32)
            print(f"Processing cluster with id {cluster_id}.")
            print(f"Number of points in the current cluster: {cluster.shape[0]}")
            # cylinder_benchmark(cluster)
            # continue

            fitting_result = fit_surface(cluster, {
                "hidden_dim": 64,
                "use_shortcut": False,
                "fraction_siren": 0.5
            }, np_rng, device,
                plane_cone_ratio_threshold = 4,
                cone_theta_tolerance_degrees = 10,
                plane_mesh_kwargs = {"mesh_dim": 100, "threshold_multiplier": 2},
                sphere_mesh_kwargs = {"dim_theta": 100, "dim_lambda": 100, "threshold_multiplier": 2},
                cylinder_mesh_kwargs = {"dim_theta": 100, "dim_height": 50, "threshold_multiplier": 2},
                cone_mesh_kwargs = {"dim_theta": 100, "dim_height": 100, "threshold_multiplier": 2},
                inr_mesh_kwargs = {"mesh_dim": 100, "uv_margin": 0.2}
            )

            surface_type_name = SURFACE_NAMES[fitting_result["surface_id"]]
            print(f"Best surface: {fitting_result['result']['surface_type']}")
            print(f"Error: {fitting_result['result']['error']}")

            meshes.append(fitting_result["mesh"])
            trimesh_meshes.append(fitting_result["trimesh_mesh"])
            surface_types.append(surface_type_name)
            cluster_points_list.append(cluster)

        os.makedirs(out_dir, exist_ok = True)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd_path = os.path.join(out_dir, "point_cloud.ply")
        o3d.io.write_point_cloud(pcd_path, pcd)
        print(f"Point cloud saved to {pcd_path}")

        # for i, (m, stype) in enumerate(zip(meshes, surface_types)):
        #     mesh_path = os.path.join(out_dir, f"fitted_{i}_{stype}.ply")
        #     o3d.io.write_triangle_mesh(mesh_path, m)

        unclipped_path = os.path.join(out_dir, "unclipped.ply")
        clipped_path = os.path.join(out_dir, "clipped.ply")
        topology_path = os.path.join(out_dir, "topology.json")

        pm_meshes = mesh_postprocessing.save_unclipped_meshes(
            trimesh_meshes, surface_types, unclipped_path
        )
        print(f"Unclipped meshes saved to {unclipped_path}")

        clipped_meshes = mesh_postprocessing.save_clipped_meshes(
            pm_meshes, cluster_points_list, surface_types, clipped_path
        )
        print(f"Clipped meshes saved to {clipped_path}")

        mesh_postprocessing.save_topology(clipped_meshes, topology_path)
        print(f"Topology saved to {topology_path}")
        print(f"All outputs saved to {out_dir}/")

    else:
        # Host mode: load saved output files and visualize
        import json
        import trimesh as tm

        pcd_path = os.path.join(out_dir, "point_cloud.ply")
        unclipped_path = os.path.join(out_dir, "unclipped.ply")
        clipped_path = os.path.join(out_dir, "clipped.ply")
        topology_path = os.path.join(out_dir, "topology.json")

        if not os.path.exists(clipped_path):
            print(f"No output files found in {out_dir}/. Run inside Docker container first.")
        else:
            pcd = o3d.io.read_point_cloud(pcd_path)
            pcd.paint_uniform_color([0.5, 0.5, 0.5])

            def trimesh_to_o3d(path):
                """Load PLY via trimesh to preserve face colors, convert to O3D with vertex colors."""
                m = tm.load(path)
                o3d_mesh = o3d.geometry.TriangleMesh()
                o3d_mesh.vertices = o3d.utility.Vector3dVector(m.vertices)
                o3d_mesh.triangles = o3d.utility.Vector3iVector(m.faces)
                o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
                    m.visual.vertex_colors[:, :3] / 255.0
                )
                o3d_mesh.compute_vertex_normals()
                return o3d_mesh

            # Build geometry lists for both windows
            unclipped_geoms = []
            clipped_geoms = []

            if os.path.exists(unclipped_path):
                unclipped_geoms.append(trimesh_to_o3d(unclipped_path))
            unclipped_geoms.append(pcd)

            clipped_geoms.append(trimesh_to_o3d(clipped_path))

            if os.path.exists(topology_path):
                with open(topology_path) as f:
                    topo = json.load(f)

                # LineSet for intersection curves, PointCloud for corners
                for curve in topo.get("curves", []):
                    pts = np.array(curve["pv_points"])
                    lines = np.array(curve["pv_lines"])
                    ls = o3d.geometry.LineSet()
                    ls.points = o3d.utility.Vector3dVector(pts)
                    ls.lines = o3d.utility.Vector2iVector(lines)
                    ls.paint_uniform_color([0.3, 0.3, 0.3])
                    clipped_geoms.append(ls)

                corners = topo.get("corners", [])
                if len(corners) > 0:
                    corner_pcd = o3d.geometry.PointCloud()
                    corner_pcd.points = o3d.utility.Vector3dVector(np.array(corners))
                    corner_pcd.paint_uniform_color([0.15, 0.15, 0.15])
                    clipped_geoms.append(corner_pcd)

            # Show both windows in parallel using poll_events loop
            vis1 = o3d.visualization.Visualizer()
            vis1.create_window(window_name="Unclipped meshes + point cloud", width=960, height=720, left=0, top=50)
            for g in unclipped_geoms:
                vis1.add_geometry(g)
            vis1.get_render_option().mesh_show_back_face = True
            vis1.get_render_option().point_size = 2.0

            vis2 = o3d.visualization.Visualizer()
            vis2.create_window(window_name="Clipped meshes + topology", width=960, height=720, left=960, top=50)
            for g in clipped_geoms:
                vis2.add_geometry(g)
            vis2.get_render_option().mesh_show_back_face = True

            running1, running2 = True, True
            while running1 and running2:
                if running1:
                    running1 = vis1.poll_events()
                    vis1.update_renderer()
                if running2:
                    running2 = vis2.poll_events()
                    vis2.update_renderer()
                time.sleep(0.01)

            vis1.destroy_window()
            vis2.destroy_window()
