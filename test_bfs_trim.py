"""
Prototype: fit surfaces and compare grid-trimmed vs BFS flood-fill trimming.

Compute:
    python test_bfs_trim.py --model_id abc_00000406

Visualize:
    python test_bfs_trim.py --model_id abc_00000406 --visualize
"""
import argparse
import glob as _glob
import os
import shutil
import time

import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def run_compute(args):
    import matplotlib.pyplot as plt
    import torch

    import point2cad.primitive_fitting_utils as pfu
    from point2cad.color_config import get_surface_color
    from point2cad.mesh_postprocessing_2 import clip_meshes_bfs, build_cluster_trees
    from point2cad.surface_fitter import SURFACE_NAMES, fit_surface

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    def normalize_points(pts):
        pts = pts - np.mean(pts, axis=0)
        S, U = np.linalg.eigh(pts.T @ pts)
        R = pfu.rotation_matrix_a_to_b(U[:, np.argmin(S)], np.array([1, 0, 0]))
        pts = (R @ pts.T).T
        extents = np.max(pts, axis=0) - np.min(pts, axis=0)
        scale = float(np.max(extents) + 1e-7)
        return (pts / scale).astype(np.float32)

    input_pattern = os.path.join(args.input_dir, args.model_id, "*.xyzc")
    xyzc_files = sorted(_glob.glob(input_pattern),
                        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    if not xyzc_files:
        print(f"No .xyzc files found matching: {input_pattern}")
        return
    if args.part >= len(xyzc_files):
        print(f"Part {args.part} out of range — model has {len(xyzc_files)} part(s)")
        return

    sample_path = xyzc_files[args.part]
    out_dir = os.path.join(args.output_dir, args.model_id, f"part_{args.part}")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        print(f"Removed old results: {out_dir}")
    os.makedirs(out_dir)

    print(f"[bfs-trim] Input:  {sample_path}")
    print(f"[bfs-trim] Output: {out_dir}")

    np_rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    data = np.loadtxt(sample_path)
    data[:, :3] = normalize_points(data[:, :3])
    unique_clusters, cluster_counts = np.unique(data[:, -1].astype(int), return_counts=True)
    n_clusters = len(unique_clusters)
    print(f"[bfs-trim] {n_clusters} clusters")

    clusters = []
    o3d_meshes = []
    surface_type_names = []

    for idx, (cid, c_count) in enumerate(zip(unique_clusters, cluster_counts)):
        cluster = data[data[:, -1] == cid][:, :3].astype(np.float32)
        clusters.append(cluster)
        np.save(os.path.join(out_dir, f"cluster_{idx}.npy"), cluster)

        print(f"[surface fitter] Cluster {cid} ({c_count} pts) fitting ...")
        res = fit_surface(
            cluster,
            {"hidden_dim": 64, "use_shortcut": True, "fraction_siren": 0.5},
            np_rng, DEVICE,
            inr_fit_kwargs={
                "max_steps": 1500,
                "noise_magnitude_3d": 0.05,
                "noise_magnitude_uv": 0.05,
                "initial_lr": 1e-1,
            },
            plane_mesh_kwargs={"mesh_dim": 200, "threshold_multiplier": 1e6,
                               "plane_sampling_deviation": 0.5},
            sphere_mesh_kwargs={"dim_theta": 200, "dim_lambda": 200,
                                "threshold_multiplier": 1e6},
            cylinder_mesh_kwargs={"dim_theta": 200, "dim_height": 100,
                                  "threshold_multiplier": 1e6, "cylinder_height_margin": 0.5},
            cone_mesh_kwargs={"dim_theta": 200, "dim_height": 200,
                              "threshold_multiplier": 1e6, "cone_height_margin": 0.5},
            inr_mesh_kwargs={"mesh_dim": 200, "uv_margin": 0.1, "threshold_multiplier": 1e6},
        )

        sid = res["surface_id"]
        stype = SURFACE_NAMES[sid]
        chosen_err = res["result"]["error"]
        all_errors = res.get("all_errors", {})
        errors_str = "  ".join(f"{name}={err:.6f}" for name, err in all_errors.items())
        print(f"[surface fitter] Cluster {cid} ({c_count} pts) → {stype}  residual={chosen_err:.6f}")
        if errors_str:
            print(f"  all errors: {errors_str}")

        o3d_meshes.append(res["mesh"])
        surface_type_names.append(stype)

        # Save unclipped mesh
        verts = np.asarray(res["mesh"].vertices)
        tris  = np.asarray(res["mesh"].triangles)
        np.savez(os.path.join(out_dir, f"surface_mesh_{idx}.npz"),
                 vertices=verts, triangles=tris)

    # Precompute KDTrees and intra-cluster NN spacings
    print("[bfs-trim] Precomputing cluster KDTrees ...")
    cluster_trees, spacings = build_cluster_trees(clusters, args.spacing_percentile)

    # BFS flood-fill trimming + post-filter
    print("[bfs-trim] Running BFS flood-fill trimming ...")
    clipped_meshes = clip_meshes_bfs(o3d_meshes, clusters, surface_type_names,
                                     cluster_trees, spacings,
                                     post_filter_threshold=args.post_filter_threshold)

    for idx, mesh in enumerate(clipped_meshes):
        verts = np.asarray(mesh.vertices)
        tris  = np.asarray(mesh.triangles)
        np.savez(os.path.join(out_dir, f"trimmed_mesh_{idx}.npz"),
                 vertices=verts, triangles=tris)

    # Metadata — cluster colors derived from fitted surface types
    cluster_colors = np.array([get_surface_color(stype) for stype in surface_type_names])
    np.savez(os.path.join(out_dir, "metadata.npz"),
             n_clusters=n_clusters,
             cluster_ids=unique_clusters,
             cluster_colors=cluster_colors,
             surface_names=np.array(surface_type_names))

    print(f"[bfs-trim] Saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Visualize
# ---------------------------------------------------------------------------

def run_visualize(args):
    out_dir = os.path.join(args.output_dir, args.model_id, f"part_{args.part}")
    meta_path = os.path.join(out_dir, "metadata.npz")
    if not os.path.exists(meta_path):
        print(f"No saved results found in {out_dir}/ — run compute first.")
        return

    meta           = np.load(meta_path, allow_pickle=True)
    n_clusters     = int(meta["n_clusters"])
    cluster_colors = meta["cluster_colors"]
    surface_names  = meta["surface_names"]

    from point2cad.color_config import get_surface_color

    # Point cloud: one colored pcd per cluster, merged
    pcd_combined = o3d.geometry.PointCloud()
    for i in range(n_clusters):
        pts = np.load(os.path.join(out_dir, f"cluster_{i}.npy"))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color(cluster_colors[i].tolist())
        pcd_combined += pcd

    # Unclipped meshes — kept as individual list for N/P cycling
    surface_meshes = []
    surface_mesh_ids = []
    for i in range(n_clusters):
        path = os.path.join(out_dir, f"surface_mesh_{i}.npz")
        if not os.path.exists(path):
            continue
        d = np.load(path)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices  = o3d.utility.Vector3dVector(d["vertices"])
        mesh.triangles = o3d.utility.Vector3iVector(d["triangles"])
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(get_surface_color(str(surface_names[i])))
        surface_meshes.append(mesh)
        surface_mesh_ids.append(i)

    # N/P highlight state: -1 = show all
    _highlight = {"idx": -1}

    def _update_highlight(vis_obj):
        idx = _highlight["idx"]
        for k, mesh in enumerate(surface_meshes):
            si = surface_mesh_ids[k]
            if idx == -1:
                mesh.paint_uniform_color(get_surface_color(str(surface_names[si])))
            elif k == idx:
                mesh.paint_uniform_color(get_surface_color(str(surface_names[si])))
            else:
                mesh.paint_uniform_color([0.3, 0.3, 0.3])
            mesh.compute_vertex_normals()
            vis_obj.update_geometry(mesh)
        vis_obj.update_renderer()
        if idx == -1:
            print("[surfaces] showing ALL clusters")
        else:
            si = surface_mesh_ids[idx]
            print(f"[surfaces] cluster {si} ({surface_names[si]})")

    def _on_key_next(vis_obj):
        n = len(surface_meshes)
        _highlight["idx"] = (_highlight["idx"] + 1) % (n + 1)
        if _highlight["idx"] == n:
            _highlight["idx"] = -1
        _update_highlight(vis_obj)
        return False

    def _on_key_prev(vis_obj):
        n = len(surface_meshes)
        _highlight["idx"] -= 1
        if _highlight["idx"] < -1:
            _highlight["idx"] = n - 1
        _update_highlight(vis_obj)
        return False

    # BFS-clipped meshes
    clipped_combined = o3d.geometry.TriangleMesh()
    for i in range(n_clusters):
        path = os.path.join(out_dir, f"trimmed_mesh_{i}.npz")
        if not os.path.exists(path):
            continue
        d = np.load(path)
        if len(d["vertices"]) == 0:
            continue
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices  = o3d.utility.Vector3dVector(d["vertices"])
        mesh.triangles = o3d.utility.Vector3iVector(d["triangles"])
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(get_surface_color(str(surface_names[i])))
        clipped_combined += mesh

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window("Point cloud", width=640, height=720, left=0, top=50)
    vis1.add_geometry(pcd_combined)
    vis1.get_render_option().point_size = 2.0

    vis2 = o3d.visualization.VisualizerWithKeyCallback()
    vis2.create_window("Unclipped (grid-trimmed) — N/P to cycle",
                       width=640, height=720, left=640, top=50)
    for mesh in surface_meshes:
        vis2.add_geometry(mesh)
    vis2.register_key_callback(ord("N"), _on_key_next)
    vis2.register_key_callback(ord("P"), _on_key_prev)
    vis2.get_render_option().mesh_show_back_face = True
    print("[surfaces] Press N/P in 'Unclipped' window to cycle clusters")

    vis3 = o3d.visualization.Visualizer()
    vis3.create_window("BFS-clipped", width=640, height=720, left=1280, top=50)
    vis3.add_geometry(clipped_combined)
    vis3.get_render_option().mesh_show_back_face = True

    vises = [vis1, vis2, vis3]
    running = [True, True, True]
    while all(running):
        for k, vis in enumerate(vises):
            if running[k]:
                running[k] = vis.poll_events()
                vis.update_renderer()
        time.sleep(0.01)

    for vis in vises:
        vis.destroy_window()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BFS flood-fill mesh trimming prototype",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--visualize", action="store_true",
                        help="Load saved results and visualize (no compute needed)")
    parser.add_argument("--model_id", type=str, required=True,
                        help="Model ID (subdirectory under input_dir / output_dir)")
    parser.add_argument("--input_dir", type=str, default="sample_clouds",
                        help="Root directory for input point cloud subdirs")
    parser.add_argument("--output_dir", type=str, default="output_bfs",
                        help="Root directory for saved results")
    parser.add_argument("--part", type=int, default=0,
                        help="Index of the .xyzc file to process (0-based)")
    parser.add_argument("--post_filter_threshold", type=float, default=1.0,
                        help="Post-BFS filter: keep faces whose barycenter is within "
                             "threshold * spacing of the nearest cluster point")
    parser.add_argument("--spacing_percentile", type=float, default=100.0,
                        help="Percentile of intra-cluster NN distances used as spacing "
                             "for the post-BFS filter (default 100 = max NN distance)")
    parser.add_argument("--seed", type=int, default=41,
                        help="Reproducibility seed")
    args = parser.parse_args()

    if args.visualize:
        run_visualize(args)
    else:
        run_compute(args)
