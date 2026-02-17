"""
Experiment: For a given point cloud, run the Point2CAD surface fitting pipeline,
and for each surface classified as INR (freeform), fit an OpenCASCADE B-spline
surface to the INR-sampled grid. Produces comparative visualizations.

Usage (inside Docker):
    python inr_to_bspline.py --input sample_clouds/abc_00470.xyzc
    python inr_to_bspline.py --input sample_clouds/abc_00470.xyzc --sampling_resolution 80 --bspline_degree_max 8

Usage (host, after running inside Docker):
    python inr_to_bspline.py --visualize --output_dir output_bspline/00470
"""

import numpy as np
import os
import time
import argparse
import torch
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import Counter

import point2cad.primitive_fitting_utils as primitive_fitting_utils
from point2cad.surface_fitter import fit_surface, SURFACE_NAMES, SURFACE_INR
from point2cad.color_config import get_surface_color

# ---------------------------------------------------------------------------
# Normalization (same as main.py)
# ---------------------------------------------------------------------------

def normalize_points(points):
    points = points - np.mean(points, axis = 0, keepdims = True)
    S, U = np.linalg.eig(points.T @ points)
    smallest_ev = U[:, np.argmin(S)]
    R = primitive_fitting_utils.rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
    points = (R @ points.T).T
    extents = np.max(points, axis = 0) - np.min(points, axis = 0)
    points = points / (np.max(extents) + 1e-7)
    return points.astype(np.float32)

def extract_pc_id(input_name):
    base_name = os.path.basename(input_name)
    return base_name.split("_")[-1].split(".")[0]

# ---------------------------------------------------------------------------
# INR sampling on a regular UV grid
# ---------------------------------------------------------------------------

def sample_inr_grid(model, uv_bb_min, uv_bb_max, cluster_mean, cluster_scale, resolution, uv_margin = 0.1):
    """
    Sample the trained INR decoder on a regular (resolution x resolution) UV grid.
    Returns:
        xyz_grid: np.ndarray of shape (resolution, resolution, 3)
        u_lin:    np.ndarray of shape (resolution,) - the u parameter values
        v_lin:    np.ndarray of shape (resolution,) - the v parameter values
    """
    uv_length = uv_bb_max - uv_bb_min
    uv_min = uv_bb_min - uv_length * uv_margin
    uv_max = uv_bb_max + uv_length * uv_margin

    if model.is_u_closed:
        uv_min[0] = max(uv_min[0], -1)
        uv_max[0] = min(uv_max[0], 1)
    if model.is_v_closed:
        uv_min[1] = max(uv_min[1], -1)
        uv_max[1] = min(uv_max[1], 1)

    device = next(model.parameters()).device

    u_lin = torch.linspace(uv_min[0], uv_max[0], resolution, device = device)
    v_lin = torch.linspace(uv_min[1], uv_max[1], resolution, device = device)
    u, v = torch.meshgrid(u_lin, v_lin, indexing = "ij")
    uv = torch.stack((u, v), dim = 2).reshape(-1, 2)

    with torch.no_grad():
        X = model.forward_decoder(uv)
        X = X * torch.tensor(cluster_scale, device = device) + torch.tensor(cluster_mean, device = device)

    xyz_grid = X.cpu().numpy().reshape(resolution, resolution, 3)
    return xyz_grid, u_lin.cpu().numpy(), v_lin.cpu().numpy()

# ---------------------------------------------------------------------------
# B-spline fitting via OpenCASCADE
# ---------------------------------------------------------------------------

def fit_bspline_surface(xyz_grid, degree_min, degree_max, continuity, tol3d):
    """
    Fit a B-spline surface to a regular grid of 3D points using OCCT.
    Args:
        xyz_grid:    np.ndarray (M, N, 3)
        degree_min:  minimum B-spline degree
        degree_max:  maximum B-spline degree
        continuity:  integer 0-3 mapping to GeomAbs_C0..C3
        tol3d:       maximum approximation tolerance
    Returns:
        bspline_surface: OCC Geom_BSplineSurface handle
        fitting_time:    seconds
    """
    from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface
    from OCC.Core.TColgp import TColgp_Array2OfPnt
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.GeomAbs import GeomAbs_C0, GeomAbs_C1, GeomAbs_C2, GeomAbs_C3
    from OCC.Core.Approx import Approx_IsoParametric

    continuity_map = {0: GeomAbs_C0, 1: GeomAbs_C1, 2: GeomAbs_C2, 3: GeomAbs_C3}
    geom_continuity = continuity_map[continuity]

    M, N, _ = xyz_grid.shape
    points = TColgp_Array2OfPnt(1, M, 1, N)
    for i in range(M):
        for j in range(N):
            x, y, z = xyz_grid[i, j]
            points.SetValue(i + 1, j + 1, gp_Pnt(float(x), float(y), float(z)))

    t0 = time.time()
    approx = GeomAPI_PointsToBSplineSurface(
        points, degree_min, degree_max, geom_continuity, tol3d
    )
    fitting_time = time.time() - t0

    bspline_surface = approx.Surface()
    return bspline_surface, fitting_time

def evaluate_bspline_on_grid(bspline_surface, M, N):
    """
    Evaluate the fitted B-spline surface on a regular (M x N) parameter grid
    spanning its full [U1, U2] x [V1, V2] domain.
    Returns:
        xyz_bspline: np.ndarray (M, N, 3)
    """
    u1, u2, v1, v2 = bspline_surface.Bounds()
    u_lin = np.linspace(u1, u2, M)
    v_lin = np.linspace(v1, v2, N)

    from OCC.Core.gp import gp_Pnt
    xyz_bspline = np.zeros((M, N, 3))
    for i, u in enumerate(u_lin):
        for j, v in enumerate(v_lin):
            pt = bspline_surface.Value(u, v)
            xyz_bspline[i, j] = [pt.X(), pt.Y(), pt.Z()]

    return xyz_bspline

def bspline_surface_info(bspline_surface):
    """Extract and print B-spline surface metadata."""
    info = {
        "u_degree": bspline_surface.UDegree(),
        "v_degree": bspline_surface.VDegree(),
        "u_num_poles": bspline_surface.NbUPoles(),
        "v_num_poles": bspline_surface.NbVPoles(),
        "u_num_knots": bspline_surface.NbUKnots(),
        "v_num_knots": bspline_surface.NbVKnots(),
    }
    u1, u2, v1, v2 = bspline_surface.Bounds()
    info["u_domain"] = (u1, u2)
    info["v_domain"] = (v1, v2)
    return info

# ---------------------------------------------------------------------------
# Error metrics (all L2 / Euclidean for consistency)
# ---------------------------------------------------------------------------

def compute_inr_error(cluster, model, cluster_mean, cluster_scale):
    """
    Pointwise autoencoder reconstruction error: encode each cluster point,
    decode it, measure Euclidean distance to the original.
    Returns:
        dists: np.ndarray (P,) per-point Euclidean error
    """
    device = next(model.parameters()).device
    cluster_norm = (cluster - cluster_mean) / (cluster_scale + 1e-6)
    X = torch.tensor(cluster_norm, dtype = torch.float32, device = device)
    cluster_mean = torch.tensor(cluster_mean, dtype = torch.float32, device = device)
    cluster_scale = torch.tensor(cluster_scale, dtype = torch.float32, device = device)

    with torch.no_grad():
        Xhat, _ = model.forward(X)
        Xhat = Xhat * cluster_scale + cluster_mean
    X_orig = torch.tensor(cluster, dtype = torch.float32, device = device)
    dists = torch.norm(X_orig - Xhat, dim = -1).cpu().numpy()
    return dists

def compute_bspline_error(cluster_points, surface_points_flat):
    """
    Per-point minimum Euclidean distance from cluster points to the B-spline
    surface (via KDTree nearest-neighbor).
    Returns:
        dists: np.ndarray (P,) per-point Euclidean distance
    """
    from scipy.spatial import KDTree
    tree = KDTree(surface_points_flat)
    dists, _ = tree.query(cluster_points)
    return dists

# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def grid_to_o3d_mesh(xyz_grid, color):
    """Convert a (M, N, 3) grid to an Open3D TriangleMesh with uniform color."""
    M, N, _ = xyz_grid.shape
    vertices = xyz_grid.reshape(-1, 3)
    triangles = []
    for i in range(M - 1):
        for j in range(N - 1):
            v00 = i * N + j
            v01 = i * N + j + 1
            v10 = (i + 1) * N + j
            v11 = (i + 1) * N + j + 1
            triangles.append([v00, v10, v01])
            triangles.append([v01, v10, v11])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh

def save_o3d_visualization(cluster, inr_grid, bspline_grid, cluster_id, out_dir):
    """Save Open3D meshes and point cloud as PLY files for each cluster."""
    # Point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cluster)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # grey
    pcd_path = os.path.join(out_dir, f"cluster_{cluster_id}_points.ply")
    o3d.io.write_point_cloud(pcd_path, pcd)

    # INR mesh
    inr_mesh = grid_to_o3d_mesh(inr_grid, get_surface_color("inr"))
    inr_path = os.path.join(out_dir, f"cluster_{cluster_id}_inr_mesh.ply")
    o3d.io.write_triangle_mesh(inr_path, inr_mesh)

    # B-spline mesh
    bsp_mesh = grid_to_o3d_mesh(bspline_grid, get_surface_color("bspline"))
    bsp_path = os.path.join(out_dir, f"cluster_{cluster_id}_bspline_mesh.ply")
    o3d.io.write_triangle_mesh(bsp_path, bsp_mesh)

    print(f"  Saved: {pcd_path}")
    print(f"  Saved: {inr_path}")
    print(f"  Saved: {bsp_path}")
    return pcd, inr_mesh, bsp_mesh

def save_error_histogram(dists_inr, dists_bspline, cluster_id, out_path):
    """Histogram of per-point Euclidean distances: INR autoencoder vs B-spline KDTree."""
    fig, ax = plt.subplots(figsize = (8, 5))
    bins = np.linspace(0, max(dists_inr.max(), dists_bspline.max()) * 1.05, 80)
    ax.hist(dists_inr, bins = bins, alpha = 0.6, label = f"INR (mean={dists_inr.mean():.5f})", color = "#FF9800")
    ax.hist(dists_bspline, bins = bins, alpha = 0.6, label = f"B-spline (mean={dists_bspline.mean():.5f})", color = "#4CAF50")
    ax.set_xlabel("Euclidean distance")
    ax.set_ylabel("Count")
    ax.set_title(f"Cluster {cluster_id}: per-point reconstruction error")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi = 150, bbox_inches = "tight")
    plt.close(fig)
    print(f"  Error histogram saved to {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "INR to B-spline fitting experiment", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type = str, help = "Path to .xyzc input file")
    parser.add_argument("--output_dir", type = str, default = "output_bspline", help = "Output directory")
    parser.add_argument("--visualize", action = "store_true", help = "Only visualize existing results (host mode)")

    # INR fitting parameters (same defaults as main.py)
    parser.add_argument("--inr_max_steps", type = int, default = 1500)
    parser.add_argument("--inr_noise_3d", type = float, default = 0.05)
    parser.add_argument("--inr_noise_uv", type = float, default = 0.05)
    parser.add_argument("--inr_lr", type = float, default = 0.1)

    # Sampling resolution for both INR evaluation and B-spline input
    parser.add_argument("--sampling_resolution", type = int, default = 50,
                        help = "Resolution of the regular UV grid (NxN) for INR sampling and B-spline fitting")

    # B-spline fitting parameters
    parser.add_argument("--bspline_degree_min", type = int, default = 3)
    parser.add_argument("--bspline_degree_max", type = int, default = 8)
    parser.add_argument("--bspline_continuity", type = int, default = 2, choices = [0, 1, 2, 3],
                        help = "Minimum continuity: 0=C0, 1=C1, 2=C2, 3=C3")
    parser.add_argument("--bspline_tol3d", type = float, default = 1e-3,
                        help = "Maximum 3D approximation tolerance for OCCT B-spline fitting")

    # Evaluation resolution (can differ from fitting resolution)
    parser.add_argument("--eval_resolution", type = int, default = 200,
                        help = "Resolution for evaluating the B-spline surface (for visualization and error computation)")

    args = parser.parse_args()

    if args.visualize:
        # Host-side visualization: open Open3D windows for saved PLY files
        import glob
        import time as _time

        out_dir = args.output_dir
        npz_files = sorted(glob.glob(os.path.join(out_dir, "cluster_*_inr.npz")))

        if not npz_files:
            print(f"No results found in {out_dir}/. Run the fitting pipeline first (inside Docker).")
            exit(1)

        for npz_path in npz_files:
            data = np.load(npz_path, allow_pickle = True)
            cluster_id = int(data["cluster_id"])
            dists_inr = data["dists_inr"]
            dists_bspline = data["dists_bspline"]
            surface_info = data["surface_info"].item()

            print(f"\nCluster {cluster_id}:")
            print(f"  INR error:      mean={dists_inr.mean():.5f}, max={dists_inr.max():.5f}, median={np.median(dists_inr):.5f}")
            print(f"  B-spline error: mean={dists_bspline.mean():.5f}, max={dists_bspline.max():.5f}, median={np.median(dists_bspline):.5f}")
            print(f"  B-spline info: {surface_info}")

            # Regenerate histogram
            hist_path = os.path.join(out_dir, f"cluster_{cluster_id}_error_hist.png")
            img = mpimg.imread(os.path.join(out_dir, f"cluster_{cluster_id}_error_hist.png"))
            plt.imshow(img)
            plt.axis("off")
            plt.show()

            # Load and display PLY files via Open3D
            pcd_path = os.path.join(out_dir, f"cluster_{cluster_id}_points.ply")
            inr_path = os.path.join(out_dir, f"cluster_{cluster_id}_inr_mesh.ply")
            bsp_path = os.path.join(out_dir, f"cluster_{cluster_id}_bspline_mesh.ply")

            geoms = []
            if os.path.exists(pcd_path):
                pcd = o3d.io.read_point_cloud(pcd_path)
                geoms.append(pcd)

            vis1 = o3d.visualization.Visualizer()
            vis1.create_window(window_name = f"Cluster {cluster_id}: Points + INR", width = 640, height = 720, left = 0, top = 50)
            if os.path.exists(pcd_path):
                vis1.add_geometry(o3d.io.read_point_cloud(pcd_path))
            if os.path.exists(inr_path):
                mesh = o3d.io.read_triangle_mesh(inr_path)
                mesh.compute_vertex_normals()
                vis1.add_geometry(mesh)
            vis1.get_render_option().mesh_show_back_face = True
            vis1.get_render_option().point_size = 3.0

            vis2 = o3d.visualization.Visualizer()
            vis2.create_window(window_name = f"Cluster {cluster_id}: Points + B-spline", width = 640, height = 720, left = 640, top = 50)
            if os.path.exists(pcd_path):
                vis2.add_geometry(o3d.io.read_point_cloud(pcd_path))
            if os.path.exists(bsp_path):
                mesh = o3d.io.read_triangle_mesh(bsp_path)
                mesh.compute_vertex_normals()
                vis2.add_geometry(mesh)
            vis2.get_render_option().mesh_show_back_face = True
            vis2.get_render_option().point_size = 3.0

            r1, r2 = True, True
            while r1 and r2:
                r1 = vis1.poll_events(); vis1.update_renderer()
                r2 = vis2.poll_events(); vis2.update_renderer()
                _time.sleep(0.01)
            vis1.destroy_window()
            vis2.destroy_window()

        exit(0)
        print("\nDone.")

    # -----------------------------------------------------------------------
    # Pipeline mode (inside Docker with OCCT available)
    # -----------------------------------------------------------------------

    assert args.input is not None, "Provide --input path to .xyzc file"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    np_rng = np.random.default_rng(41)
    torch.manual_seed(41)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(41)

    pc_id = extract_pc_id(args.input)
    out_dir = os.path.join(args.output_dir, pc_id)
    os.makedirs(out_dir, exist_ok = True)

    # Load and normalize
    data = np.loadtxt(args.input)
    points = data[:, :3]
    points = normalize_points(points)
    data[:, :3] = points
    clusters = data[:, -1].astype(int)
    unique_clusters = np.unique(clusters)

    print(f"Point cloud: {args.input}")
    print(f"Number of clusters: {len(unique_clusters)}")
    print(f"Sampling resolution: {args.sampling_resolution}x{args.sampling_resolution}")
    print(f"B-spline: degree [{args.bspline_degree_min}, {args.bspline_degree_max}], "
          f"C{args.bspline_continuity}, tol={args.bspline_tol3d}\n")

    for cluster_id in unique_clusters:
        cluster = data[data[:, 3] == cluster_id][:, :3].astype(np.float32)
        print(f"Cluster {cluster_id}: {cluster.shape[0]} points")
        if cluster.shape[0] == 0:
            continue

        fitting_result = fit_surface(cluster, {
            "hidden_dim": 64,
            "use_shortcut": True,
            "fraction_siren": 0.5
        }, np_rng, device,
            plane_cone_ratio_threshold = 4,
            cone_theta_tolerance_degrees = 5,
            inr_fit_kwargs = {
                "max_steps": args.inr_max_steps,
                "noise_magnitude_3d": args.inr_noise_3d,
                "noise_magnitude_uv": args.inr_noise_uv,
                "initial_lr": args.inr_lr,
            },
            inr_mesh_kwargs = {"mesh_dim": args.sampling_resolution, "uv_margin": 0.2, "threshold_multiplier": 0.2}
        )

        surface_name = SURFACE_NAMES[fitting_result["surface_id"]]
        print(f"  Best surface: {surface_name} (error: {fitting_result['result']['error']:.6f})")

        if fitting_result["surface_id"] != SURFACE_INR:
            print(f"  Not an INR surface, skipping B-spline fitting.\n")
            continue

        # Extract INR parameters
        params = fitting_result["result"]["params"]
        model = params["model"]
        uv_bb_min = params["uv_bb_min"]
        uv_bb_max = params["uv_bb_max"]
        cluster_mean = params["cluster_mean"]
        cluster_scale = params["cluster_scale"]

        print(f"  INR UV bounding box: [{uv_bb_min}, {uv_bb_max}]")
        print(f"  INR closedness: u_closed={model.is_u_closed}, v_closed={model.is_v_closed}")

        # Sample INR on regular grid
        print(f"  Sampling INR on {args.sampling_resolution}x{args.sampling_resolution} grid...")
        inr_grid, u_lin, v_lin = sample_inr_grid(
            model, uv_bb_min.copy(), uv_bb_max.copy(),
            cluster_mean, cluster_scale,
            args.sampling_resolution
        )

        # Fit B-spline to the INR-sampled grid
        print(f"  Fitting B-spline surface via OCCT...")
        bspline_surface, fitting_time = fit_bspline_surface(
            inr_grid,
            args.bspline_degree_min,
            args.bspline_degree_max,
            args.bspline_continuity,
            args.bspline_tol3d
        )

        info = bspline_surface_info(bspline_surface)
        print(f"  B-spline fitting time: {fitting_time:.3f}s")
        print(f"  B-spline surface: u_degree={info['u_degree']}, v_degree={info['v_degree']}, "
              f"poles=({info['u_num_poles']}x{info['v_num_poles']}), "
              f"knots=({info['u_num_knots']}x{info['v_num_knots']})")
        print(f"  INR UV bounds:     u=[{uv_bb_min[0]:.4f}, {uv_bb_max[0]:.4f}], v=[{uv_bb_min[1]:.4f}, {uv_bb_max[1]:.4f}]")
        print(f"  B-spline bounds:   u=[{info['u_domain'][0]:.4f}, {info['u_domain'][1]:.4f}], v=[{info['v_domain'][0]:.4f}, {info['v_domain'][1]:.4f}]")

        # Evaluate B-spline on a (possibly finer) grid for visualization
        eval_res = args.eval_resolution
        bspline_grid = evaluate_bspline_on_grid(bspline_surface, eval_res, eval_res)

        # Also evaluate INR at eval resolution for visualization
        inr_grid_eval, _, _ = sample_inr_grid(
            model, uv_bb_min.copy(), uv_bb_max.copy(),
            cluster_mean, cluster_scale,
            eval_res
        )

        # Error metrics
        # INR: pointwise autoencoder reconstruction (Euclidean)
        dists_inr = compute_inr_error(cluster, model, cluster_mean, cluster_scale)
        # B-spline: KDTree nearest-neighbor (Euclidean)
        bsp_flat = bspline_grid.reshape(-1, 3)
        dists_bspline = compute_bspline_error(cluster, bsp_flat)

        print(f"  Cluster → INR:      mean={dists_inr.mean():.6f}, max={dists_inr.max():.6f}, median={np.median(dists_inr):.6f}")
        print(f"  Cluster → B-spline: mean={dists_bspline.mean():.6f}, max={dists_bspline.max():.6f}, median={np.median(dists_bspline):.6f}")

        # Save results as .npz for host-side visualization
        npz_path = os.path.join(out_dir, f"cluster_{cluster_id}_inr.npz")
        np.savez(npz_path,
            cluster_id = cluster_id,
            cluster_points = cluster,
            inr_grid = inr_grid_eval,
            bspline_grid = bspline_grid,
            surface_info = info,
            dists_inr = dists_inr,
            dists_bspline = dists_bspline,
        )
        print(f"  Results saved to {npz_path}")

        # Save Open3D meshes and point cloud
        save_o3d_visualization(cluster, inr_grid_eval, bspline_grid, int(cluster_id), out_dir)

        # Error histogram
        hist_path = os.path.join(out_dir, f"cluster_{int(cluster_id)}_error_hist.png")
        save_error_histogram(dists_inr, dists_bspline, int(cluster_id), hist_path)
        print()

    print(f"All outputs saved to {out_dir}/")
