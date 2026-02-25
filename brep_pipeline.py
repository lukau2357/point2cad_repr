"""
B-Rep reconstruction pipeline.

Two execution modes, separated by the Docker / host boundary:

  Compute mode  (inside Docker container, OCC available):
    python brep_pipeline.py --input path/to/cloud.xyzc --output_dir /output/brep

  Visualize mode  (host machine, no OCC required):
    python brep_pipeline.py --visualize --visualize_id 00949 --output_dir /output/brep
"""

import argparse
import math
import os
import sys
import time
import glob as _glob

import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay, ConvexHull

# Curve-type → RGB colour used by the visualizer
CURVE_TYPE_COLORS = {
    "line":    [1.0, 0.2, 0.2],
    "circle":  [0.2, 0.85, 0.2],
    "ellipse": [0.2, 0.4,  1.0],
    "conic":   [0.8, 0.2,  0.8],
    "bspline": [1.0, 0.6,  0.0],
    "curve":   [0.8, 0.8,  0.0],
    "tangent": [1.0, 1.0,  1.0],
}


# ---------------------------------------------------------------------------
# Boundary strip visualization helper
# ---------------------------------------------------------------------------

def _boundary_mesh(pts, color=None):
    """
    Build a filled TriangleMesh from boundary strip points.

    Projects the 3D points onto their PCA best-fit plane, Delaunay-triangulates
    in 2D, then lifts the triangulation back to the original 3D coordinates.
    Falls back to a fan-triangulation of the 2D convex hull if Delaunay fails.
    Returns None when both methods fail or the input is degenerate.

    Note: ConvexHull.vertices are indices into the original pts array, so no
    remapping is needed for either triangulation method.
    """
    if color is None:
        color = [0.5, 0.5, 0.5]
    if len(pts) < 3:
        return None

    centered = pts - pts.mean(axis=0)
    _, _, Vt  = np.linalg.svd(centered, full_matrices=False)
    pts2d     = centered @ Vt[:2].T   # (N, 2) projection onto best-fit plane

    triangles = None
    try:
        triangles = Delaunay(pts2d).simplices
    except Exception as e:
        print(f"[boundary_mesh] Delaunay failed ({e}), trying convex hull fallback")
        try:
            verts = ConvexHull(pts2d).vertices   # ordered hull vertex indices
            # Fan-triangulation from verts[0]: (v0, v_i, v_{i+1})
            triangles = np.array([
                [verts[0], verts[i], verts[i + 1]]
                for i in range(1, len(verts) - 1)
            ])
        except Exception as e2:
            print(f"[boundary_mesh] Convex hull fallback also failed ({e2}), skipping")
            return None

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(pts)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh


# ---------------------------------------------------------------------------
# Visualize mode  (host, no OCC)
# ---------------------------------------------------------------------------

def run_visualize(args):
    out_dir = os.path.join(args.output_dir, args.visualize_id)

    meta         = np.load(os.path.join(out_dir, "metadata.npz"), allow_pickle=True)
    n_clusters   = int(meta["n_clusters"])
    clust_colors = meta["cluster_colors"]

    # Point clouds
    cluster_pcds = []
    for i in range(n_clusters):
        pts = np.load(os.path.join(out_dir, f"cluster_{i}.npy"))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color(clust_colors[i].tolist())
        cluster_pcds.append(pcd)

    # Intersection curves (sampled to numpy by the compute mode)
    def _lineset(pts, color):
        lines = [[m, m + 1] for m in range(len(pts) - 1)]
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines  = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
        return ls

    trimmed_linesets   = []
    untrimmed_linesets = []
    boundary_pcds      = []
    for inter_path in sorted(_glob.glob(os.path.join(out_dir, "inter_*.npz"))):
        d          = np.load(inter_path, allow_pickle=True)
        curve_type = str(d["curve_type"])
        n_curves   = int(d["n_curves"])
        n_raw      = int(d["n_untrimmed_curves"])
        ci, cj     = int(d["cluster_i"]), int(d["cluster_j"])
        print(f"({ci},{cj})  {d['surface_i_name']} ∩ {d['surface_j_name']}"
              f"  type={curve_type}  trimmed={n_curves}  raw={n_raw}")
        color = CURVE_TYPE_COLORS.get(curve_type, [0.8, 0.8, 0.8])
        for k in range(n_curves):
            trimmed_linesets.append(_lineset(d[f"curve_points_{k}"], color))
        for k in range(n_raw):
            untrimmed_linesets.append(_lineset(d[f"untrimmed_curve_points_{k}"], color))
        if "boundary_pts" in d:
            bpts = d["boundary_pts"]
            if len(bpts) > 0:
                if args.boundary_mesh:
                    bgeom = _boundary_mesh(bpts)
                    if bgeom is not None:
                        boundary_pcds.append(bgeom)
                else:
                    bpcd = o3d.geometry.PointCloud()
                    bpcd.points = o3d.utility.Vector3dVector(bpts)
                    bpcd.paint_uniform_color([0.5, 0.5, 0.5])
                    boundary_pcds.append(bpcd)

    # Fitted surface meshes (all surface types)
    surface_meshes = []
    for mesh_path in sorted(_glob.glob(os.path.join(out_dir, "surface_mesh_*.npz"))):
        ci   = int(os.path.basename(mesh_path).replace("surface_mesh_", "").replace(".npz", ""))
        d    = np.load(mesh_path)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices  = o3d.utility.Vector3dVector(d["vertices"])
        mesh.triangles = o3d.utility.Vector3iVector(d["triangles"])
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(clust_colors[ci].tolist())
        surface_meshes.append(mesh)

    # Vertices
    vertex_pcd  = None
    vertex_path = os.path.join(out_dir, "vertices.npz")
    if os.path.exists(vertex_path):
        verts = np.load(vertex_path)["vertices"]
        if len(verts) > 0:
            vertex_pcd = o3d.geometry.PointCloud()
            vertex_pcd.points = o3d.utility.Vector3dVector(verts)
            vertex_pcd.paint_uniform_color([1.0, 1.0, 0.0])
            print(f"Loaded {len(verts)} vertices")

    # 2×2 window layout (fits 1920×1080)
    W, H = 960, 490

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window("Untrimmed curves", width=W, height=H, left=0, top=50)
    for ls in untrimmed_linesets:
        vis1.add_geometry(ls)

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window("Trimmed curves + vertices", width=W, height=H, left=W, top=50)
    for ls in trimmed_linesets:
        vis2.add_geometry(ls)
    if vertex_pcd is not None:
        vis2.add_geometry(vertex_pcd)
    vis2.get_render_option().point_size = 8.0

    vis3 = o3d.visualization.Visualizer()

    vis3.create_window("Point clouds + trimmed curves + boundary strips",
                       width=W, height=H, left=0, top=50 + H + 40)
    for pcd in cluster_pcds:
        vis3.add_geometry(pcd)
    for ls in trimmed_linesets:
        vis3.add_geometry(ls)
    for bpcd in boundary_pcds:
        vis3.add_geometry(bpcd)
    vis3.get_render_option().point_size = 2.0

    vis4 = o3d.visualization.Visualizer()
    vis4.create_window("Fitted surfaces", width=W, height=H, left=W, top=50 + H + 40)
    for mesh in surface_meshes:
        vis4.add_geometry(mesh)

    vis3.get_render_option().mesh_show_back_face = True
    vis4.get_render_option().mesh_show_back_face = True
    visualizers = [vis1, vis2, vis3, vis4]
    running     = [True] * len(visualizers)
    while all(running):
        for i, vis in enumerate(visualizers):
            if running[i]:
                running[i] = vis.poll_events()
                vis.update_renderer()
        time.sleep(0.01)
    for vis in visualizers:
        vis.destroy_window()


# ---------------------------------------------------------------------------
# Compute mode  (Docker, OCC available)
# ---------------------------------------------------------------------------

def run_compute(args):
    import torch

    from point2cad.surface_fitter       import fit_surface, SURFACE_NAMES, SURFACE_INR
    from point2cad.occ_surfaces         import to_occ_surface
    from point2cad.cluster_adjacency    import compute_adjacency_matrix, adjacency_pairs
    from point2cad.color_config         import get_surface_color
    from point2cad.surface_intersection import (
        compute_all_intersections,
        trim_intersections,
        compute_vertices,
        sample_curve,
    )
    from point2cad.topology import (
        build_edge_arcs, print_edge_arcs_summary,
        face_arc_incidence, print_face_arcs_summary,
        assemble_wires, print_face_wires_summary,
        build_brep_shape, build_brep_shape_bop, export_step,
    )
    import point2cad.primitive_fitting_utils as pfu

    SAMPLE = args.input or os.path.join(
        os.path.dirname(__file__), "sample_clouds", "abc_00949.xyzc"
    )
    DEVICE  = "cuda:0" if torch.cuda.is_available() else "cpu"
    pc_id   = os.path.basename(SAMPLE).split("_")[-1].split(".")[0]
    out_dir = os.path.join(args.output_dir, pc_id)
    os.makedirs(out_dir, exist_ok=True)

    def normalize_points(pts):
        pts = pts - np.mean(pts, axis=0, keepdims=True)
        S, U = np.linalg.eigh(pts.T @ pts)
        R    = pfu.rotation_matrix_a_to_b(U[:, np.argmin(S)], np.array([1, 0, 0]))
        pts  = (R @ pts.T).T
        extents = np.max(pts, axis=0) - np.min(pts, axis=0)
        return (pts / (np.max(extents) + 1e-7)).astype(np.float32)

    data            = np.loadtxt(SAMPLE)
    data[:, :3]     = normalize_points(data[:, :3])
    unique_clusters = np.unique(data[:, -1].astype(int))

    np_rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark     = False

    # ------------------------------------------------------------------
    # Surface fitting
    # ------------------------------------------------------------------
    clusters, surface_ids, fit_results, fit_meshes, occ_surfs = [], [], [], [], []
    for cid in unique_clusters:
        cluster = data[data[:, -1].astype(int) == cid, :3].astype(np.float32)
        clusters.append(cluster)
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
            inr_mesh_kwargs={
                "mesh_dim": 200,
                "uv_margin": 0.2,
                "threshold_multiplier": 1.5,
            },
        )
        sid = res["surface_id"]
        surface_ids.append(sid)
        fit_results.append(res["result"])
        fit_meshes.append(res["mesh"])
        occ_surfs.append(
            to_occ_surface(sid, res["result"], cluster=cluster, uv_margin=0.05)
        )
        print(f"Cluster {cid}: {SURFACE_NAMES[sid]}")

    adj, threshold, spacing, boundary_strips = compute_adjacency_matrix(clusters, threshold_factor = 1.8, percentile = 98)
    print(f"\nSpacing={spacing:.5f}  threshold={threshold:.5f}")
    print(f"Adjacent pairs: {adjacency_pairs(adj)}")
    for (i, j), bpts in sorted(boundary_strips.items()):
        print(f"  boundary ({i},{j}): {len(bpts)} points")
    print()

    # Remove everything from previous runs
    for _stale in _glob.glob(os.path.join(out_dir, "*")):
        os.remove(_stale)

    # ------------------------------------------------------------------
    # Save metadata + cluster files so the visualizer always has
    # something to display even if intersection computation fails later.
    # ------------------------------------------------------------------
    np.savez(
        os.path.join(out_dir, "metadata.npz"),
        n_clusters     = len(clusters),
        surface_ids    = np.array(surface_ids),
        surface_names  = np.array([SURFACE_NAMES[s] for s in surface_ids]),
        cluster_colors = np.array(
            [get_surface_color(SURFACE_NAMES[s]).tolist() for s in surface_ids]
        ),
    )
    for i, cluster in enumerate(clusters):
        np.save(os.path.join(out_dir, f"cluster_{i}.npy"), cluster)
    for i, mesh in enumerate(fit_meshes):
        np.savez(
            os.path.join(out_dir, f"surface_mesh_{i}.npz"),
            vertices  = np.asarray(mesh.vertices),
            triangles = np.asarray(mesh.triangles),
        )
    print(f"Cluster files saved to {out_dir}/")

    # ------------------------------------------------------------------
    # Surface-surface intersection + trimming
    # ------------------------------------------------------------------
    raw_intersections   = compute_all_intersections(
        adj, surface_ids, fit_results, occ_surfs
    )
    trim_intersections_ = trim_intersections(
        raw_intersections, boundary_strips, threshold, extension_factor=0.15
    )

    # Sample curves to numpy for the visualizer (no OCC on host).
    for (i, j) in raw_intersections:
        inter_raw  = raw_intersections[(i, j)]
        inter_trim = trim_intersections_[(i, j)]
        si = SURFACE_NAMES[surface_ids[i]]
        sj = SURFACE_NAMES[surface_ids[j]]
        print(f"({i},{j})  {si} ∩ {sj}  type={inter_raw['type']}"
              f"  method={inter_raw['method']}"
              f"  raw={len(inter_raw['curves'])}  trimmed={len(inter_trim['curves'])}")
        kw = dict(
            cluster_i          = i,
            cluster_j          = j,
            surface_i_name     = si,
            surface_j_name     = sj,
            curve_type         = inter_raw["type"],
            method             = inter_raw["method"],
            n_curves           = len(inter_trim["curves"]),
            n_untrimmed_curves = len(inter_raw["curves"]),
        )
        boundary_pts = boundary_strips.get((i, j), np.empty((0, 3)))
        kw["n_boundary_pts"] = len(boundary_pts)
        if len(boundary_pts) > 0:
            kw["boundary_pts"] = boundary_pts
        for k, curve in enumerate(inter_trim["curves"]):
            t0, t1 = curve.FirstParameter(), curve.LastParameter()
            p0, p1 = curve.Value(t0), curve.Value(t1)
            endpoint_dist = math.sqrt(
                (p1.X() - p0.X()) ** 2 +
                (p1.Y() - p0.Y()) ** 2 +
                (p1.Z() - p0.Z()) ** 2
            )
            print(f"  trimmed  curve[{k}] [{t0:.6f}, {t1:.6f}]"
                  f"  endpoint_dist={endpoint_dist:.6e}")
            kw[f"curve_points_{k}"] = sample_curve(
                curve, boundary_pts=boundary_pts, threshold=threshold,
                n_points=200, extension_factor=0.15,
            )
        for k, curve in enumerate(inter_raw["curves"]):
            print(f"  raw      curve[{k}]"
                  f" [{curve.FirstParameter():.6f}, {curve.LastParameter():.6f}]")
            kw[f"untrimmed_curve_points_{k}"] = sample_curve(
                curve, n_points=200, line_extent=1.0
            )
        np.savez(os.path.join(out_dir, f"inter_{i}_{j}.npz"), **kw)

    # ------------------------------------------------------------------
    # Vertex finding
    # ------------------------------------------------------------------
    vertices, vertex_edges = compute_vertices(adj, trim_intersections_)
    print(f"Found {len(vertices)} vertices")
    np.savez(os.path.join(out_dir, "vertices.npz"), vertices=vertices)

    # ------------------------------------------------------------------
    # Step 1: vertex–edge attribution and arc splitting
    # ------------------------------------------------------------------
    trim_curves_dict              = {k: v["curves"] for k, v in trim_intersections_.items()}
    edge_arcs, vertices, vertex_edges = build_edge_arcs(
        trim_curves_dict, vertices, vertex_edges, threshold=1e-3
    )
    print_edge_arcs_summary(edge_arcs)

    # ------------------------------------------------------------------
    # Step 2: face–arc incidence
    # ------------------------------------------------------------------
    face_arcs = face_arc_incidence(edge_arcs)
    print_face_arcs_summary(face_arcs)

    # ------------------------------------------------------------------
    # Step 3: wire assembly with angular ordering at high-degree vertices
    # ------------------------------------------------------------------
    face_wires = assemble_wires(face_arcs, occ_surfs, vertices, surface_ids=surface_ids)
    print_face_wires_summary(face_wires)

    # ------------------------------------------------------------------
    # Step 4a: BRep assembly — manual arc/wire approach
    # ------------------------------------------------------------------
    step_path = os.path.join(out_dir, f"{pc_id}.step")
    shape = build_brep_shape(face_arcs, occ_surfs, vertices,
                             surface_ids=surface_ids,
                             face_wires=face_wires, tolerance=1e-3,
                             bspline_method=args.bspline_method)
    export_step(shape, step_path)

    # ------------------------------------------------------------------
    # Step 4b: BRep assembly — BOPAlgo_MakerVolume approach
    # ------------------------------------------------------------------
    # step_path_bop = os.path.join(out_dir, f"{pc_id}_bop.step")
    # shape_bop = build_brep_shape_bop(occ_surfs, clusters, adj, tolerance=1e-3, margin=0.1)
    # if shape_bop is not None:
    #     export_step(shape_bop, step_path_bop)
    print(f"\nAll results saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="B-Rep reconstruction pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--visualize", action="store_true",
                        help="Load saved results and visualize (host, no OCC needed)")
    parser.add_argument("--visualize_id", type=str, default=None,
                        help="Point-cloud ID to visualize, e.g. 00949")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to .xyzc file (compute mode)")
    parser.add_argument("--output_dir", type=str, default="output_brep",
                        help="Directory for saved results")
    parser.add_argument("-seed", type=int, default=41, help="Reproducibility seed")
    parser.add_argument("--boundary_mesh", action="store_true",
                        help="Visualize boundary strips as filled Delaunay meshes "
                             "instead of point clouds (visualize mode only)", default = True)
    parser.add_argument("--bspline_method", type=str, default="uv_bounds",
                        choices=["uv_bounds", "explicit_pcurve"],
                        help="How BSpline (INR) faces are constructed. "
                             "'uv_bounds': rectangular UV-bounds face, sewing "
                             "bridges the gap to adjacent analytical faces (default). "
                             "'explicit_pcurve': wire-based face with pcurves "
                             "computed by breplib.BuildCurve2d for each edge.")
    args = parser.parse_args()

    if args.visualize:
        if args.visualize_id is None:
            parser.error("--visualize requires --visualize_id")
        run_visualize(args)
    else:
        run_compute(args)
