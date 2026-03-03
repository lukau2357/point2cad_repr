"""
B-Rep reconstruction pipeline.

Two execution modes, separated by the Docker / host boundary:

  Compute mode  (inside Docker container, OCC available):
    python brep_pipeline.py --model_id 00000005 --input_dir sample_clouds --output_dir /output/brep

  Visualize mode  (host machine, no OCC required):
    python brep_pipeline.py --visualize --model_id 00000005 --output_dir /output/brep
"""

import argparse
import math
import os
import shutil
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
# Multi-part merge helper
# ---------------------------------------------------------------------------

def _denorm(pts, mean, R, scale):
    """Inverse of normalize_points for an (N, 3) float array.
    pts_orig = scale * pts_norm @ R + mean  (row-vector convention)
    """
    pts = np.asarray(pts, dtype=np.float64)
    return (scale * (pts @ R) + mean).astype(np.float32)


def _merge_part_dirs(part_dirs, unified_dir):
    """
    Merge per-part output directories into a single unified directory for
    the visualizer.  All per-part files are already in world-space coordinates
    (inverse normalization applied at save time), so this function only
    concatenates arrays and offsets cluster/edge indices.

    part_dirs  : list of (dir_path, cluster_offset, n_clusters)
    """
    os.makedirs(unified_dir, exist_ok=True)

    # 1. Merge metadata (concatenate per-cluster arrays)
    all_sids, all_snames, all_colors = [], [], []
    for dir_path, offset, n in part_dirs:
        meta = np.load(os.path.join(dir_path, "metadata.npz"), allow_pickle=True)
        all_sids.extend(meta["surface_ids"].tolist())
        all_snames.extend(meta["surface_names"].tolist())
        all_colors.extend(meta["cluster_colors"].tolist())
    np.savez(os.path.join(unified_dir, "metadata.npz"),
             n_clusters    = len(all_sids),
             surface_ids   = np.array(all_sids),
             surface_names = np.array(all_snames),
             cluster_colors= np.array(all_colors))

    # 2. Merge vertices (concatenate)
    all_verts = []
    for dir_path, offset, n in part_dirs:
        vpath = os.path.join(dir_path, "vertices.npz")
        if os.path.exists(vpath):
            v = np.load(vpath)["vertices"]
            if len(v):
                all_verts.append(v)
    np.savez(os.path.join(unified_dir, "vertices.npz"),
             vertices=np.concatenate(all_verts) if all_verts else np.zeros((0, 3)))

    # 3. Per-cluster files — copy with offset applied to filename
    for dir_path, offset, n in part_dirs:
        for i in range(n):
            for tmpl in (f"cluster_{i}.npy", f"surface_mesh_{i}.npz"):
                src = os.path.join(dir_path, tmpl)
                if not os.path.exists(src):
                    continue
                dst_name = tmpl.replace(f"_{i}.", f"_{i + offset}.")
                shutil.copy2(src, os.path.join(unified_dir, dst_name))

    # 4. Inter and arc files — offset indices in filename and internal fields
    for dir_path, offset, n in part_dirs:
        for fname in sorted(os.listdir(dir_path)):
            fpath = os.path.join(dir_path, fname)
            if fname.startswith("inter_") and fname.endswith(".npz"):
                d = dict(np.load(fpath, allow_pickle=True))
                i, j = int(d["cluster_i"]), int(d["cluster_j"])
                d["cluster_i"] = i + offset
                d["cluster_j"] = j + offset
                np.savez(os.path.join(unified_dir,
                                      f"inter_{i+offset}_{j+offset}.npz"), **d)
            elif fname.startswith("arcs_") and fname.endswith(".npz"):
                d = dict(np.load(fpath, allow_pickle=True))
                ei, ej = int(d["edge_i"]), int(d["edge_j"])
                d["edge_i"] = ei + offset
                d["edge_j"] = ej + offset
                np.savez(os.path.join(unified_dir,
                                      f"arcs_{ei+offset}_{ej+offset}.npz"), **d)


# ---------------------------------------------------------------------------
# Visualize mode  (host, no OCC)
# ---------------------------------------------------------------------------

def run_visualize(args):
    out_dir = os.path.join(args.output_dir, f"{args.model_id}", "unified")

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

    # Per-arc linesets (arc[0]=green, arc[1]=red, … — saved by compute mode)
    arc_linesets = []
    for arc_path in sorted(_glob.glob(os.path.join(out_dir, "arcs_*.npz"))):
        d      = np.load(arc_path, allow_pickle=True)
        n_arcs = int(d["n_arcs"])
        for k in range(n_arcs):
            pts   = d[f"arc_points_{k}"]
            color = d[f"arc_color_{k}"].tolist()
            arc_linesets.append(_lineset(pts, color))

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
    vis2.create_window("Trimmed arcs + vertices", width=W, height=H, left=W, top=50)
    for ls in (arc_linesets if arc_linesets else trimmed_linesets):
        vis2.add_geometry(ls)
    if vertex_pcd is not None:
        vis2.add_geometry(vertex_pcd)
    vis2.get_render_option().point_size = 8.0

    vis3 = o3d.visualization.Visualizer()

    vis3.create_window("Point clouds + trimmed arcs + boundary strips",
                       width=W, height=H, left=0, top=50 + H + 40)
    for pcd in cluster_pcds:
        vis3.add_geometry(pcd)
    for ls in (arc_linesets if arc_linesets else trimmed_linesets):
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
    from point2cad.cluster_adjacency    import (
        compute_adjacency_matrix, adjacency_pairs, build_cluster_proximity
    )
    from point2cad.color_config         import get_surface_color
    from point2cad.surface_intersection import (
        compute_all_intersections,
        trim_by_vertices,
        compute_vertices,
        sample_curve,
    )
    from point2cad.topology import (
        filter_vertices_by_proximity,
        build_edge_arcs, filter_curves_by_proximity, filter_arcs_by_proximity,
        print_edge_arcs_summary,
        face_arc_incidence, print_face_arcs_summary,
        assemble_wires, print_face_wires_summary,
        build_brep_shape, build_brep_shape_bop, export_step, merge_step_files,
        apply_inverse_normalization,
    )
    import point2cad.primitive_fitting_utils as pfu

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    def normalize_points(pts):
        mean    = np.mean(pts, axis=0)
        pts     = pts - mean
        S, U    = np.linalg.eigh(pts.T @ pts)
        R       = pfu.rotation_matrix_a_to_b(U[:, np.argmin(S)], np.array([1, 0, 0]))
        pts     = (R @ pts.T).T
        extents = np.max(pts, axis=0) - np.min(pts, axis=0)
        scale   = float(np.max(extents) + 1e-7)
        return (pts / scale).astype(np.float32), mean, R, scale

    np_rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark     = False

    # Determine which .xyzc files to process
    input_pattern = os.path.join(args.input_dir, f"{args.model_id}", "*.xyzc")
    part_files    = sorted(_glob.glob(input_pattern),
                           key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    if not part_files:
        print(f"No part files found matching: {input_pattern}")
        return
    model_out_dir = os.path.join(args.output_dir, f"{args.model_id}")
    if os.path.exists(model_out_dir):
        shutil.rmtree(model_out_dir)
        print(f"Removed old results: {model_out_dir}")
    os.makedirs(model_out_dir)
    print(f"Model {args.model_id}: {len(part_files)} part(s)")

    part_dirs      = []   # (out_dir, cluster_offset, n_clusters) — for final merge
    cluster_offset = 0

    for part_idx, sample_path in enumerate(part_files):
        step_stem = f"part_{part_idx}"
        out_dir   = os.path.join(model_out_dir, f"part_{part_idx}")

        print(f"\n{'='*60}")
        print(f"Part {part_idx}: {os.path.basename(sample_path)}  →  {out_dir}")
        print(f"{'='*60}")

        data = np.loadtxt(sample_path)
        data[:, :3], part_mean, part_R, part_scale = normalize_points(data[:, :3])
        unique_clusters, cluster_counts = np.unique(data[:, -1].astype(int), return_counts=True)
        os.makedirs(out_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # Surface fitting
        # ------------------------------------------------------------------
        clusters, surface_ids, fit_results, fit_meshes, occ_surfs = [], [], [], [], []
        for cid, c_count in zip(unique_clusters, cluster_counts):
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
            print(f"[surface fitter] Cluster {cid} number of points: {c_count}")
            print(f"[surface fitter] Fitted surface: {SURFACE_NAMES[sid]}")

        # Per-cluster KDTrees and NN-distance thresholds — needed by both paths.
        cluster_trees, cluster_thresholds = build_cluster_proximity(
            clusters, percentile=args.proximity_percentile
        )

        for i in range(len(cluster_thresholds)):
            cluster_thresholds[i] *= args.proximity_factor
    
        if args.full_adjacency:
            n_surf    = len(clusters)
            inter_adj = np.ones((n_surf, n_surf), dtype=bool)
            np.fill_diagonal(inter_adj, False)
            print(f"[full_adjacency] intersecting all {n_surf * (n_surf - 1) // 2} pairs\n")
        else:
            adj, _, spacing, boundary_strips, _ = compute_adjacency_matrix(
                clusters, threshold_factor=args.spacing_factor,
                spacing_percentile=args.spacing_percentile,
            )
            inter_adj = adj
            print(f"\nSpacing={spacing:.5f}  threshold={args.spacing_factor * spacing:.5f}")
            print(f"Adjacent pairs: {adjacency_pairs(adj)}")
            for (i, j), bpts in sorted(boundary_strips.items()):
                print(f"  boundary ({i},{j}): {len(bpts)} points")
            print()

        threshold_vertex = 1e-4

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
            np.save(os.path.join(out_dir, f"cluster_{i}.npy"),
                    _denorm(cluster, part_mean, part_R, part_scale))
        for i, mesh in enumerate(fit_meshes):
            np.savez(
                os.path.join(out_dir, f"surface_mesh_{i}.npz"),
                vertices  = _denorm(np.asarray(mesh.vertices), part_mean, part_R, part_scale),
                triangles = np.asarray(mesh.triangles),
            )
        print(f"Cluster files saved to {out_dir}/")

        # ------------------------------------------------------------------
        # Surface-surface intersection + trimming
        # ------------------------------------------------------------------
        raw_intersections   = compute_all_intersections(
            inter_adj, surface_ids, fit_results, occ_surfs
        )
        # Vertex-first flow:
        #   1. Find vertices on raw (untrimmed) curves — NMS is built into
        #      compute_vertices so the result is already deduplicated.
        #   2. Drop vertices too far from their constituent clusters
        #      (spurious intersections outside the physical model).
        #   3. Trim open curves using the surviving vertex parameters.
        vertices, vertex_edges = compute_vertices(
            inter_adj, raw_intersections, threshold=threshold_vertex
        )
        print(f"Found {len(vertices)} raw vertices")

        vertices, vertex_edges = filter_vertices_by_proximity(
            vertices, vertex_edges, cluster_trees, cluster_thresholds
        )
        print(f"After proximity filter: {len(vertices)} vertices")

        trim_intersections_ = trim_by_vertices(
            raw_intersections, vertices, vertex_edges,
            extension_factor=0.05,
        )
        trim_intersections_ = filter_curves_by_proximity(
            trim_intersections_, cluster_trees, cluster_thresholds
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
            boundary_pts = inter_trim.get("boundary_pts", np.empty((0, 3)))
            kw["n_boundary_pts"] = len(boundary_pts)
            if len(boundary_pts) > 0:
                kw["boundary_pts"] = _denorm(boundary_pts, part_mean, part_R, part_scale)
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
                kw[f"curve_points_{k}"] = _denorm(
                    sample_curve(curve, n_points=200), part_mean, part_R, part_scale)
            for k, curve in enumerate(inter_raw["curves"]):
                t0_raw, t1_raw = curve.FirstParameter(), curve.LastParameter()
                if abs(t0_raw) > 1e50 or abs(t1_raw) > 1e50:
                    bounds_str = "[infinite]"
                else:
                    bounds_str = f"[{t0_raw:.6f}, {t1_raw:.6f}]"
                print(f"  raw      curve[{k}] {bounds_str}")
                kw[f"untrimmed_curve_points_{k}"] = _denorm(
                    sample_curve(curve, n_points=200, line_extent=1.0),
                    part_mean, part_R, part_scale)
            np.savez(os.path.join(out_dir, f"inter_{i}_{j}.npz"), **kw)

        # ------------------------------------------------------------------
        # Save vertices (computed above, before or after trimming depending on mode)
        # ------------------------------------------------------------------
        np.savez(os.path.join(out_dir, "vertices.npz"),
                 vertices=_denorm(vertices, part_mean, part_R, part_scale))

        # ------------------------------------------------------------------
        # Step 1: vertex–edge attribution and arc splitting
        # ------------------------------------------------------------------
        trim_curves_dict              = {k: v["curves"] for k, v in trim_intersections_.items()}
        edge_arcs, vertices, vertex_edges = build_edge_arcs(
            trim_curves_dict, vertices, vertex_edges, threshold=1e-3
        )
        edge_arcs = filter_arcs_by_proximity(
            edge_arcs, cluster_trees, cluster_thresholds
        )
        print_edge_arcs_summary(edge_arcs)

        # Save arc samples for visualization (one file per edge).
        # Arc index encodes color: arc[0]=green, arc[1]=red, arc[2]=blue, arc[3]=purple
        _ARC_COLORS = [[0.2, 0.85, 0.2], [1.0, 0.2, 0.2],
                       [0.2, 0.4,  1.0], [0.8, 0.2,  0.8]]
        for (ei, ej), arcs in edge_arcs.items():
            kw_arcs = {"n_arcs": len(arcs), "edge_i": ei, "edge_j": ej}
            for k, arc in enumerate(arcs):
                kw_arcs[f"arc_points_{k}"] = _denorm(
                    sample_curve(arc["curve"], n_points=100), part_mean, part_R, part_scale)
                kw_arcs[f"arc_color_{k}"]  = _ARC_COLORS[k % len(_ARC_COLORS)]
            np.savez(os.path.join(out_dir, f"arcs_{ei}_{ej}.npz"), **kw_arcs)

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
        step_path = os.path.join(out_dir, f"{step_stem}.step")
        shape = build_brep_shape(face_arcs, occ_surfs, vertices,
                                 surface_ids=surface_ids,
                                 face_wires=face_wires, tolerance=1e-3,
                                 bspline_method=args.bspline_method)
        export_step(apply_inverse_normalization(shape, part_mean, part_R, part_scale),
                    step_path)

        # ------------------------------------------------------------------
        # Step 4b: BRep assembly — BOPAlgo_MakerVolume with cluster UV bounds
        # ------------------------------------------------------------------
        step_path_bop = os.path.join(out_dir, f"{step_stem}_bop.step")
        shape_bop = build_brep_shape_bop(
            occ_surfs, vertices, vertex_edges, face_arcs,
            surface_ids=surface_ids,
            tolerance=1e-3, rel_margin=0.5,
        )
        if shape_bop is not None:
            export_step(apply_inverse_normalization(shape_bop, part_mean, part_R, part_scale),
                        step_path_bop)
        print(f"\nAll results saved to {out_dir}/")

        if args.model_id:
            part_dirs.append((out_dir, cluster_offset, len(clusters)))
            cluster_offset += len(clusters)

    # After all parts: merge into unified/
    if part_dirs:
        unified_dir = os.path.join(model_out_dir, "unified")
        _merge_part_dirs(part_dirs, unified_dir)
        manual_paths = [
            os.path.join(d, f"part_{i}.step")
            for i, (d, _, _) in enumerate(part_dirs)
        ]
        bop_paths = [
            os.path.join(d, f"part_{i}_bop.step")
            for i, (d, _, _) in enumerate(part_dirs)
        ]
        merge_step_files(
            [p for p in manual_paths if os.path.exists(p)],
            os.path.join(unified_dir, "unified.step"),
        )
        merge_step_files(
            [p for p in bop_paths if os.path.exists(p)],
            os.path.join(unified_dir, "unified_bop.step"),
        )
        print(f"\nUnified results saved to {unified_dir}/")


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
    parser.add_argument("--model_id", type=str, default=None,
                        help="Model ID for multi-part mode (compute: globs "
                             "input_dir/model_id/*_part*.xyzc; visualize: loads "
                             "output_dir/model_id/unified/)")
    parser.add_argument("--input_dir", type=str, default="sample_clouds",
                        help="Root directory for per-model point cloud subdirs "
                             "(multi-part compute mode)")
    parser.add_argument("--output_dir", type=str, default="output_brep",
                        help="Directory for saved results")
    parser.add_argument("-seed", type=int, default=41, help="Reproducibility seed")
    parser.add_argument("--spacing_percentile", type=float, default=100.0,
                    help="Percentile of intra-cluster NN distance distribution "
                            "used for local spacing in adjacency computation "
                            "(default 100.0 = max NN distance).")
    parser.add_argument("--spacing_factor", type=float, default=2,
                        help="Adjacency detection threshold = spacing factor * "
                             "per-pair local spacing")
    parser.add_argument("--proximity_percentile", type=float, default=100,
                        help="Percentile of intra-cluster NN distance distribution "
                             "used as proximity threshold per cluster. Higher = more "
                             "generous (default 100).")
    parser.add_argument("--proximity_factor", type=float, 
                        help="Proximity detection threshold = proximity_factor * chosen intra-cluster NN percentile.", default = 3)
    parser.add_argument("--boundary_mesh", action="store_true",
                        help="Visualize boundary strips as filled Delaunay meshes "
                             "instead of point clouds (visualize mode only)", default = True)
    parser.add_argument("--full_adjacency", action="store_true",
                        help="Intersect all surface pairs, not just adjacent ones; "
                             "uses curve-proximity phantom filtering instead of "
                             "pre-computed boundary strips")
    parser.add_argument("--bspline_method", type=str, default="uv_bounds",
                        choices=["uv_bounds", "explicit_pcurve"],
                        help="How BSpline (INR) faces are constructed. "
                             "'uv_bounds': rectangular UV-bounds face, sewing "
                             "bridges the gap to adjacent analytical faces. "
                             "'explicit_pcurve': wire-based face with pcurves "
                             "computed by breplib.BuildCurve2d for each edge.")
    args = parser.parse_args()

    if args.model_id is None:
        parser.error("--model_id is required")

    if args.visualize:
        run_visualize(args)
    else:
        run_compute(args)
