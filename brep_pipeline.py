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
from collections import defaultdict

import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay, ConvexHull

from point2cad.surface_types import (
    SURFACE_PLANE, SURFACE_SPHERE, SURFACE_CYLINDER, SURFACE_CONE, SURFACE_INR,
    SURFACE_NAMES,
)

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

    # 2. Merge vertices (concatenate) — both post-filter and pre-filter
    for vfile in ("vertices.npz", "vertices_pre_filter.npz"):
        all_verts = []
        for dir_path, offset, n in part_dirs:
            vpath = os.path.join(dir_path, vfile)
            if os.path.exists(vpath):
                v = np.load(vpath)["vertices"]
                if len(v):
                    all_verts.append(v)
        np.savez(os.path.join(unified_dir, vfile),
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
                if fname.startswith("arcs_pre_filter_"):
                    dst = f"arcs_pre_filter_{ei+offset}_{ej+offset}.npz"
                else:
                    dst = f"arcs_{ei+offset}_{ej+offset}.npz"
                np.savez(os.path.join(unified_dir, dst), **d)


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

    # Per-arc linesets (post-filter: arcs_I_J.npz)
    arc_linesets = []
    for arc_path in sorted(_glob.glob(os.path.join(out_dir, "arcs_*.npz"))):
        if "pre_filter" in os.path.basename(arc_path):
            continue
        d      = np.load(arc_path, allow_pickle=True)
        n_arcs = int(d["n_arcs"])
        for k in range(n_arcs):
            pts   = d[f"arc_points_{k}"]
            color = d[f"arc_color_{k}"].tolist()
            arc_linesets.append(_lineset(pts, color))

    # Pre-filter arc linesets (arcs_pre_filter_I_J.npz)
    pre_filter_arc_linesets = []
    for arc_path in sorted(_glob.glob(os.path.join(out_dir, "arcs_pre_filter_*.npz"))):
        d      = np.load(arc_path, allow_pickle=True)
        n_arcs = int(d["n_arcs"])
        for k in range(n_arcs):
            pts   = d[f"arc_points_{k}"]
            color = d[f"arc_color_{k}"].tolist()
            pre_filter_arc_linesets.append(_lineset(pts, color))

    trimmed_linesets   = []
    untrimmed_linesets = []
    polyline_linesets  = []
    boundary_pcds      = []
    for inter_path in sorted(_glob.glob(os.path.join(out_dir, "inter_*.npz"))):
        d          = np.load(inter_path, allow_pickle=True)
        curve_type = str(d["curve_type"])
        n_curves   = int(d["n_curves"])
        n_raw      = int(d["n_untrimmed_curves"])
        ci, cj     = int(d["cluster_i"]), int(d["cluster_j"])
        print(f"({ci}, {cj})  {d['surface_i_name']} ∩ {d['surface_j_name']}"
              f"  type={curve_type}  trimmed={n_curves}  raw={n_raw}")
        color = CURVE_TYPE_COLORS.get(curve_type, [0.8, 0.8, 0.8])
        for k in range(n_curves):
            trimmed_linesets.append(_lineset(d[f"curve_points_{k}"], color))
        for k in range(n_raw):
            untrimmed_linesets.append(_lineset(d[f"untrimmed_curve_points_{k}"], color))
        # Raw polylines from mesh intersections (cyan)
        n_polys = int(d["n_polylines"]) if "n_polylines" in d else 0
        for k in range(n_polys):
            polyline_linesets.append(
                _lineset(d[f"polyline_points_{k}"], [0.0, 0.9, 0.9]))
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

    # Vertices (post-filter: after greedy oracle filter)
    vertex_pcd  = None
    vertex_path = os.path.join(out_dir, "vertices.npz")
    if os.path.exists(vertex_path):
        verts = np.load(vertex_path)["vertices"]
        if len(verts) > 0:
            vertex_pcd = o3d.geometry.PointCloud()
            vertex_pcd.points = o3d.utility.Vector3dVector(verts)
            vertex_pcd.paint_uniform_color([1.0, 1.0, 0.0])
            print(f"Loaded {len(verts)} vertices (post-filter)")

    # Vertices (pre-filter: after build_edge_arcs, before greedy oracle filter)
    pre_filter_vertex_pcd = None
    pre_filter_vertex_path = os.path.join(out_dir, "vertices_pre_filter.npz")
    if os.path.exists(pre_filter_vertex_path):
        verts_pre = np.load(pre_filter_vertex_path)["vertices"]
        if len(verts_pre) > 0:
            pre_filter_vertex_pcd = o3d.geometry.PointCloud()
            pre_filter_vertex_pcd.points = o3d.utility.Vector3dVector(verts_pre)
            pre_filter_vertex_pcd.paint_uniform_color([1.0, 0.5, 0.0])  # orange
            print(f"Loaded {len(verts_pre)} vertices (pre-filter)")

    # 3×2 window layout (fits 1920×1080)
    W, H = 640, 490

    vis1 = o3d.visualization.Visualizer()
    if polyline_linesets:
        vis1.create_window("Raw polylines (mesh)", width=W, height=H, left=0, top=50)
        for ls in polyline_linesets:
            vis1.add_geometry(ls)
    else:
        vis1.create_window("Untrimmed curves", width=W, height=H, left=0, top=50)
        for ls in untrimmed_linesets:
            vis1.add_geometry(ls)

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window("Pre-filter arcs + vertices", width=W, height=H, left=W, top=50)
    for ls in (pre_filter_arc_linesets if pre_filter_arc_linesets else trimmed_linesets):
        vis2.add_geometry(ls)
    if pre_filter_vertex_pcd is not None:
        vis2.add_geometry(pre_filter_vertex_pcd)
    vis2.get_render_option().point_size = 8.0

    vis3 = o3d.visualization.Visualizer()
    vis3.create_window("Post-filter arcs + vertices", width=W, height=H, left=2*W, top=50)
    for ls in (arc_linesets if arc_linesets else trimmed_linesets):
        vis3.add_geometry(ls)
    if vertex_pcd is not None:
        vis3.add_geometry(vertex_pcd)
    vis3.get_render_option().point_size = 8.0

    vis4 = o3d.visualization.Visualizer()
    if args.boundary_mesh:
        vis4.create_window("Point clouds + boundary meshes",
                           width=W, height=H, left=0, top=50 + H + 40)
        for pcd in cluster_pcds:
            vis4.add_geometry(pcd)
        for bpcd in boundary_pcds:
            vis4.add_geometry(bpcd)
        vis4.get_render_option().point_size = 2.0
    else:
        vis4.create_window("Point clouds",
                           width=W, height=H, left=0, top=50 + H + 40)
        for pcd in cluster_pcds:
            vis4.add_geometry(pcd)
        vis4.get_render_option().point_size = 2.0

    vis6 = None
    if not args.boundary_mesh:
        vis6 = o3d.visualization.Visualizer()
        vis6.create_window("Boundary strips",
                           width=W, height=H, left=W, top=50 + H + 40)
        for bpcd in boundary_pcds:
            vis6.add_geometry(bpcd)
        vis6.get_render_option().point_size = 10.0

    vis5 = o3d.visualization.Visualizer()
    vis5.create_window("Fitted surfaces", width=W, height=H,
                       left=W if args.boundary_mesh else 2 * W, top=50 + H + 40)
    for mesh in surface_meshes:
        vis5.add_geometry(mesh)

    vis4.get_render_option().mesh_show_back_face = True
    vis5.get_render_option().mesh_show_back_face = True
    visualizers = [vis1, vis2, vis3, vis4, vis5]
    if vis6 is not None:
        vis6.get_render_option().mesh_show_back_face = True
        visualizers.append(vis6)
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
# Coincident-surface merging
# ---------------------------------------------------------------------------

def _unit_vec(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def merge_coincident_surfaces(clusters, surface_ids, fit_results, fit_meshes, occ_surfs,
                               adj,
                               tol_angle_deg=1.0, tol_dist=1e-2, tol_radius=1e-2,
                               tol_cone_angle_deg=1.0):
    """
    Detect adjacent clusters fitted to identical (coincident) surfaces and merge them.

    Two clusters are merged when they are adjacent AND their surface type matches
    AND all geometric parameters agree within tolerance:
      Plane    : parallel normals AND same signed offset d
      Cylinder : parallel axes AND axis lines coincide AND equal radii
      Cone     : coincident apex AND parallel axes AND equal half-angles
      Sphere   : coincident centres AND equal radii

    Returns updated (clusters, surface_ids, fit_results, fit_meshes, occ_surfs).
    The representative of each merged group is the member with the most points.
    """
    cos_tol = math.cos(math.radians(tol_angle_deg))
    n = len(clusters)

    # Union-Find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    def planes_coincide(pi, pj):
        ni = _unit_vec(pi["a"])
        nj = _unit_vec(pj["a"])
        cos_a = float(np.dot(ni, nj))
        if abs(cos_a) < cos_tol:
            return False
        di = float(pi["d"])
        dj = float(pj["d"]) * (1.0 if cos_a > 0 else -1.0)
        return abs(di - dj) < tol_dist

    def cylinders_coincide(pi, pj):
        ai = _unit_vec(pi["a"])
        aj = _unit_vec(pj["a"])
        if abs(float(np.dot(ai, aj))) < cos_tol:
            return False
        if abs(float(pi["radius"]) - float(pj["radius"])) > tol_radius:
            return False
        ci = np.asarray(pi["center"], dtype=np.float64)
        cj = np.asarray(pj["center"], dtype=np.float64)
        diff = cj - ci
        perp = diff - float(np.dot(diff, ai)) * ai
        return float(np.linalg.norm(perp)) < tol_dist

    def cones_coincide(pi, pj):
        ai = _unit_vec(pi["a"])
        aj = _unit_vec(pj["a"])
        if abs(float(np.dot(ai, aj))) < cos_tol:
            return False
        if abs(float(pi["theta"]) - float(pj["theta"])) > math.radians(tol_cone_angle_deg):
            return False
        vi = np.asarray(pi["v"], dtype=np.float64)
        vj = np.asarray(pj["v"], dtype=np.float64)
        return float(np.linalg.norm(vi - vj)) < tol_dist

    def spheres_coincide(pi, pj):
        if abs(float(pi["radius"]) - float(pj["radius"])) > tol_radius:
            return False
        ci = np.asarray(pi["center"], dtype=np.float64)
        cj = np.asarray(pj["center"], dtype=np.float64)
        return float(np.linalg.norm(ci - cj)) < tol_dist

    checkers = {
        SURFACE_PLANE:    planes_coincide,
        SURFACE_CYLINDER: cylinders_coincide,
        SURFACE_CONE:     cones_coincide,
        SURFACE_SPHERE:   spheres_coincide,
    }

    for i in range(n):
        checker = checkers.get(surface_ids[i])
        if checker is None:
            continue
        pi = fit_results[i]["params"]
        for j in range(i + 1, n):
            if not adj[i, j]:
                continue
            if surface_ids[j] != surface_ids[i] or find(i) == find(j):
                continue
            if checker(pi, fit_results[j]["params"]):
                sname = SURFACE_NAMES.get(surface_ids[i], str(surface_ids[i]))
                print(f"[merge] surfaces {i} and {j} are coincident {sname}s → merging")
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    if all(len(g) == 1 for g in groups.values()):
        return clusters, surface_ids, fit_results, fit_meshes, occ_surfs

    new_clusters, new_surface_ids, new_fit_results, new_fit_meshes, new_occ_surfs = [], [], [], [], []
    visited = set()
    for i in range(n):
        root = find(i)
        if root in visited:
            continue
        visited.add(root)
        members = groups[root]
        if len(members) == 1:
            new_clusters.append(clusters[i])
            new_surface_ids.append(surface_ids[i])
            new_fit_results.append(fit_results[i])
            new_fit_meshes.append(fit_meshes[i])
            new_occ_surfs.append(occ_surfs[i])
        else:
            rep = max(members, key=lambda k: len(clusters[k]))
            merged_pts = np.concatenate([clusters[k] for k in members], axis=0)
            sname = SURFACE_NAMES.get(surface_ids[rep], str(surface_ids[rep]))
            print(f"[merge] group {members}: representative={rep} ({sname}), "
                  f"total {sum(len(clusters[k]) for k in members)} pts "
                  f"from {len(members)} clusters")
            new_clusters.append(merged_pts)
            new_surface_ids.append(surface_ids[rep])
            new_fit_results.append(fit_results[rep])
            new_fit_meshes.append(fit_meshes[rep])
            new_occ_surfs.append(occ_surfs[rep])

    return new_clusters, new_surface_ids, new_fit_results, new_fit_meshes, new_occ_surfs


# ---------------------------------------------------------------------------
# Compute mode  (Docker, OCC available)
# ---------------------------------------------------------------------------

def run_compute(args):
    import torch

    from point2cad.surface_fitter       import fit_surface
    from point2cad.occ_surfaces         import to_occ_surface
    from point2cad.cluster_adjacency    import (
        compute_adjacency_matrix, adjacency_pairs, build_cluster_proximity,
    )
    from point2cad.color_config         import get_surface_color
    from point2cad.surface_intersection import (
        compute_all_intersections,
        trim_by_vertices,
        compute_vertices_intcs,
        sample_curve,
    )
    from point2cad.mesh_intersections import (
        compute_mesh_intersections,
    )
    from point2cad.topology import (
        build_edge_arcs,
        greedy_oracle_filter,
        _score_vertex,
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
        # if part_idx not in [0]:
        #      continue
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
        # Collect clusters, compute proximity and adjacency
        # ------------------------------------------------------------------
        clusters = []
        for cid in unique_clusters:
            cluster = data[data[:, -1].astype(int) == cid, :3].astype(np.float32)
            clusters.append(cluster)

        cluster_trees, cluster_nn_percentiles = build_cluster_proximity(
            clusters, percentile=args.spacing_percentile
        )

        adj, _, spacing, boundary_strips, per_pair_thresholds, boundary_strip_trees = compute_adjacency_matrix(
            clusters, threshold_factor=args.spacing_factor,
            spacing_percentile=args.spacing_percentile,
            local_spacings=cluster_nn_percentiles,
        )
        inter_adj = adj
        print(f"[adjacency] spacing={spacing:.5f}  threshold={args.spacing_factor * spacing:.5f}")
        print(f"[adjacency] adjacent pairs: {adjacency_pairs(adj)}")
        for (i, j), bpts in sorted(boundary_strips.items()):
            print(f"  boundary ({i}, {j}): {len(bpts)} points")

        # ------------------------------------------------------------------
        # Surface fitting
        # ------------------------------------------------------------------
        surface_ids, fit_results, fit_meshes, occ_surfs = [], [], [], []
        for idx, (cid, c_count) in enumerate(zip(unique_clusters, cluster_counts)):
            cluster = clusters[idx]
            print(f"[surface fitter] Cluster {cid} ({c_count} pts) fitting ...")
            # Mesh pathway: pass per-cluster spacing so grid_trimming
            # uses the same threshold as adjacency detection.
            if args.intersection_method == "mesh":
                _spacing = cluster_nn_percentiles[idx]
                _plane_kw    = {"mesh_dim": 100, "plane_sampling_deviation": 0.5,
                                "spacing": _spacing,
                                "threshold_multiplier": 5}
                _sphere_kw   = {"dim_theta": 100, "dim_lambda": 100,
                                "spacing": _spacing,
                                "threshold_multiplier": 5}
                _cylinder_kw = {"dim_theta": 200, "dim_height": 50,
                                "cylinder_height_margin": 0.5,
                                "spacing": _spacing,
                                "threshold_multiplier": 0.5}
                _cone_kw     = {"dim_theta": 100, "dim_height": 100,
                                "cone_height_margin": 0.5,
                                "spacing": _spacing,
                                "threshold_multiplier": 5}
            else:
                _plane_kw    = None
                _sphere_kw   = None
                _cylinder_kw = None
                _cone_kw     = None

            res = fit_surface(
                cluster,
                {"hidden_dim": 64, "use_shortcut": True, "fraction_siren": 0.5},
                np_rng, DEVICE,
                # simple_error_threshold = 1e6,
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
                plane_mesh_kwargs=_plane_kw,
                sphere_mesh_kwargs=_sphere_kw,
                cylinder_mesh_kwargs=_cylinder_kw,
                cone_mesh_kwargs=_cone_kw,
            )
            sid = res["surface_id"]
            surface_ids.append(sid)
            fit_results.append(res["result"])
            fit_meshes.append(res["mesh"])
            occ_surfs.append(
                to_occ_surface(sid, res["result"], cluster=cluster, uv_margin = 0.05, grid_resolution = 50)
            )
            # Log surface fitting result with all residual errors
            chosen_err = res["result"]["error"]
            all_errors = res.get("all_errors", {})
            errors_str = "  ".join(f"{name}={err:.6f}" for name, err in all_errors.items())
            print(f"[surface fitter] Cluster {cid} ({c_count} pts) → "
                  f"{SURFACE_NAMES[sid]}  residual={chosen_err:.6f}")
            if errors_str:
                print(f"  all errors: {errors_str}")

        # Discard adjacency pairs whose boundary point count is a low outlier
        if boundary_strips and args.boundary_min_percentile > 0:
            counts = np.array([len(bpts) for bpts in boundary_strips.values()])
            cutoff = np.percentile(counts, args.boundary_min_percentile)
            removed = []
            for (i, j) in list(boundary_strips.keys()):
                if len(boundary_strips[(i, j)]) < cutoff:
                    removed.append((i, j, len(boundary_strips[(i, j)])))
                    del boundary_strips[(i, j)]
                    inter_adj[i, j] = False
                    inter_adj[j, i] = False
                    per_pair_thresholds.pop((i, j), None)
                    boundary_strip_trees.pop((i, j), None)
            if removed:
                print(f"[adjacency] boundary filter: cutoff={cutoff:.0f} pts "
                      f"(percentile={args.boundary_min_percentile})")
                for i, j, c in removed:
                    print(f"  removed ({i}, {j}): {c} points")
                print(f"[adjacency] {len(boundary_strips)} pairs remaining")

        # Merge adjacent clusters fitted to coincident surfaces.
        if args.merge_surfaces:
            prev_n = len(clusters)
            clusters, surface_ids, fit_results, fit_meshes, occ_surfs = \
                merge_coincident_surfaces(
                    clusters, surface_ids, fit_results, fit_meshes, occ_surfs,
                    adj=inter_adj,
                )
            if len(clusters) < prev_n:
                print(f"[merge] {prev_n} → {len(clusters)} clusters after merging")
                cluster_trees, cluster_nn_percentiles = build_cluster_proximity(
                    clusters, percentile=args.spacing_percentile
                )
                adj, _, spacing, boundary_strips, per_pair_thresholds, boundary_strip_trees = compute_adjacency_matrix(
                    clusters, threshold_factor=args.spacing_factor,
                    spacing_percentile=args.spacing_percentile,
                    local_spacings=cluster_nn_percentiles,
                )
                inter_adj = adj
                print(f"[adjacency] recomputed after merge: {len(adjacency_pairs(adj))} pairs")

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
        print(f"[surface fitter] cluster files saved to {out_dir}/")

        # ------------------------------------------------------------------
        # Surface-surface intersection + trimming
        # ------------------------------------------------------------------
        cluster_bboxes = None

        if args.intersection_method == "mesh":
            # ---- Mesh pathway ----
            raw_intersections, polyline_map = compute_mesh_intersections(
                inter_adj, surface_ids, fit_results, fit_meshes,
            )
            # Vertex detection via IntCS (curve ∩ surface)
            vertices, vertex_edges = compute_vertices_intcs(
                inter_adj, raw_intersections, occ_surfaces=occ_surfs,
                threshold=threshold_vertex,
            )
            print(f"[vertices] found {len(vertices)} vertices (IntCS)")

            # Score-cap filter: remove vertices too far from their clusters.
            vertex_scores = []
            for v_idx, (vpos, edges) in enumerate(zip(vertices, vertex_edges)):
                involved = set()
                for edge in edges:
                    involved.update(edge)
                score = _score_vertex(vpos, involved, cluster_trees,
                                      cluster_nn_percentiles)
                vertex_scores.append(score)

            score_cap = args.score_cap
            scores_arr = np.array(vertex_scores)

            sorted_indices = np.argsort(scores_arr)
            print(f"[vertex scores] {len(scores_arr)} vertices, "
                  f"range [{0 if scores_arr.size == 0 else scores_arr.min():.4f}, "
                  f"{0 if scores_arr.size == 0 else scores_arr.max():.4f}]")
            for rank, idx in enumerate(sorted_indices):
                edges_str = " ".join(
                    f"({min(e)}, {max(e)})" for e in vertex_edges[idx])
                print(f"  v{idx:3d}  score={scores_arr[idx]:10.4f}  "
                      f"edges=[{edges_str}]")

            keep_v = scores_arr <= score_cap
            n_drop = int(np.sum(~keep_v))
            n_keep = int(np.sum(keep_v))
            if n_drop > 0 and n_keep >= 2:
                print(f"[score-cap filter] threshold: {score_cap:.2f}  "
                      f"keeping {n_keep}, dropping {n_drop}/{len(scores_arr)} vertices")
                vertices = vertices[keep_v]
                vertex_edges = [vertex_edges[i]
                                for i in range(len(keep_v)) if keep_v[i]]
            else:
                print(f"[score-cap filter] threshold {score_cap:.2f} "
                      f"would keep {n_keep} / drop {n_drop} — skipping")
            print(f"[score-cap filter] after filter: {len(vertices)} vertices")
        else:
            # ---- Analytical pathway (OCC) ----
            raw_intersections = compute_all_intersections(
                inter_adj, surface_ids, fit_results, occ_surfs,
            )
            # Vertex-first flow:
            #   1. Find vertices on raw (untrimmed) curves — NMS is built into
            #      compute_vertices so the result is already deduplicated.
            #   2. Drop vertices too far from their constituent clusters
            #      (spurious intersections outside the physical model).
            #   3. Trim open curves using the surviving vertex parameters.
            vertices, vertex_edges = compute_vertices_intcs(
                inter_adj, raw_intersections, occ_surfaces=occ_surfs,
                threshold=threshold_vertex,
            )
            print(f"[vertices] found {len(vertices)} raw vertices")

            # Score-cap vertex pre-filter: remove vertices whose fitness score
            # exceeds a threshold.  The score d/p is already scale-invariant
            # (NN distance normalised by intra-cluster spacing), so the cap
            # has a physical meaning: score <= N means "within Nx the cluster's
            # own point spacing".
            vertex_scores = []
            for v_idx, (vpos, edges) in enumerate(zip(vertices, vertex_edges)):
                involved = set()
                for edge in edges:
                    involved.update(edge)
                score = _score_vertex(vpos, involved, cluster_trees,
                                      cluster_nn_percentiles)
                vertex_scores.append(score)

            score_cap = args.score_cap
            scores_arr = np.array(vertex_scores)

            # Log all vertex scores sorted for analysis
            sorted_indices = np.argsort(scores_arr)
            print(f"[vertex scores] {len(scores_arr)} vertices, "
                  f"range [{0 if scores_arr.size == 0 else scores_arr.min():.4f}, {0 if scores_arr.size == 0 else scores_arr.max():.4f}]")
            for rank, idx in enumerate(sorted_indices):
                edges_str = " ".join(
                    f"({min(e)}, {max(e)})" for e in vertex_edges[idx])
                print(f"  v{idx:3d}  score={scores_arr[idx]:10.4f}  "
                      f"edges=[{edges_str}]")

            keep_v = scores_arr <= score_cap
            n_drop = int(np.sum(~keep_v))
            n_keep = int(np.sum(keep_v))
            if n_drop > 0 and n_keep >= 2:
                print(f"[score-cap filter] threshold: {score_cap:.2f}  "
                      f"keeping {n_keep}, dropping {n_drop}/{len(scores_arr)} vertices")
                vertices = vertices[keep_v]
                vertex_edges = [vertex_edges[i]
                                for i in range(len(keep_v)) if keep_v[i]]
            else:
                print(f"[score-cap filter] threshold {score_cap:.2f} "
                      f"would keep {n_keep} / drop {n_drop} — skipping")
            print(f"[score-cap filter] after filter: {len(vertices)} vertices")

            trim_intersections_ = trim_by_vertices(
                raw_intersections, vertices, vertex_edges,
                extension_factor=0.05,
            )

        # Sample curves to numpy for the visualizer (no OCC on host).
        if args.intersection_method == "mesh":
            trim_intersections_ = raw_intersections  # no separate trimming step
        for (i, j) in raw_intersections:
            inter_raw  = raw_intersections[(i, j)]
            inter_trim = trim_intersections_[(i, j)]
            si = SURFACE_NAMES[surface_ids[i]]
            sj = SURFACE_NAMES[surface_ids[j]]
            print(f"[intersect] ({i}, {j})  {si} ∩ {sj}  type={inter_raw['type']}"
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
            # Save raw polylines (mesh pathway only)
            if args.intersection_method == "mesh":
                polys = polyline_map.get((i, j), [])
                kw["n_polylines"] = len(polys)
                for k, poly in enumerate(polys):
                    kw[f"polyline_points_{k}"] = _denorm(
                        poly, part_mean, part_R, part_scale)
            np.savez(os.path.join(out_dir, f"inter_{i}_{j}.npz"), **kw)

        # ------------------------------------------------------------------
        # Step 1: arc splitting
        # ------------------------------------------------------------------
        trim_curves_dict = {k: v["curves"] for k, v in trim_intersections_.items()}
        edge_arcs, vertices, vertex_edges = build_edge_arcs(
            trim_curves_dict, vertices, vertex_edges, threshold=1e-3
        )
        print_edge_arcs_summary(edge_arcs)

        # Save pre-filter vertices and arcs for visualization
        np.savez(os.path.join(out_dir, "vertices_pre_filter.npz"),
                 vertices=_denorm(vertices, part_mean, part_R, part_scale))
        for (ei, ej), arcs in edge_arcs.items():
            kw_pre = {"n_arcs": len(arcs), "edge_i": ei, "edge_j": ej}
            for k, arc in enumerate(arcs):
                kw_pre[f"arc_points_{k}"] = _denorm(
                    sample_curve(arc["curve"], n_points=100), part_mean, part_R, part_scale)
                kw_pre[f"arc_color_{k}"] = [0.2, 0.85, 0.2]
            np.savez(os.path.join(out_dir, f"arcs_pre_filter_{ei}_{ej}.npz"), **kw_pre)

        # Greedy oracle-guided filter: remove worst-scoring objects one at
        # a time until BRepCheck_Analyzer returns True.
        edge_arcs, vertices, vertex_edges, shape, brep_info = greedy_oracle_filter(
            edge_arcs, vertices, vertex_edges,
            clusters, cluster_trees, cluster_nn_percentiles,
            occ_surfaces=occ_surfs, surface_ids=surface_ids,
            bspline_method=args.bspline_method, tolerance=1e-3,
            cluster_bboxes=cluster_bboxes,
        )
        if shape is not None and not shape.IsNull():
            print_edge_arcs_summary(edge_arcs)
            face_arcs = face_arc_incidence(edge_arcs)
            print_face_arcs_summary(face_arcs)
            face_wires = assemble_wires(face_arcs, occ_surfs, vertices,
                                         surface_ids=surface_ids)
            print_face_wires_summary(face_wires)

        # Save vertices AFTER filtering so --visualize shows only final vertices
        np.savez(os.path.join(out_dir, "vertices.npz"),
                 vertices=_denorm(vertices, part_mean, part_R, part_scale))

        # Save arc samples for visualization (one file per edge).
        _ARC_COLORS = [[0.2, 0.85, 0.2], [1.0, 0.2, 0.2],
                       [0.2, 0.4,  1.0], [0.8, 0.2,  0.8]]
        for (ei, ej), arcs in edge_arcs.items():
            kw_arcs = {"n_arcs": len(arcs), "edge_i": ei, "edge_j": ej}
            for k, arc in enumerate(arcs):
                kw_arcs[f"arc_points_{k}"] = _denorm(
                    sample_curve(arc["curve"], n_points=100), part_mean, part_R, part_scale)
                kw_arcs[f"arc_color_{k}"]  = _ARC_COLORS[0]
            np.savez(os.path.join(out_dir, f"arcs_{ei}_{ej}.npz"), **kw_arcs)

        # ------------------------------------------------------------------
        # Step 4a: BRep export (shape already built by oracle filter)
        # ------------------------------------------------------------------
        step_path = os.path.join(out_dir, f"{step_stem}.step")
        if shape is None or shape.IsNull():
            print(f"[brep] build failed — 0 faces produced, skipping STEP export")
        else:
            try:
                shape_world = apply_inverse_normalization(shape, part_mean, part_R, part_scale)
            except Exception as e:
                print(f"[brep] apply_inverse_normalization failed: {e} — exporting normalized shape")
                shape_world = shape
            export_step(shape_world, step_path)

        # ------------------------------------------------------------------
        # Step 4b: BRep assembly — BOPAlgo_MakerVolume with cluster UV bounds
        # ------------------------------------------------------------------
        step_path_bop = os.path.join(out_dir, f"{step_stem}_bop.step")
        #shape_bop = build_brep_shape_bop(
        #    occ_surfs, vertices, vertex_edges, face_arcs,
        #    surface_ids=surface_ids,
        #    tolerance=1e-3, rel_margin=0.5,
        #)
        #if shape_bop is not None:
        #    export_step(apply_inverse_normalization(shape_bop, part_mean, part_R, part_scale),
        #                step_path_bop)
        print(f"\n[part] all results saved to {out_dir}/")

        if args.model_id:
            part_dirs.append((out_dir, cluster_offset, len(clusters)))
            cluster_offset += len(clusters)

    # After all parts: merge into unified/
    if part_dirs:
        unified_dir = os.path.join(model_out_dir, "unified")
        _merge_part_dirs(part_dirs, unified_dir)
        manual_paths = [
            os.path.join(d, f"{os.path.basename(d)}.step")
            for d, _, _ in part_dirs
        ]
        bop_paths = [
            os.path.join(d, f"{os.path.basename(d)}_bop.step")
            for d, _, _ in part_dirs
        ]
        merge_step_files(
            [p for p in manual_paths if os.path.exists(p)],
            os.path.join(unified_dir, "unified.step"),
        )
        #merge_step_files(
        #    [p for p in bop_paths if os.path.exists(p)],
        #    os.path.join(unified_dir, "unified_bop.step"),
        #)
        #print(f"\nUnified results saved to {unified_dir}/")


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
    parser.add_argument("--spacing_factor", type=float, default=1.5,
                        help="Adjacency detection threshold = spacing factor * "
                             "per-pair local spacing")
    parser.add_argument("--boundary_min_percentile", type=float, default=0,
                        help="Discard adjacency pairs whose boundary point count "
                             "falls below this percentile of all boundary counts. "
                             "Set to 0 to disable.")
    parser.add_argument("--score_cap", type=float, default=10,
                        help="Max vertex fitness score (d/p) to keep in pre-filter. "
                             "Score is scale-invariant: 1.0 = cluster boundary.")
    parser.add_argument("--boundary_mesh", action="store_true",
                        help="Visualize boundary strips as filled Delaunay meshes "
                             "instead of point clouds (visualize mode only)")
    parser.add_argument("--merge_surfaces", action="store_true",
                        help="Merge adjacent clusters fitted to coincident surfaces "
                             "(e.g. over-segmented planes with identical parameters)")
    parser.add_argument("--intersection_method", type=str, default="analytical",
                        choices=["analytical", "mesh"],
                        help="Surface intersection computation method. "
                             "'analytical': OCC GeomAPI_IntSS with analytical "
                             "fallbacks. 'mesh': PyVista mesh intersection "
                             "with BSpline fitting.")
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
