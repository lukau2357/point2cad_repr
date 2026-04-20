"""
Mesh generation pipeline: fit surfaces, generate meshes, clip (BFS or p2cad-style).
"""
import argparse
import glob as _glob
import json
import os
import shutil
import time

import numpy as np
import open3d as o3d
import trimesh

from point2cad.evaluation import compute_part_metrics


# ---------------------------------------------------------------------------
# Merge per-part outputs into unified/
# ---------------------------------------------------------------------------

def _denorm_points(pts, mean, R, scale):
    """Inverse PCA normalization: p_world = scale * R^T @ p_norm + mean."""
    return (scale * (R.T @ pts.T).T + mean)


def _merge_part_dirs(part_dirs, unified_dir):
    """
    Merge per-part output directories into a single unified directory for
    the visualizer.  Applies inverse normalization so all parts are in
    world-space coordinates.  Per-part files remain in normalized space.

    part_dirs : list of (dir_path, cluster_offset, n_clusters)
    """
    os.makedirs(unified_dir, exist_ok=True)

    all_snames, all_colors, all_labels = [], [], []
    for dir_path, offset, n in part_dirs:
        meta = np.load(os.path.join(dir_path, "metadata.npz"), allow_pickle=True)
        all_snames.extend(meta["surface_names"].tolist())
        all_colors.extend(meta["cluster_colors"].tolist())
        part_idx = int(os.path.basename(dir_path).split("_")[1])
        for cid in meta["cluster_ids"]:
            if len(part_dirs) == 1:
                all_labels.append(f"cluster_{cid}")
            else:
                all_labels.append(f"part_{part_idx}/cluster_{cid}")

    clip_method = "unknown"
    if part_dirs:
        meta0 = np.load(os.path.join(part_dirs[0][0], "metadata.npz"), allow_pickle=True)
        if "clip_method" in meta0:
            clip_method = str(meta0["clip_method"])

    np.savez(os.path.join(unified_dir, "metadata.npz"),
             n_clusters=len(all_snames),
             surface_names=np.array(all_snames),
             cluster_colors=np.array(all_colors),
             cluster_labels=np.array(all_labels),
             clip_method=clip_method)

    unified_combined = o3d.geometry.TriangleMesh()
    for dir_path, offset, n in part_dirs:
        meta = np.load(os.path.join(dir_path, "metadata.npz"), allow_pickle=True)
        mean = meta["norm_mean"]
        R = meta["norm_R"]
        scale = float(meta["norm_scale"])
        cluster_ids = meta["cluster_ids"]

        part_combined = o3d.geometry.TriangleMesh()
        for out_idx, cid in enumerate(cluster_ids):
            # Clusters (.npy)
            src = os.path.join(dir_path, f"cluster_{cid}.npy")
            if os.path.exists(src):
                pts = np.load(src)
                np.save(os.path.join(unified_dir, f"cluster_{out_idx + offset}.npy"),
                        _denorm_points(pts, mean, R, scale))

            # Surface meshes and trimmed meshes (.npz with vertices/triangles)
            for prefix in ("surface_mesh", "trimmed_mesh"):
                src = os.path.join(dir_path, f"{prefix}_{cid}.npz")
                if not os.path.exists(src):
                    continue
                d = np.load(src)
                verts = _denorm_points(d["vertices"], mean, R, scale)
                np.savez(os.path.join(unified_dir, f"{prefix}_{out_idx + offset}.npz"),
                         vertices=verts, triangles=d["triangles"])

                if prefix == "trimmed_mesh" and len(verts) > 0:
                    mesh = o3d.geometry.TriangleMesh()
                    mesh.vertices = o3d.utility.Vector3dVector(verts)
                    mesh.triangles = o3d.utility.Vector3iVector(d["triangles"])
                    part_combined += mesh

        # Per-part denormalized trimmed STL
        part_combined.compute_vertex_normals()
        part_stl = os.path.join(dir_path, "trimmed_denorm.stl")
        o3d.io.write_triangle_mesh(part_stl, part_combined)
        unified_combined += part_combined

    # Unified trimmed STL
    unified_combined.compute_vertex_normals()
    unified_stl = os.path.join(unified_dir, "trimmed.stl")
    o3d.io.write_triangle_mesh(unified_stl, unified_combined)
    print(f"[mesh] Unified trimmed mesh: {unified_stl}")


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def run_compute(args):
    import matplotlib.pyplot as plt
    import torch

    import point2cad.primitive_fitting_utils as pfu
    from point2cad.color_config import get_surface_color
    from point2cad.mesh_clipping import clip_meshes_bfs, clip_meshes_p2cad, build_cluster_trees
    from point2cad.surface_fitter import SURFACE_NAMES, fit_surface

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    def normalize_points(pts):
        mean = pts.mean(axis=0)
        centered = pts - mean
        S, U = np.linalg.eigh(centered.T @ centered)
        R = pfu.rotation_matrix_a_to_b(U[:, np.argmin(S)], np.array([1, 0, 0]))
        rotated = (R @ centered.T).T
        scale = float((rotated.max(axis=0) - rotated.min(axis=0)).max()) + 1e-7
        return (rotated / scale).astype(np.float32), mean, R, scale

    input_pattern = os.path.join(args.input_dir, args.model_id, "*.xyzc")
    xyzc_files = sorted(_glob.glob(input_pattern),
                        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    if not xyzc_files:
        print(f"No .xyzc files found matching: {input_pattern}")
        return

    if args.part is not None:
        if args.part >= len(xyzc_files):
            print(f"Part {args.part} out of range — model has {len(xyzc_files)} part(s)")
            return
        part_indices = [args.part]
    else:
        part_indices = list(range(len(xyzc_files)))

    print(f"Model {args.model_id}: {len(xyzc_files)} part(s), processing {len(part_indices)}")

    part_dirs = []
    cluster_offset = 0
    part_times = []
    t_total = time.perf_counter()
    for part_idx in part_indices:
        t_part = time.perf_counter()
        _run_compute_part(args, xyzc_files[part_idx], part_idx, normalize_points,
                          DEVICE)
        dt_part = time.perf_counter() - t_part
        part_times.append((part_idx, dt_part))
        print(f"[timing] part {part_idx}: {dt_part:.2f}s")
        out_dir = os.path.join(args.output_dir, args.model_id, f"part_{part_idx}")
        meta_path = os.path.join(out_dir, "metadata.npz")
        if os.path.exists(meta_path):
            n = int(np.load(meta_path, allow_pickle=True)["n_clusters"])
            part_dirs.append((out_dir, cluster_offset, n))
            cluster_offset += n

    if part_dirs:
        unified_dir = os.path.join(args.output_dir, args.model_id, "unified")
        _merge_part_dirs(part_dirs, unified_dir)

    dt_total = time.perf_counter() - t_total
    print(f"\n[timing] === mesh_pipeline summary for {args.model_id} ===")
    for pi, dt in part_times:
        print(f"[timing]   part {pi}: {dt:.2f}s")
    print(f"[timing]   total ({len(part_times)} parts): {dt_total:.2f}s")


def _run_compute_part(args, sample_path, part_idx, normalize_points, DEVICE):
    from point2cad.color_config import get_surface_color
    from point2cad.mesh_clipping import clip_meshes_bfs, clip_meshes_p2cad, build_cluster_trees
    from point2cad.surface_fitter import SURFACE_NAMES, fit_surface

    np_rng = np.random.default_rng(args.seed)
    tm = args.threshold_multiplier
    abs_th = 0.1

    out_dir = os.path.join(args.output_dir, args.model_id, f"part_{part_idx}")

    if args.clip_only:
        # Load previously saved clusters, meshes, and surface types
        meta_path = os.path.join(out_dir, "metadata.npz")
        if not os.path.exists(meta_path):
            print(f"No saved results in {out_dir}/ — run full compute first.")
            return

        print(f"\n{'='*60}")
        print(f"Part {part_idx} (clip-only)  →  {out_dir}")
        print(f"{'='*60}")

        meta = np.load(meta_path, allow_pickle=True)
        n_clusters = int(meta["n_clusters"])
        unique_clusters = meta["cluster_ids"]
        surface_type_names = [str(s) for s in meta["surface_names"]]
        part_mean = meta["norm_mean"]
        part_R = meta["norm_R"]
        part_scale = float(meta["norm_scale"])

        clusters = []
        o3d_meshes = []
        for cid in unique_clusters:
            clusters.append(np.load(os.path.join(out_dir, f"cluster_{cid}.npy")))
            d = np.load(os.path.join(out_dir, f"surface_mesh_{cid}.npz"))
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(d["vertices"])
            mesh.triangles = o3d.utility.Vector3iVector(d["triangles"])
            o3d_meshes.append(mesh)

        print(f"[mesh] Loaded {n_clusters} clusters and meshes from {out_dir}/")
        cluster_trees, spacings = build_cluster_trees(clusters, args.spacing_percentile)
        fit_time = 0.0
        primitive_fit_time = 0.0
        freeform_fit_time = 0.0
    else:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
            print(f"Removed old results: {out_dir}")
        os.makedirs(out_dir)

        print(f"\n{'='*60}")
        print(f"Part {part_idx}: {os.path.basename(sample_path)}  →  {out_dir}")
        print(f"{'='*60}")

        data = np.loadtxt(sample_path)
        pts_norm, part_mean, part_R, part_scale = normalize_points(data[:, :3])
        data[:, :3] = pts_norm
        unique_clusters, cluster_counts = np.unique(data[:, -1].astype(int), return_counts=True)
        n_clusters = len(unique_clusters)
        print(f"[mesh] {n_clusters} clusters")

        clusters = []
        for cid in unique_clusters:
            cluster = data[data[:, -1] == cid][:, :3].astype(np.float32)
            clusters.append(cluster)

        # Drop clusters too small to fit a primitive surface stably.
        # Matches HPNet/Point2CAD floor of 20 points (covers 6-DoF cone).
        MIN_CLUSTER_PTS = 20
        keep_mask = [len(c) >= MIN_CLUSTER_PTS for c in clusters]
        n_dropped = sum(1 for k in keep_mask if not k)
        if n_dropped > 0:
            dropped_info = [(int(cid), len(clusters[i]))
                            for i, cid in enumerate(unique_clusters) if not keep_mask[i]]
            print(f"[preprocess] dropped {n_dropped} cluster(s) "
                  f"with < {MIN_CLUSTER_PTS} pts: {dropped_info}")
            clusters        = [c for c, k in zip(clusters, keep_mask) if k]
            unique_clusters = np.array([cid for cid, k in zip(unique_clusters, keep_mask) if k])
            cluster_counts  = np.array([cnt for cnt, k in zip(cluster_counts, keep_mask) if k])
            n_clusters      = len(clusters)

        for idx, cid in enumerate(unique_clusters):
            np.save(os.path.join(out_dir, f"cluster_{cid}.npy"), clusters[idx])

        cluster_trees, spacings = build_cluster_trees(clusters, args.spacing_percentile)

        o3d_meshes = []
        surface_type_names = []

        t_fit = time.perf_counter()
        primitive_fit_time = 0.0
        freeform_fit_time = 0.0
        for idx, (cid, c_count) in enumerate(zip(unique_clusters, cluster_counts)):
            cluster = clusters[idx]

            print(f"[surface fitter] Cluster {cid} ({c_count} pts) fitting ...")
            res = fit_surface(
                cluster,
                {"hidden_dim": 64, "use_shortcut": True, "fraction_siren": 0.5},
                np_rng, DEVICE,
                freeform_method=args.freeform_method,
                spacing=spacings[idx],
                cluster_tree=cluster_trees[idx],
                inr_fit_kwargs={
                    "max_steps": args.inr_max_steps,
                    "noise_magnitude_3d": 0.05,
                    "noise_magnitude_uv": 0.05,
                    "initial_lr": 1e-1,
                    "polish": args.inr_polish,
                    "polish_reg_peak": args.inr_polish_reg_peak,
                    "seed": args.seed,
                },
                plane_mesh_kwargs={"mesh_dim": 200, "threshold_multiplier": tm,
                                   "plane_sampling_deviation": 2, "spacing": spacings[idx],
                                   "absolute_threshold": abs_th},
                sphere_mesh_kwargs={"dim_theta": 200, "dim_lambda": 200,
                                    "threshold_multiplier": tm, "spacing": spacings[idx],
                                    "absolute_threshold": abs_th},
                cylinder_mesh_kwargs={"dim_theta": 200, "dim_height": 100,
                                      "threshold_multiplier": tm, "cylinder_height_margin": 0.5, "spacing": spacings[idx],
                                      "absolute_threshold": abs_th},
                cone_mesh_kwargs={"dim_theta": 200, "dim_height": 200,
                                  "threshold_multiplier": tm, "cone_height_margin": 0.5, "spacing": spacings[idx],
                                  "absolute_threshold": abs_th},
                inr_mesh_kwargs={"mesh_dim": 200, "uv_margin": args.inr_uv_margin, "threshold_multiplier": tm, "spacing": spacings[idx]},
                radius_inflation=0.001
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

            primitive_fit_time += res.get("primitive_fit_time", 0.0)
            freeform_fit_time += res.get("freeform_fit_time", 0.0)

            verts = np.asarray(res["mesh"].vertices)
            tris  = np.asarray(res["mesh"].triangles)
            np.savez(os.path.join(out_dir, f"surface_mesh_{cid}.npz"),
                     vertices=verts, triangles=tris)
        fit_time = time.perf_counter() - t_fit

    t_clip = time.perf_counter()
    if args.clip_method == "bfs":
        print("[mesh] Running BFS flood-fill trimming ...")
        clipped_meshes = clip_meshes_bfs(o3d_meshes, clusters, surface_type_names,
                                         cluster_trees, spacings,
                                         post_filter_threshold=args.post_filter_threshold,
                                         cluster_ids=unique_clusters)
    else:
        print("[mesh] Running Point2CAD-style trimming ...")
        clipped_meshes = clip_meshes_p2cad(o3d_meshes, clusters, surface_type_names,
                                           cluster_trees=cluster_trees,
                                           spacings=spacings,
                                           area_multiplier=args.area_multiplier,
                                           post_filter_threshold=args.post_filter_threshold,
                                           cluster_ids=unique_clusters)
    clip_time = time.perf_counter() - t_clip

    print(f"[timing]   fit:  {fit_time:.2f}s  "
          f"(primitive: {primitive_fit_time:.2f}s, freeform: {freeform_fit_time:.2f}s)")
    print(f"[timing]   clip: {clip_time:.2f}s")
    print(f"[timing]   sum:  {fit_time + clip_time:.2f}s")

    timing_dict = {"fit_time": fit_time,
                   "primitive_fit_time": primitive_fit_time,
                   "freeform_fit_time": freeform_fit_time,
                   "clip_time": clip_time,
                   "total_time": fit_time + clip_time}
    with open(os.path.join(out_dir, "timing.json"), "w") as _f:
        json.dump(timing_dict, _f, indent=2)

    combined = o3d.geometry.TriangleMesh()
    for idx, mesh in enumerate(clipped_meshes):
        cid = int(unique_clusters[idx])
        verts = np.asarray(mesh.vertices)
        tris  = np.asarray(mesh.triangles)
        np.savez(os.path.join(out_dir, f"trimmed_mesh_{cid}.npz"),
                 vertices=verts, triangles=tris)
        combined += mesh
    combined.compute_vertex_normals()
    stl_path = os.path.join(out_dir, "trimmed.stl")
    o3d.io.write_triangle_mesh(stl_path, combined)
    print(f"[mesh] Exported {stl_path}")

    # ---- evaluation metrics (normalized space) ---------------------------
    if args.clip_only:
        raw = np.loadtxt(sample_path)
        pts_world = raw[:, :3]
        eval_points = ((part_R @ (pts_world - part_mean).T).T / part_scale).astype(np.float32)
        eval_labels = raw[:, -1].astype(int)
    else:
        eval_points = data[:, :3].astype(np.float32)
        eval_labels = data[:, -1].astype(int)

    tm_meshes = {}
    for idx, o3d_mesh in enumerate(clipped_meshes):
        cid = int(unique_clusters[idx])
        V = np.asarray(o3d_mesh.vertices)
        F = np.asarray(o3d_mesh.triangles)
        if len(V) == 0 or len(F) == 0:
            tm_meshes[cid] = None
        else:
            tm_meshes[cid] = trimesh.Trimesh(V, F, process=False)

    surface_types_dict = {int(unique_clusters[i]): surface_type_names[i]
                          for i in range(len(unique_clusters))}

    metrics = compute_part_metrics(
        input_points=eval_points,
        input_labels=eval_labels,
        cluster_meshes=tm_meshes,
        surface_types=surface_types_dict,
        timing=timing_dict,
        model_id=args.model_id,
        part_idx=part_idx,
        seed=args.seed,
    )
    with open(os.path.join(out_dir, "metrics.json"), "w") as _f:
        json.dump(metrics, _f, indent=2)
    m = metrics["metrics"]
    print(f"[eval] p_cov_p2m={m['p_coverage']:.4f}  "
          f"p_cov_m2p={m['p_coverage_mesh_to_pc']:.4f}  "
          f"resid_mean={m['residual_mean']}  "
          f"chamfer_sym={m['chamfer_sym']}  "
          f"primitive_only={metrics['is_primitive_only']}")

    cluster_colors = np.array([get_surface_color(stype) for stype in surface_type_names])
    np.savez(os.path.join(out_dir, "metadata.npz"),
             n_clusters=n_clusters,
             cluster_ids=unique_clusters,
             cluster_colors=cluster_colors,
             surface_names=np.array(surface_type_names),
             clip_method=args.clip_method,
             norm_mean=part_mean, norm_R=part_R, norm_scale=part_scale)

    print(f"[mesh] Saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Visualize
# ---------------------------------------------------------------------------

def run_visualize(args):
    """
    Five windows, all in denormalized world-space:
        1. Input point cloud
        2. Mine — unclipped surface meshes (with N/P cycling)
        3. Mine — clipped meshes
        4. Original Point2CAD — unclipped meshes
        5. Original Point2CAD — clipped meshes

    Loads our data from unified/ (already denormalized by _merge_part_dirs).
    Original Point2CAD PLYs are denormalized using normalization.npz saved
    alongside each part.
    """
    from point2cad.color_config import get_surface_color

    model_root = os.path.join(args.output_dir, args.model_id)
    unified_dir = os.path.join(model_root, "unified")
    if not os.path.isdir(unified_dir):
        print(f"No unified/ directory in {model_root}/ — run compute first.")
        return

    meta_path = os.path.join(unified_dir, "metadata.npz")
    if not os.path.exists(meta_path):
        print(f"No metadata.npz in {unified_dir}/ — run compute first.")
        return

    meta = np.load(meta_path, allow_pickle=True)
    n_clusters     = int(meta["n_clusters"])
    cluster_colors = meta["cluster_colors"]
    surface_names  = meta["surface_names"]
    cluster_labels = meta["cluster_labels"] if "cluster_labels" in meta else None
    clip_method    = str(meta["clip_method"]) if "clip_method" in meta else "unknown"

    pcd_combined     = o3d.geometry.PointCloud()
    surface_meshes   = []   # flat list across all clusters (for N/P cycling)
    surface_labels   = []   # parallel list of "cluster{i} ({stype})"
    clipped_combined = o3d.geometry.TriangleMesh()

    for i in range(n_clusters):
        stype = str(surface_names[i])
        scolor = get_surface_color(stype)

        pts_path = os.path.join(unified_dir, f"cluster_{i}.npy")
        if os.path.exists(pts_path):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.load(pts_path))
            pcd.paint_uniform_color(cluster_colors[i].tolist())
            pcd_combined += pcd

        sm_path = os.path.join(unified_dir, f"surface_mesh_{i}.npz")
        if os.path.exists(sm_path):
            d = np.load(sm_path)
            if len(d["vertices"]) > 0:
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices  = o3d.utility.Vector3dVector(d["vertices"])
                mesh.triangles = o3d.utility.Vector3iVector(d["triangles"])
                mesh.compute_vertex_normals()
                mesh.paint_uniform_color(scolor)
                surface_meshes.append(mesh)
                label = str(cluster_labels[i]) if cluster_labels is not None else f"cluster_{i}"
                surface_labels.append(f"{label} ({stype})")

        tm_path = os.path.join(unified_dir, f"trimmed_mesh_{i}.npz")
        if os.path.exists(tm_path):
            d = np.load(tm_path)
            if len(d["vertices"]) > 0:
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices  = o3d.utility.Vector3dVector(d["vertices"])
                mesh.triangles = o3d.utility.Vector3iVector(d["triangles"])
                mesh.compute_vertex_normals()
                mesh.paint_uniform_color(scolor)
                clipped_combined += mesh

    # ---- Per-part timings + metrics (mine + original Point2CAD if present) ---
    import json as _json
    def _load_json(path):
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    return _json.load(f)
            except Exception:
                return None
        return None

    part_dirs = sorted(
        d for d in os.listdir(model_root)
        if d.startswith("part_") and os.path.isdir(os.path.join(model_root, d))
    )
    mine_timings, orig_timings = [], []
    mine_metrics, orig_metrics = [], []
    orig_root = os.path.join("..", "point2cad", "output_p2cad_orig", args.model_id)

    for pd in part_dirs:
        part_dir = os.path.join(model_root, pd)
        mine_timings.append((pd, _load_json(os.path.join(part_dir, "timing.json"))))
        orig_timings.append((pd, _load_json(os.path.join(orig_root, pd, "timing.json"))))
        mine_metrics.append((pd, _load_json(os.path.join(part_dir, "metrics.json"))))
        orig_metrics.append((pd, _load_json(os.path.join(orig_root, pd, "metrics.json"))))

    # ---- Original Point2CAD outputs (per-cluster plys + types.json) ------
    def _load_orig(kind):
        """kind ∈ {'unclipped', 'clipped'}. Iterates part_*/<kind>/cluster_*.ply,
        reads types.json sidecar, applies our color scheme per cluster.
        Denormalizes using normalization.npz saved alongside each part."""
        import json as _json
        import glob as _g
        root = os.path.join("..", "point2cad", "output_p2cad_orig", args.model_id)
        combined = o3d.geometry.TriangleMesh()
        if not os.path.isdir(root):
            return combined
        for pd in sorted(d for d in os.listdir(root)
                         if d.startswith("part_")
                         and os.path.isdir(os.path.join(root, d))):
            part_dir = os.path.join(root, pd)
            kind_dir = os.path.join(part_dir, kind)
            types_path = os.path.join(kind_dir, "types.json")
            if not os.path.isfile(types_path):
                continue

            # Load normalization params for this part
            norm_path = os.path.join(part_dir, "normalization.npz")
            if os.path.isfile(norm_path):
                norm = np.load(norm_path)
                denorm = lambda pts: _denorm_points(pts, norm["mean"], norm["R"], float(norm["scale"]))
            else:
                denorm = lambda pts: pts

            with open(types_path) as f:
                types = _json.load(f)
            cluster_plys = sorted(
                _g.glob(os.path.join(kind_dir, "cluster_*.ply")),
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[1]),
            )
            for ply in cluster_plys:
                cid = int(os.path.splitext(os.path.basename(ply))[0].split("_")[1])
                if cid >= len(types):
                    continue
                m = o3d.io.read_triangle_mesh(ply)
                if len(m.vertices) == 0:
                    continue
                m.vertices = o3d.utility.Vector3dVector(denorm(np.asarray(m.vertices)))
                m.compute_vertex_normals()
                m.paint_uniform_color(get_surface_color(types[cid]))
                combined += m
        return combined

    orig_unclipped = _load_orig("unclipped")
    orig_clipped   = _load_orig("clipped")

    # ---- Print timing comparison -----------------------------------------
    def _fmt(t):
        return f"{t:7.2f}s" if t is not None else "      -"

    def _row(label, ts):
        if ts is None:
            return f"{label:>16}: {'      -':>8} {'      -':>8} {'      -':>8}"
        return (f"{label:>16}: {_fmt(ts.get('fit_time')):>8} "
                f"{_fmt(ts.get('clip_time')):>8} {_fmt(ts.get('total_time')):>8}")

    def _breakdown(ts):
        # Mine-only breakdown: primitive vs freeform time inside the fit phase.
        # Original Point2CAD doesn't record this split, so it's shown only for mine.
        if ts is None:
            return None
        prim = ts.get("primitive_fit_time")
        free = ts.get("freeform_fit_time")
        if prim is None and free is None:
            return None
        return f"{'(mine breakdown)':>16}  primitive: {_fmt(prim)}  freeform: {_fmt(free)}"

    print("\n[timing] === per-part timings ===")
    print(f"{'':>16}  {'fit':>8} {'clip':>8} {'total':>8}")
    mine_tot = {"fit": 0.0, "prim": 0.0, "free": 0.0, "clip": 0.0, "sum": 0.0}
    orig_tot = {"fit": 0.0, "clip": 0.0, "sum": 0.0}
    for (pd, mt), (_, ot) in zip(mine_timings, orig_timings):
        print(f"  {pd}")
        print("  " + _row("mine", mt))
        print("  " + _row("orig p2cad", ot))
        br = _breakdown(mt)
        if br is not None:
            print("  " + br)
        if mt and ot:
            mt_sum = mt.get("total_time") or 0.0
            ot_sum = ot.get("total_time") or 0.0
            if mt_sum > 0 and ot_sum > 0:
                print(f"  {'speedup':>16}: {ot_sum / mt_sum:.2f}x (orig/mine)")
        if mt:
            mine_tot["fit"]  += mt.get("fit_time", 0.0) or 0.0
            mine_tot["prim"] += mt.get("primitive_fit_time", 0.0) or 0.0
            mine_tot["free"] += mt.get("freeform_fit_time", 0.0) or 0.0
            mine_tot["clip"] += mt.get("clip_time", 0.0) or 0.0
            mine_tot["sum"]  += mt.get("total_time", 0.0) or 0.0
        if ot:
            orig_tot["fit"]  += ot.get("fit_time", 0.0) or 0.0
            orig_tot["clip"] += ot.get("clip_time", 0.0) or 0.0
            orig_tot["sum"]  += ot.get("total_time", 0.0) or 0.0
    print(f"\n  {'TOTAL':>16}")
    print(f"  {'mine':>16}: {mine_tot['fit']:7.2f}s {mine_tot['clip']:7.2f}s {mine_tot['sum']:7.2f}s")
    print(f"  {'orig p2cad':>16}: {orig_tot['fit']:7.2f}s {orig_tot['clip']:7.2f}s {orig_tot['sum']:7.2f}s")
    print(f"  {'(mine breakdown)':>16}  primitive: {mine_tot['prim']:7.2f}s  freeform: {mine_tot['free']:7.2f}s")
    if orig_tot['sum'] > 0 and mine_tot['sum'] > 0:
        print(f"  speedup (orig/mine): {orig_tot['sum'] / mine_tot['sum']:.2f}x")
    print()

    # ---- Per-part fidelity metrics ---------------------------------------
    def _mfmt(v, width=10, prec=4):
        if v is None:
            return "-".rjust(width)
        return f"{v:{width}.{prec}f}"

    def _metric_row(label, mj):
        if mj is None:
            dash = "-".rjust(10)
            return f"{label:>16}: {dash} {dash} {dash} {dash}"
        m = mj.get("metrics", {})
        return (f"{label:>16}: "
                f"{_mfmt(m.get('p_coverage'), 10, 4)} "
                f"{_mfmt(m.get('p_coverage_mesh_to_pc'), 10, 4)} "
                f"{_mfmt(m.get('residual_mean'), 10, 6)} "
                f"{_mfmt(m.get('chamfer_sym'), 10, 6)}")

    if any(mj is not None for _, mj in mine_metrics + orig_metrics):
        print("[metrics] === per-part fidelity metrics ===")
        print(f"{'':>16}   {'p_cov_p2m':>10} {'p_cov_m2p':>10} {'resid_mean':>10} {'chamfer_sym':>10}")
        for (pd, mj), (_, oj) in zip(mine_metrics, orig_metrics):
            print(f"  {pd}")
            print("  " + _metric_row("mine", mj))
            print("  " + _metric_row("orig p2cad", oj))
        print()

    # ---- N/P cycling over `surface_meshes` -------------------------------
    _highlight = {"idx": -1}

    def _update_highlight(vis_obj):
        idx = _highlight["idx"]
        for k, mesh in enumerate(surface_meshes):
            stype = surface_labels[k].rsplit("(", 1)[-1].rstrip(")")
            if idx == -1 or k == idx:
                mesh.paint_uniform_color(get_surface_color(stype))
            else:
                mesh.paint_uniform_color([0.3, 0.3, 0.3])
            mesh.compute_vertex_normals()
            vis_obj.update_geometry(mesh)
        vis_obj.update_renderer()
        if idx == -1:
            print("[surfaces] showing ALL clusters")
        else:
            print(f"[surfaces] {surface_labels[idx]}")

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

    # ---- Window layout ---------------------------------------------------
    W, H, top = 520, 540, 50
    row1_top, row2_top = top, top + H + 60

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window("Point cloud", width=W, height=H, left=0, top=row1_top)
    vis1.add_geometry(pcd_combined)
    vis1.get_render_option().point_size = 2.0

    vis2 = o3d.visualization.VisualizerWithKeyCallback()
    vis2.create_window("Mine — unclipped", width=W, height=H, left=W, top=row1_top)
    for mesh in surface_meshes:
        vis2.add_geometry(mesh)
    vis2.register_key_callback(ord("N"), _on_key_next)
    vis2.register_key_callback(ord("P"), _on_key_prev)
    vis2.get_render_option().mesh_show_back_face = True
    print("[surfaces] Press N/P in 'Mine — unclipped' window to cycle clusters")

    vis3 = o3d.visualization.Visualizer()
    vis3.create_window(f"Mine — clipped ({clip_method})",
                       width=W, height=H, left=2 * W, top=row1_top)
    vis3.add_geometry(clipped_combined)
    vis3.get_render_option().mesh_show_back_face = True

    vises = [vis1, vis2, vis3]

    if len(orig_unclipped.vertices) > 0:
        vis4 = o3d.visualization.Visualizer()
        vis4.create_window("Original Point2CAD — unclipped",
                           width=W, height=H, left=W, top=row2_top)
        vis4.add_geometry(orig_unclipped)
        vis4.get_render_option().mesh_show_back_face = True
        vises.append(vis4)
    else:
        print("[orig p2cad] no unclipped meshes found")

    if len(orig_clipped.vertices) > 0:
        vis5 = o3d.visualization.Visualizer()
        vis5.create_window("Original Point2CAD — clipped",
                           width=W, height=H, left=2 * W, top=row2_top)
        vis5.add_geometry(orig_clipped)
        vis5.get_render_option().mesh_show_back_face = True
        vises.append(vis5)
    else:
        print("[orig p2cad] no clipped meshes found")

    running = [True] * len(vises)
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
    parser.add_argument("--part", type=int, default=None,
                        help="Process only this part index (0-based). Default: all parts")
    parser.add_argument("--post_filter_threshold", type=float, default=1.5,
                        help="Post-BFS filter: keep faces whose barycenter is within "
                             "threshold * spacing of the nearest cluster point")
    parser.add_argument("--spacing_percentile", type=float, default=100.0,
                        help="Percentile of intra-cluster NN distances used as spacing "
                             "for the post-BFS filter (default 100 = max NN distance)")
    parser.add_argument("--freeform_method", type=str, default="inr",
                        choices=["inr", "bpa", "bpa_bspline"],
                        help="How to handle freeform clusters: 'inr' (neural autoencoder), "
                             "'bpa' (Ball Pivoting mesh only), or 'bpa_bspline' "
                             "(BPA -> LSCM -> RectBivariateSpline tensor-product B-spline)")
    parser.add_argument("--seed", type=int, default=41,
                        help="Reproducibility seed")
    parser.add_argument("--clip_method", type=str, default="p2cad",
                        choices=["bfs", "p2cad"],
                        help="Mesh trimming method: 'bfs' (BFS flood-fill) or 'p2cad' (Point2CAD-style components)")
    parser.add_argument("--area_multiplier", type=float, default=2.0,
                        help="[p2cad only] Keep components with area_per_point < best * area_multiplier")
    parser.add_argument("--clip_only", action="store_true",
                        help="Skip surface fitting; load saved untrimmed meshes and re-run clipping only")
    parser.add_argument("--trimming_mode", type=str, default="adaptive",
                        choices=["adaptive", "absolute"],
                        help="UV grid trimming mode: 'adaptive' uses threshold_multiplier * spacing, "
                             "'absolute' uses fixed threshold of 0.1 (Point2CAD-style)")
    parser.add_argument("--threshold_multiplier", type=float, default=6,
                        help="[adaptive only] Controls untrimmed mesh generation - higher values indicate more tolerance for face selection.")
    parser.add_argument("--poisson_depth", type=int, default=9,
                        help="Octree depth for screened Poisson reconstruction (higher = finer detail)")
    parser.add_argument("--poisson_density_quantile", type=float, default=0.02,
                        help="Remove Poisson vertices below this density quantile (trims outer artifacts)")
    parser.add_argument("--inr_max_steps", type=int, default=5000,
                        help="Number of training steps per INR (u,v) closedness combo")
    parser.add_argument("--inr_uv_margin", type=float, default=0.1,
                        help="Fractional UV margin used when sampling the INR mesh beyond the encoded UV bounding box")
    parser.add_argument("--inr_polish", action="store_true",
                        help="Run a post-training polish phase on the best INR "
                             "with Taylor-smoothness regularization over the UV margin / alpha-shape reject region")
    parser.add_argument("--inr_polish_reg_peak", type=float, default=0.01,
                        help="Peak regularization weight at the end of the polish ramp (0 -> reg_peak)")
    args = parser.parse_args()

    if args.visualize:
        run_visualize(args)
    else:
        run_compute(args)
