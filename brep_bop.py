"""
B-Rep reconstruction via BOPAlgo_MakerVolume.

Creates oversized bounded faces for each fitted surface, then lets OCC's
boolean kernel compute all intersection curves, split faces, and assemble
the solid.  No manual wire assembly.

Usage (inside Docker):
  python brep_bop.py --model_id 00000078 --input_dir sample_clouds --output_dir output_bop
"""

import argparse
import math
import os
import shutil
import sys
import glob as _glob

import numpy as np
import open3d as o3d

from point2cad.surface_types import (
    SURFACE_PLANE, SURFACE_SPHERE, SURFACE_CYLINDER, SURFACE_CONE, SURFACE_INR,
    SURFACE_NAMES,
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _denorm(pts, mean, R, scale):
    """Inverse of normalize_points for an (N, 3) float array."""
    pts = np.asarray(pts, dtype=np.float64)
    return (scale * (pts @ R) + mean).astype(np.float32)


def apply_inverse_normalization(shape, mean, R, scale):
    """Undo per-part normalization using decomposed gp_Trsf."""
    from OCC.Core.gp import gp_Trsf, gp_Pnt, gp_Vec
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform

    if shape is None or shape.IsNull():
        return shape

    RT = np.asarray(R, dtype=np.float64).T
    mean = np.asarray(mean, dtype=np.float64)

    trsf_scale = gp_Trsf()
    trsf_scale.SetScale(gp_Pnt(0, 0, 0), float(scale))

    trsf_rot = gp_Trsf()
    trsf_rot.SetValues(
        float(RT[0, 0]), float(RT[0, 1]), float(RT[0, 2]), 0.0,
        float(RT[1, 0]), float(RT[1, 1]), float(RT[1, 2]), 0.0,
        float(RT[2, 0]), float(RT[2, 1]), float(RT[2, 2]), 0.0,
    )

    trsf_trans = gp_Trsf()
    trsf_trans.SetTranslation(gp_Vec(float(mean[0]), float(mean[1]),
                                      float(mean[2])))

    trsf = trsf_trans.Multiplied(trsf_rot.Multiplied(trsf_scale))

    result = BRepBuilderAPI_Transform(shape, trsf, True)
    if not result.IsDone():
        print("[bop] apply_inverse_normalization failed, returning as-is")
        return shape
    return result.Shape()


# ---------------------------------------------------------------------------
# Oversized face creation
# ---------------------------------------------------------------------------

def _make_oversized_face(surface, cluster_pts, margin=0.5):
    """Create a bounded face on `surface` covering the cluster extent + margin.

    Projects cluster points to UV, expands bounds by margin * span,
    clips to the surface's natural domain, and builds the face via
    BRepBuilderAPI_MakeFace(surface, u1, u2, v1, v2, tol).

    Parameters
    ----------
    surface : Geom_Surface
    cluster_pts : (N, 3) array
    margin : float — fractional expansion of UV bounds (0.5 = 50% on each side)

    Returns (face, uv_bounds) or (None, None) on failure.
    """
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace

    # Project a subsample of cluster points to UV
    step = max(1, len(cluster_pts) // 500)
    u_vals, v_vals = [], []
    for pt in cluster_pts[::step]:
        pnt = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
        try:
            proj = GeomAPI_ProjectPointOnSurf(pnt, surface)
            if proj.NbPoints() > 0:
                u, v = proj.LowerDistanceParameters()
                u_vals.append(u)
                v_vals.append(v)
        except Exception:
            pass

    if len(u_vals) < 3:
        return None, None

    umin_r, umax_r = min(u_vals), max(u_vals)
    vmin_r, vmax_r = min(v_vals), max(v_vals)

    mu = margin * max(umax_r - umin_r, 1e-6)
    mv = margin * max(vmax_r - vmin_r, 1e-6)

    u1 = umin_r - mu
    u2 = umax_r + mu
    v1 = vmin_r - mv
    v2 = vmax_r + mv

    # Clip to the surface's natural domain
    su1, su2, sv1, sv2 = surface.Bounds()
    u1 = max(u1, su1)
    u2 = min(u2, su2)
    v1 = max(v1, sv1)
    v2 = min(v2, sv2)

    # Ensure non-degenerate
    if u2 - u1 < 1e-10 or v2 - v1 < 1e-10:
        return None, None

    try:
        face_maker = BRepBuilderAPI_MakeFace(surface, u1, u2, v1, v2, 1e-6)
        if not face_maker.IsDone():
            return None, None
        return face_maker.Face(), (u1, u2, v1, v2)
    except Exception as e:
        print(f"    MakeFace exception: {e}")
        return None, None


# ---------------------------------------------------------------------------
# BOPAlgo_MakerVolume
# ---------------------------------------------------------------------------

def run_maker_volume(faces, tolerance=1e-4):
    """Feed oversized faces into BOPAlgo_MakerVolume.

    Returns the resulting shape, or None on failure.
    """
    from OCC.Core.BOPAlgo import BOPAlgo_MakerVolume
    from OCC.Core.TopTools import TopTools_ListOfShape
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_SHELL
    from OCC.Core.Message import Message_ProgressRange

    maker = BOPAlgo_MakerVolume()
    maker.SetRunParallel(False)
    maker.SetFuzzyValue(tolerance)

    args = TopTools_ListOfShape()
    for face in faces:
        args.Append(face)
    maker.SetArguments(args)

    print(f"[bop] Running MakerVolume with {len(faces)} faces, "
          f"tolerance={tolerance} ...")
    maker.Perform(Message_ProgressRange())

    if maker.HasErrors():
        print(f"[bop] MakerVolume failed with errors")
        # Try to get error info
        report = maker.GetReport()
        if report is not None:
            print(f"[bop] Report: {report.DumpToString()}"
                  if hasattr(report, 'DumpToString') else "[bop] (no dump)")
        return None

    if maker.HasWarnings():
        print(f"[bop] MakerVolume completed with warnings")

    shape = maker.Shape()
    if shape is None or shape.IsNull():
        print(f"[bop] MakerVolume produced null shape")
        return None

    # Count results
    n_solids = 0
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        n_solids += 1
        exp.Next()

    n_faces = 0
    exp2 = TopExp_Explorer(shape, TopAbs_FACE)
    while exp2.More():
        n_faces += 1
        exp2.Next()

    n_shells = 0
    exp3 = TopExp_Explorer(shape, TopAbs_SHELL)
    while exp3.More():
        n_shells += 1
        exp3.Next()

    analyzer = BRepCheck_Analyzer(shape, True)
    print(f"[bop] Result: {n_solids} solid(s), {n_shells} shell(s), "
          f"{n_faces} face(s), valid={analyzer.IsValid()}, "
          f"shape type={shape.ShapeType()}")

    return shape


def select_solid_by_points(shape, all_points):
    """If shape contains multiple solids, select the one enclosing the most points."""
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_SOLID
    from OCC.Core.TopoDS import topods
    from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON

    solids = []
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        solids.append(topods.Solid(exp.Current()))
        exp.Next()

    if len(solids) <= 1:
        return shape

    print(f"[bop] {len(solids)} solids found — selecting by point containment")

    # Subsample points for speed
    step = max(1, len(all_points) // 200)
    sample = all_points[::step]

    best_solid = None
    best_count = -1

    for si, solid in enumerate(solids):
        classifier = BRepClass3d_SolidClassifier(solid)
        count = 0
        for pt in sample:
            classifier.Perform(gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2])),
                               1e-6)
            state = classifier.State()
            if state == TopAbs_IN or state == TopAbs_ON:
                count += 1
        print(f"  Solid {si}: {count}/{len(sample)} points inside")
        if count > best_count:
            best_count = count
            best_solid = solid

    return best_solid


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_compute(args):
    import torch

    from point2cad.surface_fitter    import fit_surface
    from point2cad.occ_surfaces      import to_occ_surface
    from point2cad.cluster_adjacency import build_cluster_proximity
    from point2cad.color_config      import get_surface_color
    from point2cad.topology          import export_step
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

    for part_idx, sample_path in enumerate(part_files):
        step_stem = f"part_{part_idx}"
        out_dir   = os.path.join(model_out_dir, f"part_{part_idx}")

        print(f"\n{'='*60}")
        print(f"Part {part_idx}: {os.path.basename(sample_path)}  →  {out_dir}")
        print(f"{'='*60}")

        data = np.loadtxt(sample_path)
        data[:, :3], part_mean, part_R, part_scale = normalize_points(data[:, :3])
        unique_clusters, cluster_counts = np.unique(
            data[:, -1].astype(int), return_counts=True)
        os.makedirs(out_dir, exist_ok=True)

        # Collect clusters
        clusters = []
        for cid in unique_clusters:
            cluster = data[data[:, -1].astype(int) == cid, :3].astype(np.float32)
            clusters.append(cluster)

        # Per-cluster spacing
        cluster_trees, cluster_nn_percentiles = build_cluster_proximity(
            clusters, percentile=100.0
        )

        # --------------------------------------------------------------
        # Surface fitting (identical to brep_pipeline)
        # --------------------------------------------------------------
        surface_ids, fit_results, fit_meshes, occ_surfs = [], [], [], []
        for idx, (cid, c_count) in enumerate(zip(unique_clusters, cluster_counts)):
            cluster = clusters[idx]
            _spacing = cluster_nn_percentiles[idx]

            print(f"[surface fitter] Cluster {cid} ({c_count} pts) fitting ...")
            _plane_kw    = {"mesh_dim": 100, "plane_sampling_deviation": 0.5,
                            "spacing": _spacing, "threshold_multiplier": 2}
            _sphere_kw   = {"dim_theta": 100, "dim_lambda": 100,
                            "spacing": _spacing, "threshold_multiplier": 2}
            _cylinder_kw = {"dim_theta": 100, "dim_height": 50,
                            "cylinder_height_margin": 0.5,
                            "spacing": _spacing, "threshold_multiplier": 2}
            _cone_kw     = {"dim_theta": 100, "dim_height": 100,
                            "cone_height_margin": 0.5,
                            "spacing": _spacing, "threshold_multiplier": 2}

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
                plane_mesh_kwargs=_plane_kw,
                sphere_mesh_kwargs=_sphere_kw,
                cylinder_mesh_kwargs=_cylinder_kw,
                cone_mesh_kwargs=_cone_kw,
                radius_inflation=0,
                angle_inflation_deg=0,
            )
            sid = res["surface_id"]
            surface_ids.append(sid)
            fit_results.append(res["result"])
            fit_meshes.append(res["mesh"])
            occ_surfs.append(
                to_occ_surface(sid, res["result"], cluster=cluster,
                               uv_margin=0.05, grid_resolution=50)
            )
            chosen_err = res["result"]["error"]
            all_errors = res.get("all_errors", {})
            errors_str = "  ".join(f"{name}={err:.6f}"
                                   for name, err in all_errors.items())
            print(f"[surface fitter] Cluster {cid} ({c_count} pts) → "
                  f"{SURFACE_NAMES[sid]}  residual={chosen_err:.6f}")
            if errors_str:
                print(f"  all errors: {errors_str}")

        # --------------------------------------------------------------
        # Save metadata + cluster files for visualization
        # --------------------------------------------------------------
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
                vertices  = _denorm(np.asarray(mesh.vertices),
                                    part_mean, part_R, part_scale),
                triangles = np.asarray(mesh.triangles),
            )

        # --------------------------------------------------------------
        # Create oversized faces
        # --------------------------------------------------------------
        print(f"\n[bop] Creating oversized faces ...")
        oversized_faces = []
        for idx in range(len(clusters)):
            surface = occ_surfs[idx]
            if surface is None:
                print(f"  Cluster {idx} ({SURFACE_NAMES[surface_ids[idx]]}): "
                      f"no OCC surface — skipping")
                continue

            face, uv_bounds = _make_oversized_face(
                surface, clusters[idx], margin=args.margin)

            if face is not None:
                oversized_faces.append(face)
                print(f"  Cluster {idx} ({SURFACE_NAMES[surface_ids[idx]]}): "
                      f"UV [{uv_bounds[0]:.3f},{uv_bounds[1]:.3f}] x "
                      f"[{uv_bounds[2]:.3f},{uv_bounds[3]:.3f}]")
            else:
                print(f"  Cluster {idx} ({SURFACE_NAMES[surface_ids[idx]]}): "
                      f"face creation FAILED")

        print(f"[bop] Created {len(oversized_faces)} oversized faces "
              f"from {len(clusters)} clusters")

        if len(oversized_faces) < 2:
            print(f"[bop] Need at least 2 faces for MakerVolume — aborting")
            continue

        # --------------------------------------------------------------
        # BOPAlgo_MakerVolume
        # --------------------------------------------------------------
        shape = run_maker_volume(oversized_faces, tolerance=args.tolerance)

        if shape is None:
            print(f"[bop] MakerVolume failed — trying compound fallback")
            # Fallback: just export the oversized faces as a compound
            from OCC.Core.BRep import BRep_Builder
            from OCC.Core.TopoDS import TopoDS_Compound
            builder = BRep_Builder()
            compound = TopoDS_Compound()
            builder.MakeCompound(compound)
            for face in oversized_faces:
                builder.Add(compound, face)
            shape = compound
            print(f"[bop] Fallback: compound with {len(oversized_faces)} faces")

        # Select best solid if multiple
        all_pts = np.vstack(clusters)
        shape = select_solid_by_points(shape, all_pts)

        # --------------------------------------------------------------
        # Export
        # --------------------------------------------------------------
        step_path = os.path.join(out_dir, f"{step_stem}.step")
        try:
            shape_world = apply_inverse_normalization(
                shape, part_mean, part_R, part_scale)
        except Exception as e:
            print(f"[bop] inverse normalization failed: {e} "
                  f"— exporting in normalized space")
            shape_world = shape
        export_step(shape_world, step_path)

        print(f"\n[part] all results saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Visualization (reuse pattern from brep_boundary)
# ---------------------------------------------------------------------------

def run_visualize(args):
    out_dir = os.path.join(args.output_dir, f"{args.model_id}", "part_0")
    if not os.path.exists(out_dir):
        print(f"Output directory not found: {out_dir}")
        return

    meta       = np.load(os.path.join(out_dir, "metadata.npz"), allow_pickle=True)
    n_clusters = int(meta["n_clusters"])
    clust_colors = meta["cluster_colors"]

    cluster_pcds = []
    for i in range(n_clusters):
        path = os.path.join(out_dir, f"cluster_{i}.npy")
        if not os.path.exists(path):
            continue
        pts = np.load(path)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color(clust_colors[i].tolist())
        cluster_pcds.append(pcd)

    surf_meshes = []
    for i in range(n_clusters):
        path = os.path.join(out_dir, f"surface_mesh_{i}.npz")
        if not os.path.exists(path):
            continue
        d = np.load(path)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices  = o3d.utility.Vector3dVector(d["vertices"])
        mesh.triangles = o3d.utility.Vector3iVector(d["triangles"])
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(clust_colors[i].tolist())
        surf_meshes.append(mesh)

    W, H = 800, 600

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window("Clusters", width=W, height=H, left=0, top=50)
    for pcd in cluster_pcds:
        vis1.add_geometry(pcd)
    vis1.get_render_option().point_size = 2.0

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window("Fitted Surfaces", width=W, height=H, left=W, top=50)
    for mesh in surf_meshes:
        vis2.add_geometry(mesh)

    visualizers = [vis1, vis2]
    running = True
    while running:
        for vis in visualizers:
            if not vis.poll_events():
                running = False
                break
            vis.update_renderer()

    for vis in visualizers:
        vis.destroy_window()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="B-Rep reconstruction via BOPAlgo_MakerVolume",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--visualize", action="store_true",
                        help="Load saved results and visualize")
    parser.add_argument("--model_id", type=str, required=True,
                        help="Model ID (e.g. 00000078)")
    parser.add_argument("--input_dir", type=str, default="sample_clouds",
                        help="Root directory for point cloud subdirs")
    parser.add_argument("--output_dir", type=str, default="output_bop",
                        help="Directory for saved results")
    parser.add_argument("-seed", type=int, default=41,
                        help="Reproducibility seed")
    parser.add_argument("--margin", type=float, default=0.5,
                        help="UV margin for oversized faces "
                             "(0.5 = 50%% expansion on each side)")
    parser.add_argument("--tolerance", type=float, default=1e-4,
                        help="Fuzzy tolerance for BOPAlgo_MakerVolume")

    args = parser.parse_args()

    if args.visualize:
        run_visualize(args)
    else:
        run_compute(args)
