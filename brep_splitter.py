"""
B-Rep reconstruction via BRepAlgoAPI_Splitter (Global Solid Partitioning).

Creates a bounding-box solid, splits it with oversized fitted-surface faces,
then classifies the resulting cells as inside/outside using point containment.

Usage (inside Docker):
  python brep_splitter.py --model_id 00000078 --input_dir sample_clouds --output_dir output_splitter
"""

import argparse
import math
import os
import shutil
import sys
import glob as _glob
import time

import numpy as np
import open3d as o3d

from point2cad.surface_types import (
    SURFACE_PLANE, SURFACE_SPHERE, SURFACE_CYLINDER, SURFACE_CONE, SURFACE_INR,
    SURFACE_NAMES,
)


# ---------------------------------------------------------------------------
# Reuse helpers from brep_bop
# ---------------------------------------------------------------------------

from brep_bop import (
    _denorm,
    apply_inverse_normalization,
)


def _make_large_face(surface, bbox_min, bbox_max):
    """Create a face on `surface` whose UV domain spans the entire bounding box.

    Projects the 8 corners of the bounding box onto the surface to find the
    UV extent, then expands generously.  For infinite surfaces (planes,
    cylinders) this ensures the face fully slices the bounding box.
    """
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace

    corners = []
    for x in (bbox_min[0], bbox_max[0]):
        for y in (bbox_min[1], bbox_max[1]):
            for z in (bbox_min[2], bbox_max[2]):
                corners.append((x, y, z))

    u_vals, v_vals = [], []
    for pt in corners:
        try:
            proj = GeomAPI_ProjectPointOnSurf(
                gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2])), surface)
            if proj.NbPoints() > 0:
                u, v = proj.LowerDistanceParameters()
                u_vals.append(u)
                v_vals.append(v)
        except Exception:
            pass

    if len(u_vals) < 2:
        return None

    umin, umax = min(u_vals), max(u_vals)
    vmin, vmax = min(v_vals), max(v_vals)

    # Expand by 50% of span on each side to ensure full coverage
    du = 0.5 * max(umax - umin, 1e-6)
    dv = 0.5 * max(vmax - vmin, 1e-6)
    u1, u2 = umin - du, umax + du
    v1, v2 = vmin - dv, vmax + dv

    # Clip to surface's natural domain (e.g. cylinder theta ∈ [0, 2π])
    su1, su2, sv1, sv2 = surface.Bounds()
    u1 = max(u1, su1)
    u2 = min(u2, su2)
    v1 = max(v1, sv1)
    v2 = min(v2, sv2)

    if u2 - u1 < 1e-10 or v2 - v1 < 1e-10:
        return None

    try:
        maker = BRepBuilderAPI_MakeFace(surface, u1, u2, v1, v2, 1e-6)
        if not maker.IsDone():
            return None
        return maker.Face()
    except Exception as e:
        print(f"    MakeFace exception: {e}")
        return None


# ---------------------------------------------------------------------------
# Splitter + cell classification
# ---------------------------------------------------------------------------

def _count_solids(shape):
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_SOLID
    n = 0
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        n += 1
        exp.Next()
    return n


def run_splitter(bounding_solid, tool_faces, face_labels, fuzzy_value=1e-2):
    """Split bounding_solid with tool_faces one at a time (incremental).

    Splits incrementally so we can see which faces actually cut.
    face_labels: list of strings for logging (e.g. "3 (plane)")

    Returns the result shape (compound of cells) or None on failure.
    """
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Splitter
    from OCC.Core.TopTools import TopTools_ListOfShape
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_FACE
    from OCC.Core.BRepCheck import BRepCheck_Analyzer

    current = bounding_solid
    n_before = _count_solids(current)
    print(f"[splitter] Starting with {n_before} solid(s), "
          f"fuzzy={fuzzy_value}")

    for i, (face, label) in enumerate(zip(tool_faces, face_labels)):
        splitter = BRepAlgoAPI_Splitter()

        args = TopTools_ListOfShape()
        args.Append(current)
        splitter.SetArguments(args)

        tools = TopTools_ListOfShape()
        tools.Append(face)
        splitter.SetTools(tools)

        splitter.SetFuzzyValue(fuzzy_value)
        splitter.SetRunParallel(False)

        t0 = time.time()
        splitter.Build()
        elapsed = time.time() - t0

        has_err = splitter.HasErrors()
        has_warn = splitter.HasWarnings()

        if has_err:
            print(f"  [{i+1}/{len(tool_faces)}] cluster {label}: "
                  f"ERROR — skipping")
            continue

        result = splitter.Shape()
        if result is None or result.IsNull():
            print(f"  [{i+1}/{len(tool_faces)}] cluster {label}: "
                  f"null result — skipping")
            continue

        n_after = _count_solids(result)
        status = "CUT" if n_after > n_before else "no effect"
        warn_str = " (warnings)" if has_warn else ""
        print(f"  [{i+1}/{len(tool_faces)}] cluster {label}: "
              f"{n_before} -> {n_after} cells  [{status}]  "
              f"{elapsed:.2f}s{warn_str}")

        current = result
        n_before = n_after

    # Final stats
    n_solids = _count_solids(current)
    n_faces = 0
    exp = TopExp_Explorer(current, TopAbs_FACE)
    while exp.More():
        n_faces += 1
        exp.Next()

    analyzer = BRepCheck_Analyzer(current, True)
    print(f"[splitter] Final: {n_solids} cell(s), {n_faces} face(s), "
          f"valid={analyzer.IsValid()}")

    return current


def classify_cells(shape, point_cloud, sample_per_cell=5):
    """Classify each solid cell as inside or outside based on point containment.

    For each cell, checks how many point cloud points lie inside it.
    Returns list of (solid, n_inside) tuples sorted by n_inside descending.
    """
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_IN, TopAbs_ON
    from OCC.Core.TopoDS import topods
    from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.gp import gp_Pnt

    solids = []
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        solids.append(topods.Solid(exp.Current()))
        exp.Next()

    print(f"[classify] {len(solids)} cells to classify against "
          f"{len(point_cloud)} points")

    # Subsample points for speed
    step = max(1, len(point_cloud) // 500)
    sample = point_cloud[::step]
    # sample = point_cloud
    print(f"[classify] Using {len(sample)} sample points")

    results = []
    for si, solid in enumerate(solids):
        # Volume for info
        props = GProp_GProps()
        brepgprop.VolumeProperties(solid, props)
        vol = props.Mass()

        classifier = BRepClass3d_SolidClassifier(solid)
        count = 0
        for pt in sample:
            classifier.Perform(
                gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2])), 1e-6)
            state = classifier.State()
            if state == TopAbs_IN or state == TopAbs_ON:
                count += 1

        results.append((solid, count, vol))
        if count > 0:
            print(f"  Cell {si}: {count}/{len(sample)} points inside, "
                  f"volume={vol:.6f}")

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def fuse_cells(cells):
    """Fuse a list of solids into one shape."""
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.TopoDS import TopoDS_Compound
    from OCC.Core.BRepCheck import BRepCheck_Analyzer

    if len(cells) == 0:
        return None
    if len(cells) == 1:
        return cells[0]

    print(f"[fuse] Fusing {len(cells)} cells ...")
    result = cells[0]
    for i in range(1, len(cells)):
        fuser = BRepAlgoAPI_Fuse(result, cells[i])
        if fuser.HasErrors():
            print(f"  Fuse step {i} failed — stopping at {i} cells")
            break
        result = fuser.Shape()

    analyzer = BRepCheck_Analyzer(result, True)
    print(f"[fuse] Result valid={analyzer.IsValid()}")
    return result


def make_bounding_box_solid(points, margin_factor=0.3):
    """Create a TopoDS_Solid box enclosing all points with margin."""
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.gp import gp_Pnt

    pts = np.asarray(points)
    pmin = pts.min(axis=0)
    pmax = pts.max(axis=0)
    extent = pmax - pmin
    margin = margin_factor * extent

    corner = pmin - margin
    size = extent + 2 * margin

    box = BRepPrimAPI_MakeBox(
        gp_Pnt(float(corner[0]), float(corner[1]), float(corner[2])),
        float(size[0]), float(size[1]), float(size[2])
    )
    return box.Solid()


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
        R       = pfu.rotation_matrix_a_to_b(
            U[:, np.argmin(S)], np.array([1, 0, 0]))
        pts     = (R @ pts.T).T
        extents = np.max(pts, axis=0) - np.min(pts, axis=0)
        scale   = float(np.max(extents) + 1e-7)
        return (pts / scale).astype(np.float32), mean, R, scale

    np_rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    input_pattern = os.path.join(args.input_dir, f"{args.model_id}", "*.xyzc")
    part_files    = sorted(_glob.glob(input_pattern),
                           key=lambda p: int(
                               os.path.splitext(os.path.basename(p))[0]))
    if not part_files:
        print(f"No part files found matching: {input_pattern}")
        return
    model_out_dir = os.path.join(args.output_dir, f"{args.model_id}")
    if os.path.exists(model_out_dir):
        shutil.rmtree(model_out_dir)
    os.makedirs(model_out_dir)
    print(f"Model {args.model_id}: {len(part_files)} part(s)")

    for part_idx, sample_path in enumerate(part_files):
        step_stem = f"part_{part_idx}"
        out_dir   = os.path.join(model_out_dir, f"part_{part_idx}")

        print(f"\n{'='*60}")
        print(f"Part {part_idx}: {os.path.basename(sample_path)}  ->  {out_dir}")
        print(f"{'='*60}")

        data = np.loadtxt(sample_path)
        data[:, :3], part_mean, part_R, part_scale = normalize_points(
            data[:, :3])
        unique_clusters, cluster_counts = np.unique(
            data[:, -1].astype(int), return_counts=True)
        os.makedirs(out_dir, exist_ok=True)

        clusters = []
        for cid in unique_clusters:
            cluster = data[data[:, -1].astype(int) == cid, :3].astype(
                np.float32)
            clusters.append(cluster)

        cluster_trees, cluster_nn_percentiles = build_cluster_proximity(
            clusters, percentile=100.0)

        # Surface fitting (identical to brep_bop / brep_pipeline)
        surface_ids, fit_results, fit_meshes, occ_surfs = [], [], [], []
        for idx, (cid, c_count) in enumerate(
                zip(unique_clusters, cluster_counts)):
            cluster = clusters[idx]
            _spacing = cluster_nn_percentiles[idx]

            print(f"[surface fitter] Cluster {cid} ({c_count} pts) ...")
            _plane_kw    = {"mesh_dim": 100,
                            "plane_sampling_deviation": 0.5,
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
                {"hidden_dim": 64, "use_shortcut": True,
                 "fraction_siren": 0.5},
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
                               uv_margin=0.05, grid_resolution=50))
            print(f"  -> {SURFACE_NAMES[sid]}  "
                  f"residual={res['result']['error']:.6f}")

        # Save metadata for visualization
        np.savez(
            os.path.join(out_dir, "metadata.npz"),
            n_clusters     = len(clusters),
            surface_ids    = np.array(surface_ids),
            surface_names  = np.array(
                [SURFACE_NAMES[s] for s in surface_ids]),
            cluster_colors = np.array(
                [get_surface_color(SURFACE_NAMES[s]).tolist()
                 for s in surface_ids]),
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

        # ------------------------------------------------------------------
        # Create large faces spanning the bounding box (tools)
        # ------------------------------------------------------------------
        all_pts = np.vstack(clusters)
        bbox_min = all_pts.min(axis=0) - 0.3 * (all_pts.max(axis=0) - all_pts.min(axis=0))
        bbox_max = all_pts.max(axis=0) + 0.3 * (all_pts.max(axis=0) - all_pts.min(axis=0))

        print(f"\n[splitter] Creating large faces spanning bounding box ...")
        tool_faces = []
        face_labels = []
        for idx in range(len(clusters)):
            surface = occ_surfs[idx]
            if surface is None:
                print(f"  Cluster {idx} ({SURFACE_NAMES[surface_ids[idx]]}): "
                      f"no OCC surface — skipping")
                continue

            face = _make_large_face(surface, bbox_min, bbox_max)

            if face is not None:
                tool_faces.append(face)
                face_labels.append(
                    f"{idx} ({SURFACE_NAMES[surface_ids[idx]]})")
                print(f"  Cluster {idx} ({SURFACE_NAMES[surface_ids[idx]]}): OK")
            else:
                print(f"  Cluster {idx} ({SURFACE_NAMES[surface_ids[idx]]}): "
                      f"FAILED")

        print(f"[splitter] {len(tool_faces)} tool faces from "
              f"{len(clusters)} clusters")

        if len(tool_faces) < 2:
            print(f"[splitter] Need at least 2 faces — aborting")
            continue

        # ------------------------------------------------------------------
        # Bounding box solid
        # ------------------------------------------------------------------
        bbox_solid = make_bounding_box_solid(all_pts, margin_factor=0.3)
        print(f"[splitter] Bounding box created")

        # ------------------------------------------------------------------
        # Split
        # ------------------------------------------------------------------
        result = run_splitter(bbox_solid, tool_faces, face_labels,
                              fuzzy_value=args.fuzzy)

        if result is None:
            print(f"[splitter] Splitter failed — aborting")
            continue

        # ------------------------------------------------------------------
        # Classify cells
        # ------------------------------------------------------------------
        cell_results = classify_cells(result, all_pts)

        # Select cells that contain points
        inside_cells = [solid for solid, count, vol in cell_results
                        if count > 0]
        print(f"[splitter] {len(inside_cells)} cells contain points "
              f"(out of {len(cell_results)} total)")

        if not inside_cells:
            print(f"[splitter] No cells contain points — aborting")
            continue

        # ------------------------------------------------------------------
        # Fuse inside cells
        # ------------------------------------------------------------------
        fused = fuse_cells(inside_cells)

        if fused is None:
            print(f"[splitter] Fusion failed — exporting best single cell")
            fused = cell_results[0][0]  # largest cell by point count

        # ------------------------------------------------------------------
        # Export
        # ------------------------------------------------------------------
        step_path = os.path.join(out_dir, f"{step_stem}.step")
        try:
            shape_world = apply_inverse_normalization(
                fused, part_mean, part_R, part_scale)
        except Exception as e:
            print(f"[splitter] inverse normalization failed: {e}")
            shape_world = fused
        export_step(shape_world, step_path)

        print(f"\n[splitter] Results saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="B-Rep via BRepAlgoAPI_Splitter (solid partitioning)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--input_dir", type=str, default="sample_clouds")
    parser.add_argument("--output_dir", type=str, default="output_splitter")
    parser.add_argument("-seed", type=int, default=41)
    parser.add_argument("--fuzzy", type=float, default=1e-2,
                        help="Fuzzy tolerance for the splitter")

    args = parser.parse_args()
    run_compute(args)
