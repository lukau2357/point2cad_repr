"""
Export oversized surface faces as a STEP compound (Format 2).

For each cluster in a segmented .xyzc file, fits the best surface (primitive
or INR/B-spline) and creates a large face whose UV domain covers the cluster's
projection plus a configurable margin. The result is a compound of untrimmed
faces intended for manual trimming by a CAD engineer.
"""

import argparse
import glob as _glob
import math
import os

import numpy as np
import torch

from point2cad.surface_fitter import fit_surface
from point2cad.occ_surfaces import to_occ_surface
from point2cad.primitive_fitting_utils import rotation_matrix_a_to_b
from point2cad.topology import export_step, apply_inverse_normalization
from point2cad.surface_types import SURFACE_NAMES


def _normalize(pts):
    """PCA normalization to unit cube. Returns (pts_norm, mean, R, scale)."""
    mean = pts.mean(axis=0)
    centered = pts - mean
    S, U = np.linalg.eigh(centered.T @ centered)
    R = rotation_matrix_a_to_b(U[:, np.argmin(S)], np.array([1.0, 0.0, 0.0]))
    rotated = (R @ centered.T).T
    scale = float((rotated.max(axis=0) - rotated.min(axis=0)).max()) + 1e-7
    return (rotated / scale).astype(np.float32), mean, R, scale


def _make_large_face(occ_surface, cluster, uv_margin, tol):
    """Create a face whose UV domain spans the cluster projection + margin."""
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace

    u_vals, v_vals = [], []
    for pt in cluster:
        try:
            proj = GeomAPI_ProjectPointOnSurf(
                gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2])), occ_surface
            )
            if proj.NbPoints() > 0:
                u, v = proj.LowerDistanceParameters()
                u_vals.append(u)
                v_vals.append(v)
        except Exception:
            pass

    if not u_vals:
        return None

    umin, umax = min(u_vals), max(u_vals)
    vmin, vmax = min(v_vals), max(v_vals)
    du = uv_margin * max(umax - umin, 1e-6)
    dv = uv_margin * max(vmax - vmin, 1e-6)

    su1, su2, sv1, sv2 = occ_surface.Bounds()
    u1 = umin - du if math.isinf(su1) else max(umin - du, su1)
    u2 = umax + du if math.isinf(su2) else min(umax + du, su2)
    v1 = vmin - dv if math.isinf(sv1) else max(vmin - dv, sv1)
    v2 = vmax + dv if math.isinf(sv2) else min(vmax + dv, sv2)

    face = BRepBuilderAPI_MakeFace(occ_surface, u1, u2, v1, v2, tol)
    return face.Face() if face.IsDone() else None


def process(xyzc_path, output_dir, args, np_rng, device):
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.TopoDS import TopoDS_Compound
    from OCC.Core.BOPAlgo import BOPAlgo_Section
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_EDGE

    data = np.loadtxt(xyzc_path)
    pts_norm, mean, R, scale = _normalize(data[:, :3])
    cluster_ids = data[:, 3].astype(int)

    faces = []
    for cid in np.unique(cluster_ids):
        cluster = pts_norm[cluster_ids == cid]

        res = fit_surface(
            cluster,
            {"hidden_dim": 64, "use_shortcut": True, "fraction_siren": 0.5},
            np_rng, device,
            inr_fit_kwargs={
                "max_steps": 1500,
                "noise_magnitude_3d": 0.05,
                "noise_magnitude_uv": 0.05,
                "initial_lr": 1e-1,
            },
        )
        sid = res["surface_id"]
        error = res["result"]["error"]
        print(f"  cluster {cid:2d}: {SURFACE_NAMES[sid]:<10s}  error={error:.6f}")

        occ_surf = to_occ_surface(sid, res["result"], cluster=cluster,
                                  uv_margin=0.05, grid_resolution=50)
        if occ_surf is None:
            print(f"             → OCC surface creation failed, skipping")
            continue

        face = _make_large_face(occ_surf, cluster, args.uv_margin, args.tol)
        if face is None:
            print(f"             → face creation failed, skipping")
            continue

        faces.append(face)

    print(f"  {len(faces)}/{len(np.unique(cluster_ids))} faces assembled")

    # Compute pairwise surface-surface intersection curves
    section_shape = None
    if not args.no_section:
        print(f"  Computing intersection curves ...")
        section = BOPAlgo_Section()
        for face in faces:
            section.AddArgument(face)
        section.Perform()
        section_shape = section.Shape()

        n_edges = 0
        exp = TopExp_Explorer(section_shape, TopAbs_EDGE)
        while exp.More():
            n_edges += 1
            exp.Next()
        print(f"  {n_edges} intersection edges computed")
    else:
        print(f"  Skipping intersection curves (--no_section)")

    # Combine faces + intersection edges into one compound
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    for face in faces:
        builder.Add(compound, face)
    if section_shape is not None:
        builder.Add(compound, section_shape)

    shape = apply_inverse_normalization(compound, mean, R, scale)
    os.makedirs(output_dir, exist_ok=True)
    part_id = os.path.splitext(os.path.basename(xyzc_path))[0]
    step_path = os.path.join(output_dir, f"part_{part_id}.step")
    export_step(shape, step_path)
    print(f"  Exported {step_path}")

    return shape


def main():
    parser = argparse.ArgumentParser(
        description="Export oversized primitive faces as a STEP compound",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_id", type=str, required=True,
                        help="Model ID (subdirectory under input_dir)")
    parser.add_argument("--input_dir", type=str, default="sample_clouds",
                        help="Root directory for input point cloud subdirs")
    parser.add_argument("--output_dir", default="./output_oversized",
                        help="Directory for output STEP files")
    parser.add_argument("--part", type=int, default=None,
                        help="Process only this part index (0-based). Default: all parts")
    parser.add_argument(
        "--uv_margin", type=float, default=0.2,
        help="Fractional UV expansion margin beyond cluster projection",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-3,
        help="OCC face construction tolerance",
    )
    parser.add_argument("--no_section", action="store_true",
                        help="Skip intersection curve computation (BOPAlgo_Section)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    np_rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    input_pattern = os.path.join(args.input_dir, args.model_id, "*.xyzc")
    xyzc_files = sorted(_glob.glob(input_pattern),
                        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    if not xyzc_files:
        parser.error(f"No .xyzc files found matching: {input_pattern}")

    if args.part is not None:
        if args.part >= len(xyzc_files):
            parser.error(f"Part {args.part} out of range — model has {len(xyzc_files)} part(s)")
        part_indices = [args.part]
    else:
        part_indices = list(range(len(xyzc_files)))

    model_out_dir = os.path.join(args.output_dir, args.model_id)
    print(f"Model {args.model_id}: {len(xyzc_files)} part(s), processing {len(part_indices)}")

    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.TopoDS import TopoDS_Compound

    denorm_shapes = []
    for part_idx in part_indices:
        print(f"\n--- Part {part_idx}: {os.path.basename(xyzc_files[part_idx])} ---")
        shape = process(xyzc_files[part_idx], model_out_dir, args, np_rng, device)
        if shape is not None and not shape.IsNull():
            denorm_shapes.append(shape)

    # Build unified STEP from denormalized shapes (always world-space)
    if denorm_shapes:
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for s in denorm_shapes:
            builder.Add(compound, s)
        unified_path = os.path.join(model_out_dir, "oversized.step")
        export_step(compound, unified_path)
        print(f"  Unified: {unified_path}")


if __name__ == "__main__":
    main()
