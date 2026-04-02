"""
Export UV alpha-shape trimmed surface faces as a STEP compound (Format 3).

For each cluster in a segmented .xyzc file, fits the best surface (primitive
or INR/B-spline), projects the cluster points to UV space, computes an
alpha-shape boundary, and builds a wire-bounded face. The result is a compound
of trimmed faces whose boundaries follow the cluster footprint.
"""

import argparse
import glob as _glob
import os

import numpy as np
import torch

from point2cad.surface_fitter import fit_surface
from point2cad.occ_surfaces import to_occ_surface
from point2cad.primitive_fitting_utils import rotation_matrix_a_to_b
from point2cad.topology import export_step, apply_inverse_normalization, build_brep_shape_bop
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


def process(xyzc_path, output_dir, args, np_rng, device):
    data = np.loadtxt(xyzc_path)
    pts_norm, mean, R, scale = _normalize(data[:, :3])
    cluster_ids = data[:, 3].astype(int)

    unique_cids = np.unique(cluster_ids)
    occ_surfaces = []
    clusters = []
    surface_ids = []

    for cid in unique_cids:
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
            occ_surfaces.append(None)
        else:
            occ_surfaces.append(occ_surf)

        clusters.append(cluster.astype(np.float64))
        surface_ids.append(sid)

    shape = build_brep_shape_bop(
        occ_surfaces,
        clusters,
        surface_ids=surface_ids,
        tolerance=args.tol,
        alpha=args.alpha,
        simplify_epsilon=args.simplify_epsilon,
        inflate=args.inflate,
    )

    if shape is None or shape.IsNull():
        print("No shape produced — aborting.")
        return None

    shape = apply_inverse_normalization(shape, mean, R, scale)
    os.makedirs(output_dir, exist_ok=True)
    part_id = os.path.splitext(os.path.basename(xyzc_path))[0]
    step_path = os.path.join(output_dir, f"part_{part_id}.step")
    export_step(shape, step_path)
    print(f"  Exported {step_path}")

    return shape


def main():
    parser = argparse.ArgumentParser(
        description="Export UV alpha-shape trimmed primitive faces as a STEP compound",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_id", type=str, required=True,
                        help="Model ID (subdirectory under input_dir)")
    parser.add_argument("--input_dir", type=str, default="sample_clouds",
                        help="Root directory for input point cloud subdirs")
    parser.add_argument("--output_dir", default="./output_uvalpha",
                        help="Directory for output STEP files")
    parser.add_argument("--part", type=int, default=None,
                        help="Process only this part index (0-based). Default: all parts")
    parser.add_argument(
        "--alpha", type=float, default=10,
        help="Alpha-shape parameter (0 = convex hull, larger = tighter boundary)",
    )
    parser.add_argument(
        "--simplify_epsilon", type=float, default=0.0001,
        help="Douglas-Peucker simplification tolerance in UV units (0 = no simplification)",
    )
    parser.add_argument(
        "--inflate", type=float, default=0.0,
        help="Fractional UV boundary inflation beyond the alpha-shape (0 = none)",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-3,
        help="OCC face construction tolerance",
    )
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
        unified_path = os.path.join(model_out_dir, "uv_alpha.step")
        export_step(compound, unified_path)
        print(f"  Unified: {unified_path}")


if __name__ == "__main__":
    main()
