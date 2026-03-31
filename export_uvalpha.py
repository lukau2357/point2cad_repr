"""
Export UV alpha-shape trimmed primitive surface faces as a STEP compound (Format 3).

For each cluster in a segmented .xyzc file, fits the best primitive surface,
projects the cluster points to UV space, computes an alpha-shape boundary, and
builds a wire-bounded face. The result is a compound of trimmed faces whose
boundaries follow the cluster footprint.

Usage (inside Docker):
    python export_uvalpha.py sample_clouds/abc_00470/0.xyzc
    python export_uvalpha.py sample_clouds/abc_00470/0.xyzc \\
        --output_dir ./output_uvalpha --alpha 0.05 --simplify_epsilon 0.01 \\
        --inflate 0.05 --tol 1e-3
"""

import argparse
import os

import numpy as np

from point2cad.primitive_fitting import (
    fit_plane_numpy, fit_sphere_numpy, fit_cylinder_optimized, fit_cone,
)
from point2cad.primitive_fitting_utils import rotation_matrix_a_to_b
from point2cad.occ_surfaces import plane_to_occ, sphere_to_occ, cylinder_to_occ, cone_to_occ
from point2cad.topology import export_step, apply_inverse_normalization, build_brep_shape_bop
from point2cad.surface_types import (
    SURFACE_PLANE, SURFACE_SPHERE, SURFACE_CYLINDER, SURFACE_CONE, SURFACE_NAMES,
)


_FITTERS = {
    SURFACE_PLANE:    fit_plane_numpy,
    SURFACE_SPHERE:   fit_sphere_numpy,
    SURFACE_CYLINDER: fit_cylinder_optimized,
    SURFACE_CONE:     fit_cone,
}

_OCC_BUILDERS = {
    SURFACE_PLANE:    lambda p, c: plane_to_occ(p),
    SURFACE_SPHERE:   lambda p, c: sphere_to_occ(p),
    SURFACE_CYLINDER: lambda p, c: cylinder_to_occ(p),
    SURFACE_CONE:     lambda p, c: cone_to_occ(p, cluster=c),
}


def _normalize(pts):
    """PCA normalization to unit cube. Returns (pts_norm, mean, R, scale)."""
    mean = pts.mean(axis=0)
    centered = pts - mean
    S, U = np.linalg.eigh(centered.T @ centered)
    R = rotation_matrix_a_to_b(U[:, np.argmin(S)], np.array([1.0, 0.0, 0.0]))
    rotated = (R @ centered.T).T
    scale = float((rotated.max(axis=0) - rotated.min(axis=0)).max()) + 1e-7
    return (rotated / scale).astype(np.float32), mean, R, scale


def _fit_best(cluster):
    """Fit all primitives and return (surface_id, params, error)."""
    results = {sid: _FITTERS[sid](cluster) for sid in _FITTERS}
    best = min(results, key=lambda sid: results[sid]["error"])
    return best, results[best]["params"], results[best]["error"]


def process(xyzc_path, output_dir, args):
    data = np.loadtxt(xyzc_path)
    pts_norm, mean, R, scale = _normalize(data[:, :3])
    cluster_ids = data[:, 3].astype(int)

    unique_cids = np.unique(cluster_ids)
    occ_surfaces = []
    clusters = []
    surface_ids = []

    for cid in unique_cids:
        cluster = pts_norm[cluster_ids == cid]
        sid, params, error = _fit_best(cluster)
        print(f"  cluster {cid:2d}: {SURFACE_NAMES[sid]:<10s}  error={error:.6f}")

        occ_surf = _OCC_BUILDERS[sid](params, cluster)
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
        return

    shape = apply_inverse_normalization(shape, mean, R, scale)
    os.makedirs(output_dir, exist_ok=True)
    model_id = os.path.basename(os.path.dirname(xyzc_path))
    part_id = os.path.splitext(os.path.basename(xyzc_path))[0]
    export_step(shape, os.path.join(output_dir, f"{model_id}_part_{part_id}.step"))


def main():
    parser = argparse.ArgumentParser(
        description="Export UV alpha-shape trimmed primitive faces as a STEP compound"
    )
    parser.add_argument("xyzc", help="Path to the segmented .xyzc file")
    parser.add_argument(
        "--output_dir", default="./output_uvalpha",
        help="Directory for output STEP files (default: ./output_uvalpha)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.0,
        help="Alpha-shape parameter (0 = convex hull, larger = tighter boundary)",
    )
    parser.add_argument(
        "--simplify_epsilon", type=float, default=0.0,
        help="Douglas-Peucker simplification tolerance in UV units (0 = no simplification)",
    )
    parser.add_argument(
        "--inflate", type=float, default=0.0,
        help="Fractional UV boundary inflation beyond the alpha-shape (0 = none)",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-3,
        help="OCC face construction tolerance (default: 1e-3)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.xyzc):
        parser.error(f"File not found: {args.xyzc}")

    print(f"Input : {args.xyzc}")
    print(f"Output: {args.output_dir}")
    process(args.xyzc, args.output_dir, args)


if __name__ == "__main__":
    main()
