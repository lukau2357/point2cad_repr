"""
Export oversized primitive surface faces as a STEP compound (Format 2).

For each cluster in a segmented .xyzc file, fits the best primitive surface
and creates a large face whose UV domain covers the cluster's projection
plus a configurable margin. The result is a compound of untrimmed faces
intended for manual trimming by a CAD engineer.

Usage (inside Docker):
    python export_oversized.py sample_clouds/abc_00470/0.xyzc
    python export_oversized.py sample_clouds/abc_00470/0.xyzc \\
        --output_dir ./output_oversized --uv_margin 0.5 --tol 1e-3
"""

import argparse
import math
import os

import numpy as np

from point2cad.primitive_fitting import (
    fit_plane_numpy, fit_sphere_numpy, fit_cylinder_optimized, fit_cone,
)
from point2cad.primitive_fitting_utils import rotation_matrix_a_to_b
from point2cad.occ_surfaces import plane_to_occ, sphere_to_occ, cylinder_to_occ, cone_to_occ
from point2cad.topology import export_step, apply_inverse_normalization
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


def process(xyzc_path, output_dir, args):
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
        sid, params, error = _fit_best(cluster)
        print(f"  cluster {cid:2d}: {SURFACE_NAMES[sid]:<10s}  error={error:.6f}")

        occ_surf = _OCC_BUILDERS[sid](params, cluster)
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

    # Combine faces + intersection edges into one compound
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    for face in faces:
        builder.Add(compound, face)
    builder.Add(compound, section_shape)

    shape = apply_inverse_normalization(compound, mean, R, scale)
    os.makedirs(output_dir, exist_ok=True)
    model_id = os.path.basename(os.path.dirname(xyzc_path))
    part_id = os.path.splitext(os.path.basename(xyzc_path))[0]
    export_step(shape, os.path.join(output_dir, f"{model_id}_part_{part_id}.step"))


def main():
    parser = argparse.ArgumentParser(
        description="Export oversized primitive faces as a STEP compound"
    )
    parser.add_argument("xyzc", help="Path to the segmented .xyzc file")
    parser.add_argument(
        "--output_dir", default="./output_oversized",
        help="Directory for output STEP files (default: ./output_oversized)",
    )
    parser.add_argument(
        "--uv_margin", type=float, default=0.2,
        help="Fractional UV expansion margin beyond cluster projection (default: 0.5)",
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
