"""
Filter ABC dataset models by surface type composition.

Iterates all STEP batch directories under --abc_dir, reads each STEP file,
decomposes into parts, and checks per-part predicates:
  - All faces have surface types in the allowed set (after _MAPPED_SURF_TYPE mapping)
  - Number of BSpline faces <= --max_bspline
  - Total number of faces <= --max_faces

Outputs a JSON manifest of qualifying (model_id, part_idx) pairs.
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopExp      import TopExp_Explorer
    from OCC.Core.TopAbs      import (TopAbs_FACE, TopAbs_COMPOUND,
                                       TopAbs_COMPSOLID)
    from OCC.Core.TopoDS      import topods, TopoDS_Iterator
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs     import (
        GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere,
        GeomAbs_Torus, GeomAbs_BezierSurface, GeomAbs_BSplineSurface,
        GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion,
        GeomAbs_OffsetSurface, GeomAbs_OtherSurface,
    )
    _OCC_AVAILABLE = True
except ImportError as err:
    print(f"OCC import error: {err}")
    _OCC_AVAILABLE = False

# ---------------------------------------------------------------------------
# Surface type classification (mirrors abc_preprocess.py)
# ---------------------------------------------------------------------------

_GEOMABS_SURF_NAMES = {
    GeomAbs_Plane:               "Plane",
    GeomAbs_Cylinder:            "Cylinder",
    GeomAbs_Cone:                "Cone",
    GeomAbs_Sphere:              "Sphere",
    GeomAbs_Torus:               "Torus",
    GeomAbs_BezierSurface:       "Bezier",
    GeomAbs_BSplineSurface:      "BSpline",
    GeomAbs_SurfaceOfRevolution: "Revolution",
    GeomAbs_SurfaceOfExtrusion:  "Extrusion",
    GeomAbs_OffsetSurface:       "Offset",
    GeomAbs_OtherSurface:        "Other",
} if _OCC_AVAILABLE else {}

# OCC type -> pipeline type (ParSeNet convention)
_MAPPED_SURF_TYPE = {
    "Plane":      "Plane",
    "Cylinder":   "Cylinder",
    "Cone":       "Cone",
    "Sphere":     "Sphere",
    "Torus":      "Torus",
    "Bezier":     "BSpline",
    "BSpline":    "BSpline",
    "Revolution": "BSpline",
    "Extrusion":  "BSpline",
    "Offset":     "BSpline",
    "Other":      "BSpline",
}

# ---------------------------------------------------------------------------
# Helpers (same as abc_preprocess.py)
# ---------------------------------------------------------------------------

def _find_step_batches(abc_dir):
    return sorted(
        os.path.join(abc_dir, e)
        for e in os.listdir(abc_dir)
        if "step" in e and os.path.isdir(os.path.join(abc_dir, e))
    )


def _load_step(step_path):
    reader = STEPControl_Reader()
    if reader.ReadFile(step_path) != 1:
        return None
    reader.TransferRoots()
    return reader.OneShape()


def _extract_parts(shape):
    if shape.ShapeType() not in (TopAbs_COMPOUND, TopAbs_COMPSOLID):
        return [shape]
    parts = []
    it = TopoDS_Iterator(shape)
    while it.More():
        parts.append(it.Value())
        it.Next()
    return parts if parts else [shape]


def _classify_faces(shape):
    """Return list of mapped surface type strings for all faces in shape."""
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    types = []
    while exp.More():
        face = topods.Face(exp.Current())
        adaptor = BRepAdaptor_Surface(face)
        raw = _GEOMABS_SURF_NAMES.get(adaptor.GetType(), "Other")
        types.append(_MAPPED_SURF_TYPE.get(raw, "BSpline"))
        exp.Next()
    return types


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _check_part(part, allowed_types, max_bspline, max_faces):
    """Check if a single part satisfies the filter predicates.
    Returns (True, info_dict) or (False, None)."""
    face_types = _classify_faces(part)
    n_faces = len(face_types)

    if n_faces == 0 or n_faces > max_faces:
        return False, None

    if any(t not in allowed_types for t in face_types):
        return False, None

    n_bspline = sum(1 for t in face_types if t == "BSpline")
    if n_bspline > max_bspline:
        return False, None

    return True, {
        "n_faces": n_faces,
        "n_bspline": n_bspline,
        "face_types": dict(Counter(face_types)),
    }


def filter_models(abc_dir, allowed_types, max_bspline, max_faces,
                  max_file_size_mb, all_parts):
    if not _OCC_AVAILABLE:
        print("ERROR: pythonocc-core is required. Run inside the Docker container.")
        sys.exit(1)

    batches = _find_step_batches(abc_dir)
    if not batches:
        print("No STEP batch directories found.")
        return []

    print(f"Found {len(batches)} batch(es): {[os.path.basename(b) for b in batches]}")

    max_file_bytes = max_file_size_mb * 1024 * 1024
    results_by_batch = {}
    n_models = 0
    n_skipped_size = 0
    n_errors = 0

    for batch_dir in batches:
        batch_name = os.path.basename(batch_dir)
        batch_results = []

        model_dirs = sorted(
            d for d in os.listdir(batch_dir)
            if os.path.isdir(os.path.join(batch_dir, d))
        )
        pbar = tqdm(model_dirs, desc=batch_name, unit="model")
        for model_id in pbar:
            model_path = os.path.join(batch_dir, model_id)
            step_files = sorted(glob.glob(os.path.join(model_path, "*.step")))
            if not step_files:
                continue

            n_models += 1
            step_path = step_files[0]

            if os.path.getsize(step_path) > max_file_bytes:
                n_skipped_size += 1
                continue

            shape = _load_step(step_path)
            if shape is None:
                n_errors += 1
                continue

            parts = _extract_parts(shape)

            if all_parts:
                # Every part must satisfy the predicate
                part_infos = []
                ok = True
                for part in parts:
                    passed, info = _check_part(part, allowed_types, max_bspline, max_faces)
                    if not passed:
                        ok = False
                        break
                    part_infos.append(info)
                if ok:
                    for part_idx, info in enumerate(part_infos):
                        batch_results.append({
                            "model_id": model_id,
                            "part_idx": part_idx,
                            "step_path": step_path,
                            "n_parts": len(parts),
                            **info,
                        })
            else:
                # Any qualifying part is included independently
                for part_idx, part in enumerate(parts):
                    passed, info = _check_part(part, allowed_types, max_bspline, max_faces)
                    if passed:
                        batch_results.append({
                            "model_id": model_id,
                            "part_idx": part_idx,
                            "step_path": step_path,
                            "n_parts": len(parts),
                            **info,
                        })

            if n_models % 200 == 0:
                pbar.write(f"  hits={len(batch_results)}  errors={n_errors}  skipped={n_skipped_size}")

        results_by_batch["batch_name"]["total_parts"] = len(batch_results)
        results_by_batch[batch_name]["parts"] = batch_results

    total_parts = sum(len(v) for v in results_by_batch.values())
    print(f"\nDone. {n_models} models, {n_skipped_size} skipped (file size), "
          f"{n_errors} read errors, {total_parts} qualifying parts across "
          f"{len(results_by_batch)} batch(es).")
    return results_by_batch


def main():
    parser = argparse.ArgumentParser(
        description="Filter ABC dataset models by surface type composition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--abc_dir", type=str,
                        default="/home/lukau/Desktop/abc_dataset",
                        help="Root directory of the ABC dataset")
    parser.add_argument("--allowed_types", type=str, nargs="+",
                        default=["Plane", "Sphere", "Cylinder", "Cone", "BSpline"],
                        help="Allowed mapped surface types")
    parser.add_argument("--max_bspline", type=int, default=5,
                        help="Max number of BSpline faces per part")
    parser.add_argument("--max_faces", type=int, default=50,
                        help="Max total faces per part")
    parser.add_argument("--max_file_size_mb", type=float, default=1.0,
                        help="Skip STEP files larger than this (MB)")
    parser.add_argument("--all_parts", action="store_true",
                        help="Require ALL parts to satisfy the filter (default: any qualifying part is included)")
    parser.add_argument("--output_dir", type=str, default="abc_filtered",
                        help="Output directory for per-batch JSON manifests")
    args = parser.parse_args()

    allowed = set(args.allowed_types)
    print(f"Allowed types: {allowed}")
    print(f"Max BSpline faces: {args.max_bspline}")
    print(f"Max total faces: {args.max_faces}")
    print(f"Max file size: {args.max_file_size_mb} MB")
    print(f"All parts must qualify: {args.all_parts}")
    print()

    results_by_batch = filter_models(args.abc_dir, allowed, args.max_bspline,
                                     args.max_faces, args.max_file_size_mb,
                                     args.all_parts)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for batch_name, entries in results_by_batch.items():
        out_path = os.path.join(output_dir, f"{batch_name}.json")
        with open(out_path, "w") as f:
            json.dump(entries, f, indent=2)
        print(f"Manifest written to {out_path} ({len(entries)} parts)")

    all_entries = [e for entries in results_by_batch.values() for e in entries]
    if all_entries:
        face_counts = [r["n_faces"] for r in all_entries]
        bspline_counts = [r["n_bspline"] for r in all_entries]
        print(f"\nSummary:")
        print(f"  Total qualifying parts: {len(all_entries)}")
        print(f"  Face count range: {min(face_counts)}-{max(face_counts)}")
        print(f"  BSpline count distribution: {dict(Counter(bspline_counts))}")

        unique_models = len(set(r["model_id"] for r in all_entries))
        print(f"  Unique models with qualifying parts: {unique_models}")


if __name__ == "__main__":
    main()
