"""
Classify ABC models as 'primitive' (every part's clusters fit a primitive
surface) or 'freeform' (at least one part contains a freeform cluster) by
running fit_surface(classify_only=True) on every cluster of every part.
Only primitive fitters are invoked — INR is never trained, so this runs on
CPU and never touches the GPU.

Writes two text files in the same format as run_abc_parts.py --ids_file:
  {primitive_out}  — one model ID per line (all parts primitive)
  {freeform_out}   — one model ID per line (>=1 freeform part)

A model with no .xyzc files under --input_dir is reported and skipped.

Granularity caveat: classification is per-part, but the output is per-model
so it can feed directly into run_abc_parts.py --ids_file. A model is
labelled "primitive" iff every one of its parts is primitive; any freeform
part promotes the whole model to "freeform".
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import point2cad.primitive_fitting_utils as pfu
from point2cad.surface_fitter import fit_surface


# Mirrors the per-part normalization closure in mesh_pipeline.run_compute
# (lines ~123-130): center → PCA-rotate so the thinnest axis aligns to +X →
# scale so the longest bbox extent is 1. Must be kept in lockstep with that
# closure so the classify_only threshold comparison sees the same error
# scale the pipeline does.
def normalize_points(pts):
    mean = pts.mean(axis=0)
    centered = pts - mean
    S, U = np.linalg.eigh(centered.T @ centered)
    R = pfu.rotation_matrix_a_to_b(U[:, np.argmin(S)], np.array([1, 0, 0]))
    rotated = (R @ centered.T).T
    scale = float((rotated.max(axis=0) - rotated.min(axis=0)).max()) + 1e-7
    return (rotated / scale).astype(np.float32), mean, R, scale


def read_ids(path):
    ids = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ids.append(s)
    return ids


def load_done(*paths):
    """Load already-classified IDs from existing output files so that
    re-running the script appends rather than reprocesses."""
    done = set()
    for p in paths:
        if os.path.isfile(p):
            for mid in read_ids(p):
                done.add(mid)
    return done


_MIN_CLUSTER_PTS = 20  # matches mesh_pipeline's HPNet/Point2CAD floor


def classify_part(xyzc_file, device, np_rng):
    data = np.loadtxt(xyzc_file).astype(np.float32)
    assert data.shape[1] == 4, (
        f"{xyzc_file}: expected 4 columns (xyz + cluster_id), got {data.shape[1]}"
    )
    points = data[:, :3]
    labels = data[:, 3].astype(int)
    points, *_ = normalize_points(points)
    for label in np.unique(labels):
        cluster = points[labels == label]
        # mesh_pipeline drops clusters with < 20 pts before fitting (cones
        # need 6 DoF worth of evidence, and small clusters make the cylinder
        # Powell minimizer return NaN). Drop them here too.
        if len(cluster) < _MIN_CLUSTER_PTS:
            continue
        if fit_surface(cluster, None, np_rng, device, classify_only=True) == "freeform":
            return "freeform"
    return "primitive"


def classify_model(model_id, input_dir, device, np_rng):
    xyzc_files = sorted(
        glob.glob(os.path.join(input_dir, model_id, "*.xyzc")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    )
    if not xyzc_files:
        return None
    for f in xyzc_files:
        if classify_part(f, device, np_rng) == "freeform":
            return "freeform"
    return "primitive"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--ids_file", required=True,
                    help="Path to a .txt file with one ABC model ID per line")
    ap.add_argument("--input_dir", default="../sample_clouds_abc_parts",
                    help="Root of per-part .xyzc inputs "
                         "(expects {input_dir}/{model_id}/*.xyzc)")
    ap.add_argument("--primitive_out", default="primitive_ids.txt",
                    help="Output path for primitive-only model IDs")
    ap.add_argument("--freeform_out", default="freeform_ids.txt",
                    help="Output path for freeform model IDs")
    ap.add_argument("--seed", type=int, default=41)
    args = ap.parse_args()

    device = torch.device("cpu")
    np_rng = np.random.default_rng(args.seed)

    ids = read_ids(args.ids_file)
    print(f"[classify] {len(ids)} model IDs from {args.ids_file}", flush=True)

    done = load_done(args.primitive_out, args.freeform_out)
    if done:
        print(f"[classify] {len(done)} IDs already classified (resume mode)", flush=True)

    n_primitive = n_freeform = n_skipped = 0
    missing = []
    t0 = time.perf_counter()
    with open(args.primitive_out, "a") as fp, open(args.freeform_out, "a") as ff:
        for model_id in tqdm.tqdm(ids, desc="classify", unit="model", dynamic_ncols=True):
            if model_id in done:
                n_skipped += 1
                continue
            label = classify_model(model_id, args.input_dir, device, np_rng)
            if label is None:
                missing.append(model_id)
                continue
            out = fp if label == "primitive" else ff
            out.write(model_id + "\n")
            out.flush()
            done.add(model_id)
            if label == "primitive":
                n_primitive += 1
            else:
                n_freeform += 1

    dt = time.perf_counter() - t0
    print(f"[classify] done in {dt:.1f}s", flush=True)
    print(f"[classify]   primitive (new): {n_primitive} -> {args.primitive_out}", flush=True)
    print(f"[classify]   freeform  (new): {n_freeform}  -> {args.freeform_out}", flush=True)
    if n_skipped:
        print(f"[classify]   skipped (already done): {n_skipped}", flush=True)
    if missing:
        print(f"[classify]   missing:   {len(missing)}  "
              f"(no .xyzc files under {args.input_dir})", flush=True)


if __name__ == "__main__":
    main()
