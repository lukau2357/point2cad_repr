"""
Aggregate per-part metrics.json files produced by mesh_pipeline.py and the
patched original Point2CAD main.py.

Pairs parts by (model_id, part_idx) across the two output trees, splits by
the `is_primitive_only` flag (taken from the mine side — the classification
is a property of the input, not of either pipeline), and emits a JSON
summary plus a per-part CSV for manual inspection.

Usage:
    python scripts/aggregate_eval.py \
        --dir_mine output_bfs \
        --dir_orig ../point2cad/output_p2cad_orig \
        --out eval_summary.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np


METRIC_FIELDS = [
    "p_coverage",
    "p_coverage_mesh_to_pc",
    "residual_mean",
    "chamfer_pc_to_mesh",
    "chamfer_mesh_to_pc",
    "chamfer_sym",
]
TIMING_FIELDS = [
    "fit_time",
    "primitive_fit_time",
    "freeform_fit_time",
    "clip_time",
    "total_time",
]


def load_metrics(root):
    """Return dict keyed by (model_id, part_idx) -> metrics dict."""
    if root is None or not os.path.isdir(root):
        return {}
    out = {}
    pattern = os.path.join(root, "*", "part_*", "metrics.json")
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path) as f:
                m = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[warn] could not read {path}: {e}")
            continue
        key = (str(m.get("model_id", "")), int(m.get("part_idx", 0)))
        out[key] = m
    return out


def classify_from_mine(m):
    """Return 'primitive_only', 'has_freeform', or None."""
    flag = m.get("is_primitive_only")
    if flag is True:
        return "primitive_only"
    if flag is False:
        return "has_freeform"
    return None


def _summarize_values(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return {"mean": None, "median": None, "std": None, "n": 0}
    arr = np.asarray(clean, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
        "n": int(arr.size),
    }


def summarize(metrics_dict, keys):
    """Aggregate metrics + timing over the given (model_id, part_idx) keys."""
    agg = {"n_parts": sum(1 for k in keys if k in metrics_dict)}
    for f in METRIC_FIELDS:
        agg[f] = _summarize_values(
            [metrics_dict[k]["metrics"].get(f) for k in keys if k in metrics_dict]
        )
    for f in TIMING_FIELDS:
        vals = [metrics_dict[k].get("timing", {}).get(f) for k in keys if k in metrics_dict]
        clean = [v for v in vals if v is not None]
        agg["timing_" + f] = {
            "mean": float(np.mean(clean)) if clean else None,
            "total": float(np.sum(clean)) if clean else None,
            "n": len(clean),
        }
    return agg


def split_keys_by_class(class_of, all_parts):
    prim, free, unk = [], [], []
    for k in all_parts:
        c = class_of.get(k)
        if c == "primitive_only":
            prim.append(k)
        elif c == "has_freeform":
            free.append(k)
        else:
            unk.append(k)
    return prim, free, unk


def write_csv(csv_path, mine, orig, class_of, all_parts):
    header = [
        "model_id", "part_idx", "split", "n_points", "n_clusters",
        "mine_p_cov", "orig_p_cov",
        "mine_p_cov_m2p", "orig_p_cov_m2p",
        "mine_residual_mean", "orig_residual_mean",
        "mine_chamfer_sym", "orig_chamfer_sym",
        "mine_chamfer_pc_to_mesh", "orig_chamfer_pc_to_mesh",
        "mine_chamfer_mesh_to_pc", "orig_chamfer_mesh_to_pc",
        "mine_fit_time", "mine_clip_time", "mine_total_time",
        "mine_primitive_fit_time", "mine_freeform_fit_time",
        "orig_fit_time", "orig_clip_time", "orig_total_time",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for k in all_parts:
            mm = mine.get(k, {})
            om = orig.get(k, {})
            mmet = mm.get("metrics", {})
            omet = om.get("metrics", {})
            mtim = mm.get("timing", {})
            otim = om.get("timing", {})
            w.writerow([
                k[0], k[1], class_of.get(k),
                mm.get("n_points") or om.get("n_points"),
                mm.get("n_clusters") or om.get("n_clusters"),
                mmet.get("p_coverage"), omet.get("p_coverage"),
                mmet.get("p_coverage_mesh_to_pc"), omet.get("p_coverage_mesh_to_pc"),
                mmet.get("residual_mean"), omet.get("residual_mean"),
                mmet.get("chamfer_sym"), omet.get("chamfer_sym"),
                mmet.get("chamfer_pc_to_mesh"), omet.get("chamfer_pc_to_mesh"),
                mmet.get("chamfer_mesh_to_pc"), omet.get("chamfer_mesh_to_pc"),
                mtim.get("fit_time"), mtim.get("clip_time"), mtim.get("total_time"),
                mtim.get("primitive_fit_time"), mtim.get("freeform_fit_time"),
                otim.get("fit_time"), otim.get("clip_time"), otim.get("total_time"),
            ])


def _fmt(v, width=8, prec=4):
    if v is None:
        return "-".rjust(width)
    return f"{v:{width}.{prec}f}"


def print_summary_table(summary):
    splits = summary["splits"]
    print("\n=== Evaluation summary ===")
    print(f"n_parts total          : {summary['n_parts_total']}")
    print(f"n_parts primitive_only : {summary['n_parts_primitive_only']}")
    print(f"n_parts has_freeform   : {summary['n_parts_has_freeform']}")
    print(f"n_parts unclassified   : {summary['n_parts_unclassified']}")

    for split_name, pair in splits.items():
        mine = pair["mine"]
        orig = pair["orig"]
        print(f"\n--- split: {split_name} ---")
        print(f"  parts: mine={mine['n_parts']}  orig={orig['n_parts']}")
        for field in METRIC_FIELDS:
            mm = mine[field]["mean"]
            om = orig[field]["mean"]
            print(f"  {field:22s}  mine={_fmt(mm)}  orig={_fmt(om)}")
        for field in ["fit_time", "clip_time", "total_time"]:
            key = "timing_" + field
            mm = mine[key]["mean"]
            om = orig[key]["mean"]
            print(f"  {field+' (mean s)':22s}  mine={_fmt(mm, 8, 3)}  orig={_fmt(om, 8, 3)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir_mine", required=True,
                    help="Root output dir of mesh_pipeline.py (contains abc_*/part_*/metrics.json)")
    ap.add_argument("--dir_orig", default=None,
                    help="Root output dir of patched original Point2CAD (optional)")
    ap.add_argument("--out", default="eval_summary.json",
                    help="Output summary JSON path")
    args = ap.parse_args()

    mine = load_metrics(args.dir_mine)
    orig = load_metrics(args.dir_orig)

    class_of = {k: classify_from_mine(m) for k, m in mine.items()}
    all_parts = sorted(set(mine) | set(orig))
    prim, free, unk = split_keys_by_class(class_of, all_parts)

    summary = {
        "dir_mine": args.dir_mine,
        "dir_orig": args.dir_orig,
        "n_parts_total": len(all_parts),
        "n_parts_primitive_only": len(prim),
        "n_parts_has_freeform": len(free),
        "n_parts_unclassified": len(unk),
        "splits": {
            "all":            {"mine": summarize(mine, all_parts),
                               "orig": summarize(orig, all_parts)},
            "primitive_only": {"mine": summarize(mine, prim),
                               "orig": summarize(orig, prim)},
            "has_freeform":   {"mine": summarize(mine, free),
                               "orig": summarize(orig, free)},
        },
    }

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.splitext(args.out)[0] + "_parts.csv"
    write_csv(csv_path, mine, orig, class_of, all_parts)

    print_summary_table(summary)
    print(f"\nWrote {args.out}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
