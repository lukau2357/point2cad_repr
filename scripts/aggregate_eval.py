"""
Aggregate per-part metrics.json files produced by mesh_pipeline.py and the
patched original Point2CAD main.py, using wrapper_status.json (written by
run_abc_parts.py) to decide which parts count as successfully completed.

Pairs parts by (model_id, part_idx) across the two output trees. A part is
counted as "ok" on a given side iff either:
  - wrapper_status.json exists on that side with status == "ok" and a valid
    metrics.json is present, OR
  - wrapper_status.json is absent (pre-wrapper run) and a valid metrics.json
    is present.
Everything else is a failure on that side.

Metric/timing summaries are computed over the **intersection** of successes —
parts where BOTH sides ran to completion — so the comparison is apples to
apples. Failure counts are reported separately (mine-only, orig-only, both).

Usage:
    python scripts/aggregate_eval.py \\
        --dir_mine output_mesh \\
        --dir_orig ../point2cad/output_p2cad_orig \\
        --out eval_summary.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

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


def _safe_read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[warn] could not read {path}: {e}")
        return None


def load_side(root):
    """Walk `{root}/*/part_*` and return {(model_id, part_idx): entry}.

    entry = {
        "status":  "ok" | "fail",
        "metrics": dict | None,   # full metrics.json contents
        "wrapper": dict | None,   # wrapper_status.json contents (None if absent)
    }

    Pre-wrapper runs (no wrapper_status.json) are treated as "ok" iff a
    valid metrics.json exists. Runs with wrapper_status.json are "ok" iff
    status == "ok" AND metrics.json is valid.
    """
    if root is None or not os.path.isdir(root):
        return {}
    out = {}
    for part_dir in sorted(glob.glob(os.path.join(root, "*", "part_*"))):
        parent = os.path.basename(os.path.dirname(part_dir))
        leaf = os.path.basename(part_dir)
        if not leaf.startswith("part_"):
            continue
        try:
            part_idx = int(leaf[len("part_"):])
        except ValueError:
            continue
        key = (parent, part_idx)

        metrics_path = os.path.join(part_dir, "metrics.json")
        wrapper_path = os.path.join(part_dir, "wrapper_status.json")

        metrics = _safe_read_json(metrics_path) if os.path.isfile(metrics_path) else None
        if metrics is not None and "metrics" not in metrics:
            metrics = None  # malformed: treat as absent

        wrapper = _safe_read_json(wrapper_path) if os.path.isfile(wrapper_path) else None

        if wrapper is None:
            status = "ok" if metrics is not None else "fail"
        else:
            status = "ok" if (wrapper.get("status") == "ok" and metrics is not None) else "fail"

        out[key] = {"status": status, "metrics": metrics, "wrapper": wrapper}
    return out


def classify_from_mine(side_entry):
    """Return 'primitive_only', 'has_freeform', or None, from the mine side."""
    if side_entry is None:
        return None
    m = side_entry.get("metrics")
    if m is None:
        return None
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


def summarize(side_dict, keys):
    """Aggregate metrics + timing over the given keys, reading from side_dict.

    `keys` is expected to be the common-success set; summaries only include
    parts whose entry has metrics is not None.
    """
    ok_keys = [k for k in keys if k in side_dict and side_dict[k].get("metrics") is not None]
    agg = {"n_parts": len(ok_keys)}
    for f in METRIC_FIELDS:
        agg[f] = _summarize_values(
            [side_dict[k]["metrics"]["metrics"].get(f) for k in ok_keys]
        )
    for f in TIMING_FIELDS:
        vals = [side_dict[k]["metrics"].get("timing", {}).get(f) for k in ok_keys]
        clean = [v for v in vals if v is not None]
        agg["timing_" + f] = {
            "mean": float(np.mean(clean)) if clean else None,
            "total": float(np.sum(clean)) if clean else None,
            "n": len(clean),
        }
    return agg


def split_keys_by_class(class_of, keys):
    prim, free, unk = [], [], []
    for k in keys:
        c = class_of.get(k)
        if c == "primitive_only":
            prim.append(k)
        elif c == "has_freeform":
            free.append(k)
        else:
            unk.append(k)
    return prim, free, unk


def _wrap_reason(entry):
    if entry is None:
        return ""
    w = entry.get("wrapper")
    if w is None:
        return ""
    return w.get("reason") or ""


def write_csv(csv_path, mine, orig, class_of, all_parts):
    header = [
        "model_id", "part_idx", "split",
        "mine_status", "orig_status",
        "mine_returncode", "orig_returncode",
        "mine_signal", "orig_signal",
        "mine_failure_reason", "orig_failure_reason",
        "n_points", "n_clusters",
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
            me = mine.get(k) or {}
            oe = orig.get(k) or {}
            mm_full = me.get("metrics") or {}
            om_full = oe.get("metrics") or {}
            mmet = mm_full.get("metrics", {})
            omet = om_full.get("metrics", {})
            mtim = mm_full.get("timing", {})
            otim = om_full.get("timing", {})
            mw = me.get("wrapper") or {}
            ow = oe.get("wrapper") or {}
            w.writerow([
                k[0], k[1], class_of.get(k),
                me.get("status"), oe.get("status"),
                mw.get("returncode"), ow.get("returncode"),
                mw.get("signal"), ow.get("signal"),
                _wrap_reason(me), _wrap_reason(oe),
                mm_full.get("n_points") or om_full.get("n_points"),
                mm_full.get("n_clusters") or om_full.get("n_clusters"),
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
    print("\n=== Evaluation summary ===")
    print(f"n_parts_union            : {summary['n_parts_union']}")
    print(f"n_mine_ok                : {summary['n_mine_ok']}")
    print(f"n_orig_ok                : {summary['n_orig_ok']}")
    print(f"n_common_ok (intersect)  : {summary['n_common_ok']}")
    print(f"n_mine_fail_orig_ok      : {summary['n_mine_fail_orig_ok']}")
    print(f"n_orig_fail_mine_ok      : {summary['n_orig_fail_mine_ok']}")
    print(f"n_both_fail              : {summary['n_both_fail']}")
    print()
    print(f"(intersection split)      primitive_only={summary['n_common_primitive_only']}  "
          f"has_freeform={summary['n_common_has_freeform']}  "
          f"unclassified={summary['n_common_unclassified']}")

    for split_name, pair in summary["splits"].items():
        mine = pair["mine"]
        orig = pair["orig"]
        print(f"\n--- split: {split_name} (intersection) ---")
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
                    help="Root output dir of mesh_pipeline.py "
                         "(contains {model_id}/part_*/metrics.json)")
    ap.add_argument("--dir_orig", default=None,
                    help="Root output dir of patched original Point2CAD (optional)")
    ap.add_argument("--out", default="eval_summary.json",
                    help="Output summary JSON path")
    args = ap.parse_args()

    mine = load_side(args.dir_mine)
    orig = load_side(args.dir_orig)

    # All keys we know about across both trees.
    all_parts = sorted(set(mine) | set(orig))

    mine_ok = {k for k in all_parts if mine.get(k, {}).get("status") == "ok"}
    orig_ok = {k for k in all_parts if orig.get(k, {}).get("status") == "ok"}
    common_ok = sorted(mine_ok & orig_ok)

    mine_fail_orig_ok = sorted(orig_ok - mine_ok)
    orig_fail_mine_ok = sorted(mine_ok - orig_ok)
    both_fail = sorted(set(all_parts) - mine_ok - orig_ok)

    # Classification (mine side) only for the apples-to-apples set.
    class_of = {k: classify_from_mine(mine.get(k)) for k in all_parts}
    prim, free, unk = split_keys_by_class(class_of, common_ok)

    summary = {
        "dir_mine": args.dir_mine,
        "dir_orig": args.dir_orig,
        "n_parts_union": len(all_parts),
        "n_mine_ok": len(mine_ok),
        "n_orig_ok": len(orig_ok),
        "n_common_ok": len(common_ok),
        "n_mine_fail_orig_ok": len(mine_fail_orig_ok),
        "n_orig_fail_mine_ok": len(orig_fail_mine_ok),
        "n_both_fail": len(both_fail),
        "n_common_primitive_only": len(prim),
        "n_common_has_freeform": len(free),
        "n_common_unclassified": len(unk),
        "failure_lists": {
            "mine_fail_orig_ok": [f"{m}/part_{p}" for (m, p) in mine_fail_orig_ok],
            "orig_fail_mine_ok": [f"{m}/part_{p}" for (m, p) in orig_fail_mine_ok],
            "both_fail":         [f"{m}/part_{p}" for (m, p) in both_fail],
        },
        "splits": {
            "all":            {"mine": summarize(mine, common_ok),
                               "orig": summarize(orig, common_ok)},
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
