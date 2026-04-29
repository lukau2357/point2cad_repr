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

Every per-split statistic — per-side mean/median/std/CI, paired delta, and
speedup — is computed on the same paired subset: parts where BOTH sides ran
to completion AND both emitted a non-None value for the metric/timing in
question. This guarantees that within a split, `mine`, `orig`, and `paired`
blocks all reduce over the exact same set of parts per field, so means,
standard errors, and deltas are directly comparable. Failure counts are
reported separately at the top level (mine-only, orig-only, both).

Stats per split (all / primitive_only / has_freeform):
  metrics:
    per side -> mean, median, std, 95% normal CI  (paired subset)
    paired   -> delta = mine - orig, mean+CI, frac_mine_better
  timings (primitive_only, has_freeform only):
    per side -> same as above
    speedup  -> arithmetic mean of orig/mine ratios with normal CI
                (see note below about geomean plot inconsistency)

Plots (saved under {out_dir}/plots):
  paired scatter per (metric, split)   — mine on x, orig on y, y=x diagonal
  timing scatter per split             — mine on x, orig on y

Uniform convention: delta = mine - orig everywhere. METRIC_DIRECTION says
which sign of delta means "mine wins" (used in plot captions + frac calc).

Usage:
    python scripts/aggregate_eval.py \\
        --dir_mine output_mesh \\
        --dir_orig ../point2cad/output_p2cad_orig \\
        --out_dir aggregator_output
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.style.use("ggplot")


METRIC_FIELDS = [
    "p_coverage",
    "p_coverage_mesh_to_pc",
    "residual_mean",
    "chamfer_sym",
]
# All non-timing metrics are plotted. Directional Chamfers are not
# tracked at all — only the symmetric Chamfer is reported.
PLOT_METRICS = list(METRIC_FIELDS)
TIMING_FIELDS = [
    "fit_time",
    "primitive_fit_time",
    "freeform_fit_time",
    "clip_time",
    "total_time",
]
TIMING_PLOT_FIELDS = ["fit_time", "clip_time", "total_time"]

# For each metric, the sign of (mine - orig) that favours ours.
METRIC_DIRECTION = {
    "p_coverage":            "higher_is_better",
    "p_coverage_mesh_to_pc": "higher_is_better",
    "residual_mean":         "lower_is_better",
    "chamfer_sym":           "lower_is_better",
}

# Plot labels in two languages. `en` for paper drafts, `me` for the
# master thesis (Montenegrin). All user-visible strings in the plots go
# through this table. summary.json / parts.csv keys are never translated.
LABELS = {
    "en": {
        "mine": "Ours",
        "orig": "Point2CAD",
        "split.all": "all models",
        "split.primitive_only": "primitive-only models",
        "split.has_freeform": "models with freeform surfaces",
        "dir.higher_is_better": "higher is better",
        "dir.lower_is_better": "lower is better",
        "dir.ours_above":  "Ours favourable above y=x",
        "dir.ours_below":  "Ours favourable below y=x",
        "axis.count": "count",
        "axis.delta_prefix": "Δ = Ours − Point2CAD",
        "axis.seconds_ci": "seconds (mean ± 95% CI)",
        "axis.speedup": "speedup = Point2CAD / Ours",
        "label.parity": "parity",
        "label.diagonal": "y = x",
        "label.mean_delta": "mean Δ = {mean:+.4g}  [CI {lo:+.4g}, {hi:+.4g}]",
        "label.geomean": "geomean = {mean:.2f}×  [CI {lo:.2f}, {hi:.2f}]",
        "label.speedup_mean": "mean speedup = {mean:.2f}×  [CI {lo:.2f}, {hi:.2f}]",
        "label.frac_ours": "Ours favourable on {pct:.0%} of parts",
        "title.paired": "Paired comparison",
        "title.delta": "Distribution of Δ",
        "title.timings": "Timing breakdown",
        "title.speedup": "Speedup distribution",
        "metric.p_coverage": "point coverage (pc → mesh)",
        "metric.p_coverage_mesh_to_pc": "mesh coverage (mesh → pc)",
        "metric.residual_mean": "mean residual",
        "metric.chamfer_sym": "symmetric Chamfer distance",
        "metric.fit_time": "fitting time",
        "metric.clip_time": "clipping time",
        "metric.total_time": "total time",
    },
    "me": {
        "mine": "Naš metod",
        "orig": "Point2CAD",
        "split.all": "svi modeli",
        "split.primitive_only": "modeli sa primitivnim površinama",
        "split.has_freeform": "modeli sa slobodnim površinama",
        "dir.higher_is_better": "više je bolje",
        "dir.lower_is_better": "manje je bolje",
        "dir.ours_above":  "naš metod bolji iznad y=x",
        "dir.ours_below":  "naš metod bolji ispod y=x",
        "axis.count": "učestanost",
        "axis.delta_prefix": "Δ = Naš − Point2CAD",
        "axis.seconds_ci": "sekunde (srednja vrijednost ± 95% IP)",
        "axis.speedup": "ubrzanje = Point2CAD / Naš",
        "label.parity": "paritet",
        "label.diagonal": "y = x",
        "label.mean_delta": "srednje Δ = {mean:+.4g}  [IP {lo:+.4g}, {hi:+.4g}]",
        "label.geomean": "geometrijska sredina = {mean:.2f}×  [IP {lo:.2f}, {hi:.2f}]",
        "label.speedup_mean": "srednje ubrzanje = {mean:.2f}×  [IP {lo:.2f}, {hi:.2f}]",
        "label.frac_ours": "naš metod je bolji na {pct:.0%} dijelova",
        "title.paired": "Uparena uporedba",
        "title.delta": "Raspodjela razlike Δ",
        "title.timings": "Uporedno vrijeme izvršavanja",
        "title.speedup": "Raspodjela ubrzanja",
        "metric.p_coverage": "pokrivenost (oblak → mreža)",
        "metric.p_coverage_mesh_to_pc": "pokrivenost (mreža → oblak)",
        "metric.residual_mean": "srednja rezidualna greška",
        "metric.chamfer_sym": "simetrična Čamferova udaljenost",
        "metric.fit_time": "vrijeme fitovanja",
        "metric.clip_time": "vrijeme isjecanja",
        "metric.total_time": "ukupno vrijeme",
    },
}


def _T(labels, key, **fmt):
    """Translate a label key. Missing keys return the key itself so a typo
    is obvious in the rendered plot rather than crashing silently."""
    s = labels.get(key, key)
    return s.format(**fmt) if fmt else s


def _metric_label(labels, metric):
    return _T(labels, f"metric.{metric}")


def _split_label(labels, split):
    return _T(labels, f"split.{split}")


def _direction_label(labels, direction):
    return _T(labels, f"dir.{direction}")


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


def _ci95_normal(arr):
    """Return (mean, lo, hi) with normal 95% CI. NaN for empty or n<2."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(arr.mean())
    if arr.size < 2:
        return m, m, m
    se = float(arr.std(ddof=1)) / np.sqrt(arr.size)
    return m, m - 1.96 * se, m + 1.96 * se


def _summarize_values(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return {"mean": None, "median": None, "std": None,
                "ci95_lo": None, "ci95_hi": None, "n": 0}
    arr = np.asarray(clean, dtype=np.float64)
    mean, lo, hi = _ci95_normal(arr)
    std = float(arr.std(ddof=1)) if arr.size >= 2 else 0.0
    return {
        "mean": mean,
        "median": float(np.median(arr)),
        "std": std,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "n": int(arr.size),
    }


def summarize_paired(mine, orig, keys):
    """Per-split aggregate for BOTH sides, restricted per metric/timing to
    the paired subset of `keys` where both sides have the value.

    Returns (mine_agg, orig_agg). Within each metric, the two sides are
    computed on the same key set, so means and standard errors are directly
    comparable and consistent with the paired-delta block.
    """
    mine_agg = {"n_parts": len(keys)}
    orig_agg = {"n_parts": len(keys)}
    for f in METRIC_FIELDS:
        mv, ov = _collect_paired(mine, orig, keys, f)
        mine_agg[f] = _summarize_values(mv.tolist())
        orig_agg[f] = _summarize_values(ov.tolist())
    for f in TIMING_FIELDS:
        mv, ov = _collect_paired_timing(mine, orig, keys, f)
        ml, ol = mv.tolist(), ov.tolist()
        mine_agg["timing_" + f] = {
            "mean": float(np.mean(ml)) if ml else None,
            "total": float(np.sum(ml)) if ml else None,
            "n": len(ml),
        }
        orig_agg["timing_" + f] = {
            "mean": float(np.mean(ol)) if ol else None,
            "total": float(np.sum(ol)) if ol else None,
            "n": len(ol),
        }
    return mine_agg, orig_agg


def _collect_paired(mine, orig, keys, metric_field):
    """Return (mine_vals, orig_vals) over keys where both have the metric."""
    mine_vals, orig_vals = [], []
    for k in keys:
        me = mine.get(k, {})
        oe = orig.get(k, {})
        mm = (me.get("metrics") or {}).get("metrics", {})
        om = (oe.get("metrics") or {}).get("metrics", {})
        a = mm.get(metric_field)
        b = om.get(metric_field)
        if a is None or b is None:
            continue
        mine_vals.append(a)
        orig_vals.append(b)
    return np.asarray(mine_vals, dtype=np.float64), np.asarray(orig_vals, dtype=np.float64)


def _collect_paired_timing(mine, orig, keys, timing_field):
    mine_vals, orig_vals = [], []
    for k in keys:
        me = mine.get(k, {})
        oe = orig.get(k, {})
        mt = (me.get("metrics") or {}).get("timing", {})
        ot = (oe.get("metrics") or {}).get("timing", {})
        a = mt.get(timing_field)
        b = ot.get(timing_field)
        if a is None or b is None or a <= 0 or b <= 0:
            continue
        mine_vals.append(a)
        orig_vals.append(b)
    return np.asarray(mine_vals, dtype=np.float64), np.asarray(orig_vals, dtype=np.float64)


def _paired_stats(mine_vals, orig_vals, direction):
    if mine_vals.size == 0:
        return {"n": 0, "delta_mean": None, "delta_ci95_lo": None,
                "delta_ci95_hi": None, "frac_mine_better": None,
                "direction": direction}
    deltas = mine_vals - orig_vals
    m, lo, hi = _ci95_normal(deltas)
    if direction == "higher_is_better":
        frac = float(np.mean(deltas > 0))
    else:
        frac = float(np.mean(deltas < 0))
    return {"n": int(deltas.size), "delta_mean": m,
            "delta_ci95_lo": lo, "delta_ci95_hi": hi,
            "frac_mine_better": frac, "direction": direction}


def _speedup_stats(mine_times, orig_times):
    if mine_times.size == 0:
        return {"n": 0, "mean_speedup": None,
                "ci95_lo": None, "ci95_hi": None}
    ratios = orig_times / mine_times
    m, lo, hi = _ci95_normal(ratios)
    return {
        "n": int(mine_times.size),
        "mean_speedup": float(m),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
    }


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
                mtim.get("fit_time"), mtim.get("clip_time"), mtim.get("total_time"),
                mtim.get("primitive_fit_time"), mtim.get("freeform_fit_time"),
                otim.get("fit_time"), otim.get("clip_time"), otim.get("total_time"),
            ])


def _fmt(v, width=8, prec=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-".rjust(width)
    return f"{v:{width}.{prec}f}"


def print_summary_table(summary):
    print("\n=== Evaluation summary ===")
    mine_rate = summary.get("mine_success_rate")
    orig_rate = summary.get("orig_success_rate")
    mine_rate_s = f"{mine_rate:.1%}" if mine_rate is not None else "-"
    orig_rate_s = f"{orig_rate:.1%}" if orig_rate is not None else "-"
    print(f"mine: attempted={summary['n_mine_attempted']}  ok={summary['n_mine_ok']}  "
          f"fail={summary['n_mine_attempted'] - summary['n_mine_ok']}  "
          f"success_rate={mine_rate_s}")
    print(f"orig: attempted={summary['n_orig_attempted']}  ok={summary['n_orig_ok']}  "
          f"fail={summary['n_orig_attempted'] - summary['n_orig_ok']}  "
          f"success_rate={orig_rate_s}")
    print(f"scope: common_attempted={summary['n_common_attempted']}  "
          f"mine_only_attempted={summary['n_mine_only_attempted']}  "
          f"orig_only_attempted={summary['n_orig_only_attempted']}")
    print(f"common_attempted breakdown: n_common_ok={summary['n_common_ok']}  "
          f"mine_fail_orig_ok={summary['n_mine_fail_orig_ok']}  "
          f"orig_fail_mine_ok={summary['n_orig_fail_mine_ok']}  "
          f"both_fail={summary['n_both_fail']}")
    print()
    print(f"(common_ok split)         primitive_only={summary['n_common_primitive_only']}  "
          f"has_freeform={summary['n_common_has_freeform']}  "
          f"unclassified={summary['n_common_unclassified']}")

    for split_name, pair in summary["splits"].items():
        mine = pair["mine"]
        orig = pair["orig"]
        paired = pair.get("paired", {})
        n_paired = pair.get("n_paired", 0)
        print(f"\n--- split: {split_name} ---")
        print(f"  side summaries: mine_ok={mine['n_parts']}  orig_ok={orig['n_parts']}  "
              f"(paired n={n_paired})")
        for field in METRIC_FIELDS:
            direction = METRIC_DIRECTION.get(field, "")
            mm = mine[field]
            om = orig[field]
            pp = paired.get(field, {})
            print(f"  {field:22s}  mine={_fmt(mm['mean'])} ± {_fmt(mm['ci95_hi'] - mm['mean'] if mm['mean'] is not None else None, 6, 4)}"
                  f"  orig={_fmt(om['mean'])} ± {_fmt(om['ci95_hi'] - om['mean'] if om['mean'] is not None else None, 6, 4)}"
                  f"  Δ={_fmt(pp.get('delta_mean'))} ({direction})")

        if split_name in ("primitive_only", "has_freeform"):
            for field in TIMING_PLOT_FIELDS:
                key = "timing_" + field
                mm = mine[key]["mean"]
                om = orig[key]["mean"]
                speedup = pair.get("speedup", {}).get(field, {})
                gm = speedup.get("mean_speedup")
                gm_lo = speedup.get("ci95_lo")
                gm_hi = speedup.get("ci95_hi")
                extra = (f"  speedup {gm:.2f}x [{gm_lo:.2f}, {gm_hi:.2f}]"
                         if gm is not None else "")
                print(f"  {field+' (mean s)':22s}  mine={_fmt(mm, 8, 3)}  orig={_fmt(om, 8, 3)}{extra}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _plot_paired_scatter(mine_vals, orig_vals, metric, split, out_path, direction, labels):
    if mine_vals.size == 0:
        return
    deltas = mine_vals - orig_vals
    if direction == "higher_is_better":
        frac = float(np.mean(deltas > 0))
        reading_key = "dir.ours_below"
    else:
        frac = float(np.mean(deltas < 0))
        reading_key = "dir.ours_above"
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(mine_vals, orig_vals, s=18, alpha=0.6)
    lo = float(min(mine_vals.min(), orig_vals.min()))
    hi = float(max(mine_vals.max(), orig_vals.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label=_T(labels, "label.diagonal"))
    ax.set_xlabel(f"{_T(labels, 'mine')}  {_metric_label(labels, metric)}")
    ax.set_ylabel(f"{_T(labels, 'orig')}  {_metric_label(labels, metric)}")
    ax.set_title(
        f"{_T(labels, 'title.paired')} — {_metric_label(labels, metric)}\n"
        f"{_split_label(labels, split)}\n"
        f"{_T(labels, reading_key)}\n"
        f"{_T(labels, 'label.frac_ours', pct=frac)}",
        fontsize=9,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_timing_paired(mine, orig, keys, split, out_path, labels,
                        timing_field="total_time"):
    mt, ot = _collect_paired_timing(mine, orig, keys, timing_field)
    if mt.size == 0:
        return
    ratios = ot / mt
    frac = float(np.mean(ratios > 1.0))
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(mt, ot, s=18, alpha=0.6)
    lo = float(min(mt.min(), ot.min()))
    hi = float(max(mt.max(), ot.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label=_T(labels, "label.diagonal"))
    ax.set_xlabel(f"{_T(labels, 'mine')}  {_metric_label(labels, timing_field)}  [s]")
    ax.set_ylabel(f"{_T(labels, 'orig')}  {_metric_label(labels, timing_field)}  [s]")
    ax.set_title(
        f"{_T(labels, 'title.paired')} — {_metric_label(labels, timing_field)}\n"
        f"{_split_label(labels, split)}\n"
        f"{_T(labels, 'dir.ours_above')}\n"
        f"{_T(labels, 'label.frac_ours', pct=frac)}",
        fontsize=9,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def generate_plots(plots_dir, mine, orig, splits_keys, labels):
    for split_name, keys in splits_keys.items():
        for metric in PLOT_METRICS:
            direction = METRIC_DIRECTION.get(metric, "higher_is_better")
            mine_vals, orig_vals = _collect_paired(mine, orig, keys, metric)
            _plot_paired_scatter(mine_vals, orig_vals, metric, split_name,
                                 os.path.join(plots_dir, f"{metric}_{split_name}_paired.png"),
                                 direction, labels)
        if split_name in ("primitive_only", "has_freeform"):
            _plot_timing_paired(mine, orig, keys, split_name,
                                os.path.join(plots_dir, f"total_time_{split_name}_paired.png"),
                                labels, timing_field="total_time")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--dir_mine", required=True,
                    help="Root output dir of mesh_pipeline.py "
                         "(contains {model_id}/part_*/metrics.json)")
    ap.add_argument("--dir_orig", default=None,
                    help="Root output dir of patched original Point2CAD (optional)")
    ap.add_argument("--out_dir", default="aggregator_output",
                    help="Output directory for summary.json, parts.csv, and plots/")
    ap.add_argument("--lang", choices=["en", "me"], default="en",
                    help="Language for plot labels (en=English, me=Montenegrin)")
    args = ap.parse_args()
    labels = LABELS[args.lang]

    mine = load_side(args.dir_mine)
    orig = load_side(args.dir_orig)

    mine_attempted_set = set(mine)
    orig_attempted_set = set(orig)
    common_attempted_set = mine_attempted_set & orig_attempted_set
    all_attempted = sorted(mine_attempted_set | orig_attempted_set)

    mine_ok_set = {k for k in mine_attempted_set if mine[k]["status"] == "ok"}
    orig_ok_set = {k for k in orig_attempted_set if orig[k]["status"] == "ok"}
    common_ok_set = mine_ok_set & orig_ok_set

    mine_ok = sorted(mine_ok_set)
    orig_ok = sorted(orig_ok_set)
    common_ok = sorted(common_ok_set)

    # Failure accounting is restricted to parts *both* sides attempted.
    # Without this restriction, parts only mine ran would be counted as
    # orig failures (and vice versa), which is wrong.
    # Per-side failure sets: every part that side attempted and exit was
    # non-zero (or metrics.json missing). Independent of the other side.
    # These are the canonical "algorithm X failed on these parts" lists.
    mine_failed_set = mine_attempted_set - mine_ok_set
    orig_failed_set = orig_attempted_set - orig_ok_set
    mine_failed = sorted(mine_failed_set)
    orig_failed = sorted(orig_failed_set)

    # Paired views: built directly from the failure sets. Since failure
    # implies attempt, each intersection below naturally lives inside
    # common_attempted — no explicit restriction needed.
    mine_fail_orig_ok = sorted(mine_failed_set & orig_ok_set)
    orig_fail_mine_ok = sorted(orig_failed_set & mine_ok_set)
    both_fail = sorted(mine_failed_set & orig_failed_set)

    mine_only_attempted = sorted(mine_attempted_set - orig_attempted_set)
    orig_only_attempted = sorted(orig_attempted_set - mine_attempted_set)

    # Prefer mine's metrics for classification; fall back to orig when mine
    # never processed the part (orig_only_attempted).
    def _classify_any(k):
        return classify_from_mine(mine.get(k)) or classify_from_mine(orig.get(k))
    class_of = {k: _classify_any(k) for k in all_attempted}

    # Every split is restricted to the paired intersection of successes —
    # both sides completed that part. Per-metric, further restricted to
    # pairs where both sides emitted a non-None value (see summarize_paired).
    prim_c, free_c, unk_c = split_keys_by_class(class_of, common_ok)

    splits_spec = {
        "all":            common_ok,
        "primitive_only": prim_c,
        "has_freeform":   free_c,
    }

    def _paired_block(keys):
        return {m: _paired_stats(*_collect_paired(mine, orig, keys, m),
                                 METRIC_DIRECTION[m])
                for m in METRIC_FIELDS}

    def _speedup_block(keys):
        out = {}
        for f in TIMING_PLOT_FIELDS:
            mt, ot = _collect_paired_timing(mine, orig, keys, f)
            out[f] = _speedup_stats(mt, ot)
        return out

    splits = {}
    for name, keys in splits_spec.items():
        mine_agg, orig_agg = summarize_paired(mine, orig, keys)
        entry = {
            "mine":     mine_agg,
            "orig":     orig_agg,
            "paired":   _paired_block(keys),
            "n_paired": len(keys),
        }
        if name in ("primitive_only", "has_freeform"):
            entry["speedup"] = _speedup_block(keys)
        splits[name] = entry

    n_mine_attempted = len(mine_attempted_set)
    n_orig_attempted = len(orig_attempted_set)
    summary = {
        "dir_mine": args.dir_mine,
        "dir_orig": args.dir_orig,
        "metric_direction": METRIC_DIRECTION,
        "n_all_attempted": len(all_attempted),
        "n_mine_attempted": n_mine_attempted,
        "n_orig_attempted": n_orig_attempted,
        "n_common_attempted": len(common_attempted_set),
        "n_mine_only_attempted": len(mine_only_attempted),
        "n_orig_only_attempted": len(orig_only_attempted),
        "n_mine_ok": len(mine_ok_set),
        "n_orig_ok": len(orig_ok_set),
        "n_mine_failed": len(mine_failed),
        "n_orig_failed": len(orig_failed),
        "n_common_ok": len(common_ok_set),
        "mine_success_rate": (len(mine_ok_set) / n_mine_attempted) if n_mine_attempted else None,
        "orig_success_rate": (len(orig_ok_set) / n_orig_attempted) if n_orig_attempted else None,
        "n_mine_fail_orig_ok": len(mine_fail_orig_ok),
        "n_orig_fail_mine_ok": len(orig_fail_mine_ok),
        "n_both_fail": len(both_fail),
        "n_common_primitive_only": len(prim_c),
        "n_common_has_freeform": len(free_c),
        "n_common_unclassified": len(unk_c),
        "failure_lists": {
            "mine_failed":         [f"{m}/part_{p}" for (m, p) in mine_failed],
            "orig_failed":         [f"{m}/part_{p}" for (m, p) in orig_failed],
            "mine_fail_orig_ok":   [f"{m}/part_{p}" for (m, p) in mine_fail_orig_ok],
            "orig_fail_mine_ok":   [f"{m}/part_{p}" for (m, p) in orig_fail_mine_ok],
            "both_fail":           [f"{m}/part_{p}" for (m, p) in both_fail],
            "mine_only_attempted": [f"{m}/part_{p}" for (m, p) in mine_only_attempted],
            "orig_only_attempted": [f"{m}/part_{p}" for (m, p) in orig_only_attempted],
        },
        "splits": splits,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "summary.json")
    csv_path = os.path.join(args.out_dir, "parts.csv")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    write_csv(csv_path, mine, orig, class_of, all_attempted)
    generate_plots(plots_dir, mine, orig, splits_spec, labels)

    print_summary_table(summary)
    print(f"\nWrote {summary_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote plots under {plots_dir}/")


if __name__ == "__main__":
    main()
