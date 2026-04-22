"""
Per-model wrapper for the mesh pipeline with split primitive / freeform phases.

Primitive-only model IDs run in a ProcessPoolExecutor (CPU-bound, no GPU
contention); freeform IDs run sequentially afterwards (GPU-bound INR
fitting).  Each (model_id) spawns exactly one subprocess invoking the
pipeline with `--write_part_status`; the subprocess handles every part of
that model internally and does its own unified/ merge.  Stdout and stderr
of each subprocess are captured to per-model log files under `{OUTPUT_DIR}/
{model_id}/stdout.log` and `stderr.log`.

Skip rule on re-run (a model is "done" iff both hold):
  (1) `{OUTPUT_DIR}/{model_id}/unified/.merge_ok` exists — written by THIS
      wrapper after observing rc=0 from the subprocess.
  (2) every part file `{input_dir}/{model_id}/*.xyzc` has a corresponding
      `{OUTPUT_DIR}/{model_id}/part_N/wrapper_status.json` — written by the
      pipeline (guarded by `--write_part_status`).

If either is missing the whole model is redone from scratch: the entire
`{model_id}/` tree is rmtree'd, then the pipeline is re-invoked.  This
handles the 'pipeline interrupted mid-merge' case cleanly: marker is
absent, model is re-run.  Cost: we redo parts that previously succeeded.
Benefit: no partial-merge ambiguity, no model-level orchestration state.

Interrupt semantics:
- The pipeline wraps `_run_compute_part` in `try/except Exception`.  SIGINT
  -> KeyboardInterrupt, which is NOT a subclass of Exception, so KI
  propagates; no status file is written for the in-flight part.  On re-run
  the model will be redone (marker missing) and that part is re-attempted.
- A deterministic pipeline bug on one part writes `status=failed`, the
  other parts in the same model still run, merge runs over the successful
  parts, and the wrapper writes `.merge_ok` — the model is considered
  'done with some failed parts' and is not re-attempted.
"""

import argparse
import glob
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import tqdm


# ---------------------------------------------------------------------------
# PIPELINE_CMD must invoke the mesh pipeline; OUTPUT_DIR must match its
# --output_dir default so paths constructed here line up with what the
# subprocess writes on disk.
PIPELINE_CMD = ["python", "-u", "mesh_pipeline.py"]
OUTPUT_DIR = "output_mesh"
# ---------------------------------------------------------------------------


def read_ids(path):
    ids = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ids.append(s)
    return ids


def part_indices_for(model_id, input_dir):
    """Indices of .xyzc part files for this model, sorted numerically."""
    pattern = os.path.join(input_dir, model_id, "*.xyzc")
    paths = glob.glob(pattern)
    return sorted(int(os.path.splitext(os.path.basename(p))[0]) for p in paths)


def model_is_done(model_id, input_dir):
    """True iff the prior run finished AND its merge committed.  Specifically:
    (1) unified/.merge_ok is present (wrapper-written on rc=0), and
    (2) every input part has its own wrapper_status.json (pipeline-written).
    """
    model_dir = os.path.join(OUTPUT_DIR, model_id)
    if not os.path.isfile(os.path.join(model_dir, "unified", ".merge_ok")):
        return False
    for pi in part_indices_for(model_id, input_dir):
        if not os.path.isfile(os.path.join(model_dir, f"part_{pi}",
                                           "wrapper_status.json")):
            return False
    return True


def _atomic_touch(path):
    """Create an empty file at `path` atomically via tmp + os.replace."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        pass
    os.replace(tmp, path)


def run_one_model(model_id, input_dir):
    """Invoke the pipeline for one model and return an outcome dict for the
    main process to log.  Workers must not call tqdm.tqdm.write — a worker's
    tqdm state is detached from the main bar, so writes bypass the bar's
    output-locking logic and corrupt its line.  All logging is done from
    main() after fut.result().

    Returns: {"outcome": "skipped"|"ok"|"failed",
              "rc": int|None, "duration_s": float|None}

    KI during subprocess.run propagates out of this function (subprocess.run's
    Popen context kills the child and re-raises).  No .merge_ok is written,
    so the model is redone from scratch on the next wrapper invocation.
    """
    if model_is_done(model_id, input_dir):
        return {"outcome": "skipped", "rc": None, "duration_s": None}

    model_dir = os.path.join(OUTPUT_DIR, model_id)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    stdout_log = os.path.join(model_dir, "stdout.log")
    stderr_log = os.path.join(model_dir, "stderr.log")

    cmd = PIPELINE_CMD + [
        "--model_id", model_id,
        "--input_dir", input_dir,
        "--output_dir", OUTPUT_DIR,
        "--no_clean_output",
        "--write_part_status",
    ]

    t0 = time.perf_counter()
    # `with` on the log handles + subprocess.run guarantees the child is
    # reaped before we leave this block.  If KI fires inside run(), Popen's
    # context manager kills the child and re-raises -> we never reach the
    # marker write below and the `with` blocks still flush+close.
    with open(stdout_log, "w") as out, open(stderr_log, "w") as err:
        proc = subprocess.run(cmd, stdout=out, stderr=err)
    dt = time.perf_counter() - t0

    if proc.returncode == 0:
        _atomic_touch(os.path.join(model_dir, "unified", ".merge_ok"))
        return {"outcome": "ok", "rc": 0, "duration_s": dt}
    else:
        return {"outcome": "failed", "rc": proc.returncode, "duration_s": dt}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--primitive_ids", default=None,
                    help="Path to .txt of primitive-only model IDs (run in parallel)")
    ap.add_argument("--freeform_ids", default=None,
                    help="Path to .txt of freeform model IDs (run sequentially)")
    ap.add_argument("--input_dir", default="./sample_clouds_abc_parts",
                    help="Root of per-part .xyzc inputs "
                         "(expects {input_dir}/{model_id}/*.xyzc)")
    ap.add_argument("--workers", type=int, default=4,
                    help="Process pool size for the primitive phase")
    args = ap.parse_args()

    if args.primitive_ids is None and args.freeform_ids is None:
        ap.error("at least one of --primitive_ids or --freeform_ids must be given")

    prim_ids = read_ids(args.primitive_ids) if args.primitive_ids else []
    free_ids = read_ids(args.freeform_ids)  if args.freeform_ids  else []
    print(f"[wrapper] primitive: {len(prim_ids)} IDs  freeform: {len(free_ids)} IDs",
          flush=True)

    total = len(prim_ids) + len(free_ids)
    bar = tqdm.tqdm(total=total, desc="models", unit="model", dynamic_ncols=True)
    counts = {"ok": 0, "failed": 0, "skipped": 0}

    def _log(mid, result):
        """Run in main only — tqdm.tqdm.write coordinates with bar redraw."""
        outcome = result["outcome"]
        counts[outcome] += 1
        if outcome == "skipped":
            msg = f"[wrapper] {mid}: already done, skipping"
        elif outcome == "ok":
            msg = f"[wrapper] {mid}: ok ({result['duration_s']:.1f}s)"
        else:
            stderr_log = os.path.join(OUTPUT_DIR, mid, "stderr.log")
            msg = (f"[wrapper] {mid}: failed rc={result['rc']} "
                   f"({result['duration_s']:.1f}s) — see {stderr_log}")
        tqdm.tqdm.write(msg)

    try:
        # ---- primitive phase ---------------------------------------------
        # workers==1 skips the pool entirely — direct in-process calls.
        # Avoids pickling, subprocess fork for the worker, and as_completed
        # bookkeeping.  Useful as a second-pass retry after an OOM'd parallel
        # run: failed models have no .merge_ok, so they get redone here with
        # full memory headroom.
        if prim_ids:
            if args.workers == 1:
                tqdm.tqdm.write(f"[wrapper] primitive phase: {len(prim_ids)} models "
                                f"(sequential, no pool)")
                for mid in prim_ids:
                    bar.set_postfix_str(mid, refresh=True)
                    try:
                        result = run_one_model(mid, args.input_dir)
                    except Exception as e:
                        tqdm.tqdm.write(f"[wrapper] {mid}: raised {e!r}")
                        counts["failed"] += 1
                    else:
                        _log(mid, result)
                    bar.update(1)
            else:
                tqdm.tqdm.write(f"[wrapper] primitive phase: {len(prim_ids)} models "
                                f"across {args.workers} worker(s)")
                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    try:
                        futs = {ex.submit(run_one_model, mid, args.input_dir): mid
                                for mid in prim_ids}
                        for fut in as_completed(futs):
                            mid = futs[fut]
                            try:
                                result = fut.result()
                            except Exception as e:
                                tqdm.tqdm.write(
                                    f"[wrapper] {mid}: worker raised {e!r}")
                                counts["failed"] += 1
                            else:
                                _log(mid, result)
                            bar.update(1)
                    except KeyboardInterrupt:
                        tqdm.tqdm.write("[wrapper] interrupted — cancelling pending jobs")
                        ex.shutdown(wait=False, cancel_futures=True)
                        raise

        # ---- freeform phase: sequential ----------------------------------
        if free_ids:
            tqdm.tqdm.write(f"[wrapper] freeform phase: {len(free_ids)} models "
                            f"(sequential)")
            for mid in free_ids:
                bar.set_postfix_str(mid, refresh=True)
                try:
                    result = run_one_model(mid, args.input_dir)
                except Exception as e:
                    tqdm.tqdm.write(f"[wrapper] {mid}: raised {e!r}")
                    counts["failed"] += 1
                else:
                    _log(mid, result)
                bar.update(1)
    finally:
        bar.close()
        print(f"[wrapper] done — ok={counts['ok']}  failed={counts['failed']}  "
              f"skipped={counts['skipped']}", flush=True)


if __name__ == "__main__":
    main()
