"""
Per-model wrapper for the mesh pipeline with split primitive / freeform phases.

Primitive-only model IDs run in a ProcessPoolExecutor (CPU-bound, no GPU
contention); freeform IDs run sequentially afterwards (GPU-bound INR
fitting).  Each (model_id) spawns exactly one subprocess invoking the
pipeline with `--write_part_status`; the subprocess handles every part of
that model internally and does its own unified/ merge.  Stdout and stderr
of each subprocess are captured to per-model log files under `{OUTPUT_DIR}/
{model_id}/stdout.log` and `stderr.log`.

Skip rule: main scans OUTPUT_DIR up front and collects the set of model IDs
whose every expected part has a `wrapper_status.json` with status == 'ok'.
Those IDs are subtracted from the input ID lists; the set difference is
what actually runs.  Any model with a missing or non-ok part — whether
interrupted, OOM'd, or hit a deterministic pipeline bug — is absent from
the done set by construction and falls into the to-process set, where it
is rmtree'd and re-run from scratch.

Interrupt semantics: SIGINT -> KeyboardInterrupt propagates out of
subprocess.run (Popen's context manager kills the child and re-raises).
The in-flight part's status file is never written, so the model falls out
of the done set on the next invocation and gets re-run.
"""

import argparse
import glob
import json
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


def _read_completed_ids(output_dir, input_dir):
    """Set of model IDs whose every expected part has a wrapper_status.json
    with status == 'ok'.  Cross-references input_dir so a model is only
    counted as done if *all* its input parts are represented on disk — not
    just the ones the pipeline managed to finish before crashing."""
    done = set()
    if not os.path.isdir(output_dir):
        return done
    for mid in os.listdir(output_dir):
        mdir = os.path.join(output_dir, mid)
        if not os.path.isdir(mdir):
            continue
        indices = part_indices_for(mid, input_dir)
        if not indices:
            continue
        all_ok = True
        for pi in indices:
            wp = os.path.join(mdir, f"part_{pi}", "wrapper_status.json")
            if not os.path.isfile(wp):
                all_ok = False
                break
            try:
                with open(wp) as f:
                    if json.load(f).get("status") != "ok":
                        all_ok = False
                        break
            except (OSError, json.JSONDecodeError):
                all_ok = False
                break
        if all_ok:
            done.add(mid)
    return done


def run_one_model(model_id, input_dir):
    """Invoke the pipeline for one model and return an outcome dict for the
    main process to log.  Workers must not call tqdm.tqdm.write — a worker's
    tqdm state is detached from the main bar and writes bypass the bar's
    output-locking logic and corrupt its line.  All logging is done from
    main() after fut.result().

    Returns: {"outcome": "ok"|"failed", "rc": int, "duration_s": float}

    KI during subprocess.run propagates out of this function (subprocess.run's
    Popen context kills the child and re-raises).
    """
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
    with open(stdout_log, "w") as out, open(stderr_log, "w") as err:
        proc = subprocess.run(cmd, stdout=out, stderr=err)
    dt = time.perf_counter() - t0

    outcome = "ok" if proc.returncode == 0 else "failed"
    return {"outcome": outcome, "rc": proc.returncode, "duration_s": dt}


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

    # Set difference: anything not fully-ok on disk (missing, failed,
    # OOM'd, interrupted) falls into the to-process set automatically.
    done = _read_completed_ids(OUTPUT_DIR, args.input_dir)
    prim_to_run = [m for m in prim_ids if m not in done]
    free_to_run = [m for m in free_ids if m not in done]
    n_skipped = (len(prim_ids) - len(prim_to_run)) + (len(free_ids) - len(free_to_run))
    if n_skipped:
        print(f"[wrapper] skipping {n_skipped} models already done on disk",
              flush=True)

    total = len(prim_ids) + len(free_ids)
    # `initial` pre-seeds the counter without contaminating elapsed time,
    # so rate / ETA reflect actual processing speed from the first real
    # completion rather than being skewed by the big-jump pre-update.
    bar = tqdm.tqdm(total=total, initial=n_skipped, desc="models",
                    unit="model", dynamic_ncols=True)
    counts = {"ok": 0, "failed": 0, "skipped": n_skipped}

    def _log(mid, result):
        """Run in main only — tqdm.tqdm.write coordinates with bar redraw."""
        outcome = result["outcome"]
        counts[outcome] += 1
        if outcome == "ok":
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
        # run: failed models fall out of `done` and get redone here with
        # full memory headroom.
        if prim_to_run:
            if args.workers == 1:
                tqdm.tqdm.write(f"[wrapper] primitive phase: {len(prim_to_run)} models "
                                f"(sequential, no pool)")
                for mid in prim_to_run:
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
                tqdm.tqdm.write(f"[wrapper] primitive phase: {len(prim_to_run)} models "
                                f"across {args.workers} worker(s)")
                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    try:
                        futs = {ex.submit(run_one_model, mid, args.input_dir): mid
                                for mid in prim_to_run}
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
        if free_to_run:
            tqdm.tqdm.write(f"[wrapper] freeform phase: {len(free_to_run)} models "
                            f"(sequential)")
            for mid in free_to_run:
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
