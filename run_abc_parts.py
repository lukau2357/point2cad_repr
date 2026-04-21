"""
Minimal per-part wrapper for the mesh pipeline.

Reads a .txt file of ABC model IDs (one per line, comments start with '#'),
globs the parts under `{input_dir}/{model_id}/*.xyzc`, and for each
(model_id, part_idx) spawns a subprocess invoking the pipeline with
`--part part_idx`. Stdout/stderr are captured to per-part log files, and a
`wrapper_status.json` is written at the end regardless of outcome. Existence
of that file is the skip signal on re-runs — status is not checked.

This wrapper is intentionally minimal and should be copy-pasted into the
original Point2CAD repo with only PIPELINE_CMD and the --input_dir default
changed (see the comment at the top of main()).
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import time

import tqdm


# ---------------------------------------------------------------------------
# The only lines that differ between the two copies of this wrapper.
#   mesh_pipeline (this repo):  ["python", "mesh_pipeline.py"]     + "output_mesh"
#   point2cad orig:             ["python", "-m", "point2cad.main"] + "output_p2cad_orig"
# OUTPUT_DIR must match the pipeline's own --output_dir default so paths
# constructed here line up with what the subprocess writes on disk.
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


def classify_result(returncode, part_dir):
    """Return (status, reason) given the subprocess exit and on-disk state."""
    if returncode != 0:
        sig = -returncode if returncode < 0 else None
        reason = f"exit {returncode}"
        if sig is not None:
            reason += f" (signal {sig})"
        return "fail", reason
    metrics_path = os.path.join(part_dir, "metrics.json")
    if not os.path.isfile(metrics_path):
        return "fail", "metrics.json missing despite exit 0"
    try:
        with open(metrics_path) as f:
            m = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return "fail", f"metrics.json unreadable: {e}"
    if "metrics" not in m:
        return "fail", "metrics.json has no 'metrics' key"
    return "ok", None


def run_one_part(model_id, part_idx, input_dir):
    part_dir = os.path.join(OUTPUT_DIR, model_id, f"part_{part_idx}")
    status_path = os.path.join(part_dir, "wrapper_status.json")
    if os.path.isfile(status_path):
        tqdm.tqdm.write(f"[wrapper] {model_id} part {part_idx}: already processed, skipping")
        return

    # Clean + recreate part_dir ourselves so the subprocess doesn't have to.
    # We pass --no_clean_output to the pipeline so it preserves the log files
    # we are about to open in this directory; otherwise the pipeline's rmtree
    # unlinks them while our file handles are still open and the logs vanish
    # when the subprocess exits.
    if os.path.exists(part_dir):
        shutil.rmtree(part_dir)
    os.makedirs(part_dir)
    stdout_log = os.path.join(part_dir, "stdout.log")
    stderr_log = os.path.join(part_dir, "stderr.log")

    # Pass --output_dir explicitly so the subprocess uses the same directory
    # we used to build `part_dir` — guards against future drift in the
    # pipeline's argparse default.
    cmd = PIPELINE_CMD + [
        "--model_id", model_id,
        "--input_dir", input_dir,
        "--output_dir", OUTPUT_DIR,
        "--part", str(part_idx),
        "--no_clean_output",
    ]
    tqdm.tqdm.write(f"[wrapper] {model_id} part {part_idx}: launching {' '.join(cmd)}")

    t0 = time.perf_counter()
    returncode = None
    try:
        # `with` on the log file handles + subprocess.run guarantees the
        # child has exited before we leave this block: run() only returns
        # after the child is reaped, and the file objects flush+close on
        # block exit. No lingering FDs or zombie children.
        with open(stdout_log, "w") as out, open(stderr_log, "w") as err:
            proc = subprocess.run(cmd, stdout=out, stderr=err)
        returncode = proc.returncode
    finally:
        duration_s = time.perf_counter() - t0

    # If a KeyboardInterrupt fired inside subprocess.run, we never reach this
    # point — the exception propagates out and wrapper_status.json is not
    # written, so the part is retried on the next run.
    status, reason = classify_result(returncode, part_dir)
    payload = {
        "model_id": model_id,
        "part_idx": part_idx,
        "status": status,
        "returncode": returncode,
        "signal": -returncode if returncode is not None and returncode < 0 else None,
        "duration_s": duration_s,
        "reason": reason,
    }
    tmp_path = status_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, status_path)

    tqdm.tqdm.write(f"[wrapper] {model_id} part {part_idx}: {status} "
                    f"(rc={returncode}, {duration_s:.1f}s)"
                    + (f"  reason={reason}" if reason else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--ids_file", required=True,
                    help="Path to a .txt file with one ABC model ID per line")
    ap.add_argument("--input_dir", default="./sample_clouds_abc_parts",
                    help="Root of per-part .xyzc inputs "
                         "(expects {input_dir}/{model_id}/*.xyzc)")
    args = ap.parse_args()

    ids = read_ids(args.ids_file)
    print(f"[wrapper] {len(ids)} model IDs from {args.ids_file}", flush=True)

    # Pre-scan: build the full (model_id, part_idx) job list up front so the
    # tqdm bar has a known total and reports ETA. Missing models are reported
    # immediately, not at their would-be turn in the loop.
    jobs = []
    for model_id in ids:
        model_in_dir = os.path.join(args.input_dir, model_id)
        part_files = sorted(
            glob.glob(os.path.join(model_in_dir, "*.xyzc")),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
        )
        if not part_files:
            print(f"[wrapper] {model_id}: no .xyzc files under {model_in_dir}, skipping",
                  flush=True)
            continue
        for p in part_files:
            jobs.append((model_id, int(os.path.splitext(os.path.basename(p))[0])))

    n_models = len({j[0] for j in jobs})
    print(f"[wrapper] {len(jobs)} parts queued across {n_models} models", flush=True)

    bar = tqdm.tqdm(jobs, desc="parts", unit="part", dynamic_ncols=True)
    for model_id, part_idx in bar:
        bar.set_postfix_str(f"{model_id} part {part_idx}", refresh=True)
        run_one_part(model_id, part_idx, args.input_dir)
    bar.close()

    print("[wrapper] done", flush=True)


if __name__ == "__main__":
    main()
