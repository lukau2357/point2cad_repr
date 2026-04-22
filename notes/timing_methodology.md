# Timing methodology: BLAS oversubscription caveat

Applies to per-part `fit_time` / `clip_time` / `total_time` numbers reported
in `metrics.json` and aggregated in `scripts/aggregate_eval.py`.

## The issue

The mesh-pipeline wrapper (`run_abc_parts.py`) runs the primitive phase with
a `ProcessPoolExecutor` of $N$ workers. Each worker launches a
`mesh_pipeline.py` subprocess. Inside that subprocess, surface fitting is
numpy-vectorized — least-squares / SVD / eigendecomposition on cluster
points — which dispatches to the BLAS backend (OpenBLAS or MKL linked
through numpy).

By default BLAS uses as many threads as there are CPU cores, **per
process**. With $N$ concurrent subprocesses on a machine with $C$ cores,
the system ends up with up to $N \cdot C$ BLAS threads fighting for $C$
physical cores. Each matrix operation then runs slower in wall-clock than
it would in isolation — classic oversubscription.

Per-part timings are taken *inside* each subprocess via
`time.perf_counter()`. That measures wall-clock, not CPU-seconds. So even
though every worker measures its own part independently, the number it
records is inflated by whatever slowdown the contention caused.

Original Point2CAD's wrapper is sequential — one subprocess at a time.
Its BLAS gets the full machine, no contention. Its per-part timings reflect
uncontended wall-clock on the same hardware.

## Direction of bias

Against our speedup claim.

- Our vectorized fits are BLAS-heavy → oversubscription hurts them a lot.
- Original's pythonic per-cluster loops barely use BLAS → uncontended
  timings are close to ideal for them.

Under the current measurement regime our numbers are inflated and
original's are not. The reported $\sim 21\times$ fit speedup is therefore a
**lower bound** on the true algorithmic gap. A clean isolated benchmark
would show a wider gap, not narrower.

This bias does not affect quality metrics (`p_coverage`, `chamfer_sym`,
`residual_mean`, ...) — those are deterministic functions of the produced
geometry and independent of the machine or concurrency level.

## Mitigations (in increasing order of effort)

1. **Methodology disclosure only.** One sentence in the thesis timing
   section: "our per-part timings were collected under 3-way concurrent
   wrapper execution, which inflates wall-clock via BLAS oversubscription;
   the reported speedup is therefore conservative." Acceptable for thesis;
   likely not enough for a publication reviewer.

2. **Pin BLAS threads per subprocess.** Export
   `OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1` in the subprocess environment
   (applied uniformly to both algorithms). Each subprocess then runs
   single-threaded BLAS; $N$ workers use $N$ threads total; no
   oversubscription. Per-part timings under this regime are clean, though
   per-operation slower than full-threaded BLAS.

3. **Dedicated isolated timing subset.** Sample $\sim 100$ representative
   models, re-run both pipelines on the same machine with `workers=1`,
   back-to-back, fresh output dirs. Full BLAS threading, no contention, no
   cross-algorithm interference. Report timing on this subset separately
   from quality on the full dataset. Strongest framing:
   "quality on $N = N_\text{common}$ paired successes; timing on an
   isolated subset of $n = 100$ under identical single-process conditions."

## When this matters

- **Master thesis.** Option 1 is sufficient.
- **Research publication.** Option 3 is the standard bar. Option 2 is a
  cheaper fallback if re-running is infeasible.

## Related notes

- `cad_metrics.md` — what the quality metrics actually measure; hardware-
  invariant unlike timings.
- `evaluation_canonical_frame.md` — per-part normalization making metrics
  comparable across parts.
