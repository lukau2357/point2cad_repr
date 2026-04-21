# brep_pipeline as a mesh_pipeline consumer

Notes captured while refactoring `mesh_pipeline.py` to serialize fitted surfaces so that a future `brep_pipeline` run can skip surface fitting.

## What mesh_pipeline writes (per part)

Inside `{output_mesh}/{model_id}/part_{p}/`:

- `surface_mesh_{cid}.npz` — unclipped mesh (vertices, triangles) in normalized space. Already existed; reused by the consumer path.
- `surface_params_{cid}.npz` — plane / sphere / cylinder / cone. Contains the raw primitive params (e.g. `a, d` for plane) plus `surface_id` and `error`.
- `surface_inr_{cid}.pt` — torch-serialized dict: `state_dict`, `network_parameters`, `is_u_closed`, `is_v_closed`, `cluster_mean`, `cluster_scale`, `uv_bb_min`, `uv_bb_max`, `surface_id`, `error`.
- `surface_types.json` — `{cid: type_name}` map (`plane` / `sphere` / `cylinder` / `cone` / `inr` / `bpa` / `bpa_bspline`).
- `metadata.npz` — already existed. Carries `cluster_ids`, `norm_mean`, `norm_R`, `norm_scale`.

Freeform `bpa` and `bpa_bspline` are intentionally not serialized — their full representation is the mesh, and the current brep pipeline cannot build BRep faces from them anyway (see gotcha below).

## Cluster-ID consistency is safe without extra bookkeeping

Both pipelines construct their cluster list identically:

1. `np.loadtxt(<xyzc>)` — deterministic.
2. `normalize_points` — identical body in both files (mesh_pipeline.py vs brep_pipeline.py). Same input → same output.
3. `np.unique(data[:, -1].astype(int))` — returns sorted unique cids.
4. Build `clusters` parallel to `unique_clusters`.
5. Apply `MIN_CLUSTER_PTS = 20` filter with `>=` (both sides).

So when brep iterates its post-filter `unique_clusters`, each cid matches on-disk filenames exactly. No extra index map needed.

`merge_coincident_surfaces` in brep_pipeline is an older artifact that defaults off (`--merge_surfaces` is `action="store_true"`, default False). Consumer path can ignore it.

## Consumer-side reconstruction is modest, not minimal

Reconstructing a `res` dict that matches what `fit_surface` returns:

- Primitive: open `.npz`, rebuild
  ```
  res = {"surface_id": sid,
         "result": {"surface_type": name, "error": err, "params": {...}},
         "mesh": <o3d mesh from surface_mesh_{cid}.npz>}
  ```
  ~15 lines.

- INR:
  1. `INRNetwork(**network_parameters, is_u_closed=..., is_v_closed=...)`
  2. `model.load_state_dict(state_dict)` + `.to(device)` + `.eval()`
  3. Repopulate `result["params"]` with `{model, cluster_mean, cluster_scale, uv_bb_min, uv_bb_max, network_parameters}` so `inr_to_occ` works unchanged.
  ~25 lines plus a new import from `point2cad.inr_fitting`.

Total: ~40–60 lines. Not a one-liner, but straightforward.

## Gotcha: `SURFACE_INR` is overloaded

`fit_surface` returns `surface_id = SURFACE_INR` for INR, BPA, *and* BPA+BSpline freeform paths (see `surface_fitter.py` lines ~461, ~475, and the INR dispatch). That means `surface_id` alone can't distinguish the three.

Current `to_occ_surface(SURFACE_INR, ...)` unconditionally calls `inr_to_occ`, which expects `params["model"]`. BPA / BPA+BSpline have `params = {}`, so the current brep_pipeline would crash on them.

This is only a problem if the consumer path encounters BPA clusters. Given the working convention of always running `--freeform_method inr`, every `SURFACE_INR` on disk has a real `.pt` sidecar and the ambiguity disappears. If the convention ever changes, the consumer must disambiguate via `surface_types.json`.

## Other determinism notes

- `np.linalg.eigh` sign ambiguity: LAPACK is deterministic for identical input, so both pipelines' `R` matrices agree byte-for-byte on the same machine. ID matching is unaffected even if the sign convention did flip (ids come from the xyzc last column, not from the eigenvectors).
- Random seeding differs between the two pipelines (brep uses `args.seed + part_idx`), but this only matters for the *fitting* fallback path. For loaded surfaces the seed is irrelevant — no random ops are invoked on reload.

## Why we stopped at write-side

Two reasons, both worth remembering:

1. Running brep_pipeline is uncertain — mentor-dependent. Write-side change is cheap and reversible; consumer refactor is not.
2. Brep has pre-existing bugs that can mask refactor errors. A clean mesh run is a clean signal; a failing brep run is undecidable (pipeline bug or refactor bug?). Only refactor brep when we need it and can validate independently.
