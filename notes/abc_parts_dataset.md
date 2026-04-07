# ABC Parts Dataset (HPNet / ParseNet) Reference

## Overview

ABC Parts is a preprocessed point cloud dataset derived from the original ABC dataset
by the **ParseNet** authors (Sharma et al., 2020). It is distributed as per-model
HDF5 files and is most easily found via the **HPNet** repository
(https://github.com/SimingYan/HPNet), which uses ParseNet's preprocessing logic and
provides the same h5 format with one model per file.

This is a **distinct artifact** from the raw ABC STEP/OBJ files documented in
`abc_dataset.md`. ABC Parts is preprocessed, normalized, and primitive-labeled,
whereas the raw ABC dataset requires its own sampling pipeline.

## File format

Each `.h5` file represents one model with 10,000 points and the following datasets:

| Key       | Shape       | Dtype   | Meaning                                      |
|-----------|-------------|---------|----------------------------------------------|
| `points`  | `(N, 3)`    | float64 | XYZ point coordinates (already normalized)   |
| `normals` | `(N, 3)`    | float64 | Per-point unit normals                       |
| `labels`  | `(N,)`      | int16   | Per-point surface instance ID                |
| `prim`    | `(N,)`      | int16   | Per-point primitive type code (see below)    |
| `T_param` | `(N, 22)`   | float64 | Per-point fitted primitive parameters        |

`N = 10000` for all models. Each cluster (set of points sharing the same `labels`
value) has homogeneous `prim` — i.e., all points in a cluster share one primitive type.

The point cloud has already been **PCA-normalized** by ParseNet's preprocessing
(minor axis aligned to x, scaled so the maximum extent is 1).

## Primitive type codes (`prim`)

The primitive type codes are documented in ParseNet's `readme_data.md`
(https://github.com/Hippogriff/parsenet-codebase/blob/master/readme_data.md) and
match HPNet's `process_abc.py` fitting switch. Note: the encoding is **many-to-one**.

| Code(s)         | Primitive type                |
|-----------------|-------------------------------|
| `1`             | Plane                         |
| `3`             | Cone                          |
| `4`             | Cylinder                      |
| `5`             | Sphere                        |
| `0, 6, 7, 9`    | Closed B-spline               |
| `2, 8`          | Open B-spline                 |

The original ABC primitive set has 10 types (circle, sphere, plane, cone, cylinder,
open spline, closed spline, **revolution, extrusion, extra**). ParseNet collapses the
last three into the B-spline categories because B-splines can approximate them, and
excluding shapes containing them would have shrunk the dataset significantly.

### Important consequence: visual ≠ encoded type

Many models in ABC Parts contain surfaces that **look like cylinders or planes
visually but are encoded as B-splines** (e.g. model 00470 has only `prim ∈ {2, 7}`,
both spline codes, despite being visually composed of planes and cylinders).

This happens because in the source ABC CAD file these surfaces were authored as:

- **Surface of Revolution** → mapped to closed B-spline by ParseNet
- **Surface of Extrusion** → mapped to closed B-spline
- **Trimmed B-spline patches** that happen to be flat or cylindrical
- **`extra` types** (rare CAD primitives like ruled, sweep, offset) → mapped to spline

Once relabeled as splines, the dataset preserves no information distinguishing a
"truly freeform" B-spline from a "geometrically primitive" one. To recover the
underlying geometric type, you must run a classifier on the points + normals
yourself (PCA-based plane/cylinder/sphere detection is one option). The dataset
labels alone are not sufficient.

## `T_param` layout

`T_param` is a per-point 22-dimensional vector where the slot range depends on the
primitive type. Only **analytical types** (1, 3, 4, 5) have non-zero parameters;
spline types leave `T_param` entirely zero (their parameters live in HPNet's
neural B-spline decoder, not in the dataset).

| Slot     | Primitive    | Layout                                  |
|----------|--------------|-----------------------------------------|
| `[0:4]`  | Sphere (5)   | center(3) + radius(1)                   |
| `[4:8]`  | Plane (1)    | normal(3) + offset(1)                   |
| `[8:15]` | Cylinder (4) | axis(3) + center(3) + radius(1)         |
| `[15:22]`| Cone (3)     | axis(3) + center(3) + half_angle(1)     |

All values are repeated identically across every point in a cluster. Empty slots
(belonging to other primitive types or to skipped/spline clusters) are zero.

Small clusters (`< 100` points) are skipped from fitting in HPNet's `process_abc.py`,
but their `prim` codes are still written. This means a cluster can have `prim=1`
(Plane) and yet have all-zero `T_param` if it was below the fitting threshold.

## File organization

```
ABC_final/
  00000.h5
  00001.h5
  ...
  00031999.h5      (~32k models)
  train_data.txt   (training split)
  val_data.txt     (validation split)
  test_data.txt    (test split)
```

The `.txt` files are split lists used by HPNet's training pipeline. They are
irrelevant for evaluation-only use.

Each `.h5` file is a **single model** (single part in our terminology). There is no
multi-model packing — the original ParseNet `train_data.h5` was monolithic, but
HPNet's `process_abc.py` decomposes it into per-model files.

## Conversion to brep_pipeline format

`scripts/convert_abc_parts.py` reads the h5 files and writes:

```
sample_clouds_abc_parts/
  00000/
    0.xyzc          (points + cluster IDs, brep_pipeline format)
    metadata.npz    (normals, prim, T_param)
    info.json       (cluster stats and primitive summary)
  00001/
    ...
```

The `info.json` reports per-cluster sizes and primitive types **as labeled by the
dataset**. Because of the spline-encoding issue, the `primitive_summary` field is
not always a faithful description of the geometric content — a model dominated by
"Spline" entries may visually contain analytical primitives.

## Resolution caveat

ABC Parts has a fixed `N = 10000` points per model. With ~10–20 clusters per model,
each cluster has only ~500–1000 points. This is **substantially sparser** than
re-sampling from the original ABC STEP files (where you typically use 50k+ points
per part with `abc_preprocess.py`). Low cluster density may affect downstream
operations that rely on density (k-NN distances for fitness scoring, intersection
vertex detection, mesh trimming).

## References

- ParseNet (Sharma et al., 2020): https://github.com/Hippogriff/parsenet-codebase
- ParseNet readme_data.md: https://github.com/Hippogriff/parsenet-codebase/blob/master/readme_data.md
- HPNet (Yan et al., 2021): https://github.com/SimingYan/HPNet
- HPNet `process_abc.py`: https://github.com/SimingYan/HPNet/blob/main/utils/process_abc.py
- HPNet `ABCDataset.py`: https://github.com/SimingYan/HPNet/blob/main/dataloader/ABCDataset.py
