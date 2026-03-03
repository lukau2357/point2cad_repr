# ABC Dataset Format Reference

## Overview

The ABC dataset contains CAD models sourced from Onshape. Each 10k chunk is organized into
subdirectories by file type. The practical model count depends on the format:

| Format | Models available | Notes |
|--------|-----------------|-------|
| STEP   | **10,000** | Full B-Rep, all models in the chunk |
| OBJ    | ~7,168 | Tessellated mesh, subset only |
| feat   | ~7,168 | Feature annotations, same subset |
| meta   | ~10,000 | Onshape document metadata |

The previously noted "10k vs 7.1k discrepancy" only applies to OBJ + feat. STEP files exist
for the full 10,000 models per batch and are the canonical ground truth for point cloud
generation and evaluation.

## Directory Structure

```
abc_dataset/
  abc_0000_step_v00/    10,000 STEP files (full CAD B-Rep, all models)
  abc_0000_stat_v00/     7,168 YAML statistics files
  obj/                   7,168 .obj files (triangle meshes, tessellated from CAD)
  feat/                  7,168 .yml files (surface/curve feature annotations, mesh-indexed)
  meta/                 ~10,000 .yml files (Onshape document metadata)
  obj_v00.txt                  archive download manifest
  meta_v00.txt                 archive download manifest
```

Each subdirectory holds up to 10,000 numbered sub-directories (00000000–00009999),
one per CAD model variant.

## File Naming Convention

```
[8-digit_ID]_[32-char_HASH]_[type]_[variant].[ext]
```

- **8-digit ID**: model index (00000000–00009999)
- **32-char HASH**: Onshape document content hash
- **type**: `step`, `trimesh`, `features`, `metadata`, `stat`
- **variant**: integer variant index (000, 001, …)
- **ext**: `.step`, `.obj`, or `.yml`

Example: `00000002_1ffb81a71e5b402e966b9341_trimesh_001.obj`

## STEP Files (canonical ground truth B-Rep)

STEP files contain the full exact B-Rep: analytical surfaces (planes, cylinders, cones,
spheres, tori) as OCC `Geom_Surface` objects, plus spline surfaces for freeform patches,
all bounded by exact edge and vertex topology.

Key properties:
- Available for all 10,000 models per batch
- Each `TopoDS_Face` corresponds to one surface patch with a known analytical type
- Face topology (wires, edges, vertices) is fully encoded

Loading with pythonocc:
```python
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp      import TopExp_Explorer
from OCC.Core.TopAbs      import TopAbs_FACE
from OCC.Core             import topods
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface

reader = STEPControl_Reader()
reader.ReadFile(step_path)
reader.TransferRoots()
shape = reader.OneShape()

exp = TopExp_Explorer(shape, TopAbs_FACE)
while exp.More():
    face    = topods.Face(exp.Current())
    adaptor = BRepAdaptor_Surface(face)
    stype   = adaptor.GetType()   # GeomAbs_Plane, GeomAbs_Cylinder, etc.
    exp.Next()
```

## OBJ Files

Standard ASCII OBJ format with vertices, normals, and triangular faces.
No texture coordinates.

```
v  x y z          # vertex position
vn x y z          # vertex normal
f  v1//vn1 v2//vn2 v3//vn3   # triangle (vertex//normal, 1-indexed)
```

Typical sizes: 18k–114k vertices, 37k–228k faces.

## FEAT (Features) Files

YAML files with `curves` and `surfaces` keys. **Mesh-indexed** — references face and
vertex indices from the corresponding OBJ, not the STEP geometry directly.

### Surfaces

```yaml
surfaces:
  - type: Cylinder
    location: [x, y, z]
    z_axis: [dx, dy, dz]
    radius: r
    face_indices: [0, 1, ...]   # 0-indexed into OBJ faces
    vert_indices: [0, 1, ...]
    vert_parameters: [...]      # per-vertex UV in the surface's parametric domain
```

Surface type frequencies across batch 0:

| Type      | Count   |
|-----------|---------|
| Plane     | 321,704 |
| Cylinder  | 155,423 |
| Torus     |  21,460 |
| Cone      |  20,054 |
| Extrusion |  10,396 |
| Sphere    |   7,376 |
| Other     |     481 |

### Curves

```yaml
curves:
  - type: Line
    sharp: true
    vert_indices: [...]
    vert_parameters: [...]
    location: [x, y, z]
    direction: [dx, dy, dz]
```

Curve type frequencies:

| Type       | Count   |
|------------|---------|
| Line       | 836,566 |
| Circle     | 368,907 |
| BSpline    | 229,118 |
| Ellipse    |  28,733 |
| Revolution |     559 |

## Point Cloud Generation: OBJ vs STEP

### OBJ + feat sampler (`--sampler obj`, ~7.1k models)

Area-weighted barycentric sampling on the tessellated mesh. Surface labels from
`feat.surfaces[i].face_indices`.

**Limitations:**
1. *Tessellation error* — OBJ triangles are planar approximations of curved surfaces.
   For a cylinder of radius r with N facets, max deviation ≈ r(1 − cos(π/N)).
2. *Non-uniform density* — CAD tessellators refine near high curvature; the mesh area
   of curved patches is slightly smaller than the true analytical area.
3. *Boundary label noise* — triangles straddling two surface patches may receive the
   wrong label.
4. *Reduced coverage* — only ~7,168 models have both OBJ and feat files.

### STEP sampler (`--sampler step`, default, 10k models)

Loads the exact B-Rep via OCC. Each `TopoDS_Face` is one cluster (cluster ID = face
index in `TopExp_Explorer` traversal order). For each face a UV grid is evaluated on
the exact `Geom_Surface`, decomposed into triangles, and points sampled area-weighted
from those triangles.

**Advantages:**
- Points lie exactly on the analytical surface — no tessellation bias
- Available for all 10,000 models per batch
- Labels come directly from B-Rep face identity

**Trade-off:** Requires pythonocc (run inside Docker). The UV-grid triangulation
approximates the area element; increasing `--grid_res` (default 50) improves accuracy
at the cost of speed.

**One-face = one-cluster assumption:** A single analytical surface can theoretically
appear as multiple `TopoDS_Face` entries in the B-Rep (e.g., a cylinder split at its
seam). In practice, Parasolid-based STEP files (the source of ABC) represent each
primitive surface as a single face. Even when splitting does occur, each face is still
a coherent patch of the same analytical type, so the fitting step is unaffected — at
worst the model is slightly over-segmented.

### Part-level sampling (default)

Many ABC STEP files are `TopAbs_COMPOUND` shapes containing multiple independent
solid parts (e.g. a row of 10 identical cylinder+plane assemblies).  Sampling all
faces globally yields 60 clusters instead of ~6, causing numerical instability
and poor BRep reconstruction.

By default `abc_preprocess.py` decomposes the top-level shape using
`TopoDS_Iterator` (immediate children only, not recursive), processes each part
independently with the existing area-weighted sampler, and saves one `.xyzc` file
per part.  A single analytical surface that spans multiple parts is not possible
in a well-formed B-Rep, so part-level decomposition is always safe.

**Output layout** (default, `--by_part` active):
```
output_dir/
  abc_{model_id}/
    part_000.xyzc
    part_001.xyzc
    ...
stats_dir/
  abc_{model_id}/
    part_000_stats.json
    part_001_stats.json
    ...
```

**Output layout** (`--no_by_part`, whole-model):
```
sample_clouds/
  abc_{model_id}.xyzc
sample_clouds_stats/
  abc_{model_id}_stats.json
```

**`--num_points`** is applied per part (not divided among parts), so each
part gets a dense-enough point cloud regardless of its relative size.

### Running the STEP sampler

```bash
# Single model
python scripts/abc_preprocess.py \
    --abc_dir /path/to/abc_dataset \
    --model_id 00000042 \
    --sampler step \
    --output_dir output/point_clouds

# Batch (all 10k models)
python scripts/abc_preprocess.py \
    --abc_dir /path/to/abc_dataset \
    --sampler step \
    --batch \
    --output_dir output/point_clouds
```
