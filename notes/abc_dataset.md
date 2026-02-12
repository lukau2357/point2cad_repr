# ABC Dataset Format Reference

## Overview

The ABC dataset contains CAD models sourced from Onshape. Each 10k chunk is organized into
three subdirectories: `obj/`, `feat/`, and `meta/`. Despite the "10k models per chunk" claim,
only ~7,168 models have complete data (OBJ + features). The remaining ~2,832 directories
either contain metadata only (e.g., directories 00000000-00000001) or are empty.

## Directory Structure

```
abc_dataset/
  obj/          7,168 .obj files (triangle meshes)
  feat/         7,168 .yml files (geometric features)
  meta/        10,001 .yml files (Onshape document metadata)
  obj_v00.txt          archive download URLs
  meta_v00.txt         archive download URLs
```

Each subdirectory contains up to 10,000 numbered directories (00000000 through 00009999),
with each directory holding files for one CAD model variant.

## File Naming Convention

All files follow the pattern:

```
[8-digit_ID]_[32-char_HASH]_[type]_[variant].[ext]
```

- **8-digit ID**: model index (00000000 - 00009999)
- **32-char HASH**: Onshape document content hash
- **type**: `trimesh` (obj), `features` (feat), `metadata` (meta)
- **variant**: integer variant index (000, 001, ...)
- **ext**: `.obj` or `.yml`

Example: `00000002_1ffb81a71e5b402e966b9341_trimesh_001.obj`

## OBJ Files

Standard ASCII OBJ format with vertices, normals, and triangular faces.
No texture coordinates are present.

### Structure

```
v  x y z          # vertex position
vn x y z          # vertex normal
f  v1//vn1 v2//vn2 v3//vn3   # triangular face (vertex//normal, 1-indexed)
```

Face format is `f v//vn` — vertex index paired with normal index, no texture coordinate slot.

### Typical Sizes

| Metric   | Range (from samples) |
|----------|---------------------|
| Vertices | 18k - 114k          |
| Normals  | 22k - 120k          |
| Faces    | 37k - 228k          |

Normal count slightly exceeds vertex count because different faces sharing
a vertex can have distinct normals (hard edges). Face count is roughly 2x vertex count.

## FEAT (Features) Files

YAML files with two top-level keys: `curves` and `surfaces`.

### Surfaces

Each entry describes a geometric surface patch and the mesh faces belonging to it.

```yaml
surfaces:
  - type: Plane              # surface type
    location: [x, y, z]      # reference point
    face_indices: [0, 1, ...]  # 0-indexed into OBJ faces
    vert_indices: [0, 1, ...]  # 0-indexed into OBJ vertices
    vert_parameters: [...]     # per-vertex 2D parametrization
    # additional fields depending on type (e.g., radius for Cylinder)
```

**Surface types and their frequency across the full chunk:**

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

Each entry describes a boundary or intersection curve on the model.

```yaml
curves:
  - type: Line               # curve type
    sharp: true               # whether the edge is sharp
    vert_indices: [0, 1, ...] # 0-indexed into OBJ vertices
    vert_parameters: [...]    # curve parameterization values
    location: [x, y, z]
    direction: [dx, dy, dz]   # Line-specific
    # additional fields depending on type (e.g., radius for Circle, knots/poles for BSpline)
```

**Curve types and their frequency:**

| Type       | Count   |
|------------|---------|
| Line       | 836,566 |
| Circle     | 368,907 |
| BSpline    | 229,118 |
| Ellipse    |  28,733 |
| Revolution |     559 |

## META (Metadata) Files

YAML files containing Onshape document metadata. Not needed for point cloud generation,
but useful for traceability.

Key fields: `id`, `name`, `description`, `createdAt`, `modifiedAt`, `createdBy`, `owner`,
`permission`, `public`, `tags`, `thumbnail`.

## The 10k vs 7.1k Discrepancy

The dataset advertises 10k models per chunk, but only 7,168 have both OBJ and features files.

| Directory      | obj | feat | meta |
|----------------|-----|------|------|
| 00000000-00000001 | no  | no   | yes  |
| 00000002-00009999 | 7,168 | 7,168 | 9,999 |

The 10k count refers to all Onshape documents in the chunk. Some documents failed geometry
extraction (no OBJ produced) or feature annotation (no FEAT produced). The practical count
for Point2CAD preprocessing is the intersection: **7,168 models** with both mesh geometry
and surface/curve annotations.

## Relevance to Point2CAD

For generating segmented point clouds (`.xyzc` format):

1. Load OBJ to get vertices and triangular faces
2. Load corresponding FEAT `.yml` to get `surfaces[i].face_indices` (0-indexed)
3. Build a face-to-surface mapping from `face_indices`
4. Sample points via area-weighted barycentric sampling, inheriting the surface label
5. Export as `.xyzc` (x, y, z, cluster_id)

Surface types Plane, Cylinder, Cone, and Sphere map directly to primitive fitters.
Torus, Extrusion, BSpline, and Other require INR fitting.
