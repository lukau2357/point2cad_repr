# Point Cloud Normalization in Point2CAD

## Where It Is Applied

The original Point2CAD applies a **global normalization** to the entire point cloud before any surface fitting.

**Call site:** `point2cad/main.py:79`
```python
points = normalize_points(points)
```

This happens after loading the `.xyzc` file (line 71) and before the fitting loop (`fn_process` at line 87). Every subsequent operation -- primitive fitting, INR fitting, mesh generation, and clipping -- operates on normalized points.

**Implementation:** `point2cad/utils.py:47-63`

## The Algorithm

Given input point cloud $P \in \mathbb{R}^{N \times 3}$:

### Step 1: Center at origin

$$P' = P - \bar{P}, \quad \bar{P} = \frac{1}{N}\sum_{i=1}^{N} P_i$$

### Step 2: PCA rotation

Compute the covariance-like matrix $C = P'^T P' \in \mathbb{R}^{3 \times 3}$ and its eigendecomposition:

$$C = U \Lambda U^T$$

where $\Lambda = \text{diag}(\lambda_1, \lambda_2, \lambda_3)$ are eigenvalues and $U = [u_1, u_2, u_3]$ the corresponding eigenvectors.

Find the smallest eigenvalue: $u_{\min} = u_{\arg\min_i \lambda_i}$.

Compute rotation matrix $R$ such that $R \cdot u_{\min} = [1, 0, 0]^T$ (align the minor principal axis with the x-axis), using the Rodrigues-like formula in `rotation_matrix_a_to_b` (`utils.py:71`).

Apply rotation:

$$P'' = (R \cdot P'^T)^T$$

**Geometric interpretation:** The direction of least variance (thinnest dimension of the point cloud) is aligned with the x-axis. For a flat CAD part, this would be the thickness direction.

### Step 3: Isotropic scaling

Compute the axis-aligned bounding box extents:

$$s = \max(P'', \text{axis}=0) - \min(P'', \text{axis}=0) \in \mathbb{R}^3$$

Scale by the maximum extent (isotropic mode, which is the default -- `anisotropic=False`):

$$P_{\text{norm}} = \frac{P''}{\max(s) + \epsilon}$$

This places the point cloud roughly in $[-0.5, 0.5]^3$, preserving the aspect ratio between axes.

## Comparison with INR-Specific Preprocessing

The INR fitting in both the original and the reproduction applies a **per-cluster** normalization:

**Original** (`fitting_one_surface.py:485-493`):
```python
points_mean = points.mean(dim=0)
points_scale = points.std(dim=0).max()
points = (points - points_mean) / points_scale
```

**Reproduction** (`inr_fitting.py:349-356`):
```python
cluster_mean = cluster.mean(axis=0)
cluster_scale = cluster_std.max()
cluster = (cluster - cluster_mean) / (cluster_scale + 1e-6)
```

These are equivalent: subtract the cluster mean, divide by the max standard deviation across axes.

**Key difference:** In the original pipeline, this INR normalization is applied **on top of** the global normalization. The INR receives already-normalized points and further normalizes per-cluster. In the reproduction, the INR normalization is the **only** normalization.

## Impact on the Pipeline

### What global normalization affects:

| Pipeline Stage | Effect |
|---|---|
| **Primitive fitting** | Numerical conditioning of least-squares solves. Fitting a plane/cylinder/sphere to points in $[-0.5, 0.5]$ vs arbitrary coordinates. |
| **Mesh generation** | Grid sampling bounds, threshold parameters (e.g., `threshold_multiplier` for grid trimming) are scale-dependent. |
| **Self-intersection resolution** | PyMesh's `resolve_self_intersection` uses tolerance-based geometric comparisons that are sensitive to scale. |
| **Proximity queries** | `trimesh.proximity.closest_point` distances are in the coordinate space of the mesh. Filtering thresholds (area_per_point) are implicitly tuned for normalized coordinates. |
| **Vertex deduplication** | `pymesh.remove_duplicated_vertices(tol=1e-6)` -- a fixed tolerance that assumes a specific scale. |

### Why the reproduction might differ without it:

1. The `tol=1e-6` in `remove_duplicated_vertices` may be too small or too large for raw coordinates.
2. The `area_multiplier=2.0` threshold in area-per-point filtering was likely tuned for normalized-scale meshes.
3. Grid trimming `threshold_multiplier` values may not generalize across different point cloud scales.

## Design Decision: To Normalize or Not

**Arguments for normalization (original approach):**
- All hardcoded tolerances (`1e-6` vertex dedup, `area_multiplier=2.0`) work correctly.
- Numerical stability for fitting and mesh operations.
- Consistent behavior across point clouds of different scales.

**Arguments against (current reproduction approach):**
- Preserves original physical dimensions -- important for engineering pipelines where metric accuracy matters.
- Avoids a non-trivial coordinate transformation that adds complexity.
- The PCA rotation is data-dependent and non-deterministic (eigenvector sign ambiguity), which could affect reproducibility.

**Possible compromise:**
Apply an isotropic scale-only normalization (no rotation) to bring points into a numerically convenient range, and store the transformation parameters to invert it after the pipeline completes. This preserves relative geometry and metric proportions while giving the algorithm well-conditioned numerics.
