# Evaluation: canonical frame for mine-vs-P2CAD comparison

**Status:** design only, not yet implemented.

## Goal

For each ABC model, compute a fairness-preserving distance between
- input point cloud $\leftrightarrow$ my reconstructed mesh
- input point cloud $\leftrightarrow$ original Point2CAD reconstructed mesh

The two sets of distances must be in the **same coordinate frame**, scale-invariant per-model so values are comparable across models of different physical size.

## Why not evaluate in either pipeline's PCA-normalized space

Both `mesh_pipeline.normalize_points` and `point2cad/utils.normalize_points` apply the same algorithm:

1. center: $p \leftarrow p - \bar{p}$
2. PCA on the covariance, pick smallest eigenvector $v$
3. rotate so $v$ maps to $+x$
4. uniform scale by $1 / (\max\text{extent} + \varepsilon)$

The math is identical, but the numerical PCA implementation differs:
- mine: `np.linalg.eigh(X^T X)`
- theirs: `np.linalg.eig(X^T X)`

Eigenvectors are defined up to sign; `eigh` and `eig` are not guaranteed to return the same sign for the smallest eigenvector. A flipped sign of $v$ gives `rotation_matrix_a_to_b(-v, +x)`, which is a *different* rotation. So the two pipelines may produce normalized clouds related by a 180° rotation (or improper rotation), even on bit-identical input. Picking either pipeline's normalization silently privileges its PCA orientation.

## Why not per-axis $[0, 1]$ normalization

$x_{\text{new}} = (x - x_{\min}) / (x_{\max} - x_{\min})$ is **anisotropic**. A long thin part is stretched into a unit cube; a 1 mm error along the short axis is rescaled differently than a 1 mm error along the long axis. Chamfer/Hausdorff after that are no longer Euclidean distances — they're distances in a non-uniformly stretched coordinate system. The metric loses geometric meaning, and the per-axis scale factor varies between models so cross-model comparison breaks down too.

## Recipe — uniform scale by input bounding-box diagonal, no translation

For each model, compute one scalar from the **input point cloud** (which already lives in world space):

$$ s = \frac{1}{\| \max(\text{cloud}) - \min(\text{cloud}) \|_2} $$

Apply this same $s$ to all three world-space inputs (no centering — chamfer/Hausdorff are translation-invariant when the same translation is applied to every set, so subtracting the mean would not affect the metric):

```python
s = 1.0 / np.linalg.norm(cloud.max(0) - cloud.min(0))

cloud_eval      = cloud * s
my_mesh_eval    = my_world_verts * s
their_mesh_eval = their_world_verts * s
```

Compute chamfer / Hausdorff / whatever distance directly on these scaled coordinates. The number is in **dimensionless units of input bbox diagonal**, comparable across models. Neither pipeline's normalization choices affect the result.

## Where to read world-space outputs

**Mine (`mesh_pipeline.py`):**
- `unified/trimmed.stl` — already world-space (denormalized by `_merge_part_dirs`)
- `unified/trimmed_mesh_{i}.npz` — already world-space
- Per-part `part_X/trimmed_mesh_{cid}.npz` — *normalized*; denorm via `_denorm_points` using `part_X/metadata.npz` (`norm_mean`, `norm_R`, `norm_scale`)

**Original P2CAD (modified clone, `../point2cad`):**
- `output_p2cad_orig/{model_id}/part_X/clipped/cluster_*.ply` — *normalized*; denorm via `_denorm_points` using `part_X/normalization.npz`
- The visualizer's `_load_orig` in `mesh_pipeline.py` already implements this loop — reuse that pattern.

## Multi-part handling

ABC parts dataset is mostly single-part, but multi-part is possible. The protocol generalizes naturally:

1. For each model part $p$:
   - load my mesh for part $p$, denorm to world space
   - load their mesh for part $p$, denorm to world space
2. Concatenate per-part vertices into a single world-space vertex array on each side
3. Apply the canonical scaling $s$ (from the *full* model's input cloud, not per part)
4. Compute the metric

For single-part models this simplifies to "read `unified/trimmed.stl` for mine, denorm `clipped/cluster_*.ply` for theirs."

## Notes / open questions

- Bbox diagonal vs max-extent: stylistic choice (literature usually uses diagonal; `mesh_pipeline` uses max-extent internally). Either works as long as the same scalar is applied to all three sets.
- Standard CAD-reconstruction metrics to consider:
  - chamfer distance (symmetric, $L_2$)
  - Hausdorff distance (worst-case)
  - F-score @ threshold $\tau$ (precision/recall of points within $\tau$ of the other set)
- Sample density: when computing point-to-mesh distances, sample $N$ points uniformly on the mesh surface (e.g., trimesh's `sample.sample_surface`) — closed-form point-to-triangle distance for every input point is fine for input cloud sizes around $10^4$.
