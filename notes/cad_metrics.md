# Pipeline fidelity metrics (point2cad_repr vs original Point2CAD)

Final evaluation suite for comparing the two mesh-generation pipelines on ABC parts. **Mesh-only**: both pipelines produce per-cluster clipped meshes; all metrics consume those meshes plus the segmented input cloud. No ground-truth STEP, no T_param sampling.

## Why mesh-only

ABC parts dataset has no easy mapping back to original ABC IDs, so GT meshes can't be retrieved for resampling. T_param-based analytical sampling was rejected: it adds complexity, requires per-primitive region/extent decisions, and is not truthful to what the methods actually produce. Both pipelines are mesh generators — evaluating the meshes they produce is the principled choice.

All metrics computed in **normalized space** (per-part PCA + unit cube), with the **same per-part normalization** applied to both pipelines so the numbers are directly comparable.

---

## 1. P-coverage (global, ParseNet / Point2CAD style)

$$P_{\text{cov}} = \frac{1}{|P|} \sum_{k=1}^{|P|} \mathbb{I}\!\left[d(p_k,\, S^r) \leq r\right], \qquad r = 0.01$$

- $P$ = input point cloud (all clusters merged).
- $S^r$ = **union of all clipped reconstructed meshes** — label-free.
- $d(p_k, S^r) = \min_{T \in F_{\text{union}}} d(p_k, T)$, exact point-to-triangle.
- Computed via `igl.point_mesh_squared_distance(input_pts, V_union, F_union)`, take square root, threshold at $r$.

### Why global, not per-cluster

ParseNet's formulation is unambiguous: $\mathbb{I}[\min_{k=1..K} D(p_i, s_k) < \varepsilon]$ — explicit minimum over all $K$ fitted surfaces, i.e. distance to the union. Point2CAD follows ParseNet but drops the index in the formula; the prose ("$S$ is all CAD reconstruction surfaces") confirms the global reading. A per-cluster variant was considered and rejected: it would be nearly redundant with per-cluster residual error (same correspondence, just thresholded vs averaged) and would collapse the only label-free signal in the suite.

### What it catches

Under-coverage of the input cloud — regions where no reconstructed surface comes within $r$. Robust to label/cluster permutations because labels are not used.

---

## 2. Per-cluster residual error

$$\text{Err}_i = \frac{1}{N_i} \sum_{s_{i,k} \in S_i^{gt}} d(s_{i,k},\, S_i^r), \qquad \overline{\text{Err}} = \frac{1}{K}\sum_{i=1}^{K} \text{Err}_i$$

- $S_i^{gt}$ proxied by **input cloud points labeled $i$** (no GT mesh available).
- $S_i^r$ = the reconstructed mesh for cluster $i$ only — **not** the union.
- $N_i$ = number of input points in cluster $i$.
- Final scalar: per-cluster mean, then averaged uniformly over clusters (not weighted by point count).

### Honest disclosure

The paper's residual error assumes $\{s_{i,k}\}$ are drawn from a known ground-truth surface. Without GT, we proxy with the segmented input points. This must be stated explicitly when reporting:

> "Residual error computed with GT samples proxied by segmented input points, since no GT mesh is available for ABC parts."

### What it catches

Per-surface fit quality and label/clipping mismatch. This is the diagnostic metric — it tells you *which* surfaces fit poorly, which the global metrics cannot. Both pipelines preserve cluster IDs from the input file, so the per-cluster correspondence is well-defined and apples-to-apples.

---

## 3. Symmetric Chamfer distance

$$\text{CD}_{\text{sym}}(P, M) \;=\; \frac{1}{2}\!\left( \frac{1}{|P|}\sum_{p \in P} d(p, M) \;+\; \frac{1}{N}\sum_{i=1}^{N} \min_{p \in P} \|q_i - p\|_2 \right)$$

where $M$ = union of clipped meshes, $\{q_i\}_{i=1}^{N}$ are uniform area-weighted samples from $M$.

### Two terms, two computations

**PC → Mesh** (left term): exact, no mesh sampling needed.

$$\frac{1}{|P|}\sum_{p \in P} d(p, M) \quad\text{via}\quad \texttt{igl.point\_mesh\_squared\_distance}$$

This direction is closed-form because point-to-triangle distance has a closed form (see below). `igl` accelerates the per-triangle search with an AABB tree.

**Mesh → PC** (right term): Monte Carlo approximation of the surface integral.

$$\frac{1}{\text{Area}(M)} \int_M \min_{p \in P} \|q - p\|_2 \, dq \;\approx\; \frac{1}{N}\sum_{i=1}^{N} \min_{p \in P} \|q_i - p\|_2$$

This integral has no closed form because $\min_{p \in P}\|q - p\|$ is piecewise-defined (Voronoi cells of $P$). Approximated by sampling $M$ uniformly via `trimesh.sample.sample_surface_even(M, N)` (Poisson-disk variant, lower variance than plain area-weighted) and KD-tree nearest neighbor in $P$.

### Sample count

$N = \max(|P|,\, 100\,000)$, **same $N$ for both pipelines**. Variance of the Monte Carlo estimator is $O(1/N)$; in practice doubling $N$ should not change the reported number meaningfully.

### What it catches

Both failure modes — the PC→Mesh term penalizes under-coverage (regions of input cloud not reached by reconstruction), the Mesh→PC term penalizes over-extension (spurious geometry the clipping didn't remove). Together they're strictly more informative than P-coverage alone (which collapses the PC→Mesh distance into a binary threshold).

---

## Mathematical foundation: point-to-mesh distance

For a triangle mesh $M = (V, F)$ viewed as a 2-manifold subset of $\mathbb{R}^3$:

$$d(p, M) = \inf_{q \in M} \|p - q\|_2 = \min_{T \in F} d(p, T)$$

The infimum is achieved (mesh is closed) and reduces to a minimum over triangles. **Point-to-triangle distance** $d(p, T)$ is computed in closed form:

1. Project $p$ onto the triangle's plane $\to p'$.
2. Compute barycentric coordinates of $p'$ w.r.t. $T$.
3. If all barycentrics $\in [0,1]$, the projection lies inside the triangle and $d(p,T) = \|p - p'\|$.
4. Otherwise the closest point is on an edge or vertex; clamp the barycentrics and recompute.

`igl.point_mesh_squared_distance` does this with an AABB tree so the per-query cost is logarithmic in $|F|$ rather than linear. **Exact, no discretization error** — the only place sampling enters the metric suite is the Mesh→PC chamfer direction, and that's a Monte Carlo integral over the surface, not a discretization of the distance function.

---

## Why these three (and not others)

| Metric | Label-aware? | Continuous? | Direction | Catches |
|---|---|---|---|---|
| P-coverage | No (union) | No (binary) | PC → mesh | Under-coverage of input cloud |
| Per-cluster residual | Yes | Yes | PC → cluster mesh | Per-surface fit quality, mis-clipping |
| Symmetric chamfer | No (union) | Yes | Both | Under- and over-extension of reconstruction |

Each catches a failure the others miss. Two of the three (P-coverage, residual error) are paper-comparable in *definition*; absolute numbers won't match prior work because of normalization and sampling-density differences, but the relative ordering between the two pipelines is what matters for this comparison.

Discarded alternatives:

- **Per-cluster P-coverage** — nearly redundant with per-cluster residual error (same correspondence, thresholded vs averaged) and collapses the only label-free signal in the suite.
- **T_param analytical sampling** — adds complexity, requires per-primitive region decisions, and isn't truthful to what mesh-generation methods produce.
- **GT-mesh Chamfer** — impossible without ABC ID mapping.
- **Volumetric IoU, Hausdorff, normal consistency, F-score @ τ** — listed in older drafts of this note but not adopted; the three above already span the relevant failure modes for this comparison.

---

## Reporting checklist

- All 3 metrics computed in normalized space (unit cube), same per-part normalization on both pipelines.
- State $r = 0.01$ for P-coverage.
- State $N$ (mesh sample count) for chamfer.
- For residual error, disclose GT-proxy = segmented input points.
- Report per-part numbers and dataset averages.
- Symmetric chamfer reported as the average of the two directions (state explicitly that it's the average, not the sum).
