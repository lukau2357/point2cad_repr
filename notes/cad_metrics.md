# Pipeline fidelity metrics (point2cad_repr vs original Point2CAD)

Final evaluation suite for comparing the two mesh-generation pipelines on ABC parts. **Mesh-only**: both pipelines produce per-cluster clipped meshes; all metrics consume those meshes plus the segmented input cloud. No ground-truth STEP, no T_param sampling.

## Why mesh-only

ABC parts dataset has no easy mapping back to original ABC IDs, so GT meshes can't be retrieved for resampling. T_param-based analytical sampling was rejected: it adds complexity, requires per-primitive region/extent decisions, and is not truthful to what the methods actually produce. Both pipelines are mesh generators — evaluating the meshes they produce is the principled choice.

All metrics computed in **normalized space** (per-part PCA + unit cube), with the **same per-part normalization** applied to both pipelines so the numbers are directly comparable.

---

## 1. P-coverage (both directions, ParseNet / Point2CAD style + reverse)

Two thresholded directions at the same radius $r = 0.01$, computed in normalized space.

### 1a. PC → Mesh (standard)

$$P^{\text{p2m}}_{\text{cov}} = \frac{1}{|P|} \sum_{k=1}^{|P|} \mathbb{I}\!\left[d(p_k,\, M) \leq r\right]$$

- $P$ = input point cloud (all clusters merged).
- $M$ = **union of all clipped reconstructed meshes** — label-free.
- $d(p_k, M) = \min_{T \in F_M} d(p_k, T)$, exact point-to-triangle distance.
- Computed via `trimesh.proximity.closest_point(M, P)` which uses an AABB-accelerated per-triangle closest-point query and returns Euclidean (not squared) distances directly — thresholded at $r$ without an extra square-root step.

### 1b. Mesh → PC (reverse — hallucination rate)

$$P^{\text{m2p}}_{\text{cov}} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\!\left[\min_{p \in P} \|q_i - p\|_2 \leq r\right]$$

- $\{q_i\}_{i=1}^N$ = uniform-area samples from $M$ (same sample set used for the Mesh→PC Chamfer term, §3).
- Interpretation: fraction of mesh surface within $r$ of any input point. The complement $1 - P^{\text{m2p}}_{\text{cov}}$ is the share of mesh area that lives far from the input cloud — spurious lobes / overfitting bumps.
- Not in the ParseNet / Point2CAD formulation; added here because the standard (PC→Mesh) direction rewards meshes that overfit to noisy input. On 40024 Point2CAD scores $P^{\text{p2m}}_{\text{cov}} \approx 0.999$ but $P^{\text{m2p}}_{\text{cov}} \approx 0.517$ — almost half its mesh surface does not correspond to any input point.

### Why global, not per-cluster

ParseNet's formulation is unambiguous: $\mathbb{I}[\min_{k=1..K} D(p_i, s_k) < \varepsilon]$ — explicit minimum over all $K$ fitted surfaces, i.e. distance to the union. Point2CAD follows ParseNet but drops the index in the formula; the prose ("$S$ is all CAD reconstruction surfaces") confirms the global reading. A per-cluster variant was considered and rejected: it would be nearly redundant with per-cluster residual error (same correspondence, just thresholded vs averaged) and would collapse the only label-free signal in the suite.

### What each catches

- **PC→Mesh**: under-coverage — input regions no reconstructed surface reached.
- **Mesh→PC**: over-extension / hallucination — mesh area that doesn't correspond to any input point.

Robust to label/cluster permutations because labels are not used.

---

## 2. Per-cluster residual error

$$\text{Err}_i = \frac{1}{N_i} \sum_{s_{i,k} \in S_i^{gt}} d(s_{i,k},\, S_i^r), \qquad \overline{\text{Err}} = \frac{1}{K}\sum_{i=1}^{K} \text{Err}_i$$

- $S_i^{gt}$ proxied by **input cloud points labeled $i$** (no GT mesh available).
- $S_i^r$ = the reconstructed mesh for cluster $i$ only — **not** the union.
- $N_i$ = number of input points in cluster $i$.
- $d(s_{i,k}, S_i^r)$ = exact point-to-triangle distance via `trimesh.proximity.closest_point(mesh_i, pts_i)` (Euclidean, AABB-accelerated).
- Final scalar: per-cluster mean, then averaged uniformly over clusters (not weighted by point count). Clusters with a missing or empty reconstructed mesh are dropped from the average and reported separately.

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

$$\frac{1}{|P|}\sum_{p \in P} d(p, M) \quad\text{via}\quad \texttt{trimesh.proximity.closest\_point}$$

This direction is closed-form because point-to-triangle distance has a closed form (see below). `trimesh` accelerates the per-triangle search with an AABB tree and returns Euclidean distances directly — the same distance vector is reused for the PC→Mesh P-coverage threshold (§1a).

**Mesh → PC** (right term): Monte Carlo approximation of the surface integral.

$$\frac{1}{\text{Area}(M)} \int_M \min_{p \in P} \|q - p\|_2 \, dq \;\approx\; \frac{1}{N}\sum_{i=1}^{N} \min_{p \in P} \|q_i - p\|_2$$

This integral has no closed form because $\min_{p \in P}\|q - p\|$ is piecewise-defined (Voronoi cells of $P$). Since $P$ is a discrete point set (not a mesh), the inner minimum is a point-to-point distance and is computed via `scipy.spatial.cKDTree(P).query(samples, k=1)` — not a point-to-triangle distance.

The sample set $\{q_i\}$ is drawn by `trimesh.sample.sample_surface_even(M, N)` (Poisson-disk variant, lower variance than plain area-weighted); if the even sampler returns zero samples (can happen on degenerate meshes), the code falls back to `trimesh.sample.sample_surface(M, N)`. The same sample set is reused for the Mesh→PC P-coverage threshold (§1b).

### Sample count

$N = 30\,000$, **same $N$ for both pipelines**. Variance of the Monte Carlo estimator is $O(1/N)$; doubling $N$ should not change the reported number meaningfully.

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

`trimesh.proximity.closest_point` does this with an AABB tree so the per-query cost is logarithmic in $|F|$ rather than linear, and returns Euclidean distances directly (no squared-distance intermediate). **Exact, no discretization error** — the only places sampling enters the metric suite are the Mesh→PC chamfer direction and the Mesh→PC P-coverage direction, both of which use the same surface Monte Carlo samples.

---

## Why these three (and not others)

| Metric | Label-aware? | Continuous? | Direction | Catches |
|---|---|---|---|---|
| P-coverage (PC→Mesh) | No (union) | No (binary) | PC → mesh | Under-coverage of input cloud |
| P-coverage (Mesh→PC) | No (union) | No (binary) | mesh → PC | Over-extension / hallucination (spurious mesh regions) |
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

- All metrics computed in normalized space (unit cube), same per-part normalization on both pipelines.
- State $r = 0.01$ for both P-coverage directions.
- Report both P-coverage directions (PC→Mesh and Mesh→PC); the asymmetry is part of the story.
- State $N = 30\,000$ (mesh sample count) for Chamfer and Mesh→PC P-coverage (same samples).
- For residual error, disclose GT-proxy = segmented input points.
- Report per-part numbers and dataset averages.
- Symmetric chamfer reported as the average of the two directions (state explicitly that it's the average, not the sum).
