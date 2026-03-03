# CAD Reconstruction Evaluation Metrics

Evaluation of a reconstructed STEP file against a ground-truth STEP file.
Both files can be loaded with OCC, giving access to exact analytical geometry.

---

## 1. Hausdorff Distance

### Mathematical definition

Let $S_{rec}$ and $S_{gt}$ denote the reconstructed and ground-truth surfaces as point sets in $\mathbb{R}^3$.

The **one-sided (directed) Hausdorff distance** from $A$ to $B$ is:

$$h(A, B) = \sup_{a \in A} \inf_{b \in B} d(a, b)$$

i.e. the worst-case nearest-neighbour distance when querying from $A$ into $B$.

The **symmetric Hausdorff distance** is:

$$H(A, B) = \max\bigl(h(A, B),\; h(B, A)\bigr)$$

### Geometric interpretation

- $h(S_{rec} \to S_{gt})$: the furthest any reconstructed surface point is from the GT. Penalises spurious or misshapen geometry that strays from the GT.
- $h(S_{gt} \to S_{rec})$: the furthest any GT point is from the reconstruction. Penalises missing surfaces or large holes.
- $H$: the maximum of both; a single outlier point can dominate.

### Relation to this problem

Because both models are exact CAD geometry (not sampled point clouds), the
point-to-set distance $\inf_{b \in B} d(a, b)$ can be computed exactly using OCC's
`GeomAPI_ProjectPointOnSurf` rather than a kd-tree approximation. This makes the
metric exact rather than sampling-density-dependent.

Practically:
1. Sample a dense point set $P_{rec}$ from $S_{rec}$ (e.g. 100k points via STEP sampler).
2. For each $p \in P_{rec}$, compute the exact distance to $S_{gt}$ using OCC projection.
3. $h(S_{rec} \to S_{gt}) = \max_{p} d(p, S_{gt})$.
4. Repeat in the other direction for $h(S_{gt} \to S_{rec})$.

**Sensitivity to outliers:** A single badly-placed reconstructed face makes $H$ large
even if 99% of the model is perfect. Always report mean/median alongside Hausdorff.

---

## 2. Point-to-Surface Mean / RMS Distance

Compute the full distribution of distances, not just the maximum:

$$\overline{d}(S_{rec} \to S_{gt}) = \frac{1}{|P_{rec}|} \sum_{p \in P_{rec}} d(p, S_{gt})$$

Report both directions and take the symmetric mean as the primary scalar metric.
This is strictly more informative than Hausdorff alone and more robust to outliers.

Using OCC projection (not point-to-point) removes the bias introduced by the
sampling density of the reference cloud.

---

## 3. F-score at Threshold $\tau$

From Tatarchenko et al. (2019), now standard in 3D reconstruction benchmarks:

$$\text{Precision}(\tau) = \frac{|\{p \in P_{rec} : d(p, S_{gt}) \leq \tau\}|}{|P_{rec}|}$$

$$\text{Recall}(\tau) = \frac{|\{p \in P_{gt} : d(p, S_{rec}) \leq \tau\}|}{|P_{gt}|}$$

$$F(\tau) = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

$\tau$ should be normalised to object scale, e.g. 1% and 2% of the bounding-box diagonal.
Plot $F(\tau)$ as a curve over $\tau$ for a full picture.

Interpretation:
- Low Precision → reconstruction contains geometry not in GT (hallucinated surfaces)
- Low Recall    → GT contains geometry missing from reconstruction
- $F$ balances both

---

## 4. Normal Consistency

Sample (point, normal) pairs from both surfaces. For each reconstructed point
projected onto the GT, compare surface normals:

$$\text{NC} = \frac{1}{|P_{rec}|} \sum_{p \in P_{rec}} \bigl|\hat{n}_{rec}(p) \cdot \hat{n}_{gt}(\text{proj}(p))\bigr|$$

Absolute value because normals may be flipped. Range $[0, 1]$; $1$ = perfect alignment.
Captures whether analytical surface orientations (plane normals, cylinder axes) are
correct, which Hausdorff alone cannot.

---

## 5. Topology Metrics (B-Rep specific)

Geometry metrics say nothing about structural correctness. Report alongside:

| Metric | Description |
|--------|-------------|
| Face count ratio | $\frac{\mid F_{rec}\mid}{\mid F_{gt} \mid} \approx 1$ |
| Surface type accuracy | Fraction of GT faces whose type (plane/cylinder/…) is matched |
| Edge count ratio | $\frac{\mid E_{rec} \mid}{\mid E_{gt} \mid}$ |
| Vertex count ratio | $\frac{\mid V_{rec} \mid}{\mid V_{gt} \mid}$ |

For surface type matching, assign each reconstructed face to the nearest GT face
(by centroid or Hausdorff) and compare `BRepAdaptor_Surface.GetType()`.

---

## 6. Volumetric IoU (if watertight)

When both models form closed solids:

$$\text{IoU} = \frac{\text{Vol}(S_{rec} \cap S_{gt})}{\text{Vol}(S_{rec} \cup S_{gt})}$$

Can be computed by voxelisation at fixed resolution (marching cubes both models)
or via OCC Boolean operations (`BRepAlgoAPI_Common`, `BRepAlgoAPI_Fuse`).
This is the single most interpretable metric: "what fraction of the 3D shape is
correctly recovered?"

OCC Boolean operations can fail on near-degenerate geometry; voxelisation is more
robust but resolution-dependent.

---

## Part-Level Reconstruction and Evaluation

### Why part-level?

Some ABC models are Compounds of N independent solid parts.  Processing the
entire Compound as one point cloud yields too many clusters (e.g. 60 for a
10-part model), causing numerical instability in surface fitting and BRep
construction.  The generation script (`abc_preprocess.py --by_part`) splits
each part into a separate `.xyzc` file; `brep_pipeline.py --model_id` runs
the reconstruction pipeline on each part independently.

### Output layout (reconstruction)

```
output_brep/
  {model_id}/
    part_000/         ← per-part npz files + {model_id}_part_000_bop.step
    part_001/
    ...
    unified/          ← merged npz files with cluster ID offsets (for visualizer)
    unified_bop.step  ← all part STEPs merged into one Compound
```

The `unified/` directory uses globally offset cluster IDs so the visualizer
(`brep_pipeline.py --visualize --model_id`) works without any changes.

### Evaluation: part-agnostic via unified STEP

Because the unified STEP (`unified_bop.step`) is a single Compound shape
containing all reconstructed parts, evaluation is identical to the single-model
case:

1. Sample N points from `unified_bop.step` (pool all faces, area-weighted)
2. Sample N points from the ground-truth STEP file (same logic)
3. Compute Chamfer / Hausdorff / F-score between the two point sets

No part-correspondence matching is needed.  Parts that failed BRep construction
produce no faces in the unified STEP, which naturally increases Chamfer distance
to the GT points in those regions — an appropriate penalty.

The GT STEP does not need to be split; it is sampled as-is.

## Recommended Evaluation Pipeline

For comparison with prior work (ComplexGen, HNC-CAD, ABC baselines):

| Priority | Metric | Tool |
|----------|--------|------|
| Primary  | Symmetric point-to-surface mean distance | OCC `GeomAPI_ProjectPointOnSurf` |
| Primary  | F-score @ 1% and 2% diagonal | numpy after OCC projection |
| Secondary | Hausdorff distance (both directions) | OCC projection |
| Secondary | Normal consistency | OCC surface normals |
| Structural | Face / edge / vertex count ratios | OCC `TopExp_Explorer` |
| Optional  | Volumetric IoU | OCC Boolean or voxelisation |

### Implementation sketch

```python
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp      import TopExp_Explorer
from OCC.Core.TopAbs      import TopAbs_FACE
from OCC.Core             import topods
from OCC.Core.BRep        import BRep_Tool
from OCC.Core.GeomAPI     import GeomAPI_ProjectPointOnSurf
from OCC.Core.gp          import gp_Pnt

def load_step_faces(path):
    reader = STEPControl_Reader()
    reader.ReadFile(path); reader.TransferRoots()
    shape = reader.OneShape()
    exp   = TopExp_Explorer(shape, TopAbs_FACE)
    faces = []
    while exp.More():
        faces.append(topods.Face(exp.Current()))
        exp.Next()
    return faces

def point_to_shape_distances(points, gt_faces):
    """Exact point-to-surface distances using OCC projection."""
    dists = np.zeros(len(points))
    for k, pt in enumerate(points):
        p    = gp_Pnt(*pt.tolist())
        best = np.inf
        for face in gt_faces:
            surf = BRep_Tool.Surface(face)
            proj = GeomAPI_ProjectPointOnSurf(p, surf)
            if proj.NbPoints() > 0:
                best = min(best, proj.LowerDistance())
        dists[k] = best
    return dists

rec_faces = load_step_faces("reconstructed.step")
gt_faces  = load_step_faces("ground_truth.step")

P_rec = sample_points_from_step("reconstructed.step", n=100_000, ...)
P_gt  = sample_points_from_step("ground_truth.step",  n=100_000, ...)

d_rec_to_gt = point_to_shape_distances(P_rec, gt_faces)
d_gt_to_rec = point_to_shape_distances(P_gt,  rec_faces)

hausdorff = max(d_rec_to_gt.max(), d_gt_to_rec.max())
mean_dist = 0.5 * (d_rec_to_gt.mean() + d_gt_to_rec.mean())

diag = np.linalg.norm(bbox_max - bbox_min)
for tau_pct in [0.01, 0.02]:
    tau       = tau_pct * diag
    precision = (d_rec_to_gt <= tau).mean()
    recall    = (d_gt_to_rec <= tau).mean()
    fscore    = 2 * precision * recall / (precision + recall + 1e-12)
    print(f"F@{tau_pct*100:.0f}%: P={precision:.3f} R={recall:.3f} F={fscore:.3f}")
```

**Performance note:** Projecting 100k points onto all GT faces naively is $O(N \times |F_{gt}|)$
OCC calls. For models with many faces, accelerate by first finding candidate faces via
a bounding-box tree (BVH or scipy kd-tree on face centroids), then projecting only
onto the nearest few faces.
