# Triangle Meshes and the Mesh Clipping Pipeline

## 1. Mathematical Definition of a Triangle Mesh

A triangle mesh $\mathcal{M} = (V, F)$ consists of:

- **Vertex set** $V = \{\mathbf{v}_0, \mathbf{v}_1, \dots, \mathbf{v}_{n-1}\} \subset \mathbb{R}^3$
- **Face set** $F = \{(i_0, j_0, k_0), (i_1, j_1, k_1), \dots, (i_{m-1}, j_{m-1}, k_{m-1})\} \subset \{0, \dots, n-1\}^3$

Each face $(i, j, k) \in F$ defines a triangle with vertices $\mathbf{v}_i, \mathbf{v}_j, \mathbf{v}_k$. The triangle itself is the convex hull of these three points:

$$T_{ijk} = \{\alpha \mathbf{v}_i + \beta \mathbf{v}_j + \gamma \mathbf{v}_k \mid \alpha, \beta, \gamma \geq 0, \; \alpha + \beta + \gamma = 1\}$$

The surface represented by the mesh is the union of all triangles:

$$S(\mathcal{M}) = \bigcup_{(i,j,k) \in F} T_{ijk}$$

In the original codebase, the vertex and face arrays are stored via Open3D's `o3d.utility.Vector3dVector` (vertices) and `o3d.utility.Vector3iVector` (faces) for primitive surfaces ([`fitting_utils.py:167-168`](point2cad/fitting_utils.py#L167)), and via `trimesh.Trimesh(vertices, faces)` for INR surfaces ([`utils.py:258-262`](point2cad/utils.py#L258)).

### Ordering Convention and Orientation

The ordering of indices within a face tuple matters. Given a face $(i, j, k)$, the **outward normal** is determined by the right-hand rule:

$$\mathbf{n} = \frac{(\mathbf{v}_j - \mathbf{v}_i) \times (\mathbf{v}_k - \mathbf{v}_i)}{\|(\mathbf{v}_j - \mathbf{v}_i) \times (\mathbf{v}_k - \mathbf{v}_i)\|}$$

A mesh is **consistently oriented** if, for every pair of adjacent triangles sharing an edge, the shared edge is traversed in opposite directions by the two triangles. That is, if triangle $A$ contains the directed edge $(i, j)$, then the adjacent triangle $B$ sharing that edge must contain $(j, i)$. This guarantees a coherent "outside" across the entire surface.

### Edges and Adjacency

The **edge set** $E$ is derived from $F$:

$$E = \{\{i, j\} \mid \exists \, (a, b, c) \in F \text{ such that } \{i, j\} \subset \{a, b, c\}\}$$

Two faces are **adjacent** if they share an edge. The **face adjacency graph** $G_F = (F, E_F)$ has faces as nodes and edges between adjacent faces. This graph is central to connected component analysis in the clipping pipeline. In the original codebase, it is computed via `trimesh.Trimesh.face_adjacency` ([`io_utils.py:58`](point2cad/io_utils.py#L58)).

### Manifoldness

A mesh is a **2-manifold** if every edge is shared by exactly 1 face (boundary edge) or exactly 2 faces (interior edge), and the faces around every vertex form a single fan (or half-fan at boundaries). Manifoldness is important because boolean operations and self-intersection resolution algorithms typically require manifold input.

## 2. How Mesh Libraries Render a Triangle Mesh

Given only $V$ and $F$, a rendering engine can produce a shaded image. The pipeline is:

### Step 1: Vertex Transformation

Each vertex $\mathbf{v} \in \mathbb{R}^3$ is transformed through a sequence of matrices:

$$\mathbf{v}_{\text{clip}} = M_{\text{proj}} \cdot M_{\text{view}} \cdot M_{\text{model}} \cdot \begin{pmatrix} \mathbf{v} \\ 1 \end{pmatrix}$$

- $M_{\text{model}}$: object-to-world (position, rotation, scale of the mesh)
- $M_{\text{view}}$: world-to-camera (where the camera is, which direction it faces)
- $M_{\text{proj}}$: camera-to-clip (perspective or orthographic projection)

After perspective division, this yields **normalized device coordinates** (NDC) in $[-1, 1]^3$.

### Step 2: Rasterization

For each face $(i, j, k)$, the triangle defined by the projected 2D positions of $\mathbf{v}_i, \mathbf{v}_j, \mathbf{v}_k$ is rasterized: the renderer determines which screen pixels fall inside the triangle. For each pixel, **barycentric coordinates** $(\alpha, \beta, \gamma)$ are computed by solving:

$$\mathbf{p}_{\text{pixel}} = \alpha \, \mathbf{v}_i' + \beta \, \mathbf{v}_j' + \gamma \, \mathbf{v}_k'$$

where primes denote screen-space positions. The pixel is inside the triangle if and only if $\alpha, \beta, \gamma \geq 0$.

### Step 3: Depth Testing

Each pixel stores a depth value (z-buffer). If the current triangle's interpolated depth $z = \alpha z_i + \beta z_j + \gamma z_k$ is closer to the camera than the stored depth, the pixel is updated. This resolves occlusion without needing to sort triangles.

### Step 4: Shading

The pixel color is computed from a lighting model. The simplest is **Lambertian shading**:

$$I = I_a + I_d \max(\mathbf{n} \cdot \mathbf{l}, \; 0)$$

where $I_a$ is ambient light, $I_d$ is the diffuse light intensity, $\mathbf{n}$ is the surface normal, and $\mathbf{l}$ is the unit direction toward the light source.

**Per-face normals** (flat shading): each face uses its geometric normal $\mathbf{n} = \text{normalize}((\mathbf{v}_j - \mathbf{v}_i) \times (\mathbf{v}_k - \mathbf{v}_i))$. This gives a faceted appearance.

**Per-vertex normals** (smooth shading): the normal at vertex $\mathbf{v}_i$ is typically the area-weighted average of the normals of all faces incident to $\mathbf{v}_i$:

$$\mathbf{n}_i = \text{normalize}\left(\sum_{(i, j, k) \in F} A_{ijk} \, \mathbf{n}_{ijk}\right)$$

where $A_{ijk}$ is the area of triangle $(i, j, k)$. During rasterization, the normal at each pixel is interpolated: $\mathbf{n}_{\text{pixel}} = \alpha \mathbf{n}_i + \beta \mathbf{n}_j + \gamma \mathbf{n}_k$. This is called **Phong interpolation** and produces smooth shading even on a coarse mesh. This is what `compute_vertex_normals()` in Open3D computes ([`fitting_utils.py:171`](point2cad/fitting_utils.py#L171)).

### Why Double-Sided Triangles Exist in the Codebase

In the reproduction's `triangulate_and_mesh`, each quad cell generates 4 triangles: 2 front-facing and 2 with reversed winding order. Without this, triangles whose normals point away from the camera are culled (**back-face culling**). By including both orientations, the mesh is visible from both sides. The cost is double the triangle count.

The original Point2CAD tessellation ([`fitting_utils.py:139-172`](point2cad/fitting_utils.py#L139)) does not include reversed triangles — it generates only 2 triangles per quad cell. The INR mesh tessellation ([`utils.py:236-252`](point2cad/utils.py#L236)) also generates only 2 triangles per quad cell. This means the original meshes are single-sided.

## 3. The Mesh Clipping Pipeline (Point2CAD)

After fitting individual surfaces and generating their meshes, the raw meshes typically overlap each other. The clipping pipeline resolves these overlaps to produce a clean assembly where each surface terminates at its boundary with neighboring surfaces.

The pipeline orchestration is in [`main.py:87-103`](point2cad/main.py#L87):

```python
out_meshes = fn_process(cfg, uniq_labels, points, labels, device)          # line 87
pm_meshes = save_unclipped_meshes(out_meshes, color_list, out_path)        # line 91
clipped_meshes = save_clipped_meshes(pm_meshes, out_meshes, color_list, out_path)  # line 97
save_topology(clipped_meshes, out_path)                                    # line 103
```

The unclipped mesh saving ([`io_utils.py:13-33`](point2cad/io_utils.py#L13)) converts individual meshes to pymesh format and exports their concatenation as a PLY file.

### Overview

```
Individual surface meshes
        │
        ▼
   Merge into one mesh (tracking face origins)
        │
        ▼
   Resolve all self-intersections
        │
        ▼
   Remove duplicate vertices
        │
        ▼
   Connected component decomposition
        │
        ▼
   Assign each component to its source surface
        │
        ▼
   For each surface: select components closest to input points
        │
        ▼
   Filter by area-per-point ratio
        │
        ▼
   Clipped meshes
```

The full implementation is in [`io_utils.py:36-121`](point2cad/io_utils.py#L36).

### Step 1: Merge Meshes

[`io_utils.py:37-39`](point2cad/io_utils.py#L37)

```python
pm_merged = pymesh.merge_meshes(pm_meshes)
face_sources_merged = pm_merged.get_attribute("face_sources").astype(np.int32)
```

All $N$ surface meshes are concatenated into a single mesh. The vertex arrays are concatenated, and face index arrays are offset accordingly. `face_sources` is an integer array of length $|F_{\text{merged}}|$ where entry $i$ stores which original surface (0 to $N-1$) face $i$ came from.

If surface $A$ has vertices $V_A$ with faces $F_A$ and surface $B$ has vertices $V_B$ with faces $F_B$, the merged mesh has:
- $V_{\text{merged}} = V_A \cup V_B$ (concatenated)
- $F_{\text{merged}} = F_A \cup (F_B + |V_A|)$ (faces of $B$ offset by vertex count of $A$)

### Step 2: Self-Intersection Resolution

[`io_utils.py:41-42`](point2cad/io_utils.py#L41)

```python
detect_pairs = pymesh.detect_self_intersection(pm_merged)
pm_resolved_ori = pymesh.resolve_self_intersection(pm_merged)
```

This is the core geometric operation. Where two triangles from different (or the same) surfaces pierce each other, the algorithm:

1. Detects all pairs of intersecting triangles.
2. Computes the exact intersection segment for each pair. Two non-coplanar triangles that intersect produce a line segment where one triangle passes through the other.
3. Inserts the intersection segments as new edges into both triangles, subdividing them. Each original triangle that participates in an intersection is split into smaller triangles such that the intersection boundary is explicitly represented in the mesh connectivity.

After resolution, no two triangles in the mesh intersect except along shared edges or vertices. The face count increases because intersecting triangles are subdivided.

The resolved mesh inherits a `face_sources` attribute that maps each new face back to the face in the pre-resolution mesh it was created from. Combined with the merged `face_sources`, this gives a two-level provenance chain: resolved face → merged face → original surface.

[`io_utils.py:50-53`](point2cad/io_utils.py#L50)

```python
face_sources_resolved_ori = pm_resolved_ori.get_attribute("face_sources").astype(np.int32)
face_sources_from_fit = face_sources_merged[face_sources_resolved_ori]
```

`face_sources_from_fit[i]` gives the original surface ID for face $i$ in the resolved mesh.

Note: `pymesh.separate_mesh(pm_resolved_ori)` is called on [`io_utils.py:44`](point2cad/io_utils.py#L44) but its result is assigned to an unused variable `a` and never referenced again.

### Step 3: Vertex Deduplication

[`io_utils.py:46-48`](point2cad/io_utils.py#L46)

```python
pm_resolved, info_dict = pymesh.remove_duplicated_vertices(pm_resolved_ori, tol=1e-6)
```

The intersection resolution step may introduce vertices that are numerically very close but not identical (e.g., two surfaces both create a vertex at the same intersection point independently). This step merges vertices within tolerance $10^{-6}$ and updates face indices accordingly. This is necessary for correct adjacency computation in the next step — without it, faces that should be adjacent might not share vertex indices.

### Step 4: Connected Component Decomposition

[`io_utils.py:55-66`](point2cad/io_utils.py#L55)

```python
tri_resolved = trimesh.Trimesh(vertices=pm_resolved.vertices, faces=pm_resolved.faces)
face_adjacency = tri_resolved.face_adjacency
connected_node_labels = trimesh.graph.connected_component_labels(
    edges=face_adjacency, node_count=len(tri_resolved.faces)
)
most_common_groupids = [item[0] for item in Counter(connected_node_labels).most_common()]
```

The **face adjacency graph** is constructed: two faces are connected if they share an edge. Then connected components are computed — groups of faces where you can walk from any face to any other face through shared edges.

After self-intersection resolution, the merged mesh is cut along all intersection curves. What was originally a single surface mesh may now be split into multiple disconnected components. For example, a plane that passes through a cylinder is cut into the part inside the cylinder and the part outside.

Each component is labeled, and components are ordered by face count (largest first).

### Step 5: Source Attribution

[`io_utils.py:68-78`](point2cad/io_utils.py#L68)

```python
submeshes = [
    trimesh.Trimesh(
        vertices=np.array(tri_resolved.vertices),
        faces=np.array(tri_resolved.faces)[np.where(connected_node_labels == item)],
    )
    for item in most_common_groupids
]
indices_sources = [
    face_sources_from_fit[connected_node_labels == item][0]
    for item in np.array(most_common_groupids)
]
```

Each connected component is extracted as a separate `trimesh.Trimesh` and assigned to the original surface it came from. Since all faces within a component originated from the same surface (intersection resolution preserves face provenance), taking the source of any face in the component suffices.

### Step 6: Proximity-Based Component Selection

[`io_utils.py:82-97`](point2cad/io_utils.py#L82)

For each original surface $p$, the algorithm must decide which of its components to keep. A surface may have been cut into multiple pieces by neighboring surfaces — typically one "correct" piece near the input point cloud and several spurious fragments.

```python
for p in range(len(out_meshes)):
    one_cluster_points = out_meshes[p]["inpoints"]
    submeshes_cur = [x for x, y in zip(submeshes, np.array(indices_sources) == p) if y and len(x.faces) > 2]
```

First, collect all components belonging to surface $p$, discarding degenerate ones with $\leq 2$ faces.

```python
    nearest_submesh = np.argmin(
        np.array([trimesh.proximity.closest_point(item, one_cluster_points)[1] for item in submeshes_cur]).transpose(),
        -1,
    )
    counter_nearest = Counter(nearest_submesh).most_common()
```

For each input point in the cluster, find which component is closest to it. `trimesh.proximity.closest_point(mesh, points)` returns, for each query point, the closest point on the mesh surface and the distance to it. The result is a matrix of shape `(num_components, num_points)` containing distances. Taking `argmin` along the component axis assigns each input point to its nearest component. Components that attract many points are likely the correct piece; small fragments far from the data attract few or none.

### Step 7: Area-Based Filtering

[`io_utils.py:99-116`](point2cad/io_utils.py#L99)

```python
    area_per_point = np.array([submeshes_cur[item[0]].area / item[1] for item in counter_nearest])
    multiplier_area = 2
    result_indices = np.array(counter_nearest)[:, 0][
        np.logical_and(
            area_per_point < area_per_point[np.nonzero(area_per_point)[0][0]] * multiplier_area,
            area_per_point != 0,
        )
    ]
    result_submesh_list = [submeshes_cur[item] for item in result_indices]
    clipped_mesh = trimesh.util.concatenate(result_submesh_list)
    clipped_mesh.visual.face_colors = color_list[p]
    clipped_meshes.append(clipped_mesh)
```

For each component, compute $\text{area} / \text{point\_count}$ — the surface area per assigned input point. This is a density metric: a correct component has a reasonable ratio, while a spurious fragment with large area but few nearby points has an inflated ratio.

Components are kept if their area-per-point ratio is within $2\times$ the ratio of the best (first nonzero) component. This filters out large stray fragments that are far from the data but happened to be nearest for a few outlier points.

The final clipped assembly is exported at [`io_utils.py:118-119`](point2cad/io_utils.py#L118).

## 4. Topology Extraction

After clipping, the final stage extracts the CAD topology: intersection curves (edges) and corners (vertices where three or more surfaces meet). The full implementation is in [`io_utils.py:124-171`](point2cad/io_utils.py#L124).

### Intersection Curves

[`io_utils.py:125-146`](point2cad/io_utils.py#L125)

```python
filtered_submeshes_pv = [pv.wrap(item) for item in clipped_meshes]
for k, pv_pair in enumerate(itertools.combinations(filtered_submeshes_pv, 2)):
    intersection, _, _ = pv_pair[0].intersection(pv_pair[1], split_first=False, split_second=False)
```

Clipped meshes are converted from trimesh to pyvista format via `pv.wrap()`. For every pair of clipped surface meshes, `pyvista.intersection()` computes the polyline where the two surfaces meet. This is the boundary curve between two adjacent CAD faces. The intersection result contains both points and line connectivity (`intersection.points` and `intersection.lines`).

### Corner Detection

[`io_utils.py:150-166`](point2cad/io_utils.py#L150)

```python
for combination_indices in itertools.combinations(range(len(intersection_curves)), 2):
    sample0 = np.array(intersection_curves[combination_indices[0]]["pv_points"])
    sample1 = np.array(intersection_curves[combination_indices[1]]["pv_points"])
    dists = scipy.spatial.distance.cdist(sample0, sample1)
    row_indices, col_indices = np.where(dists == 0)
```

Corners are points shared by two or more intersection curves. If curve $C_{AB}$ (between surfaces $A$ and $B$) and curve $C_{BC}$ (between surfaces $B$ and $C$) share a point, that point is a corner where surfaces $A$, $B$, and $C$ meet. The algorithm finds exact coincidences (`dists == 0`) between all pairs of intersection curves. Coincident points are averaged (though they should be identical): `(sample0[i] + sample1[j]) / 2` ([`io_utils.py:162-164`](point2cad/io_utils.py#L162)).

The topology (curves and corners) is exported as JSON at [`io_utils.py:170-171`](point2cad/io_utils.py#L170).

## 5. Mesh Generation in the Original Codebase

For reference, the individual mesh generation functions that feed into the clipping pipeline:

**Primitive surfaces** are meshed via `visualize_basic_mesh` ([`fitting_utils.py:294-357`](point2cad/fitting_utils.py#L294)), which:
1. Upsamples the input point cluster via `up_sample_points_torch_memory_efficient` ([`fitting_utils.py:66`](point2cad/fitting_utils.py#L66))
2. Calls `bit_mapping_points_torch` ([`fitting_utils.py:284-291`](point2cad/fitting_utils.py#L284)), which internally calls `create_grid` ([`fitting_utils.py:94-136`](point2cad/fitting_utils.py#L94)) for grid trimming followed by `tessalate_points_fast` ([`fitting_utils.py:139-172`](point2cad/fitting_utils.py#L139)) for triangulation.

**INR surfaces** are meshed via `sample_inr_mesh` ([`utils.py:222-263`](point2cad/utils.py#L222)), which:
1. Samples a regular UV grid through the decoder via `sample_inr_points` ([`utils.py:186-219`](point2cad/utils.py#L186))
2. Triangulates using UV grid adjacency (2 triangles per quad cell, no grid trimming mask)

## 6. Key Observations for Reproduction

1. The clipping pipeline requires a self-intersection resolution algorithm. The original uses `pymesh.resolve_self_intersection` ([`io_utils.py:42`](point2cad/io_utils.py#L42)), which is the hardest operation to replace. An alternative approach is to use pairwise boolean operations between surfaces via pyvista, which internally uses VTK's intersection capabilities.

2. Connected component analysis only requires the face adjacency graph, which can be constructed from $F$ directly: two faces are adjacent if they share exactly two vertex indices. This is a pure graph operation achievable with `scipy.sparse.csgraph.connected_components`.

3. The proximity-based selection (Step 6) requires computing the distance from a set of query points to the surface of a mesh. This can be approximated by building a KD-tree over the mesh vertices (ignoring the triangles), or computed exactly by projecting each query point onto every triangle face.

4. The pipeline tolerates imperfect individual meshes (INR artifacts, overly extended primitive meshes) because the clipping step cuts them to shape. Surfaces only need to extend far enough to cover their true boundary — they will be trimmed by their neighbors.

5. The `pymesh.separate_mesh` call on [`io_utils.py:44`](point2cad/io_utils.py#L44) appears to be dead code — its result is never used. The actual component decomposition happens later via trimesh's face adjacency graph ([`io_utils.py:55-62`](point2cad/io_utils.py#L55)).

6. The original INR meshing does not apply grid trimming ([`utils.py:236-252`](point2cad/utils.py#L236)) — there is no mask. It relies entirely on the downstream clipping pipeline to remove artifacts.
