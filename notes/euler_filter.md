# Progressive Euler Filter

## Problem

After computing all surface intersections, trimming curves at vertices, and
splitting curves into arcs, the resulting **face wire graphs** may not be
Eulerian. A face wire graph is Eulerian when every vertex has even degree on
that face — this is the necessary condition for assembling a closed wire
(boundary loop) for each face.

Non-Eulerian faces arise from:
- **Spurious vertices**: triple-intersection points that lie far from the
  actual geometry (e.g. tangent plane-cylinder intersections producing
  phantom vertices)
- **Phantom arcs**: extra arc segments on closed curves (circles) created
  by OCC when splitting at nearby vertices, or arcs on the "outer" half
  of an elliptical intersection that extends far beyond the actual boundary

The filter must remove these spurious objects while preserving the real
topology.

---

## Pipeline overview

The full vertex/arc pipeline in `brep_pipeline.py` has three stages:

### Stage 1: Bounding-box pre-filter (upstream, in `brep_pipeline.py`)

Each cluster gets an axis-aligned bounding box expanded by a fractional
margin (default 20%):

$$\text{bbox}_k = \bigl[\min(\text{cluster}_k) - m \cdot \text{extent}_k,\;\; \max(\text{cluster}_k) + m \cdot \text{extent}_k\bigr]$$

where $m$ is `--bbox_margin`.

A vertex is **kept** only if it lies inside the bounding boxes of **all**
clusters it is associated with (the 2–3 clusters whose surface intersection
produced it). This is a coarse filter that removes obviously wrong vertices
(e.g. 323 → 23 vertices for abc_00000006).

### Stage 2: Trim and split (upstream)

`trim_by_vertices` trims raw intersection curves to the parameter range
defined by vertex positions. `build_edge_arcs` splits trimmed curves at
vertex positions, producing a list of arcs per edge.

### Stage 3: Progressive Euler filter (`progressive_euler_filter` in `topology.py`)

This is the main algorithm described below.

---

## Algorithm

### Inputs

- `edge_arcs`: dict $(i,j) \to \text{list}[\text{arc}]$ — arcs on each edge
  (pair of adjacent clusters)
- `vertices`: $(M, 3)$ array of vertex positions
- `vertex\_edges`: $\text{list}[\text{set}]$ — which edges each vertex belongs to
- `cluster\_trees`: $\text{list}[\text{KDTree}]$ — one per cluster, for NN queries
- `cluster\_nn\_percentiles`: $\text{list}[\text{float}]$ — $p$-th percentile of
  intra-cluster NN distances (raw, not multiplied by any factor)

### Phase 1: Score all candidates

**Vertex score.** For vertex $v$ associated with clusters $K_v$ (the set
of cluster indices from its edges):

$$\text{score}(v) = \max_{k \in K_v} \frac{d(v, \text{cluster}_k)}{p_k}$$

where:
- $d(v, \text{cluster}_k)$ is the nearest-neighbour distance from $v$ to
  cluster $k$
- $p_k$ is the intra-cluster NN distance percentile for cluster $k$

The ratio $d / p$ normalises the distance by the cluster's own point
spacing. A vertex at $\text{score} = 1.0$ is exactly at the cluster's NN
distance percentile — roughly at the boundary of the cluster. Scores
$\gg 1$ indicate the vertex is far from at least one of its clusters
relative to that cluster's density.

The $\max$ over clusters means the vertex is scored by its **worst fit** —
it only takes one cluster to flag it.

**Arc score.** For an arc on edge $(i, j)$:

1. Sample $n = 10$ points along the interior of the arc
   (middle 50% of the parameter range)
2. For each sample point, compute
   $\max\!\bigl(d_i / p_i,\; d_j / p_j\bigr)$ — same ratio
   as vertices, but only against the two clusters that define the edge
3. Return the **mean** over all samples

Closed arcs (full circles/ellipses) get score 0 and are never removed.

### Phase 2: Two-pass progressive removal

Arc removal is less destructive than vertex removal: an arc affects exactly
2 faces, while a vertex can affect many faces through its incident arcs.
Therefore arcs are removed first.

**Pass 1 — Arc removal** (worst-scored first):

```
for each arc candidate in descending score order:
    recompute bad_faces = set of non-Eulerian faces
    if bad_faces is empty: stop
    if arc was already removed: skip
    if arc's faces have no overlap with bad_faces: skip
    remove arc from work_arcs
    if removing arc leaves a vertex with degree 0: mark it removed
```

**Pass 2 — Vertex removal** (worst-scored first, only if pass 1 didn't
achieve Eulerian):

```
for each vertex candidate in descending score order:
    recompute bad_faces = set of non-Eulerian faces
    if bad_faces is empty: stop
    if vertex was already removed: skip
    if vertex's faces have no overlap with bad_faces: skip
    call _merge_arcs_at_vertex(v, work_arcs)
    mark vertex as removed
    mark any newly-isolated vertices as removed
```

**Key properties:**
- Arcs are processed before vertices, preventing cascade destruction
- Within each pass, candidates are processed worst-first, so spurious
  objects (high score) are removed before real ones (low score)
- A candidate is only removed if it touches a face that is currently
  non-Eulerian — objects on already-Eulerian faces are left alone
- The loop terminates as soon as all faces are Eulerian

### Vertex removal mechanics: `_merge_arcs_at_vertex`

When vertex $v$ is removed, for each edge that has arcs touching $v$:

- If **2 arcs** on the same edge touch $v$ (one ending at $v$, one starting
  at $v$): **merge** them into a single arc spanning $[t_{\text{start},A},\; t_{\text{end},B}]$
  with endpoints $v_{\text{start},A} \to v_{\text{end},B}$
- If **1 arc** on an edge touches $v$: it's **dangling** — remove it
- If the merged arc has $v_\text{start} = v_\text{end}$: mark it as **closed**

**When does merging apply?** Only when a spurious vertex splits a single
edge into two arcs — i.e. the vertex sits in the middle of one curve. This
is rare in practice; most spurious vertices sit at **edge intersections**
(where 3 surfaces meet), touching one arc per edge on 2–3 different edges.
In that case every arc is "dangling" (no merge partner on the same edge)
and gets dropped.

### Phase 3: Rebuild

- Remove any remaining 0-degree (isolated) vertices
- Re-index surviving vertices (compact)
- Drop arcs whose endpoints were removed
- Return filtered `edge_arcs`, `vertices`, `vertex_edges`

---

## The Eulerian condition

A **face** in the BRep corresponds to one cluster. Its **wire graph** has:
- Nodes = vertices that appear in arcs on edges involving that cluster
- Edges (graph-theoretic) = open arcs on those edges

Each open arc on edge $(i, j)$ contributes to the wire graphs of both
face $i$ and face $j$. It increments the degree of its $v_\text{start}$
and $v_\text{end}$ by 1 on each face.

For a face's wire graph to admit a closed Eulerian circuit (= valid wire
boundary), every vertex must have **even degree** on that face:

$$\forall f,\; \forall v \in V_f:\quad \deg_f(v) \equiv 0 \pmod{2}$$

The function `_non_eulerian_faces_direct(work_arcs)` checks this: it builds
a $\text{face} \to \text{vertex} \to \text{degree}$ map from all open arcs,
and returns the set of faces where any vertex has odd degree.

---

## Cascade risk

Removing a vertex drops all its incident arcs. These arcs are shared
between faces (each arc belongs to faces $i$ and $j$). Dropping arcs from
a previously-Eulerian face can make it non-Eulerian, triggering further
removals in subsequent iterations.

The cascade is controlled by:
1. **Two-pass ordering**: arcs are removed before vertices, which is far
   less destructive (each arc touches exactly 2 faces)
2. **Score ordering**: within each pass, truly spurious objects have much
   higher scores than real objects, so they are removed first
3. **Face overlap check**: a candidate is only removed if it touches a
   currently non-Eulerian face

For models where the cascade is still a problem, a possible extension is a
**"do no harm" guard**: define the violation count

$$V(R) = \sum_{f} \#\{v \notin R : \deg_f(v) \text{ is odd}\}$$

and only accept a removal if $V(R') \leq V(R)$. This is not currently
implemented as the two-pass approach has been sufficient.

---

## Visualization

The pipeline saves data at two stages for side-by-side comparison:

| Stage | Vertex file | Arc files | Vertex colour |
|-------|-------------|-----------|---------------|
| Post bbox, pre-Euler | `vertices_pre_euler.npz` | `arcs_pre_euler_I_J.npz` | Orange |
| Post Euler | `vertices.npz` | `arcs_I_J.npz` | Yellow |

The visualizer (`--visualize`) shows a 3×2 grid:

| Col 1 | Col 2 | Col 3 |
|-------|-------|-------|
| Untrimmed curves | Pre-Euler arcs + vertices | Post-Euler arcs + vertices |
| Point clouds + arcs + boundaries | Fitted surfaces | Point clouds + pre-Euler arcs |

---

## Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--bbox_margin` | 0.2 | Fractional expansion of cluster bounding boxes for vertex pre-filter |
| `--proximity_percentile` | 100 | Percentile of intra-cluster NN distances used for scoring |
