# B-Rep Reconstruction via Chain Complex Optimization

## Overview

Given a segmented point cloud with fitted primitive surfaces, reconstruct a
valid B-Rep model by:

1. Computing pairwise surface intersections → edges
2. Splitting surfaces into candidate face patches using intersection edges
3. Selecting a subset of patches, edges, and vertices via Integer Linear
   Programming (ILP) such that the result is a valid, manifold B-Rep

The key insight (from PolyFit and ComplexGen) is that **topological validity
is enforced as a hard constraint** in the optimization, rather than being an
emergent property of bottom-up wire assembly.

---

## 1. Input

- $n$ fitted primitive surfaces $\{S_1, \dots, S_n\}$ (planes, cylinders, cones,
  spheres), each with an associated point cloud cluster $P_i$
- The full point cloud $P = \bigcup_i P_i$

## 2. Candidate Generation

### 2.1 Intersection Curves (Edges)

For each adjacent pair $(i, j)$, compute the intersection curve
$S_i \cap S_j$.  This produces a set of raw curves (lines, circles, ellipses,
etc.) using analytical formulas or OCC's `GeomAPI_IntSS`.

Some intersections may fail (OCC returning empty or erroring).  Unlike the
bottom-up pipeline, a missing intersection here simply means fewer candidate
patches — it does not cause a cascading failure.

### 2.2 Vertices

Vertices are points where three or more surfaces meet.  Computed via
curve–surface intersection (`GeomAPI_IntCS`) as in the existing pipeline.

### 2.3 Arcs

Each intersection curve is split at its incident vertices into **arcs** — the
segments between consecutive vertices on a curve.  Each arc lies on exactly
two surfaces $(S_i, S_j)$ and connects two vertices.

### 2.4 Candidate Face Patches

For each surface $S_i$, collect all arcs that lie on it.  Pass the surface and
its arcs to OCC's `BOPAlgo_BuilderFace`, which computes the bounded regions
(patches) formed by the arcs on the surface.

Surface $S_i$ may produce one or more patches $\{f_{i,1}, f_{i,2}, \dots\}$.
Each patch is a bounded region of $S_i$ delimited by arcs.

---

## 3. Chain Complex Structure

A valid B-Rep is a **2-chain complex** consisting of three levels of elements
connected by boundary operators:

$$\mathbb{F} \xrightarrow{\partial_2} \mathbb{E} \xrightarrow{\partial_1} \mathbb{V}$$

where $\mathbb{F}$, $\mathbb{E}$, $\mathbb{V}$ are the spaces of faces (patches),
edges (arcs), and vertices respectively, and $\partial_1 \circ \partial_2 = 0$
(boundary of a boundary is empty — i.e., face boundaries form closed loops).

### Elements

| Symbol | Description | Count |
|--------|-------------|-------|
| $F_k \in \{0, 1\}$ | Whether candidate face patch $k$ is selected | $N_f$ |
| $E_k \in \{0, 1\}$ | Whether arc $k$ is selected | $N_e$ |
| $V_k \in \{0, 1\}$ | Whether vertex $k$ is selected | $N_v$ |

### Adjacency Matrices

| Symbol | Description | Size |
|--------|-------------|------|
| $\text{FE}[i, j] \in \{0, 1\}$ | Face $i$ is adjacent to edge $j$ | $N_f \times N_e$ |
| $\text{EV}[i, j] \in \{0, 1\}$ | Edge $i$ is adjacent to vertex $j$ | $N_e \times N_v$ |

These are **decision variables** in the ILP — we jointly decide which elements
exist and how they connect.

---

## 4. Constraints

### 4.1 Edge Manifoldness

Every selected edge must be adjacent to exactly 2 selected faces (or 0 if the
edge is not selected):

$$\sum_{i} \text{FE}[i, j] = 2 \, E[j], \quad \forall \, j \in [N_e] \tag{1}$$

This ensures that every edge in the result is shared by exactly two faces —
the defining property of a manifold surface.

### 4.2 Edge Endpoint Consistency

Every selected open edge must have exactly 2 endpoint vertices:

$$\sum_{j} \text{EV}[i, j] = 2 \, E[i], \quad \forall \, i \in [N_e] \tag{2}$$

(For closed edges like full circles, the right-hand side would be 0.  In our
pipeline, closed curves are handled separately.)

### 4.3 Face Boundary Closedness

The boundary of every face must be a closed loop:

$$\text{FE} \times \text{EV} = 2 \, \text{FV} \tag{3}$$

where $\text{FV}[i, j]$ indicates that face $i$ has vertex $j$ on its boundary.
This encodes $\partial_1 \circ \partial_2 = 0$: following the boundary of a face
through its edges must return to the starting vertex.

### 4.4 Dependency Constraints

An adjacency can only exist if both elements exist:

$$\text{FE}[i, j] \leq F[i], \quad \text{FE}[i, j] \leq E[j], \quad \forall \, i, j$$

$$\text{EV}[i, j] \leq E[i], \quad \text{EV}[i, j] \leq V[j], \quad \forall \, i, j$$

These are not present in PolyFit (which has fixed adjacency) but are needed
here because we are jointly selecting elements.

---

## 5. Objective Function

### 5.1 Data Fidelity

Each candidate face patch $f_k$ lies on surface $S_i$.  Define its **support**
as the fraction of points in cluster $P_i$ that project onto patch $f_k$
within a distance threshold $\tau$:

$$\text{support}(f_k) = \frac{|\{p \in P_i : d(p, f_k) < \tau\}|}{|P_i|}$$

The data fidelity term encourages selecting patches with high point cloud
support:

$$F_{\text{data}} = \sum_{k=1}^{N_f} \text{support}(f_k) \cdot F_k$$

### 5.2 Coverage

Penalize uncovered point cloud regions.  For each cluster $P_i$, at least one
of its candidate patches should be selected:

$$F_{\text{cover}} = -\sum_{i=1}^{n} \left(1 - \min\left(1, \sum_{k \in \text{patches}(S_i)} F_k\right)\right)$$

(This can be linearized with auxiliary variables.)

### 5.3 Simplicity

Prefer simpler models with fewer elements:

$$F_{\text{simple}} = -\alpha \sum_k F_k - \beta \sum_k E_k$$

where $\alpha, \beta > 0$ are small weights penalizing unnecessary complexity.

### 5.4 Full Objective

$$\max \quad F_{\text{data}} + F_{\text{cover}} + F_{\text{simple}}$$

$$\text{s.t.} \quad \text{constraints (1)--(4)}$$

$$F_k, E_k, V_k, \text{FE}[i,j], \text{EV}[i,j] \in \{0, 1\}$$

---

## 6. Simplification: Fixed Adjacency

In our pipeline, the adjacency structure is **known from geometry** rather than
being a decision variable.  We know exactly:

- Which arcs bound which patches (from `BOPAlgo_BuilderFace`)
- Which vertices bound which arcs (from `build_edge_arcs`)

This means $\text{FE}$ and $\text{EV}$ are **fixed binary matrices**, not
decision variables.  The ILP simplifies dramatically:

### Simplified Variables

Only $F_k, E_k, V_k \in \{0, 1\}$ — one binary variable per candidate
element.

### Simplified Constraints

Using the known fixed adjacency matrices $A_{fe}$ (face-edge) and $A_{ev}$
(edge-vertex):

**(1') Edge manifoldness:**

$$\sum_{i} A_{fe}[i, j] \cdot F_i = 2 \, E_j, \quad \forall \, j$$

**(2') Edge endpoints:**

$$\sum_{j} A_{ev}[i, j] \cdot V_j = 2 \, E_i, \quad \forall \, i$$

**(3') Dependency:**

$$E_j \leq F_i \quad \text{if } A_{fe}[i, j] = 1 \quad (\text{edge exists only if its faces exist})$$

$$V_j \leq E_i \quad \text{if } A_{ev}[i, j] = 1 \quad (\text{vertex exists only if its edges exist})$$

### Simplified Objective

$$\max \quad \sum_k \text{support}(f_k) \cdot F_k - \alpha \sum_k F_k$$

This is a much smaller ILP with $N_f + N_e + N_v$ binary variables and
linear constraints only.  Tractable with PuLP + CBC for typical problem sizes
(tens to low hundreds of elements).

---

## 7. Solver

We use **PuLP** with the bundled **CBC** (COIN-OR Branch and Cut) solver.
CBC is open-source and handles ILPs of this size efficiently.

```
pip install pulp
```

---

## 8. Pipeline Summary

```
Point cloud (segmented)
    │
    ▼
Fit primitive surfaces (existing)
    │
    ▼
Compute pairwise intersections → curves (existing)
    │
    ▼
Compute vertices at triple intersections (existing)
    │
    ▼
Split curves at vertices → arcs (existing)
    │
    ▼
BOPAlgo_BuilderFace per surface → candidate patches  (new)
    │
    ▼
Extract adjacency matrices A_fe, A_ev  (new)
    │
    ▼
Compute support(f_k) per patch  (new)
    │
    ▼
Solve ILP with PuLP/CBC  (new)
    │
    ▼
Build STEP from selected patches  (new)
```

Steps 1–4 are already implemented.  Steps 5–8 are new.

---

## 9. Comparison with Existing Approaches

| Aspect | Current Pipeline | This Approach | ComplexGen |
|--------|-----------------|---------------|-----------|
| **Candidate generation** | Fitted surfaces + intersections | Same | Neural network |
| **Topology construction** | Manual wire assembly | ILP with manifold constraints | ILP with manifold constraints |
| **Manifold guarantee** | No (heuristic) | Yes (hard constraint) | Yes (hard constraint) |
| **Failure mode** | Catastrophic (0 or 1) | Graceful (partial model) | Graceful |
| **Curved surfaces** | Yes | Yes | Yes |
| **Solver** | N/A | PuLP + CBC (free) | Gurobi (commercial) |
| **Missing intersections** | Cascading failure | Fewer candidates | Fewer candidates |
