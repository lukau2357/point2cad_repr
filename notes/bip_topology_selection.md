# Binary Integer Programming for Topology Selection

## Motivation

The current pipeline is **one-shot**: MSI extraction produces candidate arcs, the oracle
filter greedily removes arcs to achieve Eulerian face graphs, and wire assembly builds
boundary loops. If the greedy filter fails (e.g. abc_00000008: 12/21 faces non-Eulerian,
all 126 arcs removed), the entire model is lost.

ComplexGen (Guo et al., 2022) solves this with a BIP that simultaneously selects which
**vertices, edges, and faces** exist, along with their mutual adjacency, subject to
structural constraints that guarantee a valid B-Rep chain complex.

## B-Rep Chain Complex (ComplexGen Section 3.1)

A B-Rep model of order 3 is $C = (V, E, F, \partial, \mathcal{P})$ with:

- 0th-order elements: **vertices** $V = \{v_i\}$
- 1st-order elements: **edges** $E = \{e_j\}$ (curves, may be open or closed)
- 2nd-order elements: **faces** $F = \{f_k\}$ (surface patches)
- Boundary operators $\partial_1, \partial_2$ connecting elements across orders
- Geometric embedding $\mathcal{P}$ (primitive types + parameters)

The boundary operators are encoded as binary adjacency matrices:

- $\mathbf{FE} \in \mathbb{B}^{N_f \times N_e}$ — face-edge adjacency (which edges bound which faces)
- $\mathbf{EV} \in \mathbb{B}^{N_e \times N_v}$ — edge-vertex adjacency (which vertices are endpoints of which edges)
- $\mathbf{FV} \in \mathbb{B}^{N_f \times N_v}$ — face-vertex adjacency (derived: which vertices touch which faces)

Plus unary existence vectors:

- $\mathbf{F}[k] \in \{0,1\}$ — does face $k$ exist?
- $\mathbf{E}[j] \in \{0,1\}$ — does edge $j$ exist?
- $\mathbf{V}[i] \in \{0,1\}$ — does vertex $i$ exist?
- $\mathbf{O}[j] \in \{0,1\}$ — is edge $j$ open? (1=open with endpoints, 0=closed loop)

## Structural Validity Constraints

For a valid closed manifold B-Rep:

$$\sum_i \mathbf{FE}[i, j] = 2\,\mathbf{E}[j], \quad \forall j \in [N_e] \tag{1}$$

**Each edge is adjacent to exactly 2 faces** (edge manifoldness). If $\mathbf{E}[j]=0$,
both sides are forced to 0.

$$\sum_j \mathbf{EV}[i, j] = 2\,\mathbf{E}[i]\,\mathbf{O}[i], \quad \forall i \in [N_e] \tag{2}$$

**An open edge has exactly 2 endpoint vertices; a closed edge has 0.** The product
$\mathbf{E}[i]\,\mathbf{O}[i]$ is linearizable since both are binary.

$$\mathbf{FE} \times \mathbf{EV} = 2\,\mathbf{FV} \tag{3}$$

**Face boundary closedness.** For each face-vertex pair, the number of boundary edges
connecting them is even — the edges around each face form closed loops. This is the
quadratic constraint that ComplexGen linearizes.

### Dependency constraints

$$\mathbf{FE}[i, j] \leq \mathbf{F}[i] \leq \sum_j \mathbf{FE}[i, j], \quad \forall i, j$$

$$\mathbf{EV}[i, j] \leq \mathbf{V}[j] \leq \sum_k \mathbf{EV}[k, j], \quad \forall i, j$$

A face exists iff it has at least one boundary edge. A vertex exists iff it's an endpoint
of at least one edge. Excluding a face forces all its FE entries to 0; excluding a vertex
forces all its EV entries to 0.

## Mapping to Our Pipeline

In our pipeline, ComplexGen's "edge" corresponds to an **arc** (a segment of an
intersection curve between two surfaces, split at junction vertices):

| ComplexGen | Our pipeline |
|-----------|-------------|
| Face $f_k$ | Fitted surface $k$ (plane, cylinder, sphere, cone) |
| Edge $e_j$ | Arc $a_j$ — segment of intersection curve between surfaces $(i_1, i_2)$ |
| Vertex $v_i$ | Junction vertex $v_i$ — point where 3+ surfaces meet |
| $\mathbf{FE}[k, j]$ | Is arc $a_j$ a boundary of face $f_k$? (Structurally: 1 iff $k \in \{i_1, i_2\}$ for arc $j$'s edge key) |
| $\mathbf{EV}[j, i]$ | Is vertex $v_i$ an endpoint of arc $a_j$? (Structurally: 1 iff $v_i \in \{v_s(j), v_e(j)\}$) |
| $\mathbf{O}[j]$ | Is arc $a_j$ open? (1 if it has endpoints, 0 if closed loop) |

### Key difference from ComplexGen

In ComplexGen, FE and EV are **free decision variables** — the neural network provides
probabilistic predictions, and the BIP decides the final topology. In our pipeline, FE
and EV are **structurally determined** by the arc's edge key and endpoint vertices:

- Arc $a_j$ with edge key $(i_1, i_2)$ has $\mathbf{FE}[i_1, j] = \mathbf{FE}[i_2, j] = \mathbf{E}[j]$
  and $\mathbf{FE}[k, j] = 0$ for all other $k$.
- Arc $a_j$ with endpoints $(v_s, v_e)$ has $\mathbf{EV}[j, v_s] = \mathbf{EV}[j, v_e] = \mathbf{E}[j]$
  (for open arcs) and $\mathbf{EV}[j, \cdot] = 0$ (for closed arcs).

This means **FE and EV are not free variables** — they are determined by the arc
existence variable $\mathbf{E}[j]$. The BIP reduces to selecting which arcs, vertices, and
faces to include, and the adjacency matrices follow.

### Simplified decision variables

- $\mathbf{E}[j] \in \{0, 1\}$ for each arc $j$ — **primary decision**
- $\mathbf{F}[k] \in \{0, 1\}$ for each face $k$ — include this face?
- $\mathbf{V}[i] \in \{0, 1\}$ for each vertex $i$ — include this vertex?

All FE, EV, FV entries are derived:

$$\mathbf{FE}[k, j] = \mathbf{E}[j] \cdot \mathbb{1}[k \in \text{faces}(a_j)]$$

$$\mathbf{EV}[j, i] = \mathbf{E}[j] \cdot \mathbb{1}[i \in \text{endpoints}(a_j)]$$

## Constraints Applied to Our Pipeline

### Constraint (1): Edge-manifoldness (automatically satisfied)

$$\sum_k \mathbf{FE}[k, j] = 2\,\mathbf{E}[j]$$

Every arc in our pipeline already borders exactly 2 faces by construction (the two
surfaces of its edge key). **This constraint is automatically satisfied** — no need to
enforce it.

### Constraint (2): Open/closed edge endpoints (automatically satisfied)

$$\sum_i \mathbf{EV}[j, i] = 2\,\mathbf{E}[j]\,\mathbf{O}[j]$$

Also automatically satisfied: open arcs have exactly 2 endpoint vertices, closed arcs
have 0. The openness $\mathbf{O}[j]$ is known from the arc data.

### Constraint (3): Face boundary closedness — THE CORE CONSTRAINT

$$\sum_j \mathbf{FE}[k, j] \cdot \mathbf{EV}[j, i] = 2\,\mathbf{FV}[k, i], \quad \forall k, i$$

Substituting our derived expressions:

$$\sum_{j : k \in \text{faces}(a_j)} \mathbf{E}[j] \cdot \mathbb{1}[i \in \text{endpoints}(a_j)] = 2\,\mathbf{FV}[k, i]$$

The left side counts: **how many selected arcs on face $k$ are incident to vertex $i$**.
The right side forces this count to be even (0, 2, 4, ...).

Define the incidence set $A(k, i) = \{j \mid k \in \text{faces}(a_j) \text{ and } i \in \text{endpoints}(a_j)\}$.
Then:

$$\sum_{j \in A(k,i)} \mathbf{E}[j] = 2\,\mathbf{FV}[k, i], \quad \mathbf{FV}[k,i] \in \mathbb{Z}_{\geq 0}$$

This is the **Eulerian parity constraint**: on every face, every vertex must have even
degree. This is exactly what the greedy oracle filter fails to achieve.

### Dependency constraints

$$\mathbf{E}[j] \leq \mathbf{F}[k] \quad \forall j, k \text{ where } k \in \text{faces}(a_j)$$

If a face is excluded, all its arcs are excluded.

$$\mathbf{F}[k] \leq \sum_{j : k \in \text{faces}(a_j)} \mathbf{E}[j] \quad \text{(for non-closed faces)}$$

If a non-closed face has no selected arcs, exclude it. Exception: faces bounded entirely
by closed arcs (e.g. a sphere cap bounded by a single closed circle) — these can exist
with only closed arcs contributing.

$$\mathbf{E}[j] \leq \mathbf{V}[i] \quad \forall j, i \text{ where } i \in \text{endpoints}(a_j) \text{ and } a_j \text{ is open}$$

If an arc is selected, its endpoint vertices must exist.

## Objective Function

### Topology fitness (prefer including elements)

$$F_{topo} = \sum_j w^E_j (2\tilde{E}[j] - 1) \cdot \mathbf{E}[j] + \sum_k w^F_k (2\tilde{F}[k] - 1) \cdot \mathbf{F}[k] + \sum_i w^V_i (2\tilde{V}[i] - 1) \cdot \mathbf{V}[i]$$

where $\tilde{E}[j], \tilde{F}[k], \tilde{V}[i] \in (0, 1)$ are confidence scores for each element.

The $(2\tilde{x} - 1)$ term means:
- If $\tilde{x} > 0.5$: positive coefficient, solver is rewarded for including the element
- If $\tilde{x} < 0.5$: negative coefficient, solver is penalized for including it

### Confidence scores for our pipeline

**Arcs:** $\tilde{E}[j] = \exp(-\sigma_j^2 / \epsilon^2)$ where $\sigma_j$ is the arc's fitness score
(lower fitness = higher confidence). Or simply $\tilde{E}[j] = 1 / (1 + \sigma_j)$.

**Faces:** $\tilde{F}[k] = 1$ for all faces (we trust the surface fitter — all fitted
surfaces should exist). Could lower this for faces with high fitting residual.

**Vertices:** $\tilde{V}[i] = 1 / (1 + s_i)$ where $s_i$ is the vertex's oracle score.

### Geometry fitness (prefer geometrically consistent elements)

$$F_{geom} = \sum_j w^E_j \, S_E[j] \cdot \mathbf{E}[j]$$

where $S_E[j]$ measures how well arc $j$'s geometry matches its adjacent faces (e.g.
curve-to-surface distance from the diagnostic).

### Combined objective

$$\max \quad w \cdot F_{topo} + (1 - w) \cdot F_{geom}$$

ComplexGen uses $w = 0.5$.

## Linearization of Constraint (3)

The original constraint (3) is $\mathbf{FE} \times \mathbf{EV} = 2\mathbf{FV}$, which is quadratic
in ComplexGen's formulation (FE and EV are both decision variables).

**In our pipeline, it's already linear.** Since FE and EV are determined by E[j], the
constraint reduces to:

$$\sum_{j \in A(k,i)} \mathbf{E}[j] = 2\,s_{k,i}, \quad s_{k,i} \in \mathbb{Z}_{\geq 0}$$

This is a standard linear constraint with integer auxiliary variables. No linearization
trick needed — a significant simplification over ComplexGen.

## What BIP Solves and What It Does Not

### Solves

- **Non-Eulerian faces**: enforces even vertex degree on every face globally, not
  greedily. Coupled decisions across faces are handled natively: removing an arc affects
  both its faces simultaneously.
- **Face exclusion**: can drop entire faces that can't be consistently bounded, rather
  than destroying the whole model. A valid model with 18/21 faces is better than 0/21.
- **Vertex exclusion**: spurious junction vertices can be excluded along with their
  incident arcs.
- **Optimal subset**: maximizes total quality subject to constraints, globally — not a
  greedy approximation.

### Does NOT solve

- **Wire assembly**: BIP determines which arcs exist, not how they're grouped into wires
  (outer vs. hole boundaries). Wire assembly still needed after BIP, but on guaranteed
  Eulerian graphs where the edge-continuity heuristic is reliable.
- **Missing geometry**: if MSI failed to extract an intersection curve (e.g. edge (2,17)
  with 0 arcs in abc_00000008), BIP cannot create it. It works with what's available.
- **Curve quality**: BIP doesn't improve B-spline curves that poorly fit their surfaces.
- **Redundant intersections**: MSI may find duplicate intersection curves (same physical
  curve extracted multiple times). BIP will keep only the consistent subset, but
  pre-filtering duplicates would reduce problem size.

## Comparison to Current Oracle Filter

| Aspect | Oracle Filter (greedy) | BIP |
|--------|----------------------|-----|
| Strategy | Remove worst arc, recheck | Global optimization |
| Coupled faces | Ignores cross-face effects | Handles natively |
| Face exclusion | No — all or nothing | Yes — can drop problematic faces |
| Guarantee | May remove everything | Finds optimal subset (if feasible) |
| Speed | Fast (linear scan) | Seconds for ~100-500 variables |
| Failure mode | Empty model (0 faces) | Infeasible (explicitly reported), or partial model |

## Problem Size

For abc_00000008 ($N_a = 126$ arcs, $N_f = 21$ faces, $N_v = 57$ vertices):

| Variable type | Count |
|--------------|-------|
| $\mathbf{E}[j]$ (arc existence) | 126 |
| $\mathbf{F}[k]$ (face existence) | 21 |
| $\mathbf{V}[i]$ (vertex existence) | 57 |
| $s_{k,i}$ (parity auxiliary) | ~300 (non-empty face-vertex pairs) |
| **Total variables** | **~500** |
| Parity constraints | ~300 |
| Dependency constraints | ~400 |
| **Total constraints** | **~700** |

This is a small ILP — solvable in seconds by any modern solver.

### Solver options (Python)

- `scipy.optimize.milp` — no extra dependencies, sufficient for this problem size
- `python-mip` (CBC backend) — pip-installable, more diagnostics
- `gurobipy` — fastest, free academic license

## After BIP: Remaining Pipeline

1. **BIP** selects arcs, faces, vertices (guaranteed Eulerian per face)
2. **Wire assembly** partitions arcs into closed loops per face
   - Degree-2 vertices: unique continuation (no ambiguity)
   - Degree-4+ vertices: edge-continuity heuristic (follow same edge_key)
3. **Wire classification**: largest area wire = outer boundary, rest = holes
4. **OCC BRep construction**: `BRepBuilderAPI_MakeFace`, sewing, shape fixing
