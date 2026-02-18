# Intersection Curve Trimming

## Overview

The intersection between two fitted surfaces yields an unbounded or fully
periodic `Geom_Curve`.  For B-Rep construction only the portion that lies on
the actual shared boundary between the two point-cloud clusters is needed.
This section describes how to determine that portion and create a
`Geom_TrimmedCurve`.

---

## Trimming strategy

### Why not the UV convex hull

An alternative approach is to compute the convex hull of each cluster in the
UV parameter domain of its surface, convert the hull vertices to 3-D, and
project them onto the intersection curve to obtain the trim interval.  This
is not used here for two reasons:

1. **Convexity assumption.**  The face boundary is convex in UV space only
   for axis-aligned rectangular domains (e.g. a plane patch or a cylinder
   cut along a generator).  For INR surfaces, whose UV domain is an
   irregular subset of $[-1,1]^2$, and for any non-convex face shape, the
   convex hull over-extends the boundary.

2. **Redundancy.**  After converting hull vertices to 3-D they must still be
   projected onto the intersection curve — reducing to the same step as
   the boundary-strip approach but with a worse choice of input points.

### Boundary-strip approach

For an adjacent pair $(i, j)$ the boundary strip is the set of cluster
points that are observed to lie near the shared edge.  These are extracted
directly from the adjacency KDTree already built during
`compute_adjacency_matrix`:

- **Strip of cluster $i$:** points in cluster $i$ whose nearest-neighbour
  distance to cluster $j$ is at most the adjacency threshold.
- **Strip of cluster $j$:** symmetrically.

The union of both strips gives the boundary points $\{\mathbf{q}_k\}$.
These points are used as input to the projection step.

This approach makes no assumption about face convexity or surface type and
works identically for planes, cylinders, cones, and INR surfaces.

---

## Point-to-curve projection

### Problem statement

Given a parametric curve $\mathbf{C} : [t_0, t_1] \to \mathbb{R}^3$ and a
point $\mathbf{P} \in \mathbb{R}^3$, find the parameter value of the closest
point on the curve:

$$t^* = \operatorname*{arg\,min}_{t \in [t_0,\, t_1]}
        f(t), \quad
        f(t) = \|\mathbf{C}(t) - \mathbf{P}\|_2^2$$

### Orthogonality condition

Differentiating $f$ and setting to zero:

$$f'(t) = 2\bigl(\mathbf{C}(t) - \mathbf{P}\bigr) \cdot \mathbf{C}'(t) = 0$$

This is the **orthogonality condition**: at the closest point the tangent
vector $\mathbf{C}'(t^*)$ is perpendicular to the vector
$\mathbf{C}(t^*) - \mathbf{P}$ from $\mathbf{P}$ to the curve.  It is a
necessary condition for an interior minimum; the boundary values $t_0$ and
$t_1$ must also be checked.

For **lines** ($\mathbf{C}(t) = \mathbf{p}_0 + t\mathbf{d}$, $\mathbf{d}$
unit) the condition is linear and gives the closed-form solution:

$$t^* = (\mathbf{P} - \mathbf{p}_0) \cdot \mathbf{d}$$

For **circles** ($\mathbf{C}(t) = \mathbf{c} + r\cos(t)\,\mathbf{u} +
r\sin(t)\,\mathbf{v}$) the closest point is the angular projection of
$\mathbf{P} - \mathbf{c}$ onto the circle plane:

$$t^* = \operatorname{atan2}\!\bigl((\mathbf{P}-\mathbf{c})\cdot\mathbf{v},\;
        (\mathbf{P}-\mathbf{c})\cdot\mathbf{u}\bigr)$$

For **arbitrary curves** (B-splines) $f'(t) = 0$ is a nonlinear scalar
equation with potentially many roots, requiring a numerical solver.

### Newton–Raphson refinement

Starting from an initial guess $t_n$, Newton–Raphson iterates:

$$t_{n+1} = t_n - \frac{f'(t_n)}{f''(t_n)}$$

The second derivative is:

$$f''(t) = \|\mathbf{C}'(t)\|^2 + \bigl(\mathbf{C}(t) - \mathbf{P}\bigr) \cdot \mathbf{C}''(t)$$

The first term is always non-negative.  The second term — the dot product of
the position error with the curve acceleration — can be negative when the
curvature is large relative to the distance to $\mathbf{P}$, making $f''$
zero or negative and the iteration non-convergent from a poor starting
point.  A good initial guess is therefore essential.

### Multiple local minima and initialisation

$f'(t) = 0$ can have multiple roots: every local closest point satisfies the
orthogonality condition.  Newton–Raphson from a single start finds only one.
The standard initialisation strategy is:

1. **Coarse sampling:** evaluate $\mathbf{C}$ at $N$ uniformly spaced
   parameter values and compute $\|\mathbf{C}(t_k) - \mathbf{P}\|$.
2. **Select the closest sample** as the starting point for Newton–Raphson.
3. **Refine** until $|t_{n+1} - t_n| < \varepsilon$ or $f'(t_n) \approx 0$.

This guarantees that the returned root is at least as good as the best
coarse sample, which is close to the global minimum when $N$ is large enough
relative to the curve's variation.

### OCC implementation: `GeomAPI_ProjectPointOnCurve`

Documentation: <https://dev.opencascade.org/doc/refman/html/class_geom_a_p_i___project_point_on_curve.html>

```python
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve

proj = GeomAPI_ProjectPointOnCurve(point, curve)   # projects onto full domain
# or with explicit bounds:
proj = GeomAPI_ProjectPointOnCurve(point, curve, t0, t1)
```

| Method | Return type | Meaning |
|--------|-------------|---------|
| `NbPoints()` | `int` | Number of local closest points found |
| `LowerDistance()` | `float` | Distance to the globally closest point |
| `LowerDistanceParameter()` | `float` | Parameter $t^*$ of the globally closest point |
| `Distance(k)` | `float` | Distance for the $k$-th extremum (1-indexed) |
| `Parameter(k)` | `float` | Parameter for the $k$-th extremum (1-indexed) |

Internally OCC uses the `Extrema_ExtPC` algorithm, which:

1. Exploits the **convex hull property** of B-spline/Bézier segments: if the
   convex hull of a segment's control polygon is entirely farther from
   $\mathbf{P}$ than the current best distance, the segment is discarded
   without evaluating the curve.
2. Splits the B-spline into its Bézier patches (via knot insertion) and
   applies recursive subdivision to isolate each local minimum to a small
   parameter interval.
3. Refines each isolated root with Newton–Raphson.
4. Reports **all** local extrema.  `LowerDistanceParameter()` returns the
   parameter of the global minimum.

For `Geom_Line` and `Geom_Circle` the same interface is used but the solver
applies the closed-form formulas above rather than the B-spline algorithm.

---

## Computing the trim interval

### Non-periodic curves (lines, B-splines)

Collect the projected parameters $\{t_k\}$ from all boundary-strip points
whose `LowerDistance()` is below a projection tolerance $\varepsilon_\text{proj}$:

$$t_\text{min} = \min_k t_k, \quad t_\text{max} = \max_k t_k$$

### Periodic curves (circles, ellipses — domain $[0, 2\pi]$)

A simple min/max is incorrect when the trimmed arc crosses the $0 / 2\pi$
seam.  Correct procedure:

1. Sort the valid parameters: $t_{(1)} \le t_{(2)} \le \cdots \le t_{(n)}$.
2. Compute the $n$ gaps between consecutive parameters (cyclically):
   $\Delta_k = t_{(k+1)} - t_{(k)}$, with $\Delta_n = 2\pi - t_{(n)} + t_{(1)}$.
3. The largest gap $\Delta_{k^*}$ is the portion of the circle **not** covered
   by the arc.
4. Set $t_\text{min} = t_{(k^*+1)}$ and $t_\text{max} = t_{(k^*)} + 2\pi$
   (i.e. shift $t_\text{max}$ by $2\pi$ if the arc wraps around the seam).

If $\Delta_{k^*}$ is close to $2\pi$ (nearly the whole circle is a gap),
fewer than two valid projections were found and the trim should be skipped or
flagged.

### Creating `Geom_TrimmedCurve`

Documentation: <https://dev.opencascade.org/doc/refman/html/class_geom___trimmed_curve.html>

```python
from OCC.Core.Geom import Geom_TrimmedCurve

trimmed = Geom_TrimmedCurve(curve, t_min, t_max)
```

`Geom_TrimmedCurve` wraps any `Geom_Curve` and overrides
`FirstParameter()` / `LastParameter()` to return $t_\text{min}$ /
$t_\text{max}$.  `sample_curve` already uses these bounds, so visualisation
requires no changes.  For lines the trimmed curve is finite and the
`line_extent` clip in `sample_curve` becomes redundant.

---

## Projection tolerance $\varepsilon_\text{proj}$

A boundary-strip point should only contribute a parameter value if it
actually lies close to the intersection curve.  Points that are in the
boundary strip (close to the other cluster) but far from the intersection
curve are on a different edge and must be excluded.  A reasonable choice is:

$$\varepsilon_\text{proj} = c \cdot \text{spacing}$$

where $\text{spacing}$ is the median nearest-neighbour distance (already
computed) and $c$ is a small constant (e.g. $c = 3$).  This adapts
automatically to the point cloud density.
