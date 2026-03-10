# Surface-Surface Intersection: OCC API and Mathematics

## Overview

Adjacent surface pairs in the B-Rep pipeline are intersected to obtain the
edge curves that bound each face.  Three pairs are handled with closed-form
analytical formulas; every other pair falls back to OpenCASCADE's numerical
solver `GeomAPI_IntSS`.

| Pair | Method | Result type |
|------|--------|-------------|
| Plane ∩ Plane | Analytical | `Geom_Line` |
| Plane ∩ Sphere | Analytical | `Geom_Circle` or tangent `gp_Pnt` |
| Sphere ∩ Sphere | Analytical (radical plane) | `Geom_Circle` or tangent `gp_Pnt` |
| Plane ∩ Cylinder (axis-parallel, tangent) | Analytical fallback after OCC empty | `Geom_Line` (generator line) |
| Cylinder ∩ Cylinder (parallel axes) | Analytical fallback after OCC `IsDone=False` | `Geom_Line`(s) (0, 1, or 2 generator lines) |
| All other pairs | OCC `GeomAPI_IntSS` | `Geom_Curve` (concrete subtype is an OCC implementation detail, not specified in the public API; inspect with `isinstance` at runtime) |

`GeomAPI_IntSS` does not expose a point query method.  When `IsDone()` is
`True` but `NbLines()` is 0, the result is treated as empty; point tangency
(two surfaces touching at a single isolated point) is geometrically
indistinguishable from no intersection at this API level.

---

## OpenCASCADE API

### `GeomAPI_IntSS`

Documentation: <https://dev.opencascade.org/doc/refman/html/class_geom_a_p_i___int_s_s.html>

```python
from OCC.Core.GeomAPI import GeomAPI_IntSS

inter = GeomAPI_IntSS(surf1, surf2, tol)   # tol is the 3-D tolerance
```

| Method | Return type | Meaning |
|--------|-------------|---------|
| `IsDone()` | `bool` | Whether the computation succeeded |
| `NbLines()` | `int` | Number of intersection curves |
| `Line(k)` | `Geom_Curve` | $k$-th curve (1-indexed) |

These are the only three public methods documented for this class.  There is
no point query (`NbPoints`, `Point`) — those belong to other intersection
classes such as `GeomAPI_IntCS` (curve–surface).  The concrete subtype
returned by `Line(k)` is not specified in the documentation; use
`isinstance` to inspect it at runtime.

### `Geom_Curve` — parameter domain and evaluation

Documentation: <https://dev.opencascade.org/doc/refman/html/class_geom___curve.html>

Every `Geom_Curve` returned by `GeomAPI_IntSS.Line(k)` or constructed
analytically is a 1-D parametric object: it defines a map $t \mapsto
\mathbf{C}(t) \in \mathbb{R}^3$ over a closed interval
$[t_0, t_1]$.  `FirstParameter()` and `LastParameter()` are the
bounds of that interval — they are not spatial coordinates but values of
the curve parameter.  Calling `Value(t)` for any $t \in [t_0, t_1]$
returns the corresponding 3-D point on the curve.

| Method | Meaning |
|--------|---------|
| `FirstParameter()` | $t_0$: lower bound of the parameter interval |
| `LastParameter()` | $t_1$: upper bound of the parameter interval |
| `Value(t)` → `gp_Pnt` | 3-D point $\mathbf{C}(t)$ on the curve |

Natural parameter intervals by concrete subtype:

| Curve type | `FirstParameter()` | `LastParameter()` | OCC docs |
|------------|-------------------|-------------------|----------|
| `Geom_Line` | $-\infty$ | $+\infty$ | [link](https://dev.opencascade.org/doc/refman/html/class_geom___line.html) |
| `Geom_Circle` | $0$ | $2\pi$ | [link](https://dev.opencascade.org/doc/refman/html/class_geom___circle.html) |
| `Geom_Ellipse` | $0$ | $2\pi$ | [link](https://dev.opencascade.org/doc/refman/html/class_geom___ellipse.html) |
| `Geom_Conic` (abstract base) | — | — | [link](https://dev.opencascade.org/doc/refman/html/class_geom___conic.html) |
| `Geom_BSplineCurve` | first knot value | last knot value | [link](https://dev.opencascade.org/doc/refman/html/class_geom___b_spline_curve.html) |

`Geom_Conic` is the abstract base of `Geom_Circle`, `Geom_Ellipse`,
`Geom_Parabola`, and `Geom_Hyperbola`.  The `Geom_Circle` check in
`_classify_occ_curve` must precede the `Geom_Conic` check because a
circle is also a conic.

For lines the interval is infinite; `sample_curve` clips it to
$[-\texttt{line\_extent},\, +\texttt{line\_extent}]$ before sampling.
`line_extent` should be on the order of the point cloud diameter (≈ 1 in
normalised coordinates).

### `Geom_Line`

Documentation: <https://dev.opencascade.org/doc/refman/html/class_geom___line.html>

Constructed from a `gp_Lin`
([docs](https://dev.opencascade.org/doc/refman/html/classgp___lin.html)):

```python
from OCC.Core.gp   import gp_Lin, gp_Pnt, gp_Dir
from OCC.Core.Geom import Geom_Line

line = Geom_Line(gp_Lin(gp_Pnt(px, py, pz), gp_Dir(dx, dy, dz)))
```

- `gp_Pnt` ([docs](https://dev.opencascade.org/doc/refman/html/classgp___pnt.html)) — a point on the line
- `gp_Dir` ([docs](https://dev.opencascade.org/doc/refman/html/classgp___dir.html)) — unit direction vector; `gp_Dir` normalises automatically

### `Geom_Circle`

Documentation: <https://dev.opencascade.org/doc/refman/html/class_geom___circle.html>

Constructed from a `gp_Circ`
([docs](https://dev.opencascade.org/doc/refman/html/classgp___circ.html)),
which requires a `gp_Ax2` coordinate system
([docs](https://dev.opencascade.org/doc/refman/html/classgp___ax2.html))
and a radius:

```python
from OCC.Core.gp   import gp_Ax2, gp_Pnt, gp_Dir, gp_Circ
from OCC.Core.Geom import Geom_Circle

ax2    = gp_Ax2(origin_pnt, normal_dir, x_dir)
circle = Geom_Circle(gp_Circ(ax2, radius))
```

`gp_Ax2` defines a right-handed coordinate frame:
- `origin_pnt` — centre of the circle
- `normal_dir` — axis of the circle (= plane normal for a plane–sphere intersection)
- `x_dir` — any unit vector **perpendicular** to `normal_dir`; fixes the
  parametric start point ($t = 0$) but does not affect geometry

---

## Analytical intersections — mathematics

### Plane ∩ Plane → Line

**Setup.**  Each plane is given by its unit normal and scalar offset:

$$P_1 : \mathbf{n}_1 \cdot \mathbf{x} = d_1 \qquad P_2 : \mathbf{n}_2 \cdot \mathbf{x} = d_2$$

**Direction of the intersection line.**  The line lies in both planes, so it
must be perpendicular to both normals:

$$\mathbf{dir} = \mathbf{n}_1 \times \mathbf{n}_2$$

If $\|\mathbf{dir}\| < \varepsilon$ the planes are parallel (or coincident)
and there is no intersection line.

**A point on the line.**  We need any point satisfying both plane equations:

$$A\mathbf{p} = \mathbf{b}, \quad
  A = \begin{pmatrix} \mathbf{n}_1^\top \\ \mathbf{n}_2^\top \end{pmatrix}
  \in \mathbb{R}^{2 \times 3}, \quad
  \mathbf{b} = \begin{pmatrix} d_1 \\ d_2 \end{pmatrix}$$

This system is underdetermined: $A$ has $m = 2$ rows and $n = 3$ columns
with $\operatorname{rank}(A) = 2$, so the solution set is a 1-D affine
subspace (the line itself).  Any member is a valid point, but we want a
canonical one.

**Step 1 — Naïve approach: minimise the residual.**  A natural first
attempt is to treat this as a least-squares problem,

$$\min_{\mathbf{p}} f(\mathbf{p}) = \|A\mathbf{p} - \mathbf{b}\|_2^2$$

Expanding and differentiating:

$$f(\mathbf{p}) = \mathbf{p}^\top A^\top A\, \mathbf{p} - 2\mathbf{b}^\top A\mathbf{p} + \mathbf{b}^\top\mathbf{b}$$

$$\nabla_\mathbf{p} f = 2A^\top A\,\mathbf{p} - 2A^\top\mathbf{b} = \mathbf{0}
  \;\Longrightarrow\; A^\top A\,\mathbf{p} = A^\top\mathbf{b}$$

The Hessian $H = 2A^\top A$ is PSD, so every stationary point is a global
minimum.  In the overdetermined (tall) case where $A$ has full column rank,
$A^\top A$ is PD (strictly positive definite) and invertible, yielding the
familiar unique solution $(A^\top A)^{-1}A^\top\mathbf{b}$.

**Step 2 — Why this fails here.**  $A^\top A \in \mathbb{R}^{3 \times 3}$
has $\operatorname{rank}(A^\top A) = \operatorname{rank}(A) = 2$.  By the
rank-nullity theorem:

$$\dim(\ker A) = n - \operatorname{rank}(A) = 3 - 2 = 1$$

so $A^\top A$ is **always singular** — not as a numerical accident but as an
inherent structural property.  The null vector of $A$ (and therefore of
$A^\top A$) is precisely $\mathbf{n}_1 \times \mathbf{n}_2$, the line
direction already computed: $A(\mathbf{n}_1 \times \mathbf{n}_2) = \mathbf{0}$
because the cross product is perpendicular to both rows of $A$.

Geometrically this is obvious: if $\mathbf{p}^*$ solves $A\mathbf{p} =
\mathbf{b}$, then so does $\mathbf{p}^* + t(\mathbf{n}_1 \times
\mathbf{n}_2)$ for any $t \in \mathbb{R}$.  Adding any multiple of the line
direction to a solution gives another solution, so the residual is zero on
the entire affine line; $f$ is flat along it.  The normal equations
$A^\top A\,\mathbf{p} = A^\top\mathbf{b}$ are therefore consistent but
rank-deficient, and $(A^\top A)^{-1}$ does not exist.

**Step 3 — Extended problem: minimum-norm solution.**  Since every point on
the line is a least-squares minimiser (residual $= 0$), we break the tie by
also minimising the norm.  The problem becomes:

$$\min_{\mathbf{p}} \tfrac{1}{2}\|\mathbf{p}\|_2^2
  \quad \text{subject to} \quad A\mathbf{p} = \mathbf{b}$$

Introduce Lagrange multipliers $\boldsymbol{\lambda} \in \mathbb{R}^2$ (one
per constraint):

$$\mathcal{L}(\mathbf{p}, \boldsymbol{\lambda})
  = \tfrac{1}{2}\|\mathbf{p}\|_2^2
  + \boldsymbol{\lambda}^\top(A\mathbf{p} - \mathbf{b})$$

Setting the gradient with respect to $\mathbf{p}$ to zero:

$$\nabla_\mathbf{p}\,\mathcal{L} = \mathbf{p} + A^\top\boldsymbol{\lambda} = \mathbf{0}
  \;\Longrightarrow\; \mathbf{p}^* = -A^\top\boldsymbol{\lambda}$$

Substituting into the primal feasibility condition $A\mathbf{p}^* =
\mathbf{b}$:

$$A(-A^\top\boldsymbol{\lambda}) = \mathbf{b}
  \;\Longrightarrow\; AA^\top\boldsymbol{\lambda} = -\mathbf{b}
  \;\Longrightarrow\; \boldsymbol{\lambda} = -(AA^\top)^{-1}\mathbf{b}$$

$AA^\top \in \mathbb{R}^{2 \times 2}$ is the Gram matrix of
$\{\mathbf{n}_1, \mathbf{n}_2\}$; it is invertible if and only if the
planes are not parallel (equivalently $\mathbf{dir} \neq \mathbf{0}$),
which is already checked.  Substituting $\boldsymbol{\lambda}$ back:

$$\mathbf{p}_0 = -A^\top\boldsymbol{\lambda} = A^\top(AA^\top)^{-1}\mathbf{b}$$

This is the **right pseudoinverse** $A^+ = A^\top(AA^\top)^{-1}$ applied to
$\mathbf{b}$.  It is the unique point on the intersection line closest to
the origin, and it arises directly from the Lagrangian stationarity
conditions — not from any unexplained formula.

### Plane ∩ Sphere → Circle

**Setup.**

$$\text{Plane} : \mathbf{n} \cdot \mathbf{x} = d \quad (\text{unit normal } \mathbf{n}) \qquad
  \text{Sphere} : \|\mathbf{x} - \mathbf{c}\| = r \quad (\text{centre } \mathbf{c},\ \text{radius } r)$$

**Signed distance from sphere centre to plane.**

$$\delta = \mathbf{n} \cdot \mathbf{c} - d$$

- $|\delta| > r$ → no intersection (sphere entirely on one side)
- $|\delta| = r$ → single tangency point $\mathbf{c} - \delta\mathbf{n}$
- $|\delta| < r$ → circle

**Circle centre.**  Project $\mathbf{c}$ orthogonally onto the plane:

$$\mathbf{p}_0 = \mathbf{c} - \delta\mathbf{n}$$

This is the foot of the perpendicular from $\mathbf{c}$ to the plane, which
lies on the plane by construction ($\mathbf{n} \cdot \mathbf{p}_0 =
\mathbf{n} \cdot \mathbf{c} - \delta = d$).

**Why $\mathbf{p}_0$ is the circle centre.**  Every point $\mathbf{x}$ on
the intersection satisfies $\mathbf{n} \cdot \mathbf{x} = d$ and
$\|\mathbf{x} - \mathbf{c}\| = r$.  Writing $\mathbf{x} = \mathbf{p}_0 +
\mathbf{v}$ where $\mathbf{v}$ is in the plane ($\mathbf{n} \cdot \mathbf{v}
= 0$):

$$\|\mathbf{p}_0 + \mathbf{v} - \mathbf{c}\|^2
  = \|\mathbf{v} - \delta\mathbf{n}\|^2
  = \|\mathbf{v}\|^2 + \delta^2 = r^2$$

so $\|\mathbf{v}\|^2 = r^2 - \delta^2$ — all admissible $\mathbf{v}$ have
the same norm, meaning $\mathbf{x}$ traces a circle centred at $\mathbf{p}_0$.

**Circle radius** (Pythagorean theorem in the right triangle
$\mathbf{c} - \mathbf{p}_0 - \mathbf{x}$):

$$r_\text{circle} = \sqrt{r^2 - \delta^2}$$

### Sphere ∩ Sphere → Circle via the Radical Plane

**Setup.**  Two spheres:

$$S_1 : \|\mathbf{x} - \mathbf{c}_1\|^2 = r_1^2 \qquad
  S_2 : \|\mathbf{x} - \mathbf{c}_2\|^2 = r_2^2$$

**Deriving the radical plane.**  Subtracting the two sphere equations:

$$\|\mathbf{x} - \mathbf{c}_1\|^2 - \|\mathbf{x} - \mathbf{c}_2\|^2 = r_1^2 - r_2^2$$

Expanding and cancelling $\|\mathbf{x}\|^2$:

$$2(\mathbf{c}_2 - \mathbf{c}_1) \cdot \mathbf{x}
  = r_1^2 - r_2^2 + \|\mathbf{c}_2\|^2 - \|\mathbf{c}_1\|^2$$

Setting $d = \|\mathbf{c}_2 - \mathbf{c}_1\|$ and
$\mathbf{n} = (\mathbf{c}_2 - \mathbf{c}_1)/d$:

$$\mathbf{n} \cdot \mathbf{x}
  = \frac{d^2 + r_1^2 - r_2^2}{2d} =: d_\text{radical}$$

This is a plane — the **radical plane** of the two spheres.  Every point
with equal power with respect to both spheres lies on it.  The intersection
circle lies entirely within this plane.

**Position along axis.**  The foot of $\mathbf{c}_1$ onto the radical plane
is at signed distance $h = (d^2 + r_1^2 - r_2^2)/(2d)$ from $\mathbf{c}_1$:

$$\mathbf{p}_0 = \mathbf{c}_1 + h\mathbf{n}$$

The circle radius within the radical plane equals $\sqrt{r_1^2 - h^2}$,
obtained by applying the plane–sphere formula to $S_1$ intersected with the
radical plane.

**Why reducing to plane–sphere is sound.**  Any point $\mathbf{x}$ in the
intersection satisfies both sphere equations; it therefore lies on the
radical plane and on $S_1$.  The set of such points is exactly the
intersection of the radical plane with $S_1$, which is a circle computed by
the plane–sphere formula.

---

### Plane ∩ Cylinder → Generator Line (tangent fallback)

`GeomAPI_IntSS` returns `NbLines() == 0` (indistinguishable from "no intersection")
when a plane is parallel to the cylinder axis and tangent to its surface.  The true
intersection is a single **generator line** (ruling of the cylinder).

**Setup.**

$$\text{Plane}: \mathbf{n}\cdot\mathbf{x} = d \quad(\text{unit normal }\mathbf{n}) \qquad
  \text{Cylinder}: \text{axis direction }\mathbf{a}\text{ (unit), point on axis }\mathbf{c},\text{ radius }r$$

**Substituting a cylinder surface point.**  A generic point on the cylinder:

$$\mathbf{p}(u,v) = \mathbf{c} + v\,\mathbf{a} + r\bigl(\cos u\,\mathbf{e}_1 + \sin u\,\mathbf{e}_2\bigr)$$

where $\mathbf{e}_1, \mathbf{e}_2$ are any orthonormal pair perpendicular to $\mathbf{a}$.
Substituting into the plane equation and letting $\alpha = \mathbf{n}\cdot\mathbf{a}$,
$D = d - \mathbf{n}\cdot\mathbf{c}$, $B = \mathbf{n}\cdot\mathbf{e}_1$,
$C = \mathbf{n}\cdot\mathbf{e}_2$:

$$v\alpha + r(B\cos u + C\sin u) = D$$

**Parallel case $\alpha \approx 0$.**  The $v$ term vanishes.  Let
$\mathbf{n}_\perp = \mathbf{n} - \alpha\mathbf{a}$ (component of $\mathbf{n}$
perpendicular to $\mathbf{a}$); then $B^2 + C^2 = |\mathbf{n}_\perp|^2$ and:

$$r\,|\mathbf{n}_\perp|\,\cos(u - \varphi) = D, \qquad \varphi = \operatorname{atan2}(C, B)$$

The perpendicular distance from the cylinder axis to the plane is
$\delta = |D|/|\mathbf{n}_\perp|$.

| $\delta$ vs $r$ | Intersection |
|-----------------|-------------|
| $\delta > r$ | empty |
| $\delta = r$ | one generator line — **tangent; OCC returns empty** |
| $\delta < r$ | two generator lines — secant; OCC handles correctly |

**Tangent foot.**  For $\delta = r$ the unique solution $u_0$ yields a tangent point
$\mathbf{q}$ on both the plane and the cylinder surface.  Without computing
$\mathbf{e}_1, \mathbf{e}_2$ explicitly:

$$\mathbf{q} = \mathbf{c} + \frac{D}{|\mathbf{n}_\perp|^2}\,\mathbf{n}_\perp$$

*Proof — lies on the plane:*

$$\mathbf{n}\cdot\mathbf{q}
  = \mathbf{n}\cdot\mathbf{c} + \frac{D}{|\mathbf{n}_\perp|^2}
    \underbrace{(\mathbf{n}\cdot\mathbf{n}_\perp)}_{=|\mathbf{n}_\perp|^2}
  = \mathbf{n}\cdot\mathbf{c} + D = d \quad\checkmark$$

($\mathbf{n}\cdot\mathbf{n}_\perp = |\mathbf{n}_\perp|^2$ because
$\mathbf{n} = \mathbf{n}_\perp + \alpha\mathbf{a}$ and
$\mathbf{a}\cdot\mathbf{n}_\perp = 0$.)

*Proof — lies on the cylinder:*

$$\operatorname{dist}(\mathbf{q}, \text{axis})
  = \left|\frac{D}{|\mathbf{n}_\perp|^2}\mathbf{n}_\perp\right|
  = \frac{|D|}{|\mathbf{n}_\perp|} = \delta = r \quad\checkmark$$

The generator line is `Geom_Line(q, a)`.  It is subsequently trimmed to the vertex-bounded
arc segment by the existing `trim_by_vertices` machinery.

**Note on plane–cone tangency.**  A plane can also be tangent to a cone along a single
generator line (when the plane passes through the cone apex and satisfies
$\mathbf{n}\cdot\mathbf{d}_\text{gen} = 0$ for a generator direction
$\mathbf{d}_\text{gen}$ with $\mathbf{d}_\text{gen}\cdot\mathbf{a} = \cos\theta$).
OCC likely has the same limitation there, but the derivation is more involved
(variable generator direction) and the case is less common in mechanical CAD.
Left as a future TODO.

### Cylinder ∩ Cylinder → Generator Lines (parallel-axis fallback)

`GeomAPI_IntSS` returns `IsDone() = False` for certain cylinder–cylinder configurations,
most notably when the axes are parallel.  In this case the intersection reduces to a
2D circle–circle problem and produces 0, 1, or 2 **generator lines** parallel to the
common axis.

**Setup.**

$$\text{Cylinder 1}: \text{axis direction } \mathbf{a}_1 \text{ (unit)}, \text{ point on axis } \mathbf{c}_1, \text{ radius } r_1$$
$$\text{Cylinder 2}: \text{axis direction } \mathbf{a}_2 \text{ (unit)}, \text{ point on axis } \mathbf{c}_2, \text{ radius } r_2$$

**Parallelism check.**  The axes are parallel when $|\mathbf{a}_1 \cdot \mathbf{a}_2| \approx 1$.
If not parallel, this fallback does not apply (the general case is a degree-4 space curve).

**Reduction to 2D.**  Let $\mathbf{a} = \mathbf{a}_1$ be the common axis direction.
Project the axis base points onto the plane perpendicular to $\mathbf{a}$:

$$\mathbf{q}_i = \mathbf{c}_i - (\mathbf{c}_i \cdot \mathbf{a})\,\mathbf{a}, \qquad i = 1, 2$$

A point $\mathbf{x}$ lies on cylinder $i$ iff its perpendicular distance to the axis
equals $r_i$:

$$|\mathbf{x}_\perp - \mathbf{q}_i| = r_i$$

where $\mathbf{x}_\perp$ is the projection of $\mathbf{x}$ onto the same perpendicular
plane.  The 3D intersection thus reduces to two circles in the perpendicular plane:

$$\text{Circle 1}: |\mathbf{p} - \mathbf{q}_1| = r_1, \qquad \text{Circle 2}: |\mathbf{p} - \mathbf{q}_2| = r_2$$

Each 2D intersection point lifts to a full 3D line parallel to $\mathbf{a}$ through
$(p_x, p_y, p_z)$, where the $\mathbf{a}$-component is restored from $\mathbf{c}_1$.

**Circle–circle intersection.**  Let $d = |\mathbf{q}_2 - \mathbf{q}_1|$ and
$\mathbf{e} = (\mathbf{q}_2 - \mathbf{q}_1)/d$ (unit vector between centres).

The intersection points lie at distance $x$ from $\mathbf{q}_1$ along $\mathbf{e}$
and distance $h$ perpendicular to $\mathbf{e}$, where:

$$x = \frac{d^2 + r_1^2 - r_2^2}{2d}, \qquad h^2 = r_1^2 - x^2$$

*Derivation.*  Any point on the intersection satisfies both circle equations:

$$|\mathbf{p} - \mathbf{q}_1|^2 = r_1^2, \qquad |\mathbf{p} - \mathbf{q}_2|^2 = r_2^2$$

Subtracting:

$$|\mathbf{p} - \mathbf{q}_1|^2 - |\mathbf{p} - \mathbf{q}_2|^2 = r_1^2 - r_2^2$$

Expanding and using $\mathbf{q}_2 - \mathbf{q}_1 = d\mathbf{e}$:

$$2d\,(\mathbf{e} \cdot (\mathbf{p} - \mathbf{q}_1)) = d^2 + r_1^2 - r_2^2$$

So the component of $\mathbf{p} - \mathbf{q}_1$ along $\mathbf{e}$ is $x = (d^2 + r_1^2 - r_2^2)/(2d)$.
The perpendicular component squared follows from Pythagoras on circle 1:
$h^2 = r_1^2 - x^2$.

**Case analysis.**

| Condition | $h^2$ | Intersection |
|-----------|-------|-------------|
| $d > r_1 + r_2$ | $h^2 < 0$ | empty (too far apart) |
| $d = r_1 + r_2$ | $h^2 = 0$ | 1 generator line (external tangent) |
| $|r_1 - r_2| < d < r_1 + r_2$ | $h^2 > 0$ | **2 generator lines** (secant) |
| $d = |r_1 - r_2|$ | $h^2 = 0$ | 1 generator line (internal tangent) |
| $d < |r_1 - r_2|$ | $h^2 < 0$ | empty (one inside the other) |

**Coaxial case $d = 0$.**  If $r_1 \neq r_2$ the cylinders are nested concentrically
(no intersection).  If $r_1 = r_2$ the cylinders are identical — this is the same-surface
seam case, handled separately by surface equivalence or not at all (no finite intersection
curve exists).

**Constructing the 3D lines.**  Let $\mathbf{f} = \mathbf{a} \times \mathbf{e}$ (unit
vector perpendicular to both $\mathbf{a}$ and $\mathbf{e}$, lying in the perpendicular
plane).  The base point in the perpendicular plane is $\mathbf{b} = \mathbf{q}_1 + x\mathbf{e}$.
The intersection point(s) in 2D are $\mathbf{b} \pm h\mathbf{f}$.  Restoring the
$\mathbf{a}$-component:

$$\mathbf{p}_{3D} = \mathbf{b} \pm h\mathbf{f} + (\mathbf{c}_1 \cdot \mathbf{a})\,\mathbf{a}$$

Each point defines a `Geom_Line` in direction $\mathbf{a}$, subsequently trimmed by
`trim_by_vertices`.

**Why OCC fails.**  `GeomAPI_IntSS` internally calls `IntPatch_ImpImpIntersection` for
quadric–quadric pairs.  For two cylinders with nearly parallel axes, the algorithm
encounters near-degenerate conditions in its Walking/Marching scheme and returns
`IsDone() = False` rather than attempting a degenerate-case analysis.  The analytical
fallback above handles exactly this gap.

---

## Why OCC for cylinder, cone and mixed pairs

For these pairs the intersection curve is no longer a simple conic.  For
example, cylinder ∩ cylinder in general position is a **space algebraic
curve of degree 4** (a Viviani-type curve).  Plane ∩ cylinder produces an
ellipse, but determining the correct sub-type (ellipse / circle / tangent
line) and its exact parameters from the implicit form requires case analysis.

OCC's `GeomAPI_IntSS` handles this internally by:
1. Classifying the surface pair type and using analytic formulae where
   possible (e.g. planar quadric curves).
2. Tracing the intersection curve numerically in the UV parameter domain of
   one surface and fitting a B-spline approximation for complex cases.

The result is always a parametric `Geom_Curve` — OCC does not expose the
implicit algebraic form $F(x, y, z) = 0$.

**Exception — plane ∩ cylinder tangent case.**  The pipeline now partially handles
plane ∩ cylinder analytically: when OCC returns empty for this pair, a fallback
checks whether the plane is axis-parallel and tangent ($\delta \approx r$) and, if
so, returns the single generator line analytically (see section above).  The general
non-degenerate plane ∩ cylinder case (circle or ellipse) still uses OCC.

**Exception — cylinder ∩ cylinder parallel-axis case.**  When OCC returns
`IsDone() = False` for a cylinder–cylinder pair, the pipeline checks whether the axes
are parallel and, if so, reduces to a 2D circle–circle intersection to produce 0, 1,
or 2 generator lines analytically (see section above).  The general non-parallel
cylinder ∩ cylinder case (degree-4 space curve) still relies on OCC.
