# Cone Sampling Algorithm - Mathematical Analysis

The algorithm samples points on a cone by finding a **generator ray**, scaling it to different heights, and rotating around the axis using **Rodrigues' rotation formula**.

## Cone Definition

A cone with vertex $\mathbf{v}$, axis direction $\mathbf{a}$ (unit vector), and half-angle $\theta$ consists of all points $\mathbf{p}$ satisfying:

$$(\mathbf{p} - \mathbf{v}) \cdot \mathbf{a} = \|\mathbf{p} - \mathbf{v}\| \cos\theta$$

Equivalently, the angle between $(\mathbf{p} - \mathbf{v})$ and $\mathbf{a}$ equals $\theta$.

### Geometric Interpretation

```
                    a (axis)
                    ↑
                   /|\
                  / | \
                 /  |  \      θ = half-angle
                /   |   \
               / θ  |    \
              /_____|_____\
                    v (vertex)
```

For a point $\mathbf{p}$ on the cone at signed height $h$ (projection onto axis):
- Height: $h = (\mathbf{p} - \mathbf{v}) \cdot \mathbf{a}$
- Distance from vertex: $\|\mathbf{p} - \mathbf{v}\| = |h| / \cos\theta$
- Radius at height $h$: $r(h) = |h| \tan\theta$

## Algorithm Overview

1. **Determine height bounds** from input cluster points
2. **Find one point on the cone surface** (a generator ray direction)
3. **Scale the generator** to different heights
4. **Rotate around axis** using Rodrigues' formula to create circles

## Algorithm Steps

### Step 1: Normalize Axis and Compute Height Bounds

```python
norm_a = np.linalg.norm(a)
a = a / norm_a                              # Normalize axis

proj = (points - c) @ a                     # Project cluster onto axis
proj_max = np.max(proj) + 0.2 * |max|       # Add 20% margin
proj_min = np.min(proj) - 0.2 * |min|
```

The projections give the signed heights of the cluster points along the axis.

### Step 2: Find a Generator Ray Direction

**Goal:** Find a unit vector $\hat{\mathbf{g}}$ from the vertex along the cone surface (a "generator ray"). This vector must make angle $\theta$ with the axis $\mathbf{a}$.

#### What Point2CAD Does (Buggy)

**Step 2a:** Find a point $\mathbf{d}$ such that $\mathbf{d} \cdot \mathbf{a} = \mathbf{v} \cdot \mathbf{a}$

```python
k = np.dot(c, a)                            # k = v · a
x = (k - a[1] - a[2]) / (a[0] + EPS)        # Solve for x given y=1, z=1
y = 1
z = 1
d = np.array([x, y, z])
```

This finds $\mathbf{d} = (x, 1, 1)$ satisfying $\mathbf{d} \cdot \mathbf{a} = k$.

**Claim:** $(\mathbf{d} - \mathbf{v}) \perp \mathbf{a}$

**Proof:**
$$(\mathbf{d} - \mathbf{v}) \cdot \mathbf{a} = \mathbf{d} \cdot \mathbf{a} - \mathbf{v} \cdot \mathbf{a} = k - k = 0 \quad \checkmark$$

So $\mathbf{r} = \mathbf{d} - \mathbf{v}$ is a radial vector perpendicular to the axis.

**Step 2b:** Construct a point on the cone surface

```python
p = a * (np.linalg.norm(d)) / (np.sin(theta) + EPS) * np.cos(theta) + d
```

This computes:
$$\mathbf{p} = \mathbf{d} + \mathbf{a} \cdot \frac{\|\mathbf{d}\| \cos\theta}{\sin\theta}$$

#### The Bug: Using $\|\mathbf{d}\|$ Instead of $\|\mathbf{d} - \mathbf{v}\|$

Let's verify if $\mathbf{p}$ actually lies on the cone. For $\mathbf{p}$ to be on the cone:
$$(\mathbf{p} - \mathbf{v}) \cdot \mathbf{a} = \|\mathbf{p} - \mathbf{v}\| \cos\theta$$

**Computing $\mathbf{p} - \mathbf{v}$:**

Let $\mathbf{r} = \mathbf{d} - \mathbf{v}$ (perpendicular to $\mathbf{a}$). Then:
$$\mathbf{p} - \mathbf{v} = (\mathbf{d} - \mathbf{v}) + \mathbf{a} \cdot \frac{\|\mathbf{d}\| \cos\theta}{\sin\theta} = \mathbf{r} + \mathbf{a} \cdot \frac{\|\mathbf{d}\| \cos\theta}{\sin\theta}$$

**Left side of cone equation:**
$$(\mathbf{p} - \mathbf{v}) \cdot \mathbf{a} = \mathbf{r} \cdot \mathbf{a} + \frac{\|\mathbf{d}\| \cos\theta}{\sin\theta} = 0 + \frac{\|\mathbf{d}\| \cos\theta}{\sin\theta} = \|\mathbf{d}\| \cot\theta$$

**Right side of cone equation:**
$$\|\mathbf{p} - \mathbf{v}\| = \sqrt{\|\mathbf{r}\|^2 + \|\mathbf{d}\|^2 \cot^2\theta}$$

(since $\mathbf{r} \perp \mathbf{a}$)

**For equality:**
$$\|\mathbf{d}\| \cot\theta = \sqrt{\|\mathbf{r}\|^2 + \|\mathbf{d}\|^2 \cot^2\theta} \cdot \cos\theta$$

Squaring both sides:
$$\|\mathbf{d}\|^2 \cot^2\theta = (\|\mathbf{r}\|^2 + \|\mathbf{d}\|^2 \cot^2\theta) \cos^2\theta$$

$$\|\mathbf{d}\|^2 \cot^2\theta (1 - \cos^2\theta) = \|\mathbf{r}\|^2 \cos^2\theta$$

$$\|\mathbf{d}\|^2 \cot^2\theta \sin^2\theta = \|\mathbf{r}\|^2 \cos^2\theta$$

$$\|\mathbf{d}\|^2 \cos^2\theta = \|\mathbf{r}\|^2 \cos^2\theta$$

$$\|\mathbf{d}\|^2 = \|\mathbf{r}\|^2 = \|\mathbf{d} - \mathbf{v}\|^2$$

**This is only true when $\mathbf{v} = \mathbf{0}$ (vertex at origin)!**

The Point2CAD code uses $\|\mathbf{d}\|$ where it should use $\|\mathbf{d} - \mathbf{v}\| = \|\mathbf{r}\|$. This is a bug that causes incorrect cone sampling when the vertex is not at the origin.

#### Correct Approach

**Method 1: Fix the existing approach**

Replace $\|\mathbf{d}\|$ with $\|\mathbf{d} - \mathbf{v}\|$:
$$\mathbf{p} = \mathbf{d} + \mathbf{a} \cdot \frac{\|\mathbf{d} - \mathbf{v}\| \cos\theta}{\sin\theta}$$

**Method 2: Direct construction (cleaner)**

1. Find any unit vector $\hat{\mathbf{r}}$ perpendicular to $\mathbf{a}$:
   ```python
   if abs(a[0]) < 0.9:
       temp = np.array([1, 0, 0])
   else:
       temp = np.array([0, 1, 0])
   r_hat = np.cross(a, temp)
   r_hat = r_hat / np.linalg.norm(r_hat)
   ```

2. Construct the generator ray direction directly:
   $$\hat{\mathbf{g}} = \cos\theta \cdot \mathbf{a} + \sin\theta \cdot \hat{\mathbf{r}}$$

   This is a unit vector (since $\mathbf{a} \perp \hat{\mathbf{r}}$ and both are unit vectors).

**Verification that $\hat{\mathbf{g}}$ makes angle $\theta$ with $\mathbf{a}$:**
$$\hat{\mathbf{g}} \cdot \mathbf{a} = \cos\theta \cdot (\mathbf{a} \cdot \mathbf{a}) + \sin\theta \cdot (\hat{\mathbf{r}} \cdot \mathbf{a}) = \cos\theta \cdot 1 + \sin\theta \cdot 0 = \cos\theta \quad \checkmark$$

### Step 3: Scale Generator to Different Heights

The unit generator ray is $\hat{\mathbf{g}}$ (making angle $\theta$ with axis).

For a point on the cone at signed height $h$ (where $h = (\mathbf{p} - \mathbf{v}) \cdot \mathbf{a}$):
- Distance from vertex: $\|\mathbf{p} - \mathbf{v}\| = h / \cos\theta$
- Point position: $\mathbf{p} = \mathbf{v} + \frac{h}{\cos\theta} \hat{\mathbf{g}}$

```python
rel_unit_vector = g_hat                              # Unit generator ray
rel_unit_vector_min = g_hat * proj_min / cos(theta)  # Vector to min height
rel_unit_vector_max = g_hat * proj_max / cos(theta)  # Vector to max height
```

**Derivation:**

If $\mathbf{p} = \mathbf{v} + t \hat{\mathbf{g}}$ for some $t > 0$, then:
$$h = (\mathbf{p} - \mathbf{v}) \cdot \mathbf{a} = t \hat{\mathbf{g}} \cdot \mathbf{a} = t \cos\theta$$

So $t = h / \cos\theta$, confirming the formula.

**Scaling to different heights:**

For a point at signed height $h$ on the cone:
- Distance from vertex: $\|\mathbf{p} - \mathbf{v}\| = h / \cos\theta$

```python
rel_unit_vector_min = rel_unit_vector * proj_min / cos(theta)
rel_unit_vector_max = rel_unit_vector * proj_max / cos(theta)
```

These are the displacement vectors from vertex to cone surface at the minimum and maximum heights.

### Step 4: Rodrigues' Rotation Formula

To create circles at each height, we rotate around the axis $\mathbf{a}$.

**Rodrigues' formula:** Rotation by angle $\phi$ around unit axis $\mathbf{a}$:

$$R(\phi) = I + \sin\phi \cdot K + (1 - \cos\phi) \cdot K^2$$

where $K$ is the skew-symmetric matrix of $\mathbf{a}$:

$$K = [\mathbf{a}]_\times = \begin{pmatrix} 0 & -a_z & a_y \\ a_z & 0 & -a_x \\ -a_y & a_x & 0 \end{pmatrix}$$

```python
K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
```

#### Derivation of Rodrigues' Formula

For a vector $\mathbf{x}$ rotated around unit axis $\mathbf{a}$ by angle $\phi$:

**Decompose $\mathbf{x}$:**
- Parallel component: $\mathbf{x}_\parallel = (\mathbf{x} \cdot \mathbf{a})\mathbf{a}$
- Perpendicular component: $\mathbf{x}_\perp = \mathbf{x} - \mathbf{x}_\parallel$

**Rotation only affects perpendicular component:**
$$R\mathbf{x} = \mathbf{x}_\parallel + \cos\phi \cdot \mathbf{x}_\perp + \sin\phi \cdot (\mathbf{a} \times \mathbf{x}_\perp)$$

**Key identity:** $\mathbf{a} \times \mathbf{x} = K\mathbf{x}$ (the cross product as matrix multiplication)

**Another identity:** $K^2\mathbf{x} = \mathbf{a}(\mathbf{a} \cdot \mathbf{x}) - \mathbf{x} = \mathbf{x}_\parallel - \mathbf{x} = -\mathbf{x}_\perp$

Substituting:
$$R\mathbf{x} = \mathbf{x}_\parallel + \cos\phi \cdot \mathbf{x}_\perp + \sin\phi \cdot K\mathbf{x}$$
$$= \mathbf{x}_\parallel + \cos\phi(\mathbf{x} - \mathbf{x}_\parallel) + \sin\phi \cdot K\mathbf{x}$$
$$= \mathbf{x} + (\cos\phi - 1)(\mathbf{x} - \mathbf{x}_\parallel) + \sin\phi \cdot K\mathbf{x}$$
$$= \mathbf{x} - (1 - \cos\phi)\mathbf{x}_\perp + \sin\phi \cdot K\mathbf{x}$$
$$= \mathbf{x} + (1 - \cos\phi)K^2\mathbf{x} + \sin\phi \cdot K\mathbf{x}$$

Therefore:
$$R = I + \sin\phi \cdot K + (1 - \cos\phi) \cdot K^2 \quad \checkmark$$

#### Proof that Rodrigues Rotation is Orthogonal

**Claim:** $R(\phi) = I + \sin\phi \cdot K + (1 - \cos\phi) \cdot K^2$ is orthogonal.

**Proof:** We verify $RR^T = I$.

Properties of $K$:
- $K^T = -K$ (skew-symmetric)
- $K^3 = -K$ (since $\mathbf{a}$ is unit)
- $(K^2)^T = K^2$

$$R^T = I - \sin\phi \cdot K + (1 - \cos\phi) \cdot K^2$$

$$RR^T = [I + \sin\phi \cdot K + (1-\cos\phi)K^2][I - \sin\phi \cdot K + (1-\cos\phi)K^2]$$

Expanding and using $K^3 = -K$, $K^4 = -K^2$:

$$= I + (1-\cos\phi)K^2 - \sin^2\phi \cdot K^2 + (1-\cos\phi)K^2 + (1-\cos\phi)^2 K^4$$
$$= I + 2(1-\cos\phi)K^2 - \sin^2\phi \cdot K^2 - (1-\cos\phi)^2 K^2$$
$$= I + K^2[2(1-\cos\phi) - \sin^2\phi - (1-\cos\phi)^2]$$

The bracket:
$$2 - 2\cos\phi - \sin^2\phi - 1 + 2\cos\phi - \cos^2\phi = 2 - 1 - (\sin^2\phi + \cos^2\phi) = 0$$

Therefore $RR^T = I$. $\checkmark$

### Step 5: Sampling Loop

```python
for j in range(100):                        # 100 height levels
    p_ = rel_unit_vector_min + (rel_unit_vector_max - rel_unit_vector_min) * 0.01 * j

    for d in range(50):                     # 50 angles (+ 1 for wrap)
        degrees = 2 * np.pi * 0.01 * d * 2  # Note: 0.01 * 2 = 0.02, so 50 steps covers full circle
        R = I + sin(degrees) * K + (1 - cos(degrees)) * K @ K
        rotate_point = R @ p_
        d_points.append(rotate_point + c)
```

**Grid structure:** 100 heights × 51 angles (including wrap-around)

**Angle step:** $\Delta\phi = 2\pi \cdot 0.02 = 0.04\pi$, so 50 steps gives $50 \times 0.04\pi = 2\pi$ (full circle).

## Mathematical Proof of Correctness

**Claim:** The sampled points lie on the cone defined by $(\mathbf{v}, \mathbf{a}, \theta)$.

**Proof:**

Let $\mathbf{g} = \mathbf{p} - \mathbf{v}$ be the initial generator ray (before scaling), where $\mathbf{p}$ is on the cone.

**Step 1:** The scaled generator $\mathbf{p}' = \lambda \hat{\mathbf{g}}$ (where $\hat{\mathbf{g}} = \mathbf{g}/\|\mathbf{g}\|$) points from vertex along the cone surface.

By construction, $\hat{\mathbf{g}}$ makes angle $\theta$ with $\mathbf{a}$:
$$\hat{\mathbf{g}} \cdot \mathbf{a} = \cos\theta$$

**Step 2:** Rodrigues' rotation around $\mathbf{a}$ preserves the angle with $\mathbf{a}$.

For any rotation $R$ around $\mathbf{a}$: $R\mathbf{a} = \mathbf{a}$ (the axis is fixed).

Since $R$ is orthogonal:
$$(R\hat{\mathbf{g}}) \cdot \mathbf{a} = (R\hat{\mathbf{g}}) \cdot (R\mathbf{a}) = \hat{\mathbf{g}} \cdot \mathbf{a} = \cos\theta$$

**Step 3:** The final point is:
$$\mathbf{p}_{\text{final}} = R\mathbf{p}' + \mathbf{v}$$

So:
$$(\mathbf{p}_{\text{final}} - \mathbf{v}) \cdot \mathbf{a} = (R\mathbf{p}') \cdot \mathbf{a} = \lambda (R\hat{\mathbf{g}}) \cdot \mathbf{a} = \lambda \cos\theta$$

And:
$$\|\mathbf{p}_{\text{final}} - \mathbf{v}\| = \|R\mathbf{p}'\| = \|\mathbf{p}'\| = |\lambda|$$

Therefore:
$$\frac{(\mathbf{p}_{\text{final}} - \mathbf{v}) \cdot \mathbf{a}}{\|\mathbf{p}_{\text{final}} - \mathbf{v}\|} = \frac{\lambda \cos\theta}{|\lambda|} = \pm\cos\theta$$

This confirms $\mathbf{p}_{\text{final}}$ lies on the cone. $\checkmark$

## Cone Normals

### Derivation from Implicit Equation

The cone equation can be written as:
$$F(\mathbf{p}) = [(\mathbf{p} - \mathbf{v}) \cdot \mathbf{a}]^2 - \|\mathbf{p} - \mathbf{v}\|^2 \cos^2\theta = 0$$

Gradient:
$$\nabla F = 2[(\mathbf{p} - \mathbf{v}) \cdot \mathbf{a}]\mathbf{a} - 2(\mathbf{p} - \mathbf{v})\cos^2\theta$$

Let $\mathbf{d} = \mathbf{p} - \mathbf{v}$ and $h = \mathbf{d} \cdot \mathbf{a}$ (the height). Then:
$$\nabla F = 2h\mathbf{a} - 2\mathbf{d}\cos^2\theta = 2[h\mathbf{a} - \mathbf{d}\cos^2\theta]$$

The unit normal (unnormalized):
$$\mathbf{n} \propto h\mathbf{a} - \mathbf{d}\cos^2\theta = h\mathbf{a} - (\mathbf{p} - \mathbf{v})\cos^2\theta$$

### Code Implementation

```python
d_normals.append(
    rotate_point
    - np.linalg.norm(rotate_point) / np.cos(theta) * a / norm_a
)
```

Here `rotate_point` = $R\mathbf{p}' = \mathbf{p}_{\text{final}} - \mathbf{v}$.

The code computes:
$$\mathbf{n}_{\text{code}} = (\mathbf{p} - \mathbf{v}) - \frac{\|\mathbf{p} - \mathbf{v}\|}{\cos\theta} \cdot \frac{\mathbf{a}}{\text{norm\_a}}$$

**Potential issue:** After normalizing $\mathbf{a}$ (line 78: `a = a / norm_a`), dividing by `norm_a` again seems incorrect. If the input axis was already unit, this is fine. If not, this introduces an error.

### Correct Normal Formula

For a point $\mathbf{p}$ on the cone with $\mathbf{d} = \mathbf{p} - \mathbf{v}$:
- Height: $h = \mathbf{d} \cdot \mathbf{a}$
- Distance: $\|\mathbf{d}\| = h / \cos\theta$

The outward normal (pointing away from axis):
$$\mathbf{n} = \mathbf{d} - h \cdot \mathbf{a} / \cos^2\theta$$

or equivalently:
$$\mathbf{n} = \mathbf{d} - \|\mathbf{d}\| / \cos\theta \cdot \mathbf{a}$$

This matches the code's formula (assuming `norm_a = 1`).

## Grid Structure

| Dimension | Parameter | Range | Samples |
|-----------|-----------|-------|---------|
| Height | $j$ | $[0, 99]$ | 100 |
| Angle | $\phi$ | $[0, 2\pi]$ | 51 (including wrap) |

**Total points before trimming:** $100 \times 51 = 5100$

**Ordering:** Height varies slow (outer loop), angle varies fast (inner loop).

## Comparison with Cylinder Sampling

| Aspect | Cone | Cylinder |
|--------|------|----------|
| Surface equation | $(\mathbf{p}-\mathbf{v})\cdot\mathbf{a} = \|\mathbf{p}-\mathbf{v}\|\cos\theta$ | $\|(\mathbf{p}-\mathbf{c}) - [(\mathbf{p}-\mathbf{c})\cdot\mathbf{a}]\mathbf{a}\| = r$ |
| Radius | Varies: $r(h) = h\tan\theta$ | Constant: $r$ |
| Rotation method | Rodrigues' formula | `rotation_matrix_a_to_b` |
| Generator | Ray from vertex | Circle at origin |
| Height interpolation | Linear in distance from vertex | Linear in height |

## Visual Representation

```
Parameter space (j, φ):              3D cone surface:

┌─────────────────────┐                    ↑ a
│ □ □ □ □ □ □ □ ... □ │ j=99              / \
│ □ □ □ □ □ □ □ ... □ │                  /   \
│ ...                 │                 /     \
│ □ □ □ □ □ □ □ ... □ │ j=0        ────●───────  (vertex)
└─────────────────────┘                 v
  φ=0              φ=2π
```

## Note on Unused Normals

Similar to sphere and cylinder sampling, the computed normals are returned but **never used** in the Point2CAD pipeline. The mesh visualization uses Open3D's `compute_vertex_normals()` instead.
