# Cylinder Sampling Algorithm - Mathematical Analysis

The algorithm samples points on a cylinder by generating a **canonical cylinder** (axis along z, centered at origin), then transforming it to the desired orientation and position.

## Cylinder Definition

A cylinder with radius $r$, axis direction $\mathbf{a}$ (unit vector), and center point $\mathbf{c}$ on the axis consists of all points $\mathbf{p}$ satisfying:

$$\|(\mathbf{p} - \mathbf{c}) - [(\mathbf{p} - \mathbf{c}) \cdot \mathbf{a}]\mathbf{a}\| = r$$

This is the perpendicular distance from $\mathbf{p}$ to the axis line.

## Canonical Cylinder (Axis Along Z)

For a cylinder with axis along the z-axis, centered at origin:

$$\mathbf{p}(\theta, h) = \begin{pmatrix} r \cos\theta \\ r \sin\theta \\ h \end{pmatrix}$$

where $\theta \in [0, 2\pi)$ and $h \in [h_{\min}, h_{\max}]$.

**Verification:** $x^2 + y^2 = r^2\cos^2\theta + r^2\sin^2\theta = r^2$ ✓

## Algorithm Steps

### Step 1: Compute Rotation Matrix

Find $R$ such that $R \cdot \mathbf{e}_z = \mathbf{a}$, where $\mathbf{e}_z = (0, 0, 1)^T$.

```python
R = rotation_matrix_a_to_b(np.array([0, 0, 1]), axis[:, 0])
```

#### Derivation of `rotation_matrix_a_to_b`

**Goal:** Find rotation $R$ such that $R\mathbf{a} = \mathbf{b}$, where $\|\mathbf{a}\| = \|\mathbf{b}\| = 1$.

**Step 1: Construct an orthonormal basis $F = [\mathbf{u}, \mathbf{v}, \mathbf{w}]$**

- $\mathbf{u} = \mathbf{a}$ (source vector)
- $\mathbf{v} = \frac{\mathbf{b} - (\mathbf{a} \cdot \mathbf{b})\mathbf{a}}{\|\mathbf{b} - (\mathbf{a} \cdot \mathbf{b})\mathbf{a}\|}$ (component of $\mathbf{b}$ perpendicular to $\mathbf{a}$, normalized)
- $\mathbf{w} = \frac{\mathbf{b} \times \mathbf{a}}{\|\mathbf{b} \times \mathbf{a}\|}$ (perpendicular to the plane of $\mathbf{a}$ and $\mathbf{b}$)

**Step 2: Express $\mathbf{a}$ and $\mathbf{b}$ in the $F$ basis**

Since $\mathbf{a} = \mathbf{u}$:

$$[\mathbf{a}]_F = (1, 0, 0)^T$$

For $\mathbf{b}$, using $\theta$ as the angle between $\mathbf{a}$ and $\mathbf{b}$:

- $\mathbf{b} \cdot \mathbf{u} = \mathbf{b} \cdot \mathbf{a} = \cos\theta$
- $\mathbf{b} \cdot \mathbf{v} = \sin\theta$ (since $\|\mathbf{b} - \cos\theta \cdot \mathbf{a}\| = \sin\theta$)
- $\mathbf{b} \cdot \mathbf{w} = 0$ (since $\mathbf{w} \perp \mathbf{b}$)

Therefore:

$$[\mathbf{b}]_F = (\cos\theta, \sin\theta, 0)^T$$

**Step 3: The rotation is simple in the $F$ basis**

In the $F$ basis, rotating $\mathbf{a}$ to $\mathbf{b}$ is just a rotation around the third axis ($\mathbf{w}$):

$$G = \begin{pmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

**Step 4: Change of basis formula**

For any vector $\mathbf{x}$ in standard coordinates:

1. Transform to $F$ basis: $[\mathbf{x}]_F = F^{-1}\mathbf{x} = F^T\mathbf{x}$ (since $F$ is orthogonal)
2. Apply rotation $G$: $G[\mathbf{x}]_F$
3. Transform back to standard basis: $F(G[\mathbf{x}]_F)$

Combined:

$$R = FGF^T$$

**Verification:**

$$R\mathbf{a} = FGF^T\mathbf{a} = FG\begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix} = F\begin{pmatrix} \cos\theta \\ \sin\theta \\ 0 \end{pmatrix} = \cos\theta \cdot \mathbf{u} + \sin\theta \cdot \mathbf{v}$$

Substituting $\mathbf{v} = \frac{\mathbf{b} - \cos\theta \cdot \mathbf{a}}{\sin\theta}$:

$$= \cos\theta \cdot \mathbf{a} + \sin\theta \cdot \frac{\mathbf{b} - \cos\theta \cdot \mathbf{a}}{\sin\theta} = \cos\theta \cdot \mathbf{a} + \mathbf{b} - \cos\theta \cdot \mathbf{a} = \mathbf{b} \quad \checkmark$$

**Code:**

```python
def rotation_matrix_a_to_b(A, B):
    cos = np.dot(A, B)                           # cos θ
    sin = np.linalg.norm(np.cross(B, A))         # sin θ = |B × A|

    u = A
    v = B - np.dot(A, B) * A                     # B - (A·B)A
    v = v / np.linalg.norm(v)                    # normalize
    w = np.cross(B, A)
    w = w / np.linalg.norm(w)                    # normalize

    F = np.stack([u, v, w], 1)                   # Orthonormal basis
    G = np.array([[cos, -sin, 0],                # Rotation around w
                  [sin, cos, 0],
                  [0, 0, 1]])

    R = F @ G @ F.T                              # F @ G @ F⁻¹ = F @ G @ Fᵀ
    return R
```

**Note:** The original code uses `np.linalg.inv(F)` with a try/except block, but since $F$ is orthogonal, $F^{-1} = F^T$. The exception handles the degenerate case where $\mathbf{a} \parallel \mathbf{b}$, causing $\mathbf{v}$ or $\mathbf{w}$ to be zero vectors.

#### Application to Cylinder Sampling: Proof that $R \cdot \mathbf{e}_z = \mathbf{a}_{\text{cylinder}}$

**Notation clarification:** In the general derivation above, $\mathbf{a}$ and $\mathbf{b}$ are the function's input vectors. In the cylinder context, we call the function with:

```python
R = rotation_matrix_a_to_b(src=e_z, dst=cylinder_axis)
```

So: $\mathbf{src} = \mathbf{e}_z$ and $\mathbf{dst} = \mathbf{a}_{\text{cylinder}}$.

**Claim:** $R \cdot \mathbf{e}_z = \mathbf{a}_{\text{cylinder}}$

**Proof:**

The basis $F = [\mathbf{u}, \mathbf{v}, \mathbf{w}]$ is constructed with $\mathbf{u} = \mathbf{src} = \mathbf{e}_z$.

Therefore $[\mathbf{e}_z]_F = (1, 0, 0)^T$ because $\mathbf{e}_z = 1 \cdot \mathbf{u} + 0 \cdot \mathbf{v} + 0 \cdot \mathbf{w}$.

Computing $R \cdot \mathbf{e}_z$:

$$R \cdot \mathbf{e}_z = (FGF^T) \cdot \mathbf{e}_z = F \cdot G \cdot (F^T \mathbf{e}_z) = F \cdot G \cdot [\mathbf{e}_z]_F$$

$$= F \cdot G \cdot \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix} = F \cdot \begin{pmatrix} \cos\theta \\ \sin\theta \\ 0 \end{pmatrix} = \cos\theta \cdot \mathbf{u} + \sin\theta \cdot \mathbf{v}$$

where $\theta$ is the angle between $\mathbf{e}_z$ and $\mathbf{a}_{\text{cylinder}}$, and:
- $\mathbf{u} = \mathbf{e}_z$
- $\mathbf{v} = \frac{\mathbf{a}_{\text{cylinder}} - \cos\theta \cdot \mathbf{e}_z}{\sin\theta}$

Substituting:

$$R \cdot \mathbf{e}_z = \cos\theta \cdot \mathbf{e}_z + \sin\theta \cdot \frac{\mathbf{a}_{\text{cylinder}} - \cos\theta \cdot \mathbf{e}_z}{\sin\theta}$$

$$= \cos\theta \cdot \mathbf{e}_z + \mathbf{a}_{\text{cylinder}} - \cos\theta \cdot \mathbf{e}_z = \mathbf{a}_{\text{cylinder}} \quad \checkmark$$

This confirms that $R \cdot \mathbf{e}_z = \mathbf{a}$ **by construction**, which is used in the proof of correctness (Step 2) where we rely on this property.

### Step 2: Determine Height Bounds

Project input cluster points onto the cylinder axis to find the extent:

```python
points_centered = points - center
projection = points_centered @ axis  # Scalar projections onto axis
min_proj = min(projection) - 0.1     # Add margin
max_proj = max(projection) + 0.1
```

### Step 3: Generate Canonical Cylinder

Sample points on z-axis-aligned cylinder at origin:

```python
theta = [0, 2π/d_theta, 4π/d_theta, ..., 0]  # d_theta angles, wrap around
circle = [(cos θ, sin θ) for θ in theta] * radius
height = linspace(min_proj, max_proj, 2 * d_height)

# Canonical points: (r cos θ, r sin θ, h)
p_canonical = concatenate([circle, height], axis=1)
```

### Step 4: Transform to Target Cylinder

```python
p_rotated = R @ p_canonical.T    # Rotate to align with axis
p_final = p_rotated.T + center   # Translate to center
```

## Mathematical Proof of Correctness

**Claim:** The transformed points $\mathbf{p}_{\text{final}}$ lie exactly on the cylinder defined by $(\mathbf{c}, \mathbf{a}, r)$.

**Proof:**

We need to show that the perpendicular distance from $\mathbf{p}_{\text{final}}$ to the axis equals $r$.

Let $\mathbf{p}_{\text{can}} = (r\cos\theta, r\sin\theta, h)^T$ be a canonical point.

**Step 1:** Compute $\mathbf{p}_{\text{final}} - \mathbf{c}$

From the code transformation (Step 4 above):

$$\mathbf{p}_{\text{final}} = R \cdot \mathbf{p}_{\text{can}} + \mathbf{c}$$

Subtracting $\mathbf{c}$ from both sides:

$$\mathbf{p}_{\text{final}} - \mathbf{c} = R \cdot \mathbf{p}_{\text{can}} + \mathbf{c} - \mathbf{c} = R \cdot \mathbf{p}_{\text{can}}$$

The center translation cancels out, leaving only the rotated canonical point.

**Step 2:** Project onto axis $\mathbf{a}$:

Since $R$ is orthogonal ($R = F G F^T$, both $F$ and $G$ are orthogonal matrices, easy to verify this by simply computing $RR^{T}$) and $R \cdot \mathbf{e}_z = \mathbf{a}$:

$$(R \cdot \mathbf{p}_{\text{can}}) \cdot \mathbf{a} = (R \cdot \mathbf{p}_{\text{can}}) \cdot (R \cdot \mathbf{e}_z) = \mathbf{p}_{\text{can}} \cdot \mathbf{e}_z = h$$

(Orthogonal matrices preserve dot products: $(Rx) \cdot (Ry) = x \cdot y$)

**Step 3:** Compute perpendicular component:

$$\mathbf{p}_{\perp} = (R \cdot \mathbf{p}_{\text{can}}) - h \cdot \mathbf{a} = R \cdot \mathbf{p}_{\text{can}} - h \cdot R \cdot \mathbf{e}_z = R \cdot (\mathbf{p}_{\text{can}} - h \cdot \mathbf{e}_z)$$

$$= R \cdot \begin{pmatrix} r\cos\theta \\ r\sin\theta \\ 0 \end{pmatrix}$$

**Step 4:** Compute norm (perpendicular distance):

$$\|\mathbf{p}_{\perp}\| = \left\| R \cdot \begin{pmatrix} r\cos\theta \\ r\sin\theta \\ 0 \end{pmatrix} \right\| = \left\| \begin{pmatrix} r\cos\theta \\ r\sin\theta \\ 0 \end{pmatrix} \right\| = r \quad \checkmark$$

(Orthogonal matrices preserve norms: $\|Rx\| = \|x\|$)

## Grid Structure

The algorithm creates a **d_theta × (2 × d_height)** grid:

| Dimension | Parameter | Values | Default |
|-----------|-----------|--------|---------|
| Angular | $\theta$ | $[0, 2\pi)$ | 60 samples |
| Axial | $h$ | $[h_{\min}, h_{\max}]$ | 200 samples |

**Ordering:** Height varies slow (outer loop), angle varies fast (inner loop).

```
Index:  0    1    2   ...  59   60   61  ...
θ:      θ₀   θ₁   θ₂  ...  θ₅₉  θ₀   θ₁  ...
h:      h₀   h₀   h₀  ...  h₀   h₁   h₁  ...
```

## Cylinder Normals

### Derivation from Implicit Equation

The implicit cylinder equation (axis along z) is:

$$F(x, y, z) = x^2 + y^2 - r^2 = 0$$

The gradient:

$$\nabla F = (2x, 2y, 0)$$

The outward unit normal:

$$\mathbf{n} = \frac{\nabla F}{\|\nabla F\|} = \frac{(x, y, 0)}{\sqrt{x^2 + y^2}} = \frac{(x, y, 0)}{r} = (\cos\theta, \sin\theta, 0)$$

### In Code

```python
# Canonical normals (before rotation)
normals = np.concatenate([circle, np.zeros((circle.shape[0], 1))], 1)
normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

# Transform normals (rotation only, no translation)
normals = (R @ normals.T).T
```

**Key insight:** Normals transform by rotation only, not translation. This is because normals are directions, not positions.

## Visual Representation

```
Canonical (axis = z):              Transformed (axis = a):

        z                                   a
        ↑                                  ↗
    ____│____                         ____/____
   /    │    \                       /   /     \
  │     │     │    ──── R ────►     │   /       │
  │     │     │                     │  /        │
  │     ●─────│──► x                │ ●─────────│
  │   origin  │                     │ c         │
   \____│____/                       \_________/
        │
```

## Comparison with Sphere Sampling

| Aspect | Cylinder | Sphere |
|--------|----------|--------|
| Parameterization | $(\theta, h)$ | $(\theta, \lambda)$ |
| Grid uniformity | Uniform in both | Uniform in params, clustered at poles |
| Transformation | Rotation + Translation | Translation only |
| Height bounds | From input points | Fixed $[-R, R]$ |
| Normal computation | $(x/r, y/r, 0)$ rotated | $\mathbf{p}/R$ |

## Note on Unused Normals

The normals computed in both `sample_sphere` and `sample_cylinder_trim` are returned but **never used** in the Point2CAD pipeline. They appear to be vestigial code, possibly intended for:

1. Future rendering with proper shading
2. Normal-based mesh smoothing
3. Debugging/visualization purposes

The mesh visualization uses `compute_vertex_normals()` from Open3D instead, which computes normals from the triangle geometry rather than the analytical surface.

## TODOs
* Think about what to do with cylinder and cone height error margins, use absolute, relative, or hybrid approach? Something like `max(height * relative_ratio, absolute_ratio)` to prevent underflows when height is too small. For now, both margins will be fixed.