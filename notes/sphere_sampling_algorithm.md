# Sphere Sampling Algorithm - Mathematical Analysis

The algorithm uses **horizontal slicing** of the sphere at different heights.

## Sphere Equation

A sphere of radius $R$ centered at the origin:

$$x^2 + y^2 + z^2 = R^2$$

## Slicing at Height $\lambda$

For a fixed height $z = \lambda$ where $\lambda \in [-R, R]$, the cross-section is a circle:

$$x^2 + y^2 = R^2 - \lambda^2$$

This circle has radius:

$$r(\lambda) = \sqrt{R^2 - \lambda^2}$$

## Parameterization

Points on the circle at height $\lambda$:

$$\mathbf{p}(\theta, \lambda) = \begin{pmatrix} r(\lambda) \cos\theta \\ r(\lambda) \sin\theta \\ \lambda \end{pmatrix} = \begin{pmatrix} \sqrt{R^2 - \lambda^2} \cos\theta \\ \sqrt{R^2 - \lambda^2} \sin\theta \\ \lambda \end{pmatrix}$$

where $\theta \in [0, 2\pi)$ and $\lambda \in [-R, R]$.

## Code Mapping

```python
# Height values: λ ∈ [-R + ε, R - ε] (100 values)
lam = np.linspace(-radius + 1e-7, radius - 1e-7, 100)

# Circle radii at each height: r(λ) = √(R² - λ²)
radii = np.sqrt(radius**2 - lam**2)

# Unit circle points: (cos θ, sin θ) for θ ∈ [0, 2π)
theta = np.arange(d_theta - 1) * 2π / d_theta  # 100 angles
circle = np.stack([np.cos(theta), np.sin(theta)], 1)  # [100, 2]

# Scale circles by their radii: (r(λ) cos θ, r(λ) sin θ)
new_circle = circle * spread_radii  # [10000, 2]

# Append heights: (r(λ) cos θ, r(λ) sin θ, λ)
points = np.concatenate([new_circle, height], 1)  # [10000, 3]
```

## Grid Structure

The algorithm creates a **100 × 100 grid**:

| Index | Height ($\lambda$) | Angle ($\theta$) |
|-------|-------------------|------------------|
| 0 | $\lambda_0$ | $\theta_0$ |
| 1 | $\lambda_0$ | $\theta_1$ |
| ... | ... | ... |
| 99 | $\lambda_0$ | $\theta_{99}$ |
| 100 | $\lambda_1$ | $\theta_0$ |
| ... | ... | ... |

**Ordering**: $\theta$ varies fast (inner loop), $\lambda$ varies slow (outer loop).

## Visual Representation

```
        z = R     ← r(λ) = 0 (pole, small circle)
          *
        * * *
      *       *   ← r(λ) = √(R² - λ²)
     *         *
    *     ●     * ← z = 0 (equator, r = R, largest circle)
     *         *
      *       *
        * * *
          *
        z = -R    ← r(λ) = 0 (pole, small circle)
```

## Why UV Triangles Translate to Valid 3D Triangles

The key is that the mapping from parameter space $(\theta, \lambda)$ to 3D is **continuous and smooth**.

### Continuity Preserves Adjacency

If two points are adjacent in parameter space:
- $(\theta_i, \lambda_j)$ and $(\theta_{i+1}, \lambda_j)$ — horizontally adjacent
- $(\theta_i, \lambda_j)$ and $(\theta_i, \lambda_{j+1})$ — vertically adjacent

Then their 3D images are also nearby because small changes in $(\theta, \lambda)$ produce small changes in $(x, y, z)$. This is precisely what continuity guarantees.

### Formal Argument

The mapping $\mathbf{p}: (\theta, \lambda) \to \mathbb{R}^3$ has continuous partial derivatives:

$$\frac{\partial \mathbf{p}}{\partial \theta} = \begin{pmatrix} -r(\lambda) \sin\theta \\ r(\lambda) \cos\theta \\ 0 \end{pmatrix}$$

$$\frac{\partial \mathbf{p}}{\partial \lambda} = \begin{pmatrix} \frac{-\lambda}{r(\lambda)} \cos\theta \\ \frac{-\lambda}{r(\lambda)} \sin\theta \\ 1 \end{pmatrix}$$

where $\frac{dr}{d\lambda} = \frac{-\lambda}{r(\lambda)}$.

These partial derivatives are bounded (except at poles), ensuring that the mapping is **Lipschitz continuous**: nearby points in parameter space map to nearby points in 3D.

### Potential Issues

1. **Poles**: At $\lambda = \pm R$, we have $r(\lambda) = 0$, so ALL $\theta$ values map to the same 3D point $(0, 0, \pm R)$. This creates **degenerate triangles** (zero area). The algorithm avoids exact poles by using $\lambda \in [-R + \epsilon, R - \epsilon]$.

2. **Wrap-around**: At $\theta = 0$ and $\theta = 2\pi$, we have the same 3D point. The algorithm handles this by including $\theta = 0$ at the end of the array.

3. **Triangle quality**: While adjacency is preserved, triangles near the poles become highly stretched and thin due to the clustering of points.

To sample from a sphere with an arbitraty center, simply sample from a sphere with center at 0, and then shift the sampled points by the desired center. This will ensure that $||samples - center||_2^2 = R^2$.

## The Area Element $dA$

The **area element** is a concept from differential geometry that tells us how to measure area on a curved surface.

### Intuition

Imagine a tiny rectangle in parameter space with sides $d\theta$ and $d\lambda$. When mapped to 3D, this becomes a tiny parallelogram on the surface. The area element $dA$ is the area of this parallelogram.

```
Parameter space:                   3D surface:
┌────────┐
│        │ dλ        ──────►      A curved parallelogram
└────────┘                         with area dA
   dθ
```

### Mathematical Definition

For a surface parameterized by $\mathbf{p}(u, v)$, the area element is:

$$dA = \left| \frac{\partial \mathbf{p}}{\partial u} \times \frac{\partial \mathbf{p}}{\partial v} \right| \, du \, dv$$

The cross product of the two tangent vectors gives a vector perpendicular to the surface, and its magnitude equals the area of the parallelogram they span.

### Derivation for the Sphere

**Step 1: Compute tangent vectors**

$$\frac{\partial \mathbf{p}}{\partial \theta} = \begin{pmatrix} -r(\lambda) \sin\theta \\ r(\lambda) \cos\theta \\ 0 \end{pmatrix}$$

$$\frac{\partial \mathbf{p}}{\partial \lambda} = \begin{pmatrix} \frac{-\lambda}{r(\lambda)} \cos\theta \\ \frac{-\lambda}{r(\lambda)} \sin\theta \\ 1 \end{pmatrix}$$

**Step 2: Compute cross product**

$$\frac{\partial \mathbf{p}}{\partial \theta} \times \frac{\partial \mathbf{p}}{\partial \lambda} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ -r\sin\theta & r\cos\theta & 0 \\ \frac{-\lambda}{r}\cos\theta & \frac{-\lambda}{r}\sin\theta & 1 \end{vmatrix}$$

Computing each component:
- $\mathbf{i}$: $(r\cos\theta)(1) - (0)(\frac{-\lambda}{r}\sin\theta) = r\cos\theta$
- $\mathbf{j}$: $(0)(\frac{-\lambda}{r}\cos\theta) - (-r\sin\theta)(1) = r\sin\theta$
- $\mathbf{k}$: $(-r\sin\theta)(\frac{-\lambda}{r}\sin\theta) - (r\cos\theta)(\frac{-\lambda}{r}\cos\theta) = \lambda\sin^2\theta + \lambda\cos^2\theta = \lambda$

Therefore:

$$\frac{\partial \mathbf{p}}{\partial \theta} \times \frac{\partial \mathbf{p}}{\partial \lambda} = \begin{pmatrix} r(\lambda) \cos\theta \\ r(\lambda) \sin\theta \\ \lambda \end{pmatrix}$$

**Step 3: Compute magnitude**

$$\left| \frac{\partial \mathbf{p}}{\partial \theta} \times \frac{\partial \mathbf{p}}{\partial \lambda} \right| = \sqrt{r^2\cos^2\theta + r^2\sin^2\theta + \lambda^2} = \sqrt{r^2 + \lambda^2}$$

Since $r^2 = R^2 - \lambda^2$:

$$= \sqrt{R^2 - \lambda^2 + \lambda^2} = \sqrt{R^2} = R$$

**Step 4: Area element**

$$dA = R \, d\theta \, d\lambda$$

### The Surprising Result: $dA$ is Constant!

The area element $dA = R \, d\theta \, d\lambda$ is **independent of $\lambda$**. This means that equal-sized rectangles in $(\theta, \lambda)$ space correspond to equal areas on the sphere surface.

### So Why Is Sampling Non-Uniform?

The non-uniformity comes from how we **distribute sample points**, not from the area element itself.

**The key insight**: Equal area $\neq$ equal point spacing.

With the algorithm:
- We use 100 values of $\theta$ uniformly spaced in $[0, 2\pi)$
- We use 100 values of $\lambda$ uniformly spaced in $[-R, R]$
- Each sample "owns" a rectangle of size $\Delta\theta \times \Delta\lambda$
- This maps to area $R \cdot \Delta\theta \cdot \Delta\lambda$ on the sphere — **the same for every sample**

**But the physical spacing differs:**

The arc length between adjacent $\theta$ samples at height $\lambda$ is:

$$\text{arc length} = r(\lambda) \cdot \Delta\theta = \sqrt{R^2 - \lambda^2} \cdot \Delta\theta$$

| Location | $\lambda$ | $r(\lambda)$ | Arc between adjacent points |
|----------|-----------|--------------|----------------------------|
| Equator | 0 | $R$ | $R \cdot \Delta\theta$ (large) |
| Near pole | $\approx R$ | $\approx 0$ | $\approx 0$ (tiny) |

### Visual Summary

```
Parameter space (θ, λ):          3D sphere surface:
┌─────────────────────┐
│ □ □ □ □ □ □ □ □ □ □ │ λ=R      ← All θ values map near north pole (clustered)
│ □ □ □ □ □ □ □ □ □ □ │          ← Stretched, thin triangles
│ □ □ □ □ □ □ □ □ □ □ │
│ □ □ □ □ □ □ □ □ □ □ │ λ=0      ← Points evenly spread (equator)
│ □ □ □ □ □ □ □ □ □ □ │
│ □ □ □ □ □ □ □ □ □ □ │          ← Stretched, thin triangles
│ □ □ □ □ □ □ □ □ □ □ │ λ=-R     ← All θ values map near south pole (clustered)
└─────────────────────┘
     θ=0        θ=2π

Uniform grid in (θ,λ)            Non-uniform point spacing on sphere!
Equal areas per cell             But unequal arc lengths between points
```

### Point Density Analysis

Point density (points per unit arc length) on each circle:

$$\rho(\lambda) = \frac{d_\theta}{2\pi r(\lambda)} = \frac{100}{2\pi \sqrt{R^2 - \lambda^2}}$$

| Region | $\lambda$ | $r(\lambda)$ | Density $\rho$ |
|--------|-----------|--------------|----------------|
| Poles | $\approx \pm R$ | $\approx 0$ | $\to \infty$ (clustered) |
| Equator | $= 0$ | $= R$ | $\frac{100}{2\pi R}$ (sparse) |

## Normal Vectors

After centering at origin, the outward normal at any point on the sphere is simply the normalized position vector:

$$\mathbf{n} = \frac{\mathbf{p}}{||\mathbf{p}||} = \frac{\mathbf{p}}{R}$$

This follows from the fact that for a sphere centered at the origin, the gradient of $f(x,y,z) = x^2 + y^2 + z^2 - R^2$ is:

$$\nabla f = (2x, 2y, 2z) = 2\mathbf{p}$$

which points radially outward.

## Summary

The algorithm is a **latitude-longitude parameterization** (like a UV-sphere in 3D modeling):

| Aspect | Description |
|--------|-------------|
| Parameterization | $(\theta, \lambda)$ where $\theta$ is azimuth, $\lambda$ is height |
| Grid structure | Regular 100 × 100 grid |
| Tessellation | Easy due to grid structure |
| Area per sample | Uniform ($R \cdot \Delta\theta \cdot \Delta\lambda$) |
| Point spacing | Non-uniform (pole clustering) |
| Triangle quality | Poor near poles (stretched, thin) |
| Use case | Visualization, mesh generation |

## Alternative: Fibonacci Sphere (Uniform Sampling)

For approximately uniform point distribution on a sphere:

```python
def sample_sphere_fibonacci(radius, center, n_points=10000):
    """Approximately uniform point distribution on sphere."""
    indices = np.arange(n_points, dtype=np.float32)
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle ≈ 2.4 radians

    y = 1 - (indices / (n_points - 1)) * 2  # y from 1 to -1
    r = np.sqrt(1 - y * y)                   # radius at height y
    theta = phi * indices                    # golden angle spiral

    x = np.cos(theta) * r
    z = np.sin(theta) * r

    points = np.stack([x, y, z], axis=1) * radius + center
    normals = (points - center) / radius

    return points.astype(np.float32), normals.astype(np.float32)
```

This produces a spiral pattern that approximates uniform coverage, but loses the regular grid structure needed for trivial tessellation.
