"""Generate a test point cloud: sphere band cut by two horizontal planes."""
import numpy as np
import os

np.random.seed(42)

R = 50.0           # sphere radius
z_lo = -10.0       # bottom cutting plane
z_hi = 20.0        # top cutting plane
N_sphere = 5000    # points on spherical band
N_plane = 2000     # points on each plane cap

# --- Cluster 0: spherical band between z_lo and z_hi ---
# Rejection-sample on the sphere
pts_sphere = []
while len(pts_sphere) < N_sphere:
    # uniform on sphere via normal distribution
    v = np.random.randn(3)
    v = v / np.linalg.norm(v) * R
    if z_lo <= v[2] <= z_hi:
        pts_sphere.append(v)
pts_sphere = np.array(pts_sphere)

# --- Cluster 1: bottom plane cap (disk at z=z_lo) ---
r_lo = np.sqrt(R**2 - z_lo**2)
angles = np.random.uniform(0, 2 * np.pi, N_plane)
radii = r_lo * np.sqrt(np.random.uniform(0, 1, N_plane))
pts_lo = np.column_stack([radii * np.cos(angles),
                          radii * np.sin(angles),
                          np.full(N_plane, z_lo)])

# --- Cluster 2: top plane cap (disk at z=z_hi) ---
r_hi = np.sqrt(R**2 - z_hi**2)
angles = np.random.uniform(0, 2 * np.pi, N_plane)
radii = r_hi * np.sqrt(np.random.uniform(0, 1, N_plane))
pts_hi = np.column_stack([radii * np.cos(angles),
                          radii * np.sin(angles),
                          np.full(N_plane, z_hi)])

# Combine with cluster labels
data = np.vstack([
    np.column_stack([pts_sphere, np.zeros(N_sphere)]),
    np.column_stack([pts_lo, np.ones(N_plane)]),
    np.column_stack([pts_hi, 2 * np.ones(N_plane)]),
])

out_dir = os.path.join("sample_clouds", "test_sphere_2planes")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "0.xyzc")
np.savetxt(out_path, data, fmt="%.6f")
print(f"Wrote {len(data)} points to {out_path}")
print(f"  Cluster 0 (sphere band): {N_sphere} pts")
print(f"  Cluster 1 (bottom cap):  {N_plane} pts, r={r_lo:.2f}")
print(f"  Cluster 2 (top cap):     {N_plane} pts, r={r_hi:.2f}")
