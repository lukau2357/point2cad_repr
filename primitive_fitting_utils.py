import numpy as np
import itertools
import open3d as o3d
import scipy
import torch

from color_config import get_surface_color

def rotation_matrix_a_to_b(A, B):
    """
    Finds rotation matrix from vector A in 3d to vector B
    in 3d.
    B = R @ A
    """
    cos = np.dot(A, B)
    sin = np.linalg.norm(np.cross(B, A))
    u = A
    v = B - np.dot(A, B) * A
    v = v / (np.linalg.norm(v))
    w = np.cross(B, A)
    w = w / (np.linalg.norm(w))
    F = np.stack([u, v, w], 1)
    G = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    # B = R @ A
    try:
        R = F @ G @ np.linalg.inv(F)
    except:
        R = np.eye(3, dtype=np.float32)
    return R

def grid_trimming(cluster, vertices, size_u, size_v, device, threshold = 0.02, k = 5):
    grid = vertices.reshape(size_u, size_v, -1) # [size_u, size_v, -1], restore [u, v] shape in first two dimensions
    grid = grid[:, :, np.newaxis, :] # [size_u, size_v, 1, -1]
    grid = grid.transpose((3, 2, 0, 1)) # [-1, 1, size_u, size_v] to match Conv2d convention
    grid = torch.tensor(grid, device = device)
    filter = torch.tensor([[0.25, 0.25], [0.25, 0.25]], dtype = torch.float32, device = device).unsqueeze(0).unsqueeze(0)
    
    cell_means = torch.nn.functional.conv2d(grid, filter) # [-1, 1, size_u - 1, size_v - 1]
    cell_means = cell_means.permute(1, 2, 3, 0).squeeze(0) # [size_u - 1, size_v - 1, 3]
    cluster = torch.tensor(cluster, dtype = torch.float32, device = device)

    cell_means = cell_means.reshape((-1, 3))
    D = torch.cdist(cell_means, cluster)
    values, _ = torch.topk(D, k, dim = -1, largest = False) # [(size_u - 1) * (size_v - 1), k]
    values = values[:, -1] < threshold
    values = values.reshape((size_u - 1, size_v - 1))

    return values

def tesselate_mesh(vertices, size_u, size_v, mask = None):
    """
    Given a grid points, this returns a tesselation of the grid using triangle.
    Furthermore, if the mask is given those grids are avoided.
    Taken from original Point2CAD implementation:

    https://github.com/prs-eth/point2cad/blob/81e15bfa952aee62cf06cdf4b0897c552fe4fb3a/point2cad/fitting_utils.py#L138C1-L172C20

    I still stand by that a correct triangle in UV space does not translate to a correct triangle in the 3D space,
    this is what the original implementation uses. This will require more investigation.
    """

    def index_to_id(i, j, size_v):
        return i * size_v + j

    triangles = []

    for i in range(0, size_u - 1):
        for j in range(0, size_v - 1):
            if mask is not None and not mask[i, j]:
                continue

            v0 = index_to_id(i, j, size_v)
            v1 = index_to_id(i + 1, j, size_v)
            v2 = index_to_id(i + 1, j + 1, size_v)
            v3 = index_to_id(i, j + 1, size_v)

            triangles.append([v0, v1, v2])
            triangles.append([v0, v2, v3])

            # Reverse orientation for back-side rendering
            # Remove potentially if it harms performance?
            triangles.append([v0, v2, v1])
            triangles.append([v0, v3, v2])

    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh

def plane_error(points, a, d):
    assert np.allclose(np.linalg.norm(a), 1, rtol = 1e-6)

    plane_projections = points - np.dot(points, a)[:, np.newaxis] * a
    plane_projections += a * d
    return np.linalg.norm(plane_projections - points, axis = 1).mean()

def sphere_error(points, center, radius):
    return np.abs(np.linalg.norm(points - center, axis = 1) - radius).mean()

def cylinder_error(points, center, axis, radius):
    assert np.allclose(np.linalg.norm(axis), 1, rtol = 1e-6)

    shifted = points - center
    plane_projection = shifted - np.dot(shifted, axis)[:, np.newaxis] * axis
    orth_distance = np.linalg.norm(plane_projection, axis = 1)
    return np.abs(orth_distance - radius).mean()

def cone_error(points : np.ndarray, vertex: np.ndarray, axis: np.ndarray, theta: float):  
    assert np.allclose(np.linalg.norm(axis), 1, rtol = 1e-6)
     
    z = points - vertex
    scalar_projections = z @ axis
    h = np.abs(scalar_projections)
    r = z - scalar_projections[:, np.newaxis] * axis
    r = np.linalg.norm(r, axis = 1)
    
    errors = np.abs(h * np.sin(theta) - r * np.cos(theta))
    return np.mean(errors)

def sample_plane(mesh_dim, a, d, mean, np_rng, scale = 0.75):
    x = np.linspace(-scale, scale, mesh_dim, dtype = np.float32)
    y = np.linspace(-scale, scale, mesh_dim, dtype = np.float32)
    grid = np.array(list(itertools.product(x, y)))
    
    mean_projected = mean - (np.dot(a, mean) - d) * a
    r1, r2 = np_rng.uniform(low = 0, high = 1, size = (2,)).astype(np.float32)
    x = ((d - a[1] * r1 - a[2] * r2) / (a[0] + 1e-8)).item()
    x = np.array([x, r1, r2], dtype = np.float32)
    x = x - d * a
    x = x / np.linalg.norm(x)

    y = np.cross(a, x)
    y = y / np.linalg.norm(y)

    return x * grid[:, 0:1] + y * grid[:, 1:2] + mean_projected

def generate_plane_mesh(mesh_dim, a, d, cluster, np_rng, device,
                        mesh_mask_threshold = 0.02,
                        mesh_topk_metric = 5,
                        point_sampling_scale = 0.75):
    
    cluster_mean = cluster.mean(axis = 0)
    vertices = sample_plane(mesh_dim, a, d, cluster_mean, np_rng, scale = point_sampling_scale)
    mask = grid_trimming(cluster, vertices, mesh_dim, mesh_dim, device,
                         threshold = mesh_mask_threshold, 
                         k = mesh_topk_metric)
    mesh = tesselate_mesh(vertices, mesh_dim, mesh_dim, mask = mask)
    color = get_surface_color("plane")
    mesh.paint_uniform_color(color)
    return mesh

def sample_sphere(radius, center):
    center = center.reshape((1, 3))
    d_theta = 100
    theta = np.arange(d_theta - 1, dtype = np.float32) * 3.14 * 2 / d_theta
    theta = np.concatenate([theta, np.zeros(1)])
    circle = np.stack([np.cos(theta), np.sin(theta)], 1, dtype = np.float32)
    lam = np.linspace(
        -radius + 1e-7, radius - 1e-7, 100, dtype = np.float32
    )  # np.linspace(-1 + 1e-7, 1 - 1e-7, 100)
    radii = np.sqrt(radius ** 2 - lam ** 2)  # radius * np.sqrt(1 - lam ** 2)
    circle = np.concatenate([circle] * lam.shape[0], 0)
    spread_radii = np.repeat(radii, d_theta, 0)
    new_circle = circle * spread_radii.reshape((-1, 1))
    height = np.repeat(lam, d_theta, 0)
    points = np.concatenate([new_circle, height.reshape((-1, 1))], 1)
    points = points - np.mean(points, 0)
    normals = points / np.linalg.norm(points, axis = 1, keepdims=True)
    points = points + center
    return points, normals

def sample_cylinder_trim(radius, center, axis, points):
    center = center.reshape((1, 3))
    axis = axis.reshape((3, 1))

    d_theta = 60
    d_height = 100

    R = rotation_matrix_a_to_b(np.array([0, 0, 1]), axis[:, 0])

    # project points on to the axis
    points = points - center

    projection = points @ axis
    arg_min_proj = np.argmin(projection)
    arg_max_proj = np.argmax(projection)

    min_proj = np.squeeze(projection[arg_min_proj]) - 0.1
    max_proj = np.squeeze(projection[arg_max_proj]) + 0.1

    theta = np.arange(d_theta - 1, dtype = np.float32) * 3.14 * 2 / d_theta

    theta = np.concatenate([theta, np.zeros(1)])
    circle = np.stack([np.cos(theta), np.sin(theta)], 1, dtype = np.float32)
    circle = np.concatenate([circle] * 2 * d_height, 0) * radius

    normals = np.concatenate([circle, np.zeros((circle.shape[0], 1), dtype = np.float32)], 1)
    normals = normals / np.linalg.norm(normals, axis = 1, keepdims = True)

    height = np.expand_dims(np.linspace(min_proj, max_proj, 2 * d_height), 1)
    height = np.repeat(height, d_theta, axis = 0)
    points = np.concatenate([circle, height], 1)
    points = R @ points.T
    points = points.T + center
    normals = (R @ normals.T).T

    return points, normals

def sample_cone_trim(c, a, theta, points):
    if c is None:
        return None, None
    c = c.reshape((3))
    a = a.reshape((3))
    norm_a = np.linalg.norm(a)
    a = a / norm_a
    proj = (points - c.reshape(1, 3)) @ a
    proj_max = np.max(proj) + 0.2 * np.abs(np.max(proj))
    proj_min = np.min(proj) - 0.2 * np.abs(np.min(proj))

    # find one point on the cone
    k = np.dot(c, a)
    # TODO: VERIFY THIS FURTHER!!!
    x = (k - a[1] - a[2]) / (a[0] + 1e-7)
    y = 1
    z = 1
    d = np.array([x, y, z])
    p = a * (np.linalg.norm(d)) / (np.sin(theta)) * np.cos(theta) + d

    # This is a point on the surface
    p = p.reshape((3, 1))

    # Now rotate the vector p around axis a by variable degree
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    points = []
    normals = []
    c = c.reshape((3, 1))
    a = a.reshape((3, 1))
    rel_unit_vector = p - c
    rel_unit_vector = (p - c) / np.linalg.norm(p - c)
    rel_unit_vector_min = rel_unit_vector * (proj_min) / (np.cos(theta))
    rel_unit_vector_max = rel_unit_vector * (proj_max) / (np.cos(theta))

    for j in range(100):
        # p_ = (p - c) * (0.01) * j
        p_ = (
            rel_unit_vector_min
            + (rel_unit_vector_max - rel_unit_vector_min) * 0.01 * j
        )

        d_points = []
        d_normals = []
        for d in range(50):
            degrees = 2 * np.pi * 0.01 * d * 2
            R = np.eye(3) + np.sin(degrees) * K + (1 - np.cos(degrees)) * K @ K
            rotate_point = R @ p_
            d_points.append(rotate_point + c)
            d_normals.append(
                rotate_point
                - np.linalg.norm(rotate_point) / np.cos(theta) * a / norm_a
            )

        # repeat the points to close the circle
        d_points.append(d_points[0])
        d_normals.append(d_normals[0])

        points += d_points
        normals += d_normals

    points = np.stack(points, 0)[:, :, 0]
    normals = np.stack(normals, 0)[:, :, 0]
    normals = normals / (np.expand_dims(np.linalg.norm(normals, axis=1), 1))

    # projecting points to the axis to trim the cone along the height.
    proj = (points - c.reshape((1, 3))) @ a
    proj = proj[:, 0]
    indices = np.logical_and(proj < proj_max, proj > proj_min)
    # project points on the axis, remove points that are beyond the limits.
    return points[indices], normals[indices]

if __name__ == "__main__":
    np_rng = np.random.default_rng(41)
    N = 1000
    radius = 1
    c = np.zeros(3, dtype = np.float32)
    a = np.array([0, 0, 1], dtype = np.float32)
    theta = np.pi / 4
    points = np_rng.standard_normal(size = (1000, 3)).astype(np.float32)
    cylinder_samples, _ = sample_cone_trim(c, a, theta, points)
    # mask = grid_trimming(sphere_samples, sphere_samples, 100, 100, "cuda:0", 0.02)
    mesh = tesselate_mesh(cylinder_samples, 100, 100)
    color = get_surface_color("cone")
    mesh.paint_uniform_color(color)
    o3d.visualization.draw_geometries([mesh])