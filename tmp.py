import open3d as o3d
import trimesh as tm
import numpy as np

'''
def trimesh_to_o3d(path):
    m = tm.load(path)
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(m.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(m.faces)
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
        m.visual.vertex_colors[:, :3] / 255.0
    )
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh

cloud_id = "00949"
# o3d_mesh = trimesh_to_o3d(f"/home/lukau/Downloads/{cloud_id}.stl")
o3d_pc = o3d.io.read_point_cloud(f"/home/lukau/Downloads/samples_{cloud_id}.ply")
geometries = [o3d_pc]
o3d.visualization.draw_geometries(geometries)
'''

np_rng = np.random.default_rng(41)
a1 = np_rng.normal(loc = 0, scale = 1, size = (3)).astype(np.float32)
a2 = np.cross(a1, np.array([1, 0, 0], dtype = np.float32))

A = np.stack([a1, a2], axis = 0)
print(A.shape)
b = np_rng.normal(loc = 0, scale = 1, size = (2,))

sol_1, _, rank, _ = np.linalg.lstsq(A, b)
sol_2 = np.linalg.inv(A.T @ A) @ A.T @ b
sol_3 = A.T @ np.linalg.solve(A @ A.T, b)

print(sol_1)
print(sol_2)
print(sol_3)

print(np.linalg.norm(sol_1 - sol_2))
print(np.linalg.norm(sol_2 - sol_3))

print(np.linalg.norm(A @ sol_1 - b))
print(np.linalg.norm(A @ sol_2 - b))
print(np.linalg.norm(A @ sol_3 - b))