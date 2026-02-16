import open3d as o3d
import trimesh as tm

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