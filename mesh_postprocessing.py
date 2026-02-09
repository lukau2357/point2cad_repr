import numpy as np
import open3d as o3d
import pymesh
import pyvista as pv
import scipy.spatial
import itertools
import json

from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from color_config import get_surface_color

SURFACE_TYPE_NAMES = ["plane", "sphere", "cylinder", "cone", "inr"]

def remove_double_sided_faces(mesh):
    faces = np.asarray(mesh.triangles)
    seen = set()
    unique = []
    for face in faces:
        key = tuple(sorted(face))
        if key not in seen:
            seen.add(key)
            unique.append(face)

    result = o3d.geometry.TriangleMesh()
    result.vertices = mesh.vertices
    result.triangles = o3d.utility.Vector3iVector(np.array(unique))
    result.remove_unreferenced_vertices()
    return result

def o3d_to_pymesh(mesh):
    vertices = np.asarray(mesh.vertices).astype(np.float64)
    faces = np.asarray(mesh.triangles).astype(np.int32)
    return pymesh.form_mesh(vertices, faces)

def pymesh_to_o3d(pm_mesh):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(pm_mesh.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(pm_mesh.faces)
    return mesh

def o3d_to_pyvista(mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    faces = np.column_stack([np.full(len(triangles), 3), triangles]).ravel()
    return pv.PolyData(vertices, faces)

def compute_face_adjacency(faces):
    edge_to_faces = defaultdict(list)
    for face_idx in range(len(faces)):
        a, b, c = int(faces[face_idx, 0]), int(faces[face_idx, 1]), int(faces[face_idx, 2])
        for edge in [(min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(a, c), max(a, c))]:
            edge_to_faces[edge].append(face_idx)

    adjacency = []
    for face_list in edge_to_faces.values():
        if len(face_list) == 2:
            adjacency.append(face_list)

    if not adjacency:
        return np.empty((0, 2), dtype = np.int32)
    return np.array(adjacency, dtype = np.int32)

def face_connected_components(faces):
    num_faces = len(faces)
    adjacency = compute_face_adjacency(faces)

    if len(adjacency) == 0:
        return np.arange(num_faces)

    rows = np.concatenate([adjacency[:, 0], adjacency[:, 1]])
    cols = np.concatenate([adjacency[:, 1], adjacency[:, 0]])
    data = np.ones(len(rows))
    graph = csr_matrix((data, (rows, cols)), shape = (num_faces, num_faces))

    _, labels = connected_components(graph, directed = False)
    return labels

def closest_vertex_distances(mesh, query_points):
    vertices = np.asarray(mesh.vertices)
    tree = scipy.spatial.KDTree(vertices)
    distances, _ = tree.query(query_points)
    return distances

def extract_submesh(mesh, face_mask):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    sub_faces = faces[face_mask]

    sub = o3d.geometry.TriangleMesh()
    sub.vertices = o3d.utility.Vector3dVector(vertices)
    sub.triangles = o3d.utility.Vector3iVector(sub_faces)
    sub.remove_unreferenced_vertices()
    return sub

def clip_meshes(meshes, clusters, surface_type_ids, area_multiplier = 2.0, dedup_tolerance = 1e-6):
    single_sided = [remove_double_sided_faces(m) for m in meshes]
    pm_meshes = [o3d_to_pymesh(m) for m in single_sided]

    # Step 1: Merge all surface meshes, tracking face provenance.
    # pymesh.merge_meshes sets the "face_sources" attribute automatically.
    # Reference: io_utils.py:37-39
    pm_merged = pymesh.merge_meshes(pm_meshes)
    face_sources_merged = pm_merged.get_attribute("face_sources").astype(np.int32)

    # Step 2: Resolve self-intersections.
    # Reference: io_utils.py:41-42
    pm_resolved_ori = pymesh.resolve_self_intersection(pm_merged)

    # Step 3: Remove duplicate vertices introduced by intersection resolution.
    # Reference: io_utils.py:46-48
    pm_resolved, _ = pymesh.remove_duplicated_vertices(pm_resolved_ori, tol = dedup_tolerance)

    # Build two-level provenance: resolved face -> merged face -> original surface.
    # Reference: io_utils.py:50-53
    face_sources_resolved = pm_resolved_ori.get_attribute("face_sources").astype(np.int32)
    face_sources_from_fit = face_sources_merged[face_sources_resolved]

    # Step 4: Connected component decomposition on the face adjacency graph.
    # Replaces trimesh.graph.connected_component_labels (io_utils.py:55-62).
    resolved_faces = pm_resolved.faces
    component_labels = face_connected_components(resolved_faces)

    component_ids, component_counts = np.unique(component_labels, return_counts = True)
    sorted_order = np.argsort(-component_counts)
    sorted_component_ids = component_ids[sorted_order]

    # Step 5: Extract submeshes and assign each component to its source surface.
    # Reference: io_utils.py:68-78
    resolved_mesh = pymesh_to_o3d(pm_resolved)
    submeshes = []
    component_sources = []

    for cid in sorted_component_ids:
        mask = component_labels == cid
        sub = extract_submesh(resolved_mesh, mask)
        submeshes.append(sub)
        component_sources.append(face_sources_from_fit[mask][0])

    component_sources = np.array(component_sources)

    # Steps 6-7: For each surface, select components by proximity and filter by area.
    # Reference: io_utils.py:82-116
    clipped_meshes = []

    for p in range(len(meshes)):
        candidates = [
            sub for sub, src in zip(submeshes, component_sources)
            if src == p and len(np.asarray(sub.triangles)) > 2
        ]

        if len(candidates) == 0:
            print(f"Warning: surface {p} has no valid components after clipping.")
            clipped_meshes.append(o3d.geometry.TriangleMesh())
            continue

        distance_matrix = np.array([
            closest_vertex_distances(sub, clusters[p]) for sub in candidates
        ])

        nearest_component = np.argmin(distance_matrix, axis = 0)
        vote_counts = Counter(nearest_component).most_common()

        area_per_point = np.array([
            candidates[idx].get_surface_area() / count
            for idx, count in vote_counts
        ])

        first_nonzero = np.nonzero(area_per_point)[0]
        if len(first_nonzero) == 0:
            clipped_meshes.append(o3d.geometry.TriangleMesh())
            continue

        threshold = area_per_point[first_nonzero[0]] * area_multiplier
        keep_indices = [
            vote_counts[i][0]
            for i in range(len(vote_counts))
            if area_per_point[i] != 0 and area_per_point[i] < threshold
        ]

        kept = [candidates[idx] for idx in keep_indices]
        combined = kept[0]
        for m in kept[1:]:
            combined += m

        surface_name = SURFACE_TYPE_NAMES[surface_type_ids[p]]
        combined.paint_uniform_color(get_surface_color(surface_name))
        combined.compute_vertex_normals()
        clipped_meshes.append(combined)

    return clipped_meshes

def extract_topology(clipped_meshes, out_path = None):
    pv_meshes = [o3d_to_pyvista(m) for m in clipped_meshes]
    pairs = list(itertools.combinations(range(len(pv_meshes)), 2))

    intersection_curves = []
    intersected_pair_indices = []

    for k, (i, j) in enumerate(pairs):
        intersection, _, _ = pv_meshes[i].intersection(
            pv_meshes[j], split_first = False, split_second = False, progress_bar = False
        )
        if intersection.n_points > 0:
            intersected_pair_indices.append(k)
            curve = {
                "pair": (i, j),
                "points": intersection.points.tolist(),
                "lines": intersection.lines.reshape(-1, 3)[:, 1:].tolist()
            }
            intersection_curves.append(curve)

    corners = []
    curve_pairs = list(itertools.combinations(range(len(intersection_curves)), 2))

    for ci, cj in curve_pairs:
        pts_a = np.array(intersection_curves[ci]["points"])
        pts_b = np.array(intersection_curves[cj]["points"])
        dists = scipy.spatial.distance.cdist(pts_a, pts_b)
        rows, cols = np.where(dists == 0)

        for r, c in zip(rows, cols):
            corners.append(((pts_a[r] + pts_b[c]) / 2).tolist())

    topology = {
        "curves": intersection_curves,
        "corners": corners
    }

    if out_path is not None:
        with open(out_path, "w") as f:
            json.dump(topology, f)

    return topology
