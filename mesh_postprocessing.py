import numpy as np
import pymesh
import pyvista as pv
import scipy.spatial
import trimesh
import itertools
import json

from collections import Counter

def save_unclipped_meshes(trimesh_meshes, color_list, out_path):
    # Reference: io_utils.py:13-33
    colored_meshes = []
    pm_meshes = []

    for s in range(len(trimesh_meshes)):
        tri_mesh = trimesh_meshes[s]
        tri_mesh.visual.face_colors = color_list[s]
        colored_meshes.append(tri_mesh)
        pm_meshes.append(
            pymesh.form_mesh(
                np.array(tri_mesh.vertices, dtype = np.float64),
                np.array(tri_mesh.faces, dtype = np.int32)
            )
        )

    combined = trimesh.util.concatenate(colored_meshes)
    combined.export(out_path)
    return pm_meshes

def save_clipped_meshes(pm_meshes, clusters, color_list, out_path, area_multiplier = 2.0):
    # Reference: io_utils.py:36-121

    # Step 1: Merge all surface meshes, tracking face provenance.
    # https://github.com/PyMesh/PyMesh/blob/main/python/pymesh/meshutils/merge_meshes.py
    pm_merged = pymesh.merge_meshes(pm_meshes)
    # For each new face, face_sources is the index of the original face the new face originated from.
    face_sources_merged = pm_merged.get_attribute("face_sources").astype(np.int32)

    # Step 2: Resolve self-intersections.
    # https://github.com/PyMesh/PyMesh/blob/main/python/pymesh/selfintersection.py
    pm_resolved_ori = pymesh.resolve_self_intersection(pm_merged)

    # Step 3: Remove duplicate vertices.
    pm_resolved, _ = pymesh.remove_duplicated_vertices(pm_resolved_ori, tol = 1e-6, importance = None)

    # Two-level provenance: resolved face -> merged face -> original surface.
    # After resolving self-intersections (edge computation), for each new face return the ID of the face 
    # that contained it in the original mesh.
    face_sources_resolved_ori = pm_resolved_ori.get_attribute("face_sources").astype(np.int32)
    # Track original face from the obtained merged face
    face_sources_from_fit = face_sources_merged[face_sources_resolved_ori]

    # Step 4: Connected component decomposition.
    tri_resolved = trimesh.Trimesh(vertices = pm_resolved.vertices, faces = pm_resolved.faces)
    # Face adjacency matrix of the deduplicated merged mesh. Using Trimesh API for reamining operations.
    face_adjacency = tri_resolved.face_adjacency
    # https://trimesh.org/trimesh.graph.html#trimesh.graph.connected_component_labels
    # Mesh graph node are faces, and two faces are adjacent if they share the same edge/line. This computes 
    # the connected components of this graph - which can be interpreted as a dual graph of the input. 
    # So the number of vertices is the number of faces - each face gets adjoined a connected component label!
    connected_node_labels = trimesh.graph.connected_component_labels(
        edges = face_adjacency, node_count = len(tri_resolved.faces)
    )

    # Order connected components by number of faces in descending order.
    most_common_groupids = [item[0] for item in Counter(connected_node_labels).most_common()]

    # Step 5: Extract submeshes and assign each to its source surface.
    submeshes = [
        trimesh.Trimesh(
            # Some vertices might not end up in the current submesh?
            vertices = np.array(tri_resolved.vertices),
            # Extract all faces belonging to the current connected component
            faces = np.array(tri_resolved.faces)[np.where(connected_node_labels == item)]
        )
        for item in most_common_groupids
    ]

    indices_sources = [
        face_sources_from_fit[connected_node_labels == item][0]
        for item in np.array(most_common_groupids)
    ]

    # Steps 6-7: For each surface, select components by proximity and filter by area.
    clipped_meshes = []

    for p in range(len(clusters)):
        one_cluster_points = clusters[p]
        submeshes_cur = [
            x for x, y in zip(submeshes, np.array(indices_sources) == p)
            if y and len(x.faces) > 2
        ]

        if len(submeshes_cur) == 0:
            print(f"Warning: surface {p} has no valid components after clipping.")
            clipped_meshes.append(trimesh.Trimesh())
            continue

        nearest_submesh = np.argmin(
            np.array([
                trimesh.proximity.closest_point(item, one_cluster_points)[1]
                for item in submeshes_cur
            ]).transpose(),
            -1,
        )

        counter_nearest = Counter(nearest_submesh).most_common()
        area_per_point = np.array([
            submeshes_cur[item[0]].area / item[1] for item in counter_nearest
        ])

        result_indices = np.array(counter_nearest)[:, 0][
            np.logical_and(
                area_per_point < area_per_point[np.nonzero(area_per_point)[0][0]] * area_multiplier,
                area_per_point != 0,
            )
        ]

        result_submesh_list = [submeshes_cur[item] for item in result_indices]
        clipped_mesh = trimesh.util.concatenate(result_submesh_list)
        clipped_mesh.visual.face_colors = color_list[p]
        clipped_meshes.append(clipped_mesh)

    clipped = trimesh.util.concatenate(clipped_meshes)
    clipped.export(out_path)
    return clipped_meshes

def save_topology(clipped_meshes, out_path):
    # Reference: io_utils.py:124-171
    pv_meshes = [pv.wrap(item) for item in clipped_meshes]
    pv_combinations = list(itertools.combinations(pv_meshes, 2))

    intersection_curves = []
    intersections = {}

    for k, pv_pair in enumerate(pv_combinations):
        intersection, _, _ = pv_pair[0].intersection(
            pv_pair[1], split_first = False, split_second = False, progress_bar = False
        )
        if intersection.n_points > 0:
            intersection_curves.append({
                "pv_points": intersection.points.tolist(),
                "pv_lines": intersection.lines.reshape(-1, 3)[:, 1:].tolist()
            })

    intersections["curves"] = intersection_curves

    intersection_corners = []
    curve_combinations = list(itertools.combinations(range(len(intersection_curves)), 2))

    for ci, cj in curve_combinations:
        sample0 = np.array(intersection_curves[ci]["pv_points"])
        sample1 = np.array(intersection_curves[cj]["pv_points"])
        dists = scipy.spatial.distance.cdist(sample0, sample1)
        row_indices, col_indices = np.where(dists == 0)

        if len(row_indices) > 0 and len(col_indices) > 0:
            corners = [
                (sample0[item[0]] + sample1[item[1]]) / 2
                for item in zip(row_indices, col_indices)
            ]
            intersection_corners.extend(corners)

    intersections["corners"] = [arr.tolist() for arr in intersection_corners]

    with open(out_path, "w") as f:
        json.dump(intersections, f)
