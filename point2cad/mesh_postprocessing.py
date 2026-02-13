import numpy as np
import pymesh
import pyvista as pv
import scipy.spatial
import trimesh
import itertools
import json

from collections import Counter
from scipy.sparse import lil_matrix, csr_matrix, eye as speye
from scipy.sparse.csgraph import connected_components as sparse_connected_components
from .color_config import get_surface_color

def _surface_color_rgba(surface_type):
    # Convert surface type string to trimesh-compatible RGBA uint8 array
    rgb = get_surface_color(surface_type)
    return np.array([*(rgb * 255).astype(np.uint8), 255], dtype=np.uint8)

def save_unclipped_meshes(trimesh_meshes, surface_types, out_path):
    # Reference: io_utils.py:13-33
    colored_meshes = []
    pm_meshes = []

    import sys
    for s in range(len(trimesh_meshes)):
        tri_mesh = trimesh_meshes[s]
        verts = np.array(tri_mesh.vertices, dtype = np.float64)
        faces = np.array(tri_mesh.faces, dtype = np.int32)
        # print(f"[DEBUG] Surface {s} ({surface_types[s]}): "
        #       f"verts={verts.shape}, faces={faces.shape}, "
        #       f"has_nan={np.any(np.isnan(verts))}, "
        #       f"face_max={faces.max() if faces.size > 0 else -1}, "
        #       f"num_verts={len(verts)}", flush=True)
        tri_mesh.visual.face_colors = _surface_color_rgba(surface_types[s])
        colored_meshes.append(tri_mesh)
        if faces.size == 0:
            print(f"Warning: surface {s} ({surface_types[s]}) has no faces, inserting degenerate mesh to preserve index alignment.")
            pm_meshes.append(pymesh.form_mesh(np.zeros((3, 3)), np.array([[0, 1, 2]])))
            continue
        sys.stdout.flush()
        pm_meshes.append(
            pymesh.form_mesh(verts, faces)
        )

    combined = trimesh.util.concatenate(colored_meshes)
    combined.export(out_path)
    return pm_meshes

def save_clipped_meshes(pm_meshes, clusters, surface_types, out_path,
                        area_multiplier = 2.0,
                        clip_method = "component",
                        spacing_percentile = 90, threshold_multiplier = 3.0,
                        component_filter = "area_per_point", support_fraction = 0.05,
                        connectivity = "edge",
                        walk_radius = 2, foreign_threshold = 0.5):
    # Reference: io_utils.py:36-121

    # Step 1: Merge all surface meshes, tracking face provenance.
    # https://github.com/PyMesh/PyMesh/blob/main/python/pymesh/meshutils/merge_meshes.py
    pm_merged = pymesh.merge_meshes(pm_meshes)
    # For each new face, face_sources contains the index of the input mesh/surface from which the new face originated from.
    # (Surface level mapping)
    face_sources_merged = pm_merged.get_attribute("face_sources").astype(np.int32)

    # Step 2: Resolve self-intersections.
    # https://github.com/PyMesh/PyMesh/blob/main/python/pymesh/selfintersection.py
    pm_resolved_ori = pymesh.resolve_self_intersection(pm_merged)

    # Step 3: Remove duplicate vertices.
    pm_resolved, _ = pymesh.remove_duplicated_vertices(pm_resolved_ori, tol = 1e-6, importance = None)

    # For each face after self intersection resolution, returns the index of the face from which the new face was subdivided from
    # If no subdivision is performed, the the face points to itself.
    # Two-level provenance: resolved face -> merged face -> original surface/mesh
    face_sources_resolved_ori = pm_resolved_ori.get_attribute("face_sources").astype(np.int32)
    # Track original face from the obtained merged face
    # length of face_sources_from_fit - length of faces after self-intersections are resolved.
    # This maps the new faces after self intersetction resolution to the faces before merging, via the exaplined
    # two-level provenance!
    face_sources_from_fit = face_sources_merged[face_sources_resolved_ori]

    # Diagnostic: count connected components at each stage to identify where fragmentation occurs.
    # tri_merged = trimesh.Trimesh(vertices = pm_merged.vertices, faces = pm_merged.faces, process = False)
    # labels1 = trimesh.graph.connected_component_labels(tri_merged.face_adjacency, len(tri_merged.faces))
    # n1 = len(np.unique(labels1))

    # tri_resolved_ori = trimesh.Trimesh(vertices = pm_resolved_ori.vertices, faces = pm_resolved_ori.faces, process = False)
    # labels2 = trimesh.graph.connected_component_labels(tri_resolved_ori.face_adjacency, len(tri_resolved_ori.faces))
    # n2 = len(np.unique(labels2))

    tri_resolved = trimesh.Trimesh(vertices = pm_resolved.vertices, faces = pm_resolved.faces, process = False)
    # labels3 = trimesh.graph.connected_component_labels(tri_resolved.face_adjacency, len(tri_resolved.faces))
    # n3 = len(np.unique(labels3))

    # print(f"Components after merge: {n1}, after resolve: {n2}, after dedup: {n3}")

    if clip_method == "distance":
        clipped_meshes = _clip_per_face(tri_resolved, face_sources_from_fit, clusters, surface_types, spacing_percentile, threshold_multiplier)
    elif clip_method == "cluster_mismatch":
        clipped_meshes = _clip_nearest(tri_resolved, face_sources_from_fit, clusters, surface_types)
    elif clip_method == "component_based":
        clipped_meshes = _clip_per_component(tri_resolved, face_sources_from_fit, clusters, surface_types, area_multiplier, component_filter, support_fraction, connectivity)
    elif clip_method == "provenance_walk":
        clipped_meshes = _clip_provenance_walk(tri_resolved, face_sources_from_fit, clusters, surface_types, walk_radius, foreign_threshold)
    else:
        raise ValueError(f"Unknown clip_method '{clip_method}'. Must be one of: 'distance', 'cluster_mismatch', 'component_based', 'provenance_walk'.")

    clipped = trimesh.util.concatenate(clipped_meshes)
    if len(clipped.vertices) > 0:
        clipped.export(out_path)
    else:
        print(f"Warning: clipped mesh has 0 vertices, skipping export to {out_path}")
    return clipped_meshes

def _clip_per_face(tri_resolved, face_sources_from_fit, clusters, surface_types,
                    spacing_percentile, threshold_multiplier):
    # Triangle center - Barycentric coordinates (1/3, 1/3, 1/3)
    centroids = tri_resolved.triangles_center
    clipped_meshes = []

    for p in range(len(clusters)):
        face_mask = face_sources_from_fit == p
        face_indices = np.where(face_mask)[0]

        if len(face_indices) == 0:
            print(f"Warning: surface {p} has no faces after resolution.")
            clipped_meshes.append(trimesh.Trimesh())
            continue

        # Compute nearest-neighbor distances within the cluster to infer point spacing
        cluster_dists = scipy.spatial.distance.cdist(clusters[p], clusters[p])
        np.fill_diagonal(cluster_dists, np.inf)
        nn_dists = cluster_dists.min(axis = 1)
        spacing = np.percentile(nn_dists, spacing_percentile)
        threshold = spacing * threshold_multiplier

        # For each face, compute the distance from its centroid to the closest cluster point
        surface_centroids = centroids[face_indices]
        dists = scipy.spatial.distance.cdist(surface_centroids, clusters[p])
        min_dists = dists.min(axis = 1)

        # Diagnostics: understand the distance distribution
        print(f"Surface {p} ({surface_types[p]}):")
        print(f"  Cluster: {len(clusters[p])} points, NN spacing p{spacing_percentile}={spacing:.6f}")
        print(f"  Faces: {len(face_indices)}, min_dist range: [{min_dists.min():.6f}, {min_dists.max():.6f}], "
              f"mean={min_dists.mean():.6f}, median={np.median(min_dists):.6f}")
        print(f"  Threshold: {threshold:.6f}")

        # Discard faces whose centroid is farther than the threshold
        keep = min_dists <= threshold

        kept_faces = tri_resolved.faces[face_indices[keep]]
        print(f"  Result: {keep.sum()}/{len(face_indices)} faces kept")

        mesh = trimesh.Trimesh(vertices = tri_resolved.vertices, faces = kept_faces)
        mesh.visual.face_colors = _surface_color_rgba(surface_types[p])
        clipped_meshes.append(mesh)

    return clipped_meshes

def _clip_nearest(tri_resolved, face_sources_from_fit, clusters, surface_types):
    # Triangle center - Barycentric coordinates (1/3, 1/3, 1/3)
    centroids = tri_resolved.triangles_center

    # Build a single KD-tree from all cluster points with their surface labels.
    # For each face centroid, the tree answers "which cluster point is closest?"
    # If the closest point's surface label does not match the face's provenance
    # (face_sources_from_fit), the face extends into another surface's territory
    # after intersection resolution and is discarded from the clipped mesh.
    all_points = np.concatenate(clusters)
    all_labels = np.concatenate([np.full(len(c), i, dtype = np.int32) for i, c in enumerate(clusters)])
    tree = scipy.spatial.cKDTree(all_points)
    _, nn_indices = tree.query(centroids)
    nearest_surface = all_labels[nn_indices]

    clipped_meshes = []

    for p in range(len(clusters)):
        face_mask = face_sources_from_fit == p
        face_indices = np.where(face_mask)[0]

        if len(face_indices) == 0:
            print(f"Warning: surface {p} has no faces after resolution.")
            clipped_meshes.append(trimesh.Trimesh())
            continue

        # Keep only faces whose nearest cluster point belongs to surface p
        keep = nearest_surface[face_indices] == p
        kept_faces = tri_resolved.faces[face_indices[keep]]

        print(f"Surface {p} ({surface_types[p]}): {keep.sum()}/{len(face_indices)} faces kept")

        mesh = trimesh.Trimesh(vertices = tri_resolved.vertices, faces = kept_faces)
        mesh.visual.face_colors = _surface_color_rgba(surface_types[p])
        clipped_meshes.append(mesh)

    return clipped_meshes

def _clip_provenance_walk(tri_resolved, face_sources_from_fit, clusters, surface_types,
                           walk_radius, foreign_threshold):
    """Per-face filtering using BFS neighborhood provenance voting.

    For each face F attributed to surface p, consider all faces reachable within
    `walk_radius` edge-hops of F. Among these, compute the fraction belonging to a
    different surface (the "foreign ratio"). If this ratio exceeds `foreign_threshold`,
    F is likely an excess face extending past an intersection boundary into foreign
    territory and is discarded.

    Interior faces are surrounded by same-surface faces (low foreign ratio).
    Excess faces at intersection boundaries are surrounded by the other surface's
    faces (high foreign ratio).
    """
    num_faces = len(tri_resolved.faces)
    num_vertices = len(tri_resolved.vertices)

    # Vertex-based adjacency: two faces are adjacent if they share at least one vertex.
    # Edge-based adjacency (face_adjacency) has 0 cross-surface pairs after
    # resolve_self_intersection + dedup, because PyMesh keeps each surface's faces
    # topologically isolated even when they share vertex positions at intersection
    # boundaries. Vertex-based adjacency bridges across surfaces through shared
    # intersection vertices, enabling the BFS walk to detect foreign faces.
    #
    # Computed via sparse incidence matrix: F[i, v] = 1 if face i contains vertex v.
    # A = F @ F^T gives A[i, j] = number of shared vertices between faces i and j.
    face_indices = np.repeat(np.arange(num_faces), 3)
    vertex_indices = tri_resolved.faces.flatten()
    F = csr_matrix(
        (np.ones(len(face_indices), dtype = np.float64), (face_indices, vertex_indices)),
        shape = (num_faces, num_vertices)
    )
    A = F @ F.T
    A.setdiag(0)
    A.eliminate_zeros()
    # Binarize: we only care about adjacency, not the count of shared vertices
    A = (A > 0).astype(np.float64)

    # Diagnostic: cross-surface adjacency check
    adj_coo = A.tocoo()
    src_a = face_sources_from_fit[adj_coo.row]
    src_b = face_sources_from_fit[adj_coo.col]
    cross_count = (src_a != src_b).sum() // 2  # each pair counted twice (symmetric)
    total_pairs = adj_coo.nnz // 2
    print(f"[provenance_walk] Vertex-based adjacency pairs: {total_pairs}")
    print(f"[provenance_walk] Cross-surface pairs: {cross_count} / {total_pairs}")
    print(f"[provenance_walk] Num faces: {num_faces}")

    # Compute r-hop reachability matrix: M[i, j] > 0 iff face j is within
    # walk_radius edge-hops of face i. M is symmetric (if i reaches j, j reaches i),
    # so the foreign ratio computation exploits this automatically.
    I = speye(num_faces, format = "csr")
    M = I.copy()
    power = I.copy()
    for _ in range(walk_radius):
        power = power @ A
        M = M + power
    # Binarize: we only care about reachability, not path counts
    M = (M > 0).astype(np.float64)

    # Total neighborhood size for each face (including itself)
    total_counts = np.array(M.sum(axis = 1)).flatten()
    print(f"[provenance_walk] Neighborhood sizes (walk_radius={walk_radius}): "
          f"min={total_counts.min():.0f}, max={total_counts.max():.0f}, "
          f"mean={total_counts.mean():.1f}, median={np.median(total_counts):.0f}")
    print(f"[provenance_walk] Faces with neighborhood size 1 (only self): {(total_counts == 1).sum()}")

    # For each surface, count how many neighbors share the same provenance
    clipped_meshes = []

    for p in range(len(clusters)):
        face_mask = face_sources_from_fit == p
        face_indices = np.where(face_mask)[0]

        if len(face_indices) == 0:
            print(f"Warning: surface {p} has no faces after resolution.")
            clipped_meshes.append(trimesh.Trimesh())
            continue

        # Indicator vector: 1 for faces belonging to surface p, 0 otherwise
        indicator = face_mask.astype(np.float64)
        # For each face, count how many of its r-hop neighbors belong to surface p
        same_surface_counts = np.array(M @ indicator).flatten()

        # Foreign ratio: fraction of neighbors NOT from surface p
        surface_foreign = 1.0 - same_surface_counts[face_indices] / total_counts[face_indices]
        keep = surface_foreign <= foreign_threshold
        kept_faces = tri_resolved.faces[face_indices[keep]]

        print(f"Surface {p} ({surface_types[p]}): {keep.sum()}/{len(face_indices)} faces kept "
              f"(foreign ratio: mean={surface_foreign.mean():.3f}, max={surface_foreign.max():.3f})")

        mesh = trimesh.Trimesh(vertices = tri_resolved.vertices, faces = kept_faces)
        mesh.visual.face_colors = _surface_color_rgba(surface_types[p])
        clipped_meshes.append(mesh)

    return clipped_meshes

def _vertex_based_component_labels(faces, num_vertices):
    """Compute connected component labels using vertex-based adjacency.
    Two faces are connected if they share at least one vertex. This bridges
    T-junctions where faces share a vertex but not a full edge (see Section 7
    of mesh_clipping_and_topology.md)."""
    num_faces = len(faces)
    # For each vertex, collect all faces containing it
    vert_to_faces = [[] for _ in range(num_vertices)]
    for fi in range(num_faces):
        for vi in faces[fi]:
            vert_to_faces[vi].append(fi)

    # Build sparse adjacency: faces sharing any vertex are connected
    adj = lil_matrix((num_faces, num_faces), dtype = bool)
    for face_list in vert_to_faces:
        for i in range(len(face_list)):
            for j in range(i + 1, len(face_list)):
                adj[face_list[i], face_list[j]] = True

    _, labels = sparse_connected_components(adj, directed = False)
    return labels

def _clip_per_component(tri_resolved, face_sources_from_fit, clusters, surface_types,
                        area_multiplier, component_filter, support_fraction, connectivity):
    # Step 4: Connected component decomposition.
    if connectivity == "edge":
        # Face adjacency matrix of the deduplicated merged mesh. Using Trimesh API for remaining operations.
        face_adjacency = tri_resolved.face_adjacency

        # https://trimesh.org/trimesh.graph.html#trimesh.graph.connected_component_labels
        # Mesh graph nodes are faces, and two faces are adjacent if they share the same edge/line. This computes
        # the connected components of this graph - which can be interpreted as a dual graph of the input.
        # So the number of vertices is the number of faces - each face gets adjoined a connected component label!
        connected_node_labels = trimesh.graph.connected_component_labels(
            edges = face_adjacency, node_count = len(tri_resolved.faces)
        )
    elif connectivity == "vertex":
        # Vertex-based adjacency: two faces are connected if they share at least one vertex.
        # This bridges T-junctions introduced by resolve_self_intersection, where split faces
        # and their non-split neighbors share corner vertices but not full edges.
        connected_node_labels = _vertex_based_component_labels(
            tri_resolved.faces, len(tri_resolved.vertices)
        )
    else:
        raise ValueError(f"Unknown connectivity '{connectivity}'. Must be one of: 'edge', 'vertex'.")

    # Order connected components by number of faces in descending order.
    most_common_groupids = [item[0] for item in Counter(connected_node_labels).most_common()]
    print(f"Connected components ({connectivity}-based): {len(most_common_groupids)}")

    # Step 5: Extract submeshes and assign each to its source surface.
    submeshes = [
        trimesh.Trimesh(
            vertices = np.array(tri_resolved.vertices),
            # Extract all faces belonging to the current connected component
            faces = np.array(tri_resolved.faces)[np.where(connected_node_labels == item)]
        )
        for item in most_common_groupids
    ]

    indices_sources = [
        # face_sources_from_fit[connected_node_labels == item] - all input meshes that correspond to the current component
        # selects the first one as representative. Reasoning behind this, is this correct?
        # Naturally, a single connected component should be covered by exactly one input mesh?
        face_sources_from_fit[connected_node_labels == item][0]
        for item in np.array(most_common_groupids)
    ]

    # Steps 6-7: For each surface, select components by proximity and filter.
    clipped_meshes = []

    for p in range(len(clusters)):
        # (cluster_size, 3)
        one_cluster_points = clusters[p]

        # All submeshes associated with the current cluster/fitted surface/original mesh
        # Which are not degenerate, have at most 2 faces.
        submeshes_cur = [
            x for x, y in zip(submeshes, np.array(indices_sources) == p)
            if y and len(x.faces) > 2
        ]

        if len(submeshes_cur) == 0:
            print(f"Warning: surface {p} has no valid components after clipping.")
            clipped_meshes.append(trimesh.Trimesh())
            continue

        # https://trimesh.org/trimesh.proximity.html#trimesh.proximity.closest_point
        # Then for each point of the cluster extract the id of the closest mesh in submesh_cur
        # The resulting array is of shape (cluster_size)
        nearest_submesh = np.argmin(
            # Inner array is (cluster_size, len(submesh_cur))
            np.array([
                trimesh.proximity.closest_point(item, one_cluster_points)[1]
                for item in submeshes_cur
            ]).transpose(),
            -1,
        )

        counter_nearest = Counter(nearest_submesh).most_common()

        if component_filter == "area_per_point":
            result_indices = _filter_area_per_point(counter_nearest, submeshes_cur, area_multiplier, p)
        elif component_filter == "min_support":
            result_indices = _filter_min_support(counter_nearest, support_fraction, p)
        else:
            raise ValueError(f"Unknown component_filter '{component_filter}'. Must be one of: 'area_per_point', 'min_support'.")

        result_submesh_list = [submeshes_cur[item] for item in result_indices]
        if len(result_submesh_list) == 0:
            print(f"Warning: surface {p} has no surviving components after filtering.")
            clipped_meshes.append(trimesh.Trimesh())
            continue
        clipped_mesh = trimesh.util.concatenate(result_submesh_list)
        clipped_mesh.visual.face_colors = _surface_color_rgba(surface_types[p])
        clipped_meshes.append(clipped_mesh)

    return clipped_meshes

def _filter_area_per_point(counter_nearest, submeshes_cur, area_multiplier, surface_idx):
    """Original filtering criterion: keep components whose area-per-point ratio
    is within area_multiplier of the best (lowest) ratio."""
    # Compute the ratio submesh_area / number of points it contains - area per point
    # This is of shape [K], where K <= len(submesh_cur)
    # Well fitting submesh should have low area per point.
    area_per_point = np.array([
        submeshes_cur[item[0]].area / item[1] for item in counter_nearest
    ])

    # np.array(counter_nearest) is of shape (K, 2). First entry is the cluster_id, the second entry
    # is the number of supporting points
    nonzero_indices = np.nonzero(area_per_point)
    best_app = area_per_point[nonzero_indices].min()

    if len(nonzero_indices[0]) == 0:
        print(f"Warning: surface {surface_idx} has only zero-area components.")
        return []

    return np.array(counter_nearest)[:, 0][
        np.logical_and(
            # Compare each area_per_point with the first non-zero element of area_per_point multiplied by area_multiplier
            # First element of area_per_point will contain the best fitting fragment/submesh, with respect to the number
            # of points closest to it. Allowance is best_area_per_point * 2 (default value of area_multiplier)
            # but this is parametrizable. Remember, smaller area_per_point is better
            # Of course, we disallow fragments with zero area - covered by the second condition.
            area_per_point < best_app * area_multiplier,
            area_per_point != 0,
        )
    ]

def _filter_min_support(counter_nearest, support_fraction, surface_idx):
    """Support-based filtering: keep components whose number of supporting points
    is at least support_fraction of the best (most supported) component's count.
    This penalizes small components with few supporting points regardless of their area."""
    # counter_nearest is sorted by count descending (from Counter.most_common())
    # counter_nearest[i] = (submesh_index, num_supporting_points)
    support_counts = np.array([item[1] for item in counter_nearest])
    print(support_counts)
    best_support = support_counts[0]
    threshold = best_support * support_fraction

    keep_mask = support_counts >= threshold
    return np.array(counter_nearest)[:, 0][keep_mask]

def save_topology(clipped_meshes, out_path):
    # Reference: io_utils.py:124-171
    import vtk
    vtk.vtkObject.GlobalWarningDisplayOff()
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
        # Zero positions are indicated by two separate arrays
        # Row indices - contain i positions of zeros
        # Col indices - contain j positions of zeros
        # These should not be averaged?
        row_indices, col_indices = np.where(dists == 0)

        if len(row_indices) > 0 and len(col_indices) > 0:
            # corners = [
            #    (sample0[item[0]] + sample1[item[1]]) / 2
            #   for item in zip(row_indices, col_indices)
            # ]
            corners = [sample0[idx] for idx in row_indices]
            intersection_corners.extend(corners)

    intersections["corners"] = [arr.tolist() for arr in intersection_corners]

    with open(out_path, "w") as f:
        json.dump(intersections, f, indent = 4)
