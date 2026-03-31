import numpy as np
import open3d as o3d
import igl
import igl.copyleft.cgal
import trimesh
from collections import Counter, deque
from scipy.spatial import cKDTree

from .color_config import get_surface_color


def o3d_mesh_to_numpy(o3d_mesh):
    vertices = np.asarray(o3d_mesh.vertices, dtype=np.float64)
    triangles = np.asarray(o3d_mesh.triangles, dtype=np.int64)
    # triangulate_and_mesh stores 4 triangles per grid cell (2 front + 2 back).
    # Keep only the first 2 (front-facing) for the self-intersection pipeline.
    mask = np.arange(len(triangles)) % 4 < 2
    triangles = triangles[mask]
    return vertices, triangles


def _merge_meshes(mesh_list):
    all_V = []
    all_F = []
    face_sources = []
    vertex_offset = 0

    for idx, (V, F) in enumerate(mesh_list):
        all_V.append(np.asarray(V, dtype=np.float64))
        all_F.append(np.asarray(F, dtype=np.int64) + vertex_offset)
        face_sources.append(np.full(len(F), idx, dtype=np.int32))
        vertex_offset += len(V)

    V = np.vstack(all_V) if all_V else np.empty((0, 3), dtype=np.float64)
    F = np.vstack(all_F) if all_F else np.empty((0, 3), dtype=np.int64)
    face_sources = np.concatenate(face_sources) if face_sources else np.empty(0, dtype=np.int32)

    return V, F, face_sources


def _build_edge_face_map(F):
    """Map each undirected edge (min_v, max_v) to the list of face indices containing it."""
    edge_to_faces = {}
    for fi in range(len(F)):
        for k in range(3):
            v0 = int(F[fi, k])
            v1 = int(F[fi, (k + 1) % 3])
            edge = (min(v0, v1), max(v0, v1))
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(fi)
    return edge_to_faces


def build_cluster_trees(clusters, spacing_percentile=100.0):
    """
    Precompute per-cluster KDTree and intra-cluster NN spacing.

    Parameters
    ----------
    clusters           : list of (N, 3) float arrays
    spacing_percentile : percentile of intra-cluster NN distances (default 100 = max)

    Returns
    -------
    trees    : list[cKDTree]
    spacings : list[float] — one scalar per cluster
    """
    trees, spacings = [], []
    for c in clusters:
        tree = cKDTree(c)
        d, _ = tree.query(c, k=2)   # k=2: column 0 is self (dist=0), column 1 is NN
        spacings.append(float(np.percentile(d[:, 1], spacing_percentile)))
        trees.append(tree)
    return trees, spacings


def _resolve_intersections(o3d_meshes, tag="clip"):
    """
    Merge all surface meshes, resolve self-intersections via libigl/CGAL,
    deduplicate vertices.

    Returns
    -------
    VV             : (N, 3) float64 — resolved vertices
    FF             : (M, 3) int64   — resolved faces
    face_provenance: (M,)   int32   — original surface index per resolved face
    """
    mesh_list = []
    for m in o3d_meshes:
        V_i, F_i = o3d_mesh_to_numpy(m)
        mesh_list.append((V_i.astype(np.float64), F_i.astype(np.int32)))
    V_merged, F_merged, face_sources_merged = _merge_meshes(mesh_list)
    print(f"[{tag}] Merged: {len(V_merged)} vertices, {len(F_merged)} faces")

    # J[i] = merged face that resolved face i came from (one-level provenance)
    # IM[i] = canonical vertex index for vertex i (deduplication map)
    VV_raw, FF_raw, _IF, J, IM = igl.copyleft.cgal.remesh_self_intersections(
        V_merged, F_merged.astype(np.int32)
    )
    face_provenance = face_sources_merged[J].astype(np.int32)

    FF_dedup = IM[FF_raw]
    VV, FF_clean, _I, _J2 = igl.remove_unreferenced(VV_raw, FF_dedup)
    FF = FF_clean.astype(np.int64)
    print(f"[{tag}] Resolved: {len(VV)} vertices, {len(FF)} faces")
    return VV, FF, face_provenance


def clip_meshes_p2cad(o3d_meshes, clusters, surface_types,
                      area_multiplier=2.0):
    """
    Point2CAD-style mesh clipping: connected components of the resolved mesh
    filtered by area-per-supporting-point ratio.

    For each surface s, the resolved mesh is split into connected components
    (edge-based adjacency). Each cluster point votes for its nearest component.
    Components whose area/vote_count is within area_multiplier of the best
    (lowest) ratio are kept.

    Parameters
    ----------
    o3d_meshes     : list of open3d.geometry.TriangleMesh, one per cluster
    clusters       : list of (N, 3) float arrays — input point clouds
    surface_types  : list of str — surface type names (for coloring)
    area_multiplier: keep components with area_per_point < best * area_multiplier

    Returns
    -------
    clipped : list of open3d.geometry.TriangleMesh, one per cluster
    """
    VV, FF, face_provenance = _resolve_intersections(o3d_meshes, tag="p2cad-clip")

    tri_resolved = trimesh.Trimesh(vertices=VV, faces=FF, process=False)
    connected_labels = trimesh.graph.connected_component_labels(
        edges=tri_resolved.face_adjacency,
        node_count=len(FF)
    )
    unique_labels = [item[0] for item in Counter(connected_labels).most_common()]
    print(f"[p2cad-clip] {len(unique_labels)} connected components")

    # Build submeshes and attribute each to a surface via majority provenance
    submeshes = []
    submesh_surface = []
    for lbl in unique_labels:
        face_idx = np.where(connected_labels == lbl)[0]
        if len(face_idx) <= 2:
            continue
        sub = trimesh.Trimesh(vertices=VV,
                              faces=FF[face_idx],
                              process=False)
        submeshes.append(sub)
        submesh_surface.append(int(face_provenance[face_idx[0]]))

    clipped = []
    for s in range(len(clusters)):
        cluster_pts = clusters[s].astype(np.float64)
        subs = [sub for sub, sid in zip(submeshes, submesh_surface) if sid == s]

        if len(subs) == 0:
            print(f"[p2cad-clip] surface {s} ({surface_types[s]}): no components")
            clipped.append(o3d.geometry.TriangleMesh())
            continue

        # For each cluster point, find its nearest submesh
        nearest = np.argmin(
            np.array([trimesh.proximity.closest_point(sub, cluster_pts)[1]
                      for sub in subs]).T,
            axis=1
        )
        counter = Counter(nearest).most_common()
        area_per_point = np.array([subs[idx].area / count
                                   for idx, count in counter])

        nonzero = np.nonzero(area_per_point)[0]
        if len(nonzero) == 0:
            print(f"[p2cad-clip] surface {s} ({surface_types[s]}): all zero-area components")
            clipped.append(o3d.geometry.TriangleMesh())
            continue

        best = area_per_point[nonzero[0]]
        keep_idx = np.array(counter)[:, 0][
            (area_per_point < best * area_multiplier) & (area_per_point != 0)
        ]

        kept = trimesh.util.concatenate([subs[i] for i in keep_idx])
        print(f"[p2cad-clip] surface {s} ({surface_types[s]}): "
              f"{len(keep_idx)}/{len(subs)} components kept")

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices  = o3d.utility.Vector3dVector(np.array(kept.vertices))
        mesh.triangles = o3d.utility.Vector3iVector(np.array(kept.faces))
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(get_surface_color(surface_types[s]))
        clipped.append(mesh)

    return clipped


def clip_meshes_bfs(o3d_meshes, clusters, surface_types,
                    cluster_trees, spacings,
                    post_filter_threshold=1.0):
    """
    Trim each surface mesh to its correct region using BFS flood-fill on the
    CGAL-resolved unified mesh, stopping at cross-provenance edges.

    After CGAL resolves self-intersections, each face carries a provenance label
    (which original surface it came from). Edges shared by faces of different
    provenances are intersection boundaries — the BFS does not cross them.
    The seed face is the one with provenance s whose centroid is closest to any
    cluster point (robust to partially visible surfaces).

    Parameters
    ----------
    o3d_meshes            : list of open3d.geometry.TriangleMesh, one per cluster
    clusters              : list of (N, 3) float arrays — input point clouds
    surface_types         : list of str — surface type names (for coloring)
    cluster_trees         : list[cKDTree] — precomputed, one per cluster
    spacings              : list[float]   — intra-cluster NN spacing per cluster
    post_filter_threshold : multiplier for the optional post-BFS distance filter

    Returns
    -------
    clipped : list of open3d.geometry.TriangleMesh, one per cluster
    """
    # Merge all meshes into one, tracking face provenance
    mesh_list = []
    for m in o3d_meshes:
        V_i, F_i = o3d_mesh_to_numpy(m)
        mesh_list.append((V_i.astype(np.float64), F_i.astype(np.int32)))
    V_merged, F_merged, face_sources_merged = _merge_meshes(mesh_list)
    print(f"[bfs-clip] Merged: {len(V_merged)} vertices, {len(F_merged)} faces")

    # J[i] = merged face that resolved face i came from (one-level provenance)
    # IM[i] = canonical vertex index for vertex i (deduplication map)
    VV_raw, FF_raw, _IF, J, IM = igl.copyleft.cgal.remesh_self_intersections(
        V_merged, F_merged.astype(np.int32)
    )
    face_provenance = face_sources_merged[J].astype(np.int32)

    FF_dedup = IM[FF_raw]
    VV, FF_clean, _I, _J2 = igl.remove_unreferenced(VV_raw, FF_dedup)
    FF = FF_clean.astype(np.int64)
    print(f"[bfs-clip] Resolved: {len(VV)} vertices, {len(FF)} faces")

    # Build edge → [face indices] map on the resolved mesh
    edge_to_faces = _build_edge_face_map(FF)

    # Log adjacent surface pairs detected from cross-provenance edges
    adj_pairs = set()
    for face_list in edge_to_faces.values():
        provs = {face_provenance[fi] for fi in face_list}
        if len(provs) > 1:
            provs_sorted = sorted(provs)
            for i in range(len(provs_sorted)):
                for j in range(i + 1, len(provs_sorted)):
                    adj_pairs.add((provs_sorted[i], provs_sorted[j]))
    for a, b in sorted(adj_pairs):
        print(f"[bfs-clip] adjacent surfaces: {a} ({surface_types[a]}) ↔ "
              f"{b} ({surface_types[b]})")

    # Per-face adjacency: same-provenance 2-face edges only.
    # Cross-provenance seam edges are non-manifold (4 faces) and excluded,
    # so BFS naturally cannot cross the seam between surfaces.
    n_faces = len(FF)
    face_adj = [[] for _ in range(n_faces)]
    for edge, face_list in edge_to_faces.items():
        if len(face_list) != 2:
            continue
        fi, fj = face_list
        if face_provenance[fi] == face_provenance[fj]:
            face_adj[fi].append(fj)
            face_adj[fj].append(fi)

    # Face centroids: mean of the 3 vertex positions for each resolved face
    centroids = VV[FF].mean(axis=1)  # (n_faces, 3)

    clipped = []
    for s in range(len(clusters)):
        prov_mask = face_provenance == s
        prov_faces = np.where(prov_mask)[0]

        if len(prov_faces) == 0:
            print(f"[bfs-clip] surface {s} ({surface_types[s]}): "
                  f"no faces after resolution")
            clipped.append(o3d.geometry.TriangleMesh())
            continue

        # Seed: face with provenance s whose centroid is closest to any cluster point.
        prov_centroids = centroids[prov_faces]
        dists, _ = cluster_trees[s].query(prov_centroids)
        seed = prov_faces[np.argmin(dists)]

        # BFS over same-provenance 2-face edges.
        visited = set()
        queue = deque([seed])
        visited.add(seed)
        while queue:
            fi = queue.popleft()
            for fj in face_adj[fi]:
                if fj not in visited:
                    visited.add(fj)
                    queue.append(fj)

        kept_faces = FF[list(visited)]
        print(f"[bfs-clip] surface {s} ({surface_types[s]}): "
              f"{len(visited)}/{len(prov_faces)} faces after BFS")

        # Optional post-BFS distance filter: keep only faces whose barycenter
        # is within post_filter_threshold * spacings[s] of the nearest cluster point.
        thr = post_filter_threshold * spacings[s]
        barycenters = VV[kept_faces].mean(axis=1)
        bar_dists, _ = cluster_trees[s].query(barycenters)
        kept_faces = kept_faces[bar_dists <= thr]

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(VV)
        mesh.triangles = o3d.utility.Vector3iVector(kept_faces)
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(get_surface_color(surface_types[s]))
        clipped.append(mesh)

    return clipped
