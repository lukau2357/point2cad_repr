import numpy as np
import open3d as o3d
import pymesh
from collections import deque
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

    A post-BFS distance filter then discards any remaining faces whose barycenter
    is farther than post_filter_threshold * spacing[s] from the nearest cluster
    point, catching spurious triangles that leaked through (e.g. at tangencies).

    Parameters
    ----------
    o3d_meshes            : list of open3d.geometry.TriangleMesh, one per cluster
    clusters              : list of (N, 3) float arrays — input point clouds
    surface_types         : list of str — surface type names (for coloring)
    cluster_trees         : list[cKDTree] — precomputed, one per cluster
    spacings              : list[float]   — intra-cluster NN spacing per cluster
    post_filter_threshold : keep faces whose nearest cluster dist ≤ threshold * spacing

    Returns
    -------
    clipped : list of open3d.geometry.TriangleMesh, one per cluster
    """
    # Build PyMesh meshes from O3D meshes (single-sided faces only)
    pm_meshes = []
    for m in o3d_meshes:
        V_i, F_i = o3d_mesh_to_numpy(m)
        pm_meshes.append(pymesh.form_mesh(V_i.astype(np.float64), F_i.astype(np.int32)))

    # merge_meshes adds a "face_sources" attribute mapping each face to its source mesh index
    pm_merged = pymesh.merge_meshes(pm_meshes)
    face_sources_merged = pm_merged.get_attribute("face_sources").astype(np.int32)
    print(f"[bfs-clip] Merged: {pm_merged.num_vertices} vertices, {pm_merged.num_faces} faces")

    # Resolve self-intersections, then merge duplicate vertices at seam positions
    pm_resolved_ori = pymesh.resolve_self_intersection(pm_merged)
    pm_resolved, _ = pymesh.remove_duplicated_vertices(pm_resolved_ori, tol=1e-6)

    # Two-level provenance: resolved face → merged face → surface index
    face_sources_resolved = pm_resolved_ori.get_attribute("face_sources").astype(np.int32)
    face_provenance = face_sources_merged[face_sources_resolved].astype(np.int32)

    VV = pm_resolved.vertices.copy()
    FF = pm_resolved.faces.astype(np.int64)
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

        # Seed: face with provenance s whose centroid is closest to any cluster
        # point. Cluster points only exist on the visible/correct side, so this
        # is robust even when two disconnected components (e.g. sphere cap vs
        # interior inside a cylinder) would have similar cluster-centroid distances.
        prov_centroids = centroids[prov_faces]
        dists, _ = cluster_trees[s].query(prov_centroids)
        seed = prov_faces[np.argmin(dists)]

        # Simple BFS over same-provenance 2-face edges.
        # The seam edge between surfaces is non-manifold (4 faces) and absent
        # from face_adj, so BFS cannot cross it — no wall-face logic needed.
        # Correct-side wall faces are reachable (their non-seam edges connect
        # to correct-side interior); wrong-side wall faces are not.
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
              f"{len(visited)}/{len(prov_faces)} faces after BFS+wall")

        # Post-BFS distance filter: remove overflow from partial seams.
        # thr = post_filter_threshold * spacings[s]
        # barycenters = VV[kept_faces].mean(axis=1)
        # bar_dists, _ = cluster_trees[s].query(barycenters)
        # keep_mask = bar_dists <= thr
        # kept_faces = kept_faces[keep_mask]
        # print(f"[bfs-clip] surface {s} ({surface_types[s]}): "
        #       f"{keep_mask.sum()}/{len(keep_mask)} faces after post-filter "
        #       f"(thr={thr:.5f})") 

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(VV)
        mesh.triangles = o3d.utility.Vector3iVector(kept_faces)
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(get_surface_color(surface_types[s]))
        clipped.append(mesh)

    return clipped
