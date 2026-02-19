import numpy as np
from scipy.spatial import KDTree


def _reference_spacing(clusters):
    """Median nearest-neighbour distance across all cluster points."""
    nn_dists = []
    for cluster in clusters:
        tree = KDTree(cluster)
        d, _ = tree.query(cluster, k = 2)  # k=2: skip self-match
        # if for point x in cluster C y is the closest, then for point y x is the closest point in C
        # if distance(x, y) = l, l will be repeated in the resuling array two times. Could perhaps bias median computation?
        nn_dists.append(d[:, 1])
    return float(np.median(np.concatenate(nn_dists)))


def compute_adjacency_matrix(clusters, percentile = 98, threshold_factor = 1.5, spacing = None):
    """
    Compute the symmetric cluster adjacency matrix.

    For each ordered pair (i, j) with i < j, build a KDTree from the larger
    cluster and query every point in the smaller cluster. The representative
    distance is the (100 - percentile)th percentile of those NN distances
    (e.g. percentile=98 → 2nd percentile = robust minimum). Clusters are
    declared adjacent when that distance is within threshold_factor * spacing.

    Args:
        clusters:         list of np.ndarray (Ni, 3)
        percentile:       configurable robustness knob, e.g. 98 or 99
        threshold_factor: adjacency threshold = threshold_factor * spacing
        spacing:          reference point spacing; computed from clusters if None

    Returns:
        adj:             (n, n) bool array, symmetric, diagonal is False
        threshold:       distance threshold used
        spacing:         reference spacing used
        boundary_strips: dict (i, j) i<j -> (N, 3) float32 array, union of
                         boundary points from both clusters (points in smaller
                         within threshold of larger, plus their NNs in larger).
                         Only populated for adjacent pairs.
    """
    n = len(clusters)

    if spacing is None:
        spacing = _reference_spacing(clusters)
    threshold = threshold_factor * spacing

    low_pct = 100 - percentile  # e.g. 98 → 2nd percentile of NN distances
    adj = np.zeros((n, n), dtype = bool)
    boundary_strips = {}

    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = clusters[i], clusters[j]
            larger, smaller = (ci, cj) if len(ci) >= len(cj) else (cj, ci)

            tree = KDTree(larger)
            nn_dists, nn_idx = tree.query(smaller, k = 1)
            close_dist = np.percentile(nn_dists, low_pct)

            if close_dist <= threshold:
                adj[i, j] = adj[j, i] = True

                mask         = nn_dists <= threshold
                strip_smaller = smaller[mask]
                strip_larger  = larger[nn_idx[mask]]
                boundary_strips[(i, j)] = np.vstack(
                    [strip_smaller, strip_larger]
                ).astype(np.float32)

    return adj, threshold, spacing, boundary_strips

def adjacency_pairs(adj):
    """Return list of (i, j) pairs with i < j where adj[i, j] is True."""
    n = adj.shape[0]
    return [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j]]

def adjacency_triangles(adj):
    """Return list of (i, j, k) triples with i < j < k where all three pairs are adjacent."""
    n = adj.shape[0]
    triangles = []
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i, j]:
                continue
            for k in range(j + 1, n):
                if adj[i, k] and adj[j, k]:
                    triangles.append((i, j, k))
    return triangles

if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import torch
    import point2cad.primitive_fitting_utils as pfu
    from point2cad.surface_fitter import fit_surface, SURFACE_NAMES

    SAMPLE = os.path.join(os.path.dirname(__file__), "..", "sample_clouds", "abc_00470.xyzc")
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    def normalize_points(points):
        points = points - np.mean(points, axis = 0, keepdims = True)
        S, U = np.linalg.eig(points.T @ points)
        smallest_ev = U[:, np.argmin(S)]
        R = pfu.rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
        points = (R @ points.T).T
        extents = np.max(points, axis = 0) - np.min(points, axis = 0)
        return (points / (np.max(extents) + 1e-7)).astype(np.float32)

    data = np.loadtxt(SAMPLE)
    data[:, :3] = normalize_points(data[:, :3])
    unique_clusters = np.unique(data[:, -1].astype(int))
    clusters = [data[data[:, -1].astype(int) == cid, :3].astype(np.float32) for cid in unique_clusters]
    cluster_sizes = [int((data[:, -1] == cid).sum()) for cid in unique_clusters]
    adj, threshold, spacing, boundary_strips = compute_adjacency_matrix(clusters)

    print(f"Reference spacing: {spacing:.6f}")
    print(f"Adjacency threshold: {threshold:.6f}")
    print(f"Adjacency matrix:\n{adj.astype(int)}")
    for i, j in adjacency_pairs(adj):
        print(f"  Clusters {unique_clusters[i]} and {unique_clusters[j]} are adjacent")
        print(f"  Cluster sizes: {cluster_sizes[i]} {cluster_sizes[j]}")