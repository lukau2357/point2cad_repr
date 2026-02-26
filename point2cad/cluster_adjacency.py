import numpy as np
from scipy.spatial import KDTree


def _reference_spacing(clusters):
    """Median nearest-neighbour distance across all cluster points."""
    nn_dists = []
    for cluster in clusters:
        tree = KDTree(cluster)
        d, _ = tree.query(cluster, k=2)
        nn_dists.append(d[:, 1])
    return float(np.median(np.concatenate(nn_dists)))


def _local_spacing(cluster):
    """Median nearest-neighbour distance within a single cluster."""
    tree = KDTree(cluster)
    d, _ = tree.query(cluster, k=2)
    return float(np.median(d[:, 1]))


def compute_adjacency_matrix(clusters, threshold_factor=1.5, spacing=None):
    """
    Compute the symmetric cluster adjacency matrix.

    Adjacency is determined per cluster pair using an adaptive threshold:

        threshold_ij = threshold_factor * max(local_spacing_i, local_spacing_j)

    where local_spacing_k is the median NN distance within cluster k alone.
    Two clusters are adjacent when the minimum NN distance from the smaller
    to the larger is within threshold_ij.  Using the per-cluster local spacing
    handles uneven sampling densities: sparse clusters get a proportionally
    larger threshold so narrow shared boundaries are not missed.

    Args:
        clusters:         list of np.ndarray (Ni, 3)
        threshold_factor: adjacency threshold = threshold_factor * local_spacing
        spacing:          if provided, used as the global reference spacing
                          returned for downstream consumers; does not affect
                          the per-pair adaptive detection logic.

    Returns:
        adj:             (n, n) bool array, symmetric, diagonal is False
        threshold:       global threshold (threshold_factor * global_spacing),
                         returned for backward compatibility
        spacing:         global reference spacing
        boundary_strips: dict (i, j) i<j -> (N, 3) float32 array, union of
                         boundary points from both clusters (points in smaller
                         within threshold_ij of larger, plus their NNs in larger).
                         Only populated for adjacent pairs.
    """
    n = len(clusters)

    if spacing is None:
        spacing = _reference_spacing(clusters)
    global_threshold = threshold_factor * spacing

    local_spacings = [_local_spacing(c) for c in clusters]

    adj = np.zeros((n, n), dtype=bool)
    boundary_strips = {}
    per_pair_thresholds = {}

    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = clusters[i], clusters[j]
            larger, smaller = (ci, cj) if len(ci) >= len(cj) else (cj, ci)

            tree = KDTree(larger)
            nn_dists, nn_idx = tree.query(smaller, k=1)

            threshold_ij = threshold_factor * max(local_spacings[i], local_spacings[j])
            if nn_dists.min() <= threshold_ij:
                adj[i, j] = adj[j, i] = True
                per_pair_thresholds[(i, j)] = threshold_ij

                mask = nn_dists <= threshold_ij
                strip_smaller = smaller[mask]
                strip_larger = larger[nn_idx[mask]]
                boundary_strips[(i, j)] = np.vstack(
                    [strip_smaller, strip_larger]
                ).astype(np.float32)

    return adj, global_threshold, spacing, boundary_strips, per_pair_thresholds

def build_cluster_proximity(clusters, chi2_threshold=14.16, reg=1e-6):
    """
    Build per-cluster Mahalanobis distance test for proximity queries.

    A point x is "close to cluster C" when its squared Mahalanobis distance
    is within the chi-squared threshold:

        (x - mu)^T  Sigma_inv  (x - mu)  <=  chi2_threshold

    The default chi2_threshold=14.16 corresponds to chi^2(3, 0.997) — the
    3-sigma equivalent for a 3D Gaussian.

    Parameters
    ----------
    clusters        : list of (Nk, 3) float arrays
    chi2_threshold  : float — chi-squared cutoff (default: 3-sigma in 3D)
    reg             : float — L2 regularisation for singular covariance

    Returns
    -------
    cluster_means     : list of (3,) arrays
    cluster_sigma_inv : list of (3, 3) arrays — regularised precision matrices
    chi2_threshold    : float — passed through for use in queries
    """
    means, sigma_invs = [], []
    for cluster in clusters:
        mu = cluster.mean(axis=0)
        cov = np.cov(cluster, rowvar=False)
        cov += reg * np.eye(3)
        sigma_invs.append(np.linalg.inv(cov))
        means.append(mu)
    return means, sigma_invs, chi2_threshold


def build_cluster_bboxes(clusters, margin_percentile=50.0):
    """
    Compute an axis-aligned bounding box for each cluster, extended by a margin.

    The margin is the *margin_percentile*-th percentile of intra-cluster
    nearest-neighbour distances (default: median, i.e. 50th percentile).
    This accounts for the finite sampling gap at cluster boundaries so that
    genuine corner vertices — which lie at the extreme edge of each
    constituent surface — are not falsely excluded.

    Parameters
    ----------
    clusters          : list of (Nk, 3) float arrays
    margin_percentile : float in [0, 100]

    Returns
    -------
    list of (lo, hi) pairs — (3,) float64 arrays for the min / max corners
    of the extended bounding box, one pair per cluster.
    """
    bboxes = []
    for cluster in clusters:
        tree = KDTree(cluster)
        d, _ = tree.query(cluster, k=2)
        margin = float(np.percentile(d[:, 1], margin_percentile))
        lo = cluster.min(axis=0).astype(np.float64) - margin
        hi = cluster.max(axis=0).astype(np.float64) + margin
        bboxes.append((lo, hi))
    return bboxes


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