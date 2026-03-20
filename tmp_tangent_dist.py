"""
Prototype: for a given model, find adjacent pairs with empty mesh
intersections and plot the distribution of nearest-neighbour distances
between the two meshes' vertices.

Usage (inside Docker):
    python tmp_tangent_dist.py --model_id 00000949 --input_dir sample_clouds
"""

import argparse
import glob as _glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

import torch

from point2cad.cluster_adjacency import (
    compute_adjacency_matrix, build_cluster_proximity, adjacency_pairs,
)
from point2cad.surface_fitter import fit_surface
from point2cad.surface_types import SURFACE_NAMES
from point2cad.mesh_intersections import compute_mesh_intersections
import point2cad.primitive_fitting_utils as pfu

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def normalize_points(pts):
    mean = np.mean(pts, axis=0)
    pts = pts - mean
    S, U = np.linalg.eigh(pts.T @ pts)
    R = pfu.rotation_matrix_a_to_b(U[:, np.argmin(S)], np.array([1, 0, 0]))
    pts = (R @ pts.T).T
    extents = np.max(pts, axis=0) - np.min(pts, axis=0)
    scale = float(np.max(extents) + 1e-7)
    return (pts / scale).astype(np.float32), mean, R, scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--input_dir", type=str, default="sample_clouds")
    parser.add_argument("--spacing_factor", type=float, default=3.0)
    parser.add_argument("--spacing_percentile", type=float, default=5.0)
    args = parser.parse_args()

    np_rng = np.random.default_rng(42)

    # Load part files
    input_pattern = os.path.join(args.input_dir, args.model_id, "*.xyzc")
    part_files = sorted(_glob.glob(input_pattern),
                        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    if not part_files:
        print(f"No part files found: {input_pattern}")
        return

    for part_idx, sample_path in enumerate(part_files):
        print(f"\n{'='*60}")
        print(f"Part {part_idx}: {os.path.basename(sample_path)}")
        print(f"{'='*60}")

        data = np.loadtxt(sample_path)
        data[:, :3], mean, R, scale = normalize_points(data[:, :3])
        unique_clusters = np.unique(data[:, -1].astype(int))

        clusters = []
        for cid in unique_clusters:
            cluster = data[data[:, -1].astype(int) == cid, :3].astype(np.float32)
            clusters.append(cluster)

        cluster_trees, cluster_nn_percentiles = build_cluster_proximity(
            clusters, percentile=args.spacing_percentile
        )

        adj, _, spacing, _, _, _ = compute_adjacency_matrix(
            clusters, threshold_factor=args.spacing_factor,
            spacing_percentile=args.spacing_percentile,
            local_spacings=cluster_nn_percentiles,
        )
        pairs = adjacency_pairs(adj)
        print(f"Adjacent pairs: {pairs}")

        # Fit surfaces
        surface_ids, fit_results, fit_meshes = [], [], []
        for idx, cluster in enumerate(clusters):
            res = fit_surface(
                cluster,
                {"hidden_dim": 64, "use_shortcut": True, "fraction_siren": 0.5},
                np_rng, DEVICE,
            )
            surface_ids.append(res["surface_id"])
            fit_results.append(res["result"])
            fit_meshes.append(res["mesh"])
            print(f"  cluster {idx}: {SURFACE_NAMES[res['surface_id']]}  "
                  f"error={res['result']['error']:.6f}")

        # Compute mesh intersections
        raw_intersections, polyline_map = compute_mesh_intersections(
            adj, surface_ids, fit_results, fit_meshes,
        )

        # Find pairs with empty intersections
        empty_pairs = []
        for (i, j), result in raw_intersections.items():
            if len(result["curves"]) == 0:
                empty_pairs.append((i, j))

        if not empty_pairs:
            print("No empty intersections — all adjacent pairs have curves.")
            continue

        print(f"\nEmpty intersection pairs: {empty_pairs}")

        # Plot NN distance distributions
        n_plots = len(empty_pairs)
        fig, axes = plt.subplots(n_plots, 2, figsize=(12, 4 * n_plots),
                                 squeeze=False)

        for row, (i, j) in enumerate(empty_pairs):
            verts_i = np.asarray(fit_meshes[i].vertices)
            verts_j = np.asarray(fit_meshes[j].vertices)

            tree_j = cKDTree(verts_j)
            tree_i = cKDTree(verts_i)

            dists_ij, _ = tree_j.query(verts_i, k=1)
            dists_ji, _ = tree_i.query(verts_j, k=1)

            si = SURFACE_NAMES[surface_ids[i]]
            sj = SURFACE_NAMES[surface_ids[j]]

            ax_left = axes[row, 0]
            ax_left.hist(dists_ij, bins=100, color="steelblue", edgecolor="none")
            ax_left.set_title(f"({i},{j}) {si}→{sj}: NN dist mesh {i} → mesh {j}")
            ax_left.set_xlabel("NN distance")
            ax_left.set_ylabel("count")
            ax_left.axvline(np.percentile(dists_ij, 5), color="red", ls="--",
                            label=f"p5={np.percentile(dists_ij, 5):.4f}")
            ax_left.axvline(np.percentile(dists_ij, 1), color="orange", ls="--",
                            label=f"p1={np.percentile(dists_ij, 1):.4f}")
            ax_left.legend(fontsize=8)

            ax_right = axes[row, 1]
            ax_right.hist(dists_ji, bins=100, color="coral", edgecolor="none")
            ax_right.set_title(f"({i},{j}) {sj}→{si}: NN dist mesh {j} → mesh {i}")
            ax_right.set_xlabel("NN distance")
            ax_right.set_ylabel("count")
            ax_right.axvline(np.percentile(dists_ji, 5), color="red", ls="--",
                            label=f"p5={np.percentile(dists_ji, 5):.4f}")
            ax_right.axvline(np.percentile(dists_ji, 1), color="orange", ls="--",
                            label=f"p1={np.percentile(dists_ji, 1):.4f}")
            ax_right.legend(fontsize=8)

            print(f"\n({i},{j}) {si} ∩ {sj}:")
            print(f"  mesh_i: {len(verts_i)} verts,  mesh_j: {len(verts_j)} verts")
            print(f"  i→j: min={dists_ij.min():.6f}  p1={np.percentile(dists_ij, 1):.6f}"
                  f"  p5={np.percentile(dists_ij, 5):.6f}  median={np.median(dists_ij):.6f}")
            print(f"  j→i: min={dists_ji.min():.6f}  p1={np.percentile(dists_ji, 1):.6f}"
                  f"  p5={np.percentile(dists_ji, 5):.6f}  median={np.median(dists_ji):.6f}")

            sorted_ij = np.sort(dists_ij)[:20]
            sorted_ji = np.sort(dists_ji)[:20]
            print(f"  i→j smallest 20: {np.array2string(sorted_ij, precision=5)}")
            print(f"  j→i smallest 20: {np.array2string(sorted_ji, precision=5)}")

        plt.tight_layout()
        out_path = f"tangent_nn_dists_{args.model_id}_part{part_idx}.png"
        plt.savefig(out_path, dpi=150)
        print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
