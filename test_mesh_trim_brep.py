#!/usr/bin/env python3
"""
Prototype: Mesh-boundary trimmed BRep.

Uses the resolved mesh boundary as trim curves on fitted parametric surfaces.
After CGAL self-intersection resolution, each surface's mesh region has clean
boundary edges. These are projected to UV, fit with a B-spline, and used as
the trim wire. Only the longest boundary loop per face is used (no holes).

Usage:
    python test_mesh_trim_brep.py <xyzc_file> [--output OUTPUT]
"""

import argparse
import math
import os
import sys
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Mesh-boundary trimmed BRep")
    parser.add_argument("xyzc", help="Path to .xyzc file")
    parser.add_argument("--output", type=str, default="mesh_trim.step",
                        help="Output STEP file path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ---- Load point cloud ----
    data = np.loadtxt(args.xyzc)
    points = data[:, :3]
    labels = data[:, 3].astype(int)

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]

    clusters = []
    for lbl in unique_labels:
        mask = labels == lbl
        clusters.append(points[mask])
    print(f"Loaded {len(points)} points, {len(clusters)} clusters")

    # ---- Normalize ----
    import point2cad.primitive_fitting_utils as pfu

    def normalize_points(pts):
        mean = np.mean(pts, axis=0)
        pts = pts - mean
        S, U = np.linalg.eigh(pts.T @ pts)
        R = pfu.rotation_matrix_a_to_b(U[:, np.argmin(S)], np.array([1, 0, 0]))
        pts = (R @ pts.T).T
        extents = np.max(pts, axis=0) - np.min(pts, axis=0)
        scale = float(np.max(extents) + 1e-7)
        return (pts / scale).astype(np.float32), mean, R, scale

    all_pts = np.vstack(clusters)
    _, mean, R, scale = normalize_points(all_pts)
    clusters_norm = []
    for c in clusters:
        c_centered = c - mean
        c_rotated = (R @ c_centered.T).T
        c_scaled = c_rotated / scale
        clusters_norm.append(c_scaled.astype(np.float32))

    # ---- Fit surfaces ----
    from point2cad.surface_fitter import fit_surface
    from point2cad.occ_surfaces import to_occ_surface
    from point2cad.surface_types import SURFACE_NAMES

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np_rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    occ_surfaces = []
    surface_ids = []
    fit_results = []
    fit_meshes = []

    for idx, cluster in enumerate(clusters_norm):
        print(f"[fit] Cluster {idx} ({len(cluster)} pts) ...")

        from scipy.spatial import cKDTree
        tree = cKDTree(cluster)
        dists, _ = tree.query(cluster, k=2)
        spacing = float(np.percentile(dists[:, 1], 50))

        mesh_kw = {"spacing": spacing, "threshold_multiplier": 5}
        plane_kw = {**mesh_kw, "mesh_dim": 100, "plane_sampling_deviation": 0.5}
        sphere_kw = {**mesh_kw, "dim_theta": 100, "dim_lambda": 100}
        cylinder_kw = {**mesh_kw, "dim_theta": 100, "dim_height": 50,
                       "cylinder_height_margin": 0.5}
        cone_kw = {**mesh_kw, "dim_theta": 100, "dim_height": 100,
                   "cone_height_margin": 0.5}

        res = fit_surface(
            cluster,
            {"hidden_dim": 64, "use_shortcut": True, "fraction_siren": 0.5},
            np_rng, DEVICE,
            inr_fit_kwargs={"max_steps": 1500, "noise_magnitude_3d": 0.05,
                            "noise_magnitude_uv": 0.05, "initial_lr": 1e-1},
            inr_mesh_kwargs={"mesh_dim": 200, "uv_margin": 0.2,
                             "threshold_multiplier": 1.5},
            plane_mesh_kwargs=plane_kw,
            sphere_mesh_kwargs=sphere_kw,
            cylinder_mesh_kwargs=cylinder_kw,
            cone_mesh_kwargs=cone_kw,
        )
        sid = res["surface_id"]
        surface_ids.append(sid)
        fit_results.append(res["result"])
        fit_meshes.append(res["mesh"])
        occ_surf = to_occ_surface(sid, res["result"], cluster=cluster,
                                   uv_margin=0.05, grid_resolution=50)
        occ_surfaces.append(occ_surf)

        err = res["result"]["error"]
        print(f"[fit] Cluster {idx} → {SURFACE_NAMES[sid]}  residual={err:.6f}")

    # ---- Resolve mesh self-intersections ----
    from point2cad.msi_extraction import _merge_meshes
    from collections import defaultdict

    print("\n[mesh] Merging meshes and resolving self-intersections ...")
    mesh_list = [(np.asarray(m.vertices), np.asarray(m.triangles)) for m in fit_meshes]
    V, F, face_sources = _merge_meshes(mesh_list)
    print(f"[mesh] Merged: {len(V)} vertices, {len(F)} faces")

    from igl.copyleft.cgal import remesh_self_intersections
    SV, SF, IF, J, IM = remesh_self_intersections(V, F)
    print(f"[mesh] Resolved: {len(SV)} vertices, {len(SF)} faces")

    # Deduplicate vertices
    SF = IM[SF]
    used_verts, inv = np.unique(SF.ravel(), return_inverse=True)
    SF = inv.reshape(-1, 3)
    SV = SV[used_verts]

    # Face provenance
    resolved_face_sources = face_sources[J]
    print(f"[mesh] After dedup: {len(SV)} vertices, {len(SF)} faces")

    # ---- Extract per-surface boundary edges ----
    # An edge is a boundary of surface S if it is shared by a face of S
    # and a face of a different surface (or is a manifold boundary edge).
    print("\n[boundary] Extracting per-surface boundary edges ...")

    edge_to_faces = defaultdict(list)
    for fi in range(len(SF)):
        f = SF[fi]
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            e = (min(int(f[a]), int(f[b])), max(int(f[a]), int(f[b])))
            edge_to_faces[e].append(fi)

    # Collect boundary edges per surface
    surface_boundary_edges = defaultdict(list)
    for edge, face_list in edge_to_faces.items():
        surfaces = set(int(resolved_face_sources[fi]) for fi in face_list)
        if len(surfaces) >= 2:
            # Cross-surface edge — boundary for all involved surfaces
            for s in surfaces:
                surface_boundary_edges[s].append(edge)
        elif len(face_list) == 1:
            # Manifold boundary (edge of the mesh)
            s = int(resolved_face_sources[face_list[0]])
            surface_boundary_edges[s].append(edge)

    for s in sorted(surface_boundary_edges.keys()):
        print(f"  surface {s}: {len(surface_boundary_edges[s])} boundary edges")

    # ---- Chain boundary edges into loops per surface ----
    def chain_into_loops(edges, vertices):
        """Chain edges into closed loops. Returns list of (K,3) arrays."""
        if not edges:
            return []

        graph = defaultdict(set)
        for v0, v1 in edges:
            graph[v0].add(v1)
            graph[v1].add(v0)

        visited_edges = set()
        loops = []

        def edge_key(a, b):
            return (min(a, b), max(a, b))

        # Try to trace closed loops
        all_verts = list(graph.keys())
        for start in all_verts:
            for neighbor in list(graph[start]):
                ek = edge_key(start, neighbor)
                if ek in visited_edges:
                    continue

                chain = [start]
                current = start
                next_v = neighbor

                while True:
                    ek = edge_key(current, next_v)
                    if ek in visited_edges:
                        break
                    visited_edges.add(ek)
                    chain.append(next_v)
                    current = next_v

                    # Check if we closed the loop
                    if current == start:
                        break

                    neighbors = graph[current] - {chain[-2]}
                    if not neighbors:
                        break
                    # Pick neighbor — prefer one that closes the loop
                    if start in neighbors:
                        next_v = start
                    else:
                        next_v = min(neighbors)  # deterministic

                if len(chain) >= 4 and chain[-1] == chain[0]:
                    # Closed loop (remove duplicate start/end)
                    loop_pts = vertices[chain[:-1]]
                    loops.append(loop_pts)

        return loops

    # ---- Build faces from boundary loops ----
    from OCC.Core.gp import gp_Pnt, gp_Pnt2d
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.TopoDS import TopoDS_Compound
    from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeFace,
                                          BRepBuilderAPI_MakeEdge,
                                          BRepBuilderAPI_MakeWire,
                                          BRepBuilderAPI_Sewing)
    from OCC.Core.BRepLib import breplib
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCC.Core.Geom2dAPI import Geom2dAPI_Interpolate
    from OCC.Core.TColgp import TColgp_HArray1OfPnt2d
    from OCC.Core.ShapeFix import ShapeFix_Shape
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from point2cad.topology import export_step, apply_inverse_normalization

    print("\n[brep] Building faces ...")

    occ_faces = []
    n_surfaces = len(occ_surfaces)

    for si in range(n_surfaces):
        surface = occ_surfaces[si]
        if surface is None:
            continue

        edges = surface_boundary_edges.get(si, [])
        if not edges:
            print(f"  face {si} ({SURFACE_NAMES[surface_ids[si]]}): no boundary edges — skipping")
            continue

        loops = chain_into_loops(edges, SV)
        if not loops:
            print(f"  face {si} ({SURFACE_NAMES[surface_ids[si]]}): no closed loops — skipping")
            continue

        # Use longest loop only
        longest = max(loops, key=len)
        print(f"  face {si} ({SURFACE_NAMES[surface_ids[si]]}): "
              f"{len(loops)} loops, using longest ({len(longest)} pts)")

        # Subsample if too many points (>100)
        max_boundary_pts = 100
        if len(longest) > max_boundary_pts:
            step = len(longest) / max_boundary_pts
            indices = [int(round(i * step)) for i in range(max_boundary_pts)]
            indices = [min(idx, len(longest) - 1) for idx in indices]
            longest = longest[indices]
            print(f"    subsampled to {len(longest)} pts")

        # Project 3D boundary points to UV
        uv_pts = []
        for pt in longest:
            pnt = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
            try:
                proj = GeomAPI_ProjectPointOnSurf(pnt, surface)
                if proj.NbPoints() > 0:
                    u, v = proj.LowerDistanceParameters()
                    uv_pts.append([u, v])
            except Exception:
                pass

        if len(uv_pts) < 3:
            print(f"  face {si}: UV projection failed ({len(uv_pts)} pts)")
            continue

        uv_arr = np.array(uv_pts, dtype=np.float64)
        print(f"    UV range: u=[{uv_arr[:,0].min():.4f}, {uv_arr[:,0].max():.4f}] "
              f"v=[{uv_arr[:,1].min():.4f}, {uv_arr[:,1].max():.4f}]")

        # Simplify UV boundary with Douglas-Peucker
        from point2cad.topology import _douglas_peucker
        uv_arr = _douglas_peucker(uv_arr, epsilon=0.01)
        print(f"    after DP simplification: {len(uv_arr)} pts")

        if len(uv_arr) < 3:
            print(f"  face {si}: too few points after simplification")
            continue

        # Fit closed B-spline through UV boundary
        try:
            n_pts = len(uv_arr)
            arr = TColgp_HArray1OfPnt2d(1, n_pts)
            for k in range(n_pts):
                arr.SetValue(k + 1, gp_Pnt2d(float(uv_arr[k, 0]),
                                               float(uv_arr[k, 1])))
            interp = Geom2dAPI_Interpolate(arr, True, 1e-6)  # True = periodic
            interp.Perform()
            if not interp.IsDone():
                print(f"  face {si}: B-spline interpolation failed")
                continue
            curve2d = interp.Curve()
        except Exception as exc:
            print(f"  face {si}: B-spline error: {exc}")
            continue

        # Build edge and wire
        try:
            edge = BRepBuilderAPI_MakeEdge(curve2d, surface).Edge()
            breplib.BuildCurves3d(edge)

            wire_builder = BRepBuilderAPI_MakeWire()
            wire_builder.Add(edge)
            if not wire_builder.IsDone():
                print(f"  face {si}: wire construction failed")
                continue
            wire = wire_builder.Wire()

            face_maker = BRepBuilderAPI_MakeFace(surface, wire)
            if not face_maker.IsDone():
                print(f"  face {si}: MakeFace failed (error {face_maker.Error()})")
                continue

            occ_faces.append(face_maker.Face())
            print(f"  face {si}: OK")

        except Exception as exc:
            print(f"  face {si}: {exc}")

    if not occ_faces:
        print("[brep] No faces built — aborting")
        sys.exit(1)

    # ---- Sew faces ----
    print(f"\n[brep] Sewing {len(occ_faces)} faces ...")
    sewing = BRepBuilderAPI_Sewing(1e-3)
    for face in occ_faces:
        sewing.Add(face)
    sewing.Perform()
    shape = sewing.SewedShape()

    if shape is None or shape.IsNull():
        print("[brep] Sewing produced no shape")
        sys.exit(1)

    # Heal
    try:
        breplib.BuildCurves3d(shape)
    except Exception:
        pass
    fixer = ShapeFix_Shape(shape)
    fixer.SetPrecision(1e-3)
    fixer.Perform()
    shape = fixer.Shape()

    # Stats
    n_output = 0
    fexp = TopExp_Explorer(shape, TopAbs_FACE)
    while fexp.More():
        n_output += 1
        fexp.Next()

    analyzer = BRepCheck_Analyzer(shape)
    valid = analyzer.IsValid()
    print(f"[brep] {n_output} faces, valid={valid}")

    # Apply inverse normalization
    shape = apply_inverse_normalization(shape, mean, R, scale)

    # Export
    export_step(shape, args.output)


if __name__ == "__main__":
    main()
