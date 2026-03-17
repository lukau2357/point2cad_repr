"""
B-Rep reconstruction via mesh boundary extraction + B-spline fitting.

Instead of computing analytical intersection curves, detecting vertices, and
assembling wires, this pipeline:
  1. Fits analytical surfaces to each cluster (reuses brep_pipeline logic)
  2. Builds a trimesh mesh per cluster
  3. Extracts the boundary polygon of each mesh
  4. Projects boundary points onto the fitted analytical surface
  5. Simplifies the boundary (Douglas-Peucker)
  6. Fits a periodic B-spline through the projected boundary
  7. Builds an OCC face: MakeFace(surface, wire)
  8. Sews all faces into a shell
  9. Exports STEP

Usage (inside Docker):
  python brep_boundary.py --model_id 00000078 --input_dir sample_clouds --output_dir output_boundary
"""

import argparse
import math
import os
import shutil
import sys
import glob as _glob

import numpy as np
import open3d as o3d
import trimesh

from point2cad.surface_types import (
    SURFACE_PLANE, SURFACE_SPHERE, SURFACE_CYLINDER, SURFACE_CONE, SURFACE_INR,
    SURFACE_NAMES,
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _denorm(pts, mean, R, scale):
    """Inverse of normalize_points for an (N, 3) float array."""
    pts = np.asarray(pts, dtype=np.float64)
    return (scale * (pts @ R) + mean).astype(np.float32)


def apply_inverse_normalization(shape, mean, R, scale):
    """Undo per-part normalization using gp_Trsf (not gp_GTrsf).

    Inverse of normalize_points:
        pts_orig = scale * R^T @ pts_norm + mean
             = translate(mean) . rotate(R^T) . scale(s) . pts_norm

    Uses gp_Trsf which preserves analytical surface types (plane, cylinder,
    cone, sphere).  gp_GTrsf forces conversion to B-spline, which crashes
    on infinite underlying surfaces from MakeFace(surface, wire).

    The transform is decomposed into three elementary gp_Trsf operations
    (scale, rotation, translation) so OCC correctly classifies each one.
    Passing the combined matrix via SetValues can mis-classify the form
    and corrupt curved surfaces.
    """
    from OCC.Core.gp import gp_Trsf, gp_Pnt, gp_Vec
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform

    if shape is None or shape.IsNull():
        return shape

    RT = np.asarray(R, dtype=np.float64).T
    mean = np.asarray(mean, dtype=np.float64)

    # 1. Uniform scale around origin
    trsf_scale = gp_Trsf()
    trsf_scale.SetScale(gp_Pnt(0, 0, 0), float(scale))

    # 2. Rotation (R^T is orthogonal → OCC classifies as gp_Rotation)
    trsf_rot = gp_Trsf()
    trsf_rot.SetValues(
        float(RT[0, 0]), float(RT[0, 1]), float(RT[0, 2]), 0.0,
        float(RT[1, 0]), float(RT[1, 1]), float(RT[1, 2]), 0.0,
        float(RT[2, 0]), float(RT[2, 1]), float(RT[2, 2]), 0.0,
    )

    # 3. Translation by mean
    trsf_trans = gp_Trsf()
    trsf_trans.SetTranslation(gp_Vec(float(mean[0]), float(mean[1]),
                                      float(mean[2])))

    # Compose: translate . rotate . scale
    trsf = trsf_trans.Multiplied(trsf_rot.Multiplied(trsf_scale))

    result = BRepBuilderAPI_Transform(shape, trsf, True)
    if not result.IsDone():
        print("[brep] apply_inverse_normalization failed, returning as-is")
        return shape
    return result.Shape()


def _douglas_peucker(pts, epsilon):
    """Simplify a polyline (N, 3) using the Douglas-Peucker algorithm.
    Returns the simplified polyline preserving start and end points.
    """
    if len(pts) <= 2:
        return pts

    # Find point with max distance from the line start->end
    start, end = pts[0], pts[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-15:
        # Degenerate: all points essentially the same
        dists = np.linalg.norm(pts - start, axis=1)
        idx_max = np.argmax(dists)
        if dists[idx_max] < epsilon:
            return pts[[0, -1]]
    else:
        line_dir = line_vec / line_len
        vecs = pts - start
        proj = np.outer(vecs @ line_dir, line_dir)
        perp = vecs - proj
        dists = np.linalg.norm(perp, axis=1)
        idx_max = np.argmax(dists)

    if dists[idx_max] > epsilon:
        left = _douglas_peucker(pts[:idx_max + 1], epsilon)
        right = _douglas_peucker(pts[idx_max:], epsilon)
        return np.vstack([left[:-1], right])
    else:
        return pts[[0, -1]]


# ---------------------------------------------------------------------------
# Fine mesh trimming
# ---------------------------------------------------------------------------

def fine_trim_mesh(tmesh, cluster, spacing, threshold_multiplier=1.5):
    """Trim a mesh to closely follow the cluster boundary.

    Unlike grid_trimming (which keeps/removes entire grid cells), this operates
    at the vertex level: each mesh vertex gets a distance to the nearest cluster
    point.  Triangles fully outside the threshold are removed.  Boundary
    triangles (some vertices in, some out) are clipped by linearly interpolating
    new vertices along the edges at the threshold crossing.

    Parameters
    ----------
    tmesh : trimesh.Trimesh
    cluster : (N, 3) array of cluster points
    spacing : float — max NN distance for this cluster (from build_cluster_proximity)
    threshold_multiplier : float — threshold = spacing * threshold_multiplier

    Returns a new trimesh.Trimesh with clean boundaries.
    """
    from scipy.spatial import cKDTree

    cluster = np.asarray(cluster, dtype=np.float64)
    verts = np.asarray(tmesh.vertices, dtype=np.float64)
    faces = np.asarray(tmesh.faces)

    tree = cKDTree(cluster)
    dists, _ = tree.query(verts, k=1)

    threshold = threshold_multiplier * spacing
    inside = dists <= threshold  # per-vertex boolean

    new_verts = list(verts)
    new_faces = []
    edge_cache = {}  # (v_in, v_out) -> new_vertex_idx

    def _interp_vertex(vi_in, vi_out):
        """Interpolate a new vertex on the edge vi_in->vi_out at the threshold."""
        key = (min(vi_in, vi_out), max(vi_in, vi_out))
        if key in edge_cache:
            return edge_cache[key]

        d_in = dists[vi_in]
        d_out = dists[vi_out]
        denom = d_out - d_in
        if abs(denom) < 1e-15:
            t = 0.5
        else:
            t = (threshold - d_in) / denom
        t = np.clip(t, 0.01, 0.99)

        new_pt = (1 - t) * verts[vi_in] + t * verts[vi_out]
        new_idx = len(new_verts)
        new_verts.append(new_pt)
        edge_cache[key] = new_idx
        return new_idx

    for face in faces:
        v0, v1, v2 = face
        in0, in1, in2 = inside[v0], inside[v1], inside[v2]
        n_in = int(in0) + int(in1) + int(in2)

        if n_in == 3:
            new_faces.append([v0, v1, v2])
        elif n_in == 0:
            continue
        elif n_in == 2:
            # Two inside, one outside → clip to quad (2 triangles)
            if not in0:
                v_out, va, vb = v0, v1, v2
            elif not in1:
                v_out, va, vb = v1, v2, v0
            else:
                v_out, va, vb = v2, v0, v1

            na = _interp_vertex(va, v_out)
            nb = _interp_vertex(vb, v_out)
            new_faces.append([va, na, vb])
            new_faces.append([na, nb, vb])
        else:
            # One inside, two outside → clip to single triangle
            if in0:
                v_in, va, vb = v0, v1, v2
            elif in1:
                v_in, va, vb = v1, v2, v0
            else:
                v_in, va, vb = v2, v0, v1

            na = _interp_vertex(v_in, va)
            nb = _interp_vertex(v_in, vb)
            new_faces.append([v_in, na, nb])

    if not new_faces:
        return tmesh

    new_mesh = trimesh.Trimesh(
        vertices=np.array(new_verts),
        faces=np.array(new_faces),
        process=False,
    )
    new_mesh.remove_unreferenced_vertices()
    return new_mesh


# ---------------------------------------------------------------------------
# Mesh boundary extraction
# ---------------------------------------------------------------------------

def extract_mesh_boundary(tmesh):
    """Extract ordered boundary polygon(s) from a trimesh.Trimesh.

    Returns a list of (M, 3) arrays, each an ordered closed boundary loop
    (first point == last point).
    """
    # Boundary edges: edges that belong to exactly one face
    edges = tmesh.edges_sorted
    # Count how many faces each edge belongs to
    from collections import Counter
    edge_tuples = [tuple(e) for e in edges]
    edge_counts = Counter(edge_tuples)

    boundary_edges = set()
    for edge, count in edge_counts.items():
        if count == 1:
            boundary_edges.add(edge)

    if not boundary_edges:
        return []

    # Build adjacency for boundary vertices
    from collections import defaultdict
    adj = defaultdict(list)
    for (v0, v1) in boundary_edges:
        adj[v0].append(v1)
        adj[v1].append(v0)

    # Traverse to form ordered loops
    visited_edges = set()
    loops = []

    for start_edge in boundary_edges:
        v0, v1 = start_edge
        if (v0, v1) in visited_edges or (v1, v0) in visited_edges:
            continue

        # Start a new loop from v0
        loop = [v0]
        prev = -1
        current = v0
        while True:
            neighbors = adj[current]
            # Pick next unvisited neighbor
            next_v = None
            for n in neighbors:
                edge_key = (min(current, n), max(current, n))
                if n != prev and edge_key not in visited_edges:
                    next_v = n
                    break
            if next_v is None:
                break

            edge_key = (min(current, next_v), max(current, next_v))
            visited_edges.add(edge_key)
            loop.append(next_v)
            prev = current
            current = next_v

            if current == v0:
                break

        if len(loop) >= 3:
            loop_pts = tmesh.vertices[loop]
            loops.append(loop_pts)

    return loops


# ---------------------------------------------------------------------------
# OCC face construction
# ---------------------------------------------------------------------------

def _laplacian_smooth(pts, iterations=3, alpha=0.5):
    """Laplacian smoothing on a closed polyline (N, 3).
    Each vertex moves toward the average of its two neighbors.
    """
    pts = pts.copy()
    n = len(pts)
    for _ in range(iterations):
        new_pts = pts.copy()
        for i in range(n):
            prev_pt = pts[(i - 1) % n]
            next_pt = pts[(i + 1) % n]
            new_pts[i] = (1 - alpha) * pts[i] + alpha * 0.5 * (prev_pt + next_pt)
        pts = new_pts
    return pts


def _boundary_to_wire(occ_surface, boundary_pts,
                      dp_epsilon=0.005, n_bspline_pts=80,
                      smooth_iterations=5):
    """Process a boundary polygon into an OCC wire on the given surface.

    Steps:
      1. Project boundary points onto the OCC surface
      2. Laplacian smoothing (removes zig-zag from mesh boundaries)
      3. Simplify with Douglas-Peucker
      4. Subsample to n_bspline_pts if still too dense
      5. Fit periodic B-spline
      6. MakeEdge → MakeWire

    Returns (wire, projected_boundary) or (None, None) on failure.
    """
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.GeomAPI import (
        GeomAPI_ProjectPointOnSurf,
        GeomAPI_Interpolate,
    )
    from OCC.Core.TColgp import TColgp_HArray1OfPnt
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeWire,
    )

    if len(boundary_pts) < 3:
        return None, None

    # Ensure open polyline (no duplicate last point)
    if np.linalg.norm(boundary_pts[0] - boundary_pts[-1]) < 1e-10:
        boundary_pts = boundary_pts[:-1]

    # 1. Project onto OCC surface
    projected = []
    for pt in boundary_pts:
        gp_pt = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
        proj = GeomAPI_ProjectPointOnSurf(gp_pt, occ_surface)
        if proj.IsDone() and proj.NbPoints() > 0:
            pp = proj.NearestPoint()
            projected.append([pp.X(), pp.Y(), pp.Z()])
        else:
            projected.append(pt.tolist())
    projected = np.array(projected, dtype=np.float64)

    # 2. Laplacian smoothing
    smoothed = _laplacian_smooth(projected, iterations=smooth_iterations)

    # 3. Re-project after smoothing (smoothing may pull points off surface)
    for i in range(len(smoothed)):
        gp_pt = gp_Pnt(float(smoothed[i][0]), float(smoothed[i][1]),
                        float(smoothed[i][2]))
        proj = GeomAPI_ProjectPointOnSurf(gp_pt, occ_surface)
        if proj.IsDone() and proj.NbPoints() > 0:
            pp = proj.NearestPoint()
            smoothed[i] = [pp.X(), pp.Y(), pp.Z()]

    # 4. Simplify with Douglas-Peucker
    simplified = _douglas_peucker(smoothed, dp_epsilon)
    print(f"    boundary: {len(boundary_pts)} pts → smoothed → "
          f"simplified {len(simplified)}")

    if len(simplified) < 3:
        print(f"    WARNING: boundary too simple after Douglas-Peucker "
              f"({len(simplified)} pts)")
        return None, None

    # 5. Subsample if too dense
    if len(simplified) > n_bspline_pts:
        indices = np.linspace(0, len(simplified) - 1, n_bspline_pts, dtype=int)
        indices = np.unique(indices)
        simplified = simplified[indices]

    # 6. Fit periodic B-spline (closed curve)
    n = len(simplified)
    h_array = TColgp_HArray1OfPnt(1, n)
    for i in range(n):
        h_array.SetValue(i + 1, gp_Pnt(
            float(simplified[i][0]),
            float(simplified[i][1]),
            float(simplified[i][2]),
        ))

    try:
        interp = GeomAPI_Interpolate(h_array, True, 1e-6)  # periodic=True
        interp.Perform()
        if not interp.IsDone():
            print(f"    WARNING: GeomAPI_Interpolate failed (IsDone=False)")
            return None, None
        bspline_curve = interp.Curve()
    except Exception as e:
        print(f"    WARNING: B-spline interpolation failed: {e}")
        return None, None

    # 7. MakeEdge → MakeWire
    edge_builder = BRepBuilderAPI_MakeEdge(bspline_curve)
    if not edge_builder.IsDone():
        print(f"    WARNING: MakeEdge failed")
        return None, None

    wire_builder = BRepBuilderAPI_MakeWire(edge_builder.Edge())
    if not wire_builder.IsDone():
        print(f"    WARNING: MakeWire failed")
        return None, None

    return wire_builder.Wire(), projected


def build_face_from_loops(occ_surface, outer_loop, inner_loops=None,
                          dp_epsilon=0.005, n_bspline_pts=80,
                          smooth_iterations=5):
    """Build an OCC TopoDS_Face with outer wire and optional inner wires (holes).

    Returns (face, projected_outer) or (None, None) on failure.
    """
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace

    outer_wire, proj_outer = _boundary_to_wire(
        occ_surface, outer_loop,
        dp_epsilon=dp_epsilon, n_bspline_pts=n_bspline_pts,
        smooth_iterations=smooth_iterations,
    )
    if outer_wire is None:
        return None, None

    face_builder = BRepBuilderAPI_MakeFace(occ_surface, outer_wire, True)
    if not face_builder.IsDone():
        err = face_builder.Error()
        print(f"    WARNING: MakeFace failed (error code: {err})")
        return None, None

    # Add inner wires (holes)
    if inner_loops:
        for i, inner_loop in enumerate(inner_loops):
            inner_wire, _ = _boundary_to_wire(
                occ_surface, inner_loop,
                dp_epsilon=dp_epsilon, n_bspline_pts=n_bspline_pts,
                smooth_iterations=smooth_iterations,
            )
            if inner_wire is not None:
                face_builder.Add(inner_wire)
                if not face_builder.IsDone():
                    print(f"    WARNING: adding inner wire {i} failed")

    return face_builder.Face(), proj_outer


# ---------------------------------------------------------------------------
# Shell assembly
# ---------------------------------------------------------------------------

def sew_faces(faces, tolerance=1e-3):
    """Sew a list of TopoDS_Face into a shape (shell or compound)."""
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing
    from OCC.Core.BRepCheck import BRepCheck_Analyzer

    sewer = BRepBuilderAPI_Sewing(tolerance)
    for face in faces:
        sewer.Add(face)
    sewer.Perform()
    shape = sewer.SewedShape()

    n_free = sewer.NbFreeEdges()
    n_multi = sewer.NbMultipleEdges()
    n_degen = sewer.NbDegeneratedShapes()
    print(f"[sewing] free edges: {n_free}  multiple edges: {n_multi}  "
          f"degenerated: {n_degen}")

    analyzer = BRepCheck_Analyzer(shape, True)
    print(f"[sewing] BRepCheck valid: {analyzer.IsValid()}")

    return shape


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_compute(args):
    import torch

    from point2cad.surface_fitter    import fit_surface
    from point2cad.occ_surfaces      import to_occ_surface
    from point2cad.cluster_adjacency import build_cluster_proximity
    from point2cad.color_config      import get_surface_color
    from point2cad.topology          import export_step
    import point2cad.primitive_fitting_utils as pfu

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    def normalize_points(pts):
        mean    = np.mean(pts, axis=0)
        pts     = pts - mean
        S, U    = np.linalg.eigh(pts.T @ pts)
        R       = pfu.rotation_matrix_a_to_b(U[:, np.argmin(S)], np.array([1, 0, 0]))
        pts     = (R @ pts.T).T
        extents = np.max(pts, axis=0) - np.min(pts, axis=0)
        scale   = float(np.max(extents) + 1e-7)
        return (pts / scale).astype(np.float32), mean, R, scale

    np_rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Determine which .xyzc files to process
    input_pattern = os.path.join(args.input_dir, f"{args.model_id}", "*.xyzc")
    part_files    = sorted(_glob.glob(input_pattern),
                           key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    if not part_files:
        print(f"No part files found matching: {input_pattern}")
        return
    model_out_dir = os.path.join(args.output_dir, f"{args.model_id}")
    if os.path.exists(model_out_dir):
        shutil.rmtree(model_out_dir)
        print(f"Removed old results: {model_out_dir}")
    os.makedirs(model_out_dir)
    print(f"Model {args.model_id}: {len(part_files)} part(s)")

    for part_idx, sample_path in enumerate(part_files):
        step_stem = f"part_{part_idx}"
        out_dir   = os.path.join(model_out_dir, f"part_{part_idx}")

        print(f"\n{'='*60}")
        print(f"Part {part_idx}: {os.path.basename(sample_path)}  →  {out_dir}")
        print(f"{'='*60}")

        data = np.loadtxt(sample_path)
        data[:, :3], part_mean, part_R, part_scale = normalize_points(data[:, :3])
        unique_clusters, cluster_counts = np.unique(
            data[:, -1].astype(int), return_counts=True)
        os.makedirs(out_dir, exist_ok=True)

        # Collect clusters
        clusters = []
        for cid in unique_clusters:
            cluster = data[data[:, -1].astype(int) == cid, :3].astype(np.float32)
            clusters.append(cluster)

        # Per-cluster spacing (KDTree-based, same as brep_pipeline)
        cluster_trees, cluster_nn_percentiles = build_cluster_proximity(
            clusters, percentile=100.0
        )

        # ------------------------------------------------------------------
        # Surface fitting
        # ------------------------------------------------------------------
        surface_ids, fit_results, fit_meshes, trimeshes, occ_surfs = [], [], [], [], []
        for idx, (cid, c_count) in enumerate(zip(unique_clusters, cluster_counts)):
            cluster = clusters[idx]
            _spacing = cluster_nn_percentiles[idx]

            print(f"[surface fitter] Cluster {cid} ({c_count} pts) fitting ...")
            _plane_kw    = {"mesh_dim": 100, "plane_sampling_deviation": 0.5,
                            "spacing": _spacing, "threshold_multiplier": 2}
            _sphere_kw   = {"dim_theta": 100, "dim_lambda": 100,
                            "spacing": _spacing, "threshold_multiplier": 2}
            _cylinder_kw = {"dim_theta": 100, "dim_height": 50,
                            "cylinder_height_margin": 0.5,
                            "spacing": _spacing, "threshold_multiplier": 2}
            _cone_kw     = {"dim_theta": 100, "dim_height": 100,
                            "cone_height_margin": 0.5,
                            "spacing": _spacing, "threshold_multiplier": 2}

            res = fit_surface(
                cluster,
                {"hidden_dim": 64, "use_shortcut": True, "fraction_siren": 0.5},
                np_rng, DEVICE,
                inr_fit_kwargs={
                    "max_steps": 1500,
                    "noise_magnitude_3d": 0.05,
                    "noise_magnitude_uv": 0.05,
                    "initial_lr": 1e-1,
                },
                inr_mesh_kwargs={
                    "mesh_dim": 200,
                    "uv_margin": 0.2,
                    "threshold_multiplier": 1.5,
                },
                plane_mesh_kwargs=_plane_kw,
                sphere_mesh_kwargs=_sphere_kw,
                cylinder_mesh_kwargs=_cylinder_kw,
                cone_mesh_kwargs=_cone_kw,
                radius_inflation=0,
                angle_inflation_deg=0,
            )
            sid = res["surface_id"]
            surface_ids.append(sid)
            fit_results.append(res["result"])
            fit_meshes.append(res["mesh"])
            # Fine-trim the mesh to get clean boundaries
            trimmed = fine_trim_mesh(
                res["trimesh_mesh"], cluster,
                spacing=_spacing,
                threshold_multiplier=args.trim_multiplier,
            )
            trimeshes.append(trimmed)
            occ_surfs.append(
                to_occ_surface(sid, res["result"], cluster=cluster,
                               uv_margin=0.05, grid_resolution=50)
            )
            chosen_err = res["result"]["error"]
            all_errors = res.get("all_errors", {})
            errors_str = "  ".join(f"{name}={err:.6f}"
                                   for name, err in all_errors.items())
            print(f"[surface fitter] Cluster {cid} ({c_count} pts) → "
                  f"{SURFACE_NAMES[sid]}  residual={chosen_err:.6f}")
            if errors_str:
                print(f"  all errors: {errors_str}")

        # ------------------------------------------------------------------
        # Save metadata + cluster files for visualization
        # ------------------------------------------------------------------
        np.savez(
            os.path.join(out_dir, "metadata.npz"),
            n_clusters     = len(clusters),
            surface_ids    = np.array(surface_ids),
            surface_names  = np.array([SURFACE_NAMES[s] for s in surface_ids]),
            cluster_colors = np.array(
                [get_surface_color(SURFACE_NAMES[s]).tolist() for s in surface_ids]
            ),
        )
        for i, cluster in enumerate(clusters):
            np.save(os.path.join(out_dir, f"cluster_{i}.npy"),
                    _denorm(cluster, part_mean, part_R, part_scale))
        for i, mesh in enumerate(fit_meshes):
            np.savez(
                os.path.join(out_dir, f"surface_mesh_{i}.npz"),
                vertices  = _denorm(np.asarray(mesh.vertices),
                                    part_mean, part_R, part_scale),
                triangles = np.asarray(mesh.triangles),
            )

        # ------------------------------------------------------------------
        # Mesh boundary extraction + B-spline face construction
        # (everything in normalized space)
        # ------------------------------------------------------------------
        print(f"\n[boundary] Extracting mesh boundaries ...")
        faces_occ = []
        boundary_data = {}  # for visualization: cluster_idx -> list of loops

        for idx in range(len(clusters)):
            tmesh = trimeshes[idx]
            sname = SURFACE_NAMES[surface_ids[idx]]
            print(f"\n  Cluster {idx} ({sname}):")

            loops = extract_mesh_boundary(tmesh)
            if not loops:
                print(f"    No boundary loops found — skipping")
                continue

            print(f"    Found {len(loops)} boundary loop(s): "
                  f"{[len(l) for l in loops]} pts")

            # Longest loop = outer boundary, rest = inner (holes)
            outer_idx = max(range(len(loops)), key=lambda i: len(loops[i]))
            outer_loop = loops[outer_idx]
            inner_loops = [loops[i] for i in range(len(loops))
                           if i != outer_idx]

            print(f"    Outer loop: {len(outer_loop)} pts, "
                  f"{len(inner_loops)} inner loop(s)")

            face, proj_pts = build_face_from_loops(
                occ_surfs[idx], outer_loop, inner_loops,
                dp_epsilon=args.dp_epsilon,
                n_bspline_pts=args.max_boundary_pts,
                smooth_iterations=args.smooth_iterations,
            )

            if face is not None:
                faces_occ.append(face)
                print(f"    → face built successfully")
            else:
                print(f"    → face construction FAILED")

            # Save boundary data for visualization (all in normalized space,
            # _denorm applied when saving to disk)
            boundary_data[idx] = [{
                "loop": outer_loop,
                "projected": proj_pts,
                "tag": "outer",
            }]
            for li, inner_loop in enumerate(inner_loops):
                boundary_data[idx].append({
                    "loop": inner_loop,
                    "projected": None,
                    "tag": f"inner_{li}",
                })

        print(f"\n[boundary] Built {len(faces_occ)} OCC faces "
              f"out of {len(clusters)} clusters")

        # ------------------------------------------------------------------
        # Assembly + export
        # ------------------------------------------------------------------
        if faces_occ:
            from OCC.Core.BRep import BRep_Builder
            from OCC.Core.TopoDS import TopoDS_Compound

            builder = BRep_Builder()
            compound = TopoDS_Compound()
            builder.MakeCompound(compound)
            for face in faces_occ:
                builder.Add(compound, face)

            step_path = os.path.join(out_dir, f"{step_stem}.step")
            try:
                shape_world = apply_inverse_normalization(
                    compound, part_mean, part_R, part_scale)
            except Exception as e:
                print(f"[brep] inverse normalization failed: {e} "
                      f"— exporting in normalized space")
                shape_world = compound
            export_step(shape_world, step_path)
        else:
            print(f"[brep] no faces built — skipping STEP export")

        # Save boundary loops for visualization
        for idx, loops_info in boundary_data.items():
            for li, linfo in enumerate(loops_info):
                np.savez(
                    os.path.join(out_dir,
                                 f"boundary_{idx}_loop_{li}.npz"),
                    loop=_denorm(linfo["loop"], part_mean, part_R, part_scale),
                    projected=(_denorm(linfo["projected"],
                                       part_mean, part_R, part_scale)
                               if linfo["projected"] is not None
                               else np.empty((0, 3))),
                    tag=linfo["tag"],
                )

        print(f"\n[part] all results saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def run_visualize(args):
    out_dir = os.path.join(args.output_dir, f"{args.model_id}", "part_0")
    if not os.path.exists(out_dir):
        # Try unified
        out_dir = os.path.join(args.output_dir, f"{args.model_id}", "unified")
    if not os.path.exists(out_dir):
        print(f"Output directory not found: {out_dir}")
        return

    meta       = np.load(os.path.join(out_dir, "metadata.npz"), allow_pickle=True)
    n_clusters = int(meta["n_clusters"])
    clust_colors = meta["cluster_colors"]

    # Load cluster point clouds
    cluster_pcds = []
    all_cluster_pts = []
    for i in range(n_clusters):
        path = os.path.join(out_dir, f"cluster_{i}.npy")
        if not os.path.exists(path):
            continue
        pts = np.load(path)
        all_cluster_pts.append(pts)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color(clust_colors[i].tolist())
        cluster_pcds.append(pcd)

    def _lineset(pts, color):
        lines = [[m, m + 1] for m in range(len(pts) - 1)]
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines  = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
        return ls

    # Load boundary loops
    boundary_linesets_raw = []
    boundary_linesets_proj = []
    boundary_files = sorted(_glob.glob(
        os.path.join(out_dir, "boundary_*_loop_*.npz")))
    for bf in boundary_files:
        bd = np.load(bf, allow_pickle=True)
        loop = bd["loop"]
        proj = bd["projected"]
        tag  = str(bd["tag"])

        # Parse cluster index from filename
        base = os.path.basename(bf)
        parts = base.replace(".npz", "").split("_")
        cidx = int(parts[1])
        color = clust_colors[cidx].tolist() if cidx < len(clust_colors) else [1, 1, 1]

        if len(loop) >= 2:
            boundary_linesets_raw.append(_lineset(loop, color))
        if len(proj) >= 2:
            boundary_linesets_proj.append(_lineset(proj, [1.0, 1.0, 0.0]))

    # Load surface meshes
    surf_meshes = []
    for i in range(n_clusters):
        path = os.path.join(out_dir, f"surface_mesh_{i}.npz")
        if not os.path.exists(path):
            continue
        d = np.load(path)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices  = o3d.utility.Vector3dVector(d["vertices"])
        mesh.triangles = o3d.utility.Vector3iVector(d["triangles"])
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(clust_colors[i].tolist())
        surf_meshes.append(mesh)

    # Create visualization windows
    W, H = 800, 600

    # Window 1: Point clouds + raw boundary loops
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window("Clusters + Mesh Boundaries", width=W, height=H,
                       left=0, top=50)
    for pcd in cluster_pcds:
        vis1.add_geometry(pcd)
    for ls in boundary_linesets_raw:
        vis1.add_geometry(ls)
    vis1.get_render_option().point_size = 2.0

    # Window 2: Fitted surfaces + projected boundaries
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window("Surfaces + Projected Boundaries", width=W, height=H,
                       left=W, top=50)
    for mesh in surf_meshes:
        vis2.add_geometry(mesh)
    for ls in boundary_linesets_proj:
        vis2.add_geometry(ls)

    # Window 3: Boundaries only
    vis3 = o3d.visualization.Visualizer()
    vis3.create_window("Boundary Loops Only", width=W, height=H,
                       left=2*W, top=50)
    for ls in boundary_linesets_raw:
        vis3.add_geometry(ls)
    for ls in boundary_linesets_proj:
        vis3.add_geometry(ls)

    # Run all windows — closing any one window exits cleanly
    visualizers = [vis1, vis2, vis3]
    running = True
    while running:
        for vis in visualizers:
            if not vis.poll_events():
                running = False
                break
            vis.update_renderer()

    for vis in visualizers:
        vis.destroy_window()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="B-Rep reconstruction via mesh boundary + B-spline fitting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--visualize", action="store_true",
                        help="Load saved results and visualize (no OCC needed)")
    parser.add_argument("--model_id", type=str, required=True,
                        help="Model ID (e.g. 00000078)")
    parser.add_argument("--input_dir", type=str, default="sample_clouds",
                        help="Root directory for point cloud subdirs")
    parser.add_argument("--output_dir", type=str, default="output_boundary",
                        help="Directory for saved results")
    parser.add_argument("-seed", type=int, default=41,
                        help="Reproducibility seed")
    parser.add_argument("--trim_multiplier", type=float, default=1.5,
                        help="Fine mesh trimming threshold = spacing * multiplier "
                             "(consistent with adjacency detection)")
    parser.add_argument("--dp_epsilon", type=float, default=0.005,
                        help="Douglas-Peucker simplification tolerance")
    parser.add_argument("--max_boundary_pts", type=int, default=80,
                        help="Max boundary points after simplification "
                             "(subsampled if exceeded)")
    parser.add_argument("--smooth_iterations", type=int, default=5,
                        help="Laplacian smoothing iterations on boundary "
                             "before B-spline fitting")
    parser.add_argument("--sewing_tolerance", type=float, default=1e-3,
                        help="BRepBuilderAPI_Sewing tolerance")

    args = parser.parse_args()

    if args.visualize:
        run_visualize(args)
    else:
        run_compute(args)
