"""
Mesh-based surface intersection for the Point2CAD B-Rep pipeline.

Alternative to the analytical/OCC pathway in surface_intersection.py.
Computes intersection polylines via PyVista boolean intersection of
untrimmed surface meshes, then fits parametric curves to each polyline
using GeomAPI_PointsToBSpline.

Returns the same dict format as compute_all_intersections() so the
downstream vertex / trimming / arc-splitting pipeline is unchanged.
"""

import math
import numpy as np
from collections import defaultdict

import pyvista as pv
import vtk

try:
    from .surface_types import (
        SURFACE_PLANE, SURFACE_SPHERE, SURFACE_CYLINDER, SURFACE_CONE, SURFACE_INR,
        SURFACE_NAMES,
    )
    from .cluster_adjacency import adjacency_pairs
except ImportError:
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from point2cad.surface_types import (
        SURFACE_PLANE, SURFACE_SPHERE, SURFACE_CYLINDER, SURFACE_CONE, SURFACE_INR,
        SURFACE_NAMES,
    )
    from point2cad.cluster_adjacency import adjacency_pairs

from scipy.spatial import cKDTree

try:
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.TColgp import TColgp_Array1OfPnt
    from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline, GeomAPI_ProjectPointOnCurve
    from OCC.Core.GeomAbs import GeomAbs_C2
    from OCC.Core.Geom import Geom_TrimmedCurve
    OCC_AVAILABLE = True
except ImportError:
    OCC_AVAILABLE = False


def _result(curves, points, curve_type, method):
    return {"curves": curves, "points": points, "type": curve_type, "method": method}


# ---------------------------------------------------------------------------
# Open3D → PyVista conversion
# ---------------------------------------------------------------------------

def _o3d_to_pyvista(o3d_mesh):
    """
    Convert an Open3D TriangleMesh to a PyVista PolyData.

    triangulate_and_mesh() creates 4 triangles per quad cell: indices 0,1
    are the original orientation, 2,3 are reversed duplicates for double-
    sided rendering.  We keep only the original-orientation triangles
    (i % 4 < 2) to avoid degenerate self-intersections in PyVista.
    """
    vertices = np.asarray(o3d_mesh.vertices)
    triangles = np.asarray(o3d_mesh.triangles)

    # Filter double-sided duplicates
    mask = np.arange(len(triangles)) % 4 < 2
    triangles = triangles[mask]

    # PyVista faces format: [3, v0, v1, v2, 3, v0, v1, v2, ...]
    faces = np.column_stack([
        np.full(len(triangles), 3, dtype=np.int64),
        triangles,
    ]).ravel()

    return pv.PolyData(vertices, faces)


# ---------------------------------------------------------------------------
# Polyline extraction from PyVista intersection result
# ---------------------------------------------------------------------------

def _extract_polylines(intersection_polydata):
    """
    Parse VTK intersection segments into ordered polyline point arrays.

    vtkIntersectionPolyDataFilter returns 2-point line segments [2, p0, p1],
    one per triangle-triangle intersection.  Adjacent segments share point
    indices, forming a path graph (every node has degree <= 2).  We parse
    these into an adjacency list and walk each connected chain.
    """
    points = np.array(intersection_polydata.points)
    if intersection_polydata.n_cells == 0:
        return []

    lines = intersection_polydata.lines

    # Parse VTK connectivity: [2, p0, p1, 2, p0, p1, ...]
    graph = defaultdict(set)
    idx = 0
    while idx < len(lines):
        npts = lines[idx]
        seg = lines[idx + 1 : idx + 1 + npts]
        for k in range(len(seg) - 1):
            graph[seg[k]].add(seg[k + 1])
            graph[seg[k + 1]].add(seg[k])
        idx += 1 + npts

    # Check path graph assumption (all degrees <= 2)
    high_degree = {n: len(nb) for n, nb in graph.items() if len(nb) > 2}
    if high_degree:
        print(f"[polyline] WARNING: {len(high_degree)} node(s) with degree > 2 "
              f"(max {max(high_degree.values())})")

    # Walk open chains (start from degree-1 endpoints)
    visited = set()
    polylines = []

    for start in graph:
        if start in visited or len(graph[start]) != 1:
            continue
        ordered = [start]
        visited.add(start)
        current = start
        while True:
            neighbors = graph[current] - visited
            if not neighbors:
                break
            current = neighbors.pop()
            visited.add(current)
            ordered.append(current)
        polylines.append(points[ordered])

    # Walk closed loops (all remaining nodes have degree 2)
    for start in graph:
        if start in visited:
            continue
        ordered = [start]
        visited.add(start)
        current = start
        while True:
            neighbors = graph[current] - visited
            if not neighbors:
                break
            current = neighbors.pop()
            visited.add(current)
            ordered.append(current)
        ordered.append(start)  # close the loop
        polylines.append(points[ordered])

    polylines = _merge_nearby_polylines(polylines, points)
    return polylines


def _merge_nearby_polylines(polylines, points, tol_factor=3.0):
    """
    Merge polylines whose endpoints are within a few median segment lengths.

    VTK mesh intersection can produce gaps where triangle–triangle segments
    are spatially adjacent but have distinct point indices.  This stitches
    them back into a single continuous polyline.

    The tolerance is tol_factor × median segment length across all polylines
    of this intersection pair, adapting to mesh resolution automatically.
    """
    if len(polylines) <= 1:
        return polylines

    all_seg_lengths = []
    for poly in polylines:
        if len(poly) >= 2:
            all_seg_lengths.append(np.linalg.norm(np.diff(poly, axis=0), axis=1))
    if not all_seg_lengths:
        return polylines
    tol = tol_factor * float(np.median(np.concatenate(all_seg_lengths)))

    merged = True
    while merged:
        merged = False
        for i in range(len(polylines)):
            for j in range(i + 1, len(polylines)):
                pi = polylines[i]
                pj = polylines[j]
                # Try all four endpoint pairings
                pairs = [
                    ("end_start",   pi[-1], pj[0]),   # A_end   ↔ B_start
                    ("end_end",     pi[-1], pj[-1]),   # A_end   ↔ B_end
                    ("start_start", pi[0],  pj[0]),    # A_start ↔ B_start
                    ("start_end",   pi[0],  pj[-1]),   # A_start ↔ B_end
                ]
                for tag, pa, pb in pairs:
                    if np.linalg.norm(pa - pb) < tol:
                        if tag == "end_start":
                            combined = np.concatenate([pi, pj], axis=0)
                        elif tag == "end_end":
                            combined = np.concatenate([pi, pj[::-1]], axis=0)
                        elif tag == "start_start":
                            combined = np.concatenate([pi[::-1], pj], axis=0)
                        elif tag == "start_end":
                            combined = np.concatenate([pj, pi], axis=0)
                        polylines[i] = combined
                        polylines.pop(j)
                        merged = True
                        break
                if merged:
                    break
            if merged:
                break

    return polylines


# ---------------------------------------------------------------------------
# Curve fitting
# ---------------------------------------------------------------------------

_CLOSURE_TOL = 5e-3


def _fit_bspline(pts):
    """Fit a BSpline curve to an ordered point array via GeomAPI_PointsToBSpline.

    If the polyline is closed (first ≈ last point), the last point is snapped
    to exactly the first point before fitting, and the resulting BSpline is
    made periodic so that downstream arc splitting can use wrap-around
    parameters instead of introducing artificial seam vertices.
    """
    is_closed = len(pts) >= 3 and np.linalg.norm(pts[0] - pts[-1]) < _CLOSURE_TOL

    # Snap closed polylines: replace last point with exact copy of first
    if is_closed:
        pts = pts.copy()
        pts[-1] = pts[0]

    arr = TColgp_Array1OfPnt(1, len(pts))
    for k, p in enumerate(pts):
        arr.SetValue(k + 1, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))

    deg_min = 1 if len(pts) <= 3 else 3
    approx = GeomAPI_PointsToBSpline(arr, deg_min, 8, GeomAbs_C2, 1e-4)
    if not approx.IsDone():
        return None
    curve = approx.Curve()

    if is_closed:
        try:
            curve.SetPeriodic()
        except Exception:
            pass  # non-critical: seam-split fallback still works

    return curve


# ---------------------------------------------------------------------------
# Public API — intersection computation
# ---------------------------------------------------------------------------

def compute_mesh_intersections(adj, surface_ids, results, fit_meshes,
                               tol=1e-6, pairs=None):
    """
    Compute intersections for every adjacent pair using mesh representations.

    Parameters
    ----------
    adj        : (n, n) bool adjacency matrix
    surface_ids: list of surface type ids (SURFACE_PLANE, etc.)
    results    : list of fitting result dicts (with "params" key)
    fit_meshes : list of Open3D TriangleMesh, one per cluster
    tol        : unused (kept for API compatibility with compute_all_intersections)
    pairs      : optional list of (i, j) pairs to compute; if None, all
                 adjacent pairs from adj are used.

    Returns
    -------
    intersections : dict (i, j) i<j → result dict  (same format as
                    compute_all_intersections)
    polyline_map  : dict (i, j) i<j → list of (N, 3) arrays  (raw polylines,
                    for vertex detection)
    """
    vtk.vtkObject.GlobalWarningDisplayOff()

    out = {}
    polyline_map = {}

    iter_pairs = pairs if pairs is not None else adjacency_pairs(adj)
    for i, j in iter_pairs:
        si, sj = surface_ids[i], surface_ids[j]
        label = f"({i}, {j}) {SURFACE_NAMES[si]}∩{SURFACE_NAMES[sj]}"

        pv_i = _o3d_to_pyvista(fit_meshes[i])
        pv_j = _o3d_to_pyvista(fit_meshes[j])

        try:
            intersection, _, _ = pv_i.intersection(
                pv_j, split_first=False, split_second=False, progress_bar=False,
            )
        except Exception as e:
            print(f"  [mesh-intersect] {label}: FAILED (exception: {e})")
            out[(i, j)] = _result([], [], "failed", "mesh")
            polyline_map[(i, j)] = []
            continue

        if intersection.n_points == 0:
            print(f"  [mesh-intersect] {label}: empty (0 points)")
            out[(i, j)] = _result([], [], "empty", "mesh")
            polyline_map[(i, j)] = []
            continue

        polylines = _extract_polylines(intersection)
        polyline_map[(i, j)] = polylines

        # Fit BSpline to each polyline
        curves = []
        for poly in polylines:
            if len(poly) < 2:
                continue
            curve = _fit_bspline(poly)
            if curve is not None:
                curves.append(curve)

        if curves:
            result = _result(curves, [], "bspline", "mesh")
        else:
            result = _result([], [], "failed", "mesh")
        out[(i, j)] = result

        print(f"  [mesh-intersect] {label}: {result['type']}  "
              f"polylines={len(polylines)}  curves={len(result['curves'])}")

    return out, polyline_map


def tangent_fallback(raw_intersections, polyline_map, boundary_strips,
                     min_count, min_variance_ratio=0.9, extension=0.5):
    """
    For adjacent pairs with empty mesh intersections, attempt to recover a
    tangent contact line from boundary strip points.

    Fits a PCA line to the boundary strip, guarded by two gates:
    1. Point count ≥ *min_count* (precomputed percentile of all boundary
       strip sizes, filters false positive adjacencies).
    2. PCA explained variance ratio ≥ *min_variance_ratio* (confirms
       points are collinear, i.e. the tangency is a line).

    Produces a 2-point polyline (single line segment) extended by
    *extension* × span on each side, in the same format as mesh
    intersection polylines.
    """
    for (i, j), result in list(raw_intersections.items()):
        if result["type"] != "empty":
            continue

        bpts = boundary_strips.get((i, j))
        if bpts is None or len(bpts) < max(3, min_count):
            print(f"  [tangent] ({i}, {j}): {0 if bpts is None else len(bpts)} "
                  f"boundary pts < {max(3, min_count)} — skipping")
            continue

        # PCA
        centroid = bpts.mean(axis=0)
        centered = bpts - centroid
        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # eigh returns ascending order; largest eigenvalue is last
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]

        total = eigenvalues.sum()
        if total < 1e-15:
            continue
        variance_ratio = eigenvalues[0] / total

        if variance_ratio < min_variance_ratio:
            print(f"  [tangent] ({i}, {j}): variance ratio {variance_ratio:.4f} "
                  f"< {min_variance_ratio} — skipping")
            continue

        # Fit line: centroid + t * direction
        direction = eigenvectors[:, 0]
        projections = centered @ direction
        t_min, t_max = float(projections.min()), float(projections.max())
        span = t_max - t_min
        if span < 1e-10:
            continue

        # Extend
        t_min -= extension * span
        t_max += extension * span

        # 2-point polyline
        p0 = centroid + t_min * direction
        p1 = centroid + t_max * direction
        polyline = np.array([p0, p1], dtype=np.float64)

        # Fit BSpline (trivial for 2 points — degree 1 line)
        curve = _fit_bspline(polyline)
        if curve is None:
            continue

        raw_intersections[(i, j)] = _result([curve], [], "line", "tangent")
        polyline_map[(i, j)] = [polyline]
        print(f"  [tangent] ({i}, {j}): line from {len(bpts)} boundary pts  "
              f"variance_ratio={variance_ratio:.4f}  span={span:.4f}")


# ---------------------------------------------------------------------------
# Segment-segment closest-point computation
# ---------------------------------------------------------------------------

def _closest_point_segments(p0, p1, q0, q1):
    """
    Find the closest points between segments p0–p1 and q0–q1.

    Returns (point_on_p, point_on_q, distance).
    """
    d1 = p1 - p0   # direction of segment P
    d2 = q1 - q0   # direction of segment Q
    r  = p0 - q0

    a = float(np.dot(d1, d1))  # |d1|^2
    e = float(np.dot(d2, d2))  # |d2|^2
    f = float(np.dot(d2, r))

    EPS = 1e-12

    if a < EPS and e < EPS:
        # Both segments degenerate to points
        return p0.copy(), q0.copy(), float(np.linalg.norm(r))

    if a < EPS:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = float(np.dot(d1, r))
        if e < EPS:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = float(np.dot(d1, d2))
            denom = a * e - b * b
            if abs(denom) > EPS:
                s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
            else:
                s = 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)

    pt_p = p0 + s * d1
    pt_q = q0 + t * d2
    dist = float(np.linalg.norm(pt_p - pt_q))
    return pt_p, pt_q, dist


def compute_vertices_from_segment_intersection(polyline_map, threshold=5e-3,
                                                crossing_threshold=None,
                                                cluster_radius=5e-3,
                                                bbox_margin=None):
    """
    Find B-Rep vertices by intersecting polyline segments pairwise.

    For each pair of edges (i,j) and (i,k) sharing exactly one face,
    test all segment pairs from their respective polylines and find the
    closest approach.  If the minimum distance is below `crossing_threshold`,
    a vertex candidate is generated at the midpoint.

    Parameters
    ----------
    polyline_map        : dict (i, j) → list of (N, 3) arrays
    threshold           : KDTree search radius for pre-filtering segment pairs
    crossing_threshold  : max segment distance to accept a crossing candidate
                          (default: threshold)
    cluster_radius      : greedy deduplication radius for final vertices
    bbox_margin         : deprecated, unused

    Returns
    -------
    vertices     : (M, 3) float64 array
    vertex_edges : list[set] of length M
    """
    if crossing_threshold is None:
        crossing_threshold = threshold

    edge_keys = list(polyline_map.keys())
    candidates = []  # (position, distance, edge_a, edge_b)

    n_pairs_tested = 0
    n_segments_tested = 0
    n_segments_skipped = 0

    for a_idx in range(len(edge_keys)):
        for b_idx in range(a_idx + 1, len(edge_keys)):
            edge_a = edge_keys[a_idx]
            edge_b = edge_keys[b_idx]
            # Only consider edges sharing exactly one face
            shared = set(edge_a) & set(edge_b)
            if len(shared) != 1:
                continue

            n_pairs_tested += 1

            for poly_a in polyline_map[edge_a]:
                if len(poly_a) < 2:
                    continue
                for poly_b in polyline_map[edge_b]:
                    if len(poly_b) < 2:
                        continue

                    # KDTree pre-filter: find point pairs within reach,
                    # then only test segments adjacent to those points.
                    max_seg_a = float(np.max(np.linalg.norm(
                        np.diff(poly_a, axis=0), axis=1)))
                    max_seg_b = float(np.max(np.linalg.norm(
                        np.diff(poly_b, axis=0), axis=1)))
                    search_r = threshold + max_seg_a + max_seg_b

                    tree_b = cKDTree(poly_b)
                    # Query all points of poly_a against poly_b
                    close_b_indices = tree_b.query_ball_point(
                        poly_a, r=search_r)

                    # Collect candidate segment index pairs (si, sj)
                    seg_pairs = set()
                    for ai, b_indices in enumerate(close_b_indices):
                        if not b_indices:
                            continue
                        # Segments adjacent to point ai: (ai-1, ai) and (ai, ai+1)
                        segs_a = set()
                        if ai > 0:
                            segs_a.add(ai - 1)
                        if ai < len(poly_a) - 1:
                            segs_a.add(ai)
                        for bi in b_indices:
                            segs_b = set()
                            if bi > 0:
                                segs_b.add(bi - 1)
                            if bi < len(poly_b) - 1:
                                segs_b.add(bi)
                            for sa in segs_a:
                                for sb in segs_b:
                                    seg_pairs.add((sa, sb))

                    n_segments_skipped += (
                        (len(poly_a) - 1) * (len(poly_b) - 1)
                        - len(seg_pairs))

                    for si, sj in seg_pairs:
                        n_segments_tested += 1
                        pt_p, pt_q, dist = _closest_point_segments(
                            poly_a[si], poly_a[si + 1],
                            poly_b[sj], poly_b[sj + 1])

                        if dist < crossing_threshold:
                            candidates.append(
                                (0.5 * (pt_p + pt_q), dist, edge_a, edge_b))

    print(f"[seg-intersect] {n_pairs_tested} edge pairs, "
          f"{n_segments_tested} segment pairs tested, "
          f"{n_segments_skipped} skipped by KDTree filter")
    print(f"[seg-intersect] {len(candidates)} raw candidates "
          f"(crossing_threshold={crossing_threshold:.1e})")

    if not candidates:
        return np.empty((0, 3), dtype=np.float64), []

    # Sort by distance (best first)
    candidates.sort(key=lambda c: c[1])

    # Greedy clustering
    positions = np.array([c[0] for c in candidates], dtype=np.float64)
    used = np.zeros(len(positions), dtype=bool)
    vertices = []
    vertex_edges = []

    for idx in range(len(positions)):
        if used[idx]:
            continue
        dists_arr = np.linalg.norm(positions - positions[idx], axis=1)
        close = (dists_arr < cluster_radius) & ~used
        close_indices = np.where(close)[0]

        edges = set()
        for ci in close_indices:
            _, _, ea, eb = candidates[ci]
            edges.add(ea)
            edges.add(eb)

        if len(edges) < 2:
            used[close] = True
            continue

        pos = positions[close].mean(axis=0)
        best_score = candidates[close_indices[0]][1]
        v_idx = len(vertices)
        print(f"  v{v_idx}: ({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f})  "
              f"dist={best_score:.6e}  "
              f"edges={sorted(edges)}  merged={len(close_indices)}")
        vertices.append(pos)
        vertex_edges.append(edges)
        used[close] = True

    print(f"[seg-intersect] {len(vertices)} vertices after clustering")

    # Post-clustering attribution: for each vertex, check if nearby
    # polylines (sharing a face with an existing edge) pass through it.
    # This catches edges missed by the segment crossing test (e.g. when
    # the crossing distance was slightly above crossing_threshold).
    for v_idx in range(len(vertices)):
        pos = vertices[v_idx]
        existing_faces = set()
        for ea in vertex_edges[v_idx]:
            existing_faces.update(ea)

        for edge_key, polys in polyline_map.items():
            if edge_key in vertex_edges[v_idx]:
                continue
            # Edge must share at least one face with the vertex
            if not existing_faces & set(edge_key):
                continue
            for poly in polys:
                if len(poly) < 2:
                    continue
                dists = np.linalg.norm(poly - pos, axis=1)
                min_dist = float(np.min(dists))
                if min_dist < cluster_radius:
                    vertex_edges[v_idx].add(edge_key)
                    print(f"  v{v_idx}: attributed edge {edge_key} "
                          f"(polyline dist={min_dist:.6e})")
                    break

    verts_arr = (np.array(vertices, dtype=np.float64) if vertices
                 else np.empty((0, 3), dtype=np.float64))
    return verts_arr, vertex_edges


# ---------------------------------------------------------------------------
# Arc construction from polylines + vertices
# ---------------------------------------------------------------------------

def _find_polyline_index(vertex_pos, poly, threshold=5e-3):
    """Find the index in poly closest to vertex_pos, or None if too far."""
    dists = np.linalg.norm(poly - vertex_pos, axis=1)
    best = np.argmin(dists)
    if dists[best] > threshold:
        return None
    return int(best)


def build_arcs_from_polylines(polyline_map, vertices, vertex_edges,
                              threshold=1e-3):
    """
    Build B-Rep arcs by fitting BSplines to polyline segments between vertices.

    For each edge's polyline:
    1. Find incident vertices and their positions along the polyline.
    2. Take the two outermost as endpoint vertices; trim polyline to them,
       snap endpoints to exact vertex positions.
    3. Fit one BSpline to the trimmed polyline.
    4. Split the BSpline at interior vertex parameters to produce arcs.

    Closed polylines with 0 incident vertices get a full closed BSpline.
    Open polylines with 0 incident vertices get a full open BSpline.

    Parameters
    ----------
    polyline_map  : dict (i, j) → list of (N, 3) arrays
    vertices      : (M, 3) float64 array
    vertex_edges  : list[set] of length M
    threshold     : max distance for vertex-to-polyline attribution

    Returns
    -------
    edge_arcs     : dict (i,j) → list[arc_dict]
    vertices      : (M', 3) array (unchanged)
    vertex_edges  : list[set] of length M'
    """
    edge_arcs = {}

    for edge_key, polylines in polyline_map.items():
        arcs_for_edge = []

        for poly in polylines:
            if len(poly) < 2:
                continue

            is_closed = np.linalg.norm(poly[0] - poly[-1]) < _CLOSURE_TOL

            # Find incident vertices for this edge + polyline
            incident = []  # (polyline_index, vertex_index)
            for v_idx, v_edges in enumerate(vertex_edges):
                if edge_key not in v_edges:
                    continue
                pi = _find_polyline_index(vertices[v_idx], poly, threshold)
                if pi is not None:
                    incident.append((pi, v_idx))
                else:
                    d = float(np.linalg.norm(
                        poly - vertices[v_idx], axis=1).min())
                    print(f"    [arcs] WARNING: v{v_idx} claimed by edge "
                          f"{edge_key} but min_dist={d:.6e} > {threshold}")

            incident.sort(key=lambda x: x[0])
            k = len(incident)

            if k == 0:
                # No incident vertices: fit full BSpline
                curve = _fit_bspline(poly)
                if curve is None:
                    continue
                arcs_for_edge.append({
                    "curve": curve,
                    "v_start": None,
                    "v_end": None,
                    "t_start": curve.FirstParameter(),
                    "t_end": curve.LastParameter(),
                    "closed": is_closed,
                    "edge_key": edge_key,
                })
                continue

            if k == 1:
                # Single vertex — shouldn't produce useful arcs, fit full
                curve = _fit_bspline(poly)
                if curve is None:
                    continue
                _, v0 = incident[0]
                arcs_for_edge.append({
                    "curve": curve,
                    "v_start": v0,
                    "v_end": v0,
                    "t_start": curve.FirstParameter(),
                    "t_end": curve.LastParameter(),
                    "closed": True,
                    "edge_key": edge_key,
                })
                continue

            # k >= 2: trim polyline to outermost vertices and snap
            # ALL incident vertex positions into the polyline so the
            # fitted BSpline passes exactly through every vertex.
            first_pi, first_vi = incident[0]
            last_pi, last_vi = incident[-1]

            trimmed = poly[first_pi:last_pi + 1].copy()
            if len(trimmed) < 2:
                continue

            # Snap all incident vertices into the trimmed polyline
            # (indices shifted by -first_pi after trimming)
            for pi, vi in incident:
                local_idx = pi - first_pi
                if 0 <= local_idx < len(trimmed):
                    trimmed[local_idx] = vertices[vi]

            curve = _fit_bspline(trimmed)
            if curve is None:
                continue

            t_min = curve.FirstParameter()
            t_max = curve.LastParameter()

            if k == 2:
                # No interior vertices — single arc
                arcs_for_edge.append({
                    "curve": Geom_TrimmedCurve(curve, t_min, t_max),
                    "v_start": first_vi,
                    "v_end": last_vi,
                    "t_start": t_min,
                    "t_end": t_max,
                    "closed": False,
                    "edge_key": edge_key,
                })
                continue

            # k > 2: project interior vertices onto BSpline, split into arcs.
            # Since the vertex positions were snapped into the polyline before
            # fitting, projection lands exactly on the interpolation point.
            interior = incident[1:-1]
            arc_params = [(t_min, first_vi)]
            for _, v_idx in interior:
                vpos = vertices[v_idx]
                proj = GeomAPI_ProjectPointOnCurve(
                    gp_Pnt(float(vpos[0]), float(vpos[1]), float(vpos[2])),
                    curve, t_min, t_max,
                )
                if proj.NbPoints() > 0:
                    arc_params.append(
                        (float(proj.LowerDistanceParameter()), v_idx))
            arc_params.append((t_max, last_vi))
            arc_params.sort(key=lambda x: x[0])

            for m in range(len(arc_params) - 1):
                t_a, v_a = arc_params[m]
                t_b, v_b = arc_params[m + 1]
                if t_b <= t_a + 1e-10:
                    continue
                arcs_for_edge.append({
                    "curve": Geom_TrimmedCurve(curve, t_a, t_b),
                    "v_start": v_a,
                    "v_end": v_b,
                    "t_start": t_a,
                    "t_end": t_b,
                    "closed": False,
                    "edge_key": edge_key,
                })

        for arc_i, arc in enumerate(arcs_for_edge):
            arc["arc_idx"] = arc_i
        edge_arcs[edge_key] = arcs_for_edge

        n_arcs = len(arcs_for_edge)
        v_set = set()
        for arc in arcs_for_edge:
            if arc["v_start"] is not None:
                v_set.add(arc["v_start"])
            if arc["v_end"] is not None:
                v_set.add(arc["v_end"])
        print(f"  [arcs] edge {edge_key}: {n_arcs} arcs, "
              f"vertices={sorted(v_set) if v_set else 'none'}")

    return edge_arcs, vertices, vertex_edges
