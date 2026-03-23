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

        n_valid = sum(1 for p in polylines if len(p) >= 2)
        if n_valid > 0:
            out[(i, j)] = _result([], [], "polyline", "mesh")
        else:
            out[(i, j)] = _result([], [], "failed", "mesh")

        print(f"  [mesh-intersect] {label}: {out[(i, j)]['type']}  "
              f"polylines={len(polylines)}  valid={n_valid}")

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

    Returns
    -------
    tangent_spreads : dict (i, j) → float
        Perpendicular spread (sqrt of second eigenvalue) for each tangent
        edge.  Used as a data-derived attribution threshold in
        post-clustering vertex attribution.
    """
    tangent_spreads = {}

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

        # Perpendicular spread: sqrt of second eigenvalue
        spread = float(np.sqrt(eigenvalues[1])) if eigenvalues[1] > 0 else 0.0

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

        raw_intersections[(i, j)] = _result([], [], "line", "tangent")
        polyline_map[(i, j)] = [polyline]
        tangent_spreads[(i, j)] = spread
        print(f"  [tangent] ({i}, {j}): line from {len(bpts)} boundary pts  "
              f"variance_ratio={variance_ratio:.4f}  span={span:.4f}  "
              f"spread={spread:.4f}")

    return tangent_spreads


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
                                                bbox_margin=None,
                                                tangent_edges=None,
                                                tangent_threshold=1e-2):
    """
    Find B-Rep vertices by intersecting polyline segments pairwise.

    For each pair of edges (i,j) and (i,k) sharing exactly one face,
    test all segment pairs from their respective polylines and find the
    closest approach.  If the minimum distance is below `crossing_threshold`,
    a vertex candidate is generated at the midpoint.

    Tracks which specific polyline (by index within the edge's polyline list)
    produced each crossing.  During greedy clustering, the merged vertex
    inherits the union of all (edge, poly_idx) pairs from its constituents.

    Parameters
    ----------
    polyline_map        : dict (i, j) -> list of (N, 3) arrays
    threshold           : KDTree search radius for pre-filtering segment pairs
    crossing_threshold  : max segment distance to accept a crossing candidate
                          (default: threshold)
    cluster_radius      : greedy deduplication radius for final vertices
    bbox_margin         : deprecated, unused

    Returns
    -------
    vertices      : (M, 3) float64 array
    vertex_edges  : list[set] of length M — set of (edge_key) tuples
    vertex_polys  : list[set] of length M — set of (edge_key, poly_idx) tuples
    """
    if crossing_threshold is None:
        crossing_threshold = threshold

    edge_keys = list(polyline_map.keys())
    # Each candidate: (position, distance, edge_a, poly_idx_a, edge_b, poly_idx_b)
    candidates = []

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

            for pa_idx, poly_a in enumerate(polyline_map[edge_a]):
                if len(poly_a) < 2:
                    continue
                for pb_idx, poly_b in enumerate(polyline_map[edge_b]):
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
                    close_b_indices = tree_b.query_ball_point(
                        poly_a, r=search_r)

                    seg_pairs = set()
                    for ai, b_indices in enumerate(close_b_indices):
                        if not b_indices:
                            continue
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
                            candidates.append((
                                0.5 * (pt_p + pt_q), dist,
                                edge_a, pa_idx, edge_b, pb_idx,
                            ))

    print(f"[seg-intersect] {n_pairs_tested} edge pairs, "
          f"{n_segments_tested} segment pairs tested, "
          f"{n_segments_skipped} skipped by KDTree filter")
    print(f"[seg-intersect] {len(candidates)} raw candidates "
          f"(crossing_threshold={crossing_threshold:.1e})")

    if not candidates:
        return np.empty((0, 3), dtype=np.float64), [], []

    # Sort by distance (best first)
    candidates.sort(key=lambda c: c[1])

    # Greedy clustering
    positions = np.array([c[0] for c in candidates], dtype=np.float64)
    used = np.zeros(len(positions), dtype=bool)
    vertices = []
    vertex_edges = []
    vertex_polys = []

    for idx in range(len(positions)):
        if used[idx]:
            continue
        dists_arr = np.linalg.norm(positions - positions[idx], axis=1)
        close = (dists_arr < cluster_radius) & ~used
        close_indices = np.where(close)[0]

        edges = set()
        polys = set()
        for ci in close_indices:
            _, _, ea, pa, eb, pb = candidates[ci]
            edges.add(ea)
            edges.add(eb)
            polys.add((ea, pa))
            polys.add((eb, pb))

        if len(edges) < 2:
            used[close] = True
            continue

        pos = positions[close].mean(axis=0)
        best_score = candidates[close_indices[0]][1]
        v_idx = len(vertices)
        print(f"  v{v_idx}: ({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f})  "
              f"dist={best_score:.6e}  "
              f"edges={sorted(edges)}  polys={sorted(polys)}  "
              f"merged={len(close_indices)}")
        vertices.append(pos)
        vertex_edges.append(edges)
        vertex_polys.append(polys)
        used[close] = True

    print(f"[seg-intersect] {len(vertices)} vertices after clustering")

    # Post-clustering attribution: for each vertex, check all 2-combinations
    # of its incident surfaces for missing edges.  Use distance-based
    # attribution with cluster_radius for mesh polylines and a wider
    # tangent_threshold for tangent-generated polylines.
    if tangent_edges is None:
        tangent_edges = set()
    from itertools import combinations
    for v_idx in range(len(vertices)):
        pos = vertices[v_idx]
        incident_faces = set()
        for ea in vertex_edges[v_idx]:
            incident_faces.update(ea)

        for fi, fj in combinations(sorted(incident_faces), 2):
            edge_key = (fi, fj)
            if edge_key in vertex_edges[v_idx]:
                continue
            if edge_key not in polyline_map:
                continue
            polys_list = polyline_map[edge_key]
            valid_polys = [(pi, p) for pi, p in enumerate(polys_list)
                           if len(p) >= 2]

            thr = tangent_threshold if edge_key in tangent_edges else cluster_radius
            for pi, poly in valid_polys:
                _, _, dist = _nearest_segment_index(pos, poly)
                if dist < thr:
                    vertex_edges[v_idx].add(edge_key)
                    vertex_polys[v_idx].add((edge_key, pi))
                    print(f"  v{v_idx}: post-attributed edge {edge_key} "
                          f"poly {pi} (dist={dist:.6e}, thr={thr:.1e})")
                    break

    verts_arr = (np.array(vertices, dtype=np.float64) if vertices
                 else np.empty((0, 3), dtype=np.float64))
    return verts_arr, vertex_edges, vertex_polys


# ---------------------------------------------------------------------------
# Arc construction from polylines + vertices
# ---------------------------------------------------------------------------

def _nearest_segment_index(vertex_pos, poly):
    """Find the segment index in poly closest to vertex_pos.

    Returns (seg_idx, t, dist) where t in [0,1] is the parameter along
    segment poly[seg_idx] -> poly[seg_idx+1], and dist is the distance.
    """
    best_seg = 0
    best_t = 0.0
    best_dist = float("inf")
    for k in range(len(poly) - 1):
        d = poly[k + 1] - poly[k]
        seg_len_sq = float(np.dot(d, d))
        if seg_len_sq < 1e-30:
            t = 0.0
        else:
            t = float(np.clip(np.dot(vertex_pos - poly[k], d) / seg_len_sq,
                              0.0, 1.0))
        proj = poly[k] + t * d
        dist = float(np.linalg.norm(vertex_pos - proj))
        if dist < best_dist:
            best_seg = k
            best_t = t
            best_dist = dist
    return best_seg, best_t, best_dist


def build_arcs_from_polylines(polyline_map, vertices, vertex_edges,
                              vertex_polys):
    """
    Build B-Rep arcs by splitting polylines at vertices and fitting B-splines.

    Uses polyline-level vertex attribution (vertex_polys) to know exactly
    which vertices are incident to which polyline.

    For each edge, for each polyline:
    1. Collect incident vertices from vertex_polys.
    2. Discard if fewer than 2 incident vertices.
    3. For each incident vertex, find nearest segment and insert the exact
       vertex position into the polyline point sequence.
    4. Sort incident vertices by arc-length along the polyline.
    5. Trim polyline to outermost vertices.
    6. Split into sub-polylines between consecutive vertices.
    7. Fit a B-spline to each sub-polyline.

    Parameters
    ----------
    polyline_map  : dict (i, j) -> list of (N, 3) arrays
    vertices      : (M, 3) float64 array
    vertex_edges  : list[set] of length M — set of edge_key tuples
    vertex_polys  : list[set] of length M — set of (edge_key, poly_idx) tuples

    Returns
    -------
    edge_arcs     : dict (i,j) -> list[arc_dict]
    vertices      : (M, 3) array (unchanged)
    vertex_edges  : list[set] of length M
    """
    edge_arcs = {}

    for edge_key, polylines in polyline_map.items():
        arcs_for_edge = []

        for poly_idx, poly in enumerate(polylines):
            if len(poly) < 2:
                continue

            is_closed = (len(poly) >= 3 and
                         np.linalg.norm(poly[0] - poly[-1]) < _CLOSURE_TOL)

            # Collect incident vertices for this specific polyline
            incident = []  # (v_idx,)
            for v_idx in range(len(vertices)):
                if (edge_key, poly_idx) in vertex_polys[v_idx]:
                    incident.append(v_idx)

            if not is_closed and len(incident) < 2:
                # Open polyline with 0 or 1 vertices — discard
                continue

            if is_closed and len(incident) == 0:
                # Closed polyline with 0 vertices — keep as full B-spline
                curve = _fit_bspline(poly)
                if curve is not None:
                    arcs_for_edge.append({
                        "curve": curve,
                        "v_start": None,
                        "v_end": None,
                        "t_start": curve.FirstParameter(),
                        "t_end": curve.LastParameter(),
                        "closed": True,
                        "edge_key": edge_key,
                    })
                continue

            # For each incident vertex, find its position along the polyline
            # (segment index + parameter within that segment)
            vertex_positions = []  # (seg_idx, t, v_idx)
            for v_idx in incident:
                seg_idx, t, dist = _nearest_segment_index(
                    vertices[v_idx], poly)
                vertex_positions.append((seg_idx, t, v_idx))

            # Sort by position along polyline
            vertex_positions.sort(key=lambda x: (x[0], x[1]))

            # Insert vertex positions into polyline, working from last to
            # first to keep indices stable.
            # Build insertion list: (insert_after_index, vertex_position)
            insertions = []  # (insert_point, v_idx, vertex_pos)
            for seg_idx, t, v_idx in vertex_positions:
                if t < 1e-6:
                    # Snap to existing point at seg_idx
                    insertions.append(("snap", seg_idx, v_idx))
                elif t > 1 - 1e-6:
                    # Snap to existing point at seg_idx + 1
                    insertions.append(("snap", seg_idx + 1, v_idx))
                else:
                    # Insert new point after seg_idx
                    insertions.append(("insert", seg_idx, v_idx))

            # Apply insertions to build modified polyline
            # Process in reverse order to keep indices stable
            mod_poly = poly.copy()
            # Track where each vertex ends up in the modified polyline
            vertex_mod_indices = {}  # v_idx -> index in mod_poly

            # First pass: snaps (no index shifting)
            for action, idx, v_idx in insertions:
                if action == "snap":
                    mod_poly[idx] = vertices[v_idx]
                    vertex_mod_indices[v_idx] = idx

            # Second pass: inserts (process in reverse order)
            # Collect inserts sorted by position descending
            insert_ops = [(idx, v_idx) for action, idx, v_idx in insertions
                          if action == "insert"]
            insert_ops.sort(key=lambda x: x[0], reverse=True)

            for seg_idx, v_idx in insert_ops:
                insert_pos = seg_idx + 1
                mod_poly = np.insert(mod_poly, insert_pos,
                                     vertices[v_idx], axis=0)
                # Shift indices of previously placed vertices
                for vk in vertex_mod_indices:
                    if vertex_mod_indices[vk] >= insert_pos:
                        vertex_mod_indices[vk] += 1
                vertex_mod_indices[v_idx] = insert_pos

            # Sort vertices by their position in the modified polyline
            sorted_verts = sorted(vertex_mod_indices.items(),
                                  key=lambda x: x[1])

            if is_closed:
                # Closed polyline: no trimming, split at all vertices
                # including wrap-around arc from last vertex back to first.
                n_sv = len(sorted_verts)
                for m in range(n_sv):
                    v_a, idx_a = sorted_verts[m]
                    v_b, idx_b = sorted_verts[(m + 1) % n_sv]

                    if idx_b > idx_a:
                        sub_poly = mod_poly[idx_a:idx_b + 1]
                    else:
                        # Wrap-around: end of polyline + start of polyline
                        # Skip the duplicated closure point (last == first)
                        sub_poly = np.concatenate([
                            mod_poly[idx_a:],
                            mod_poly[1:idx_b + 1],
                        ], axis=0)

                    if len(sub_poly) < 2:
                        continue

                    curve = _fit_bspline(sub_poly)
                    if curve is None:
                        continue

                    t_min = curve.FirstParameter()
                    t_max = curve.LastParameter()
                    arcs_for_edge.append({
                        "curve": Geom_TrimmedCurve(curve, t_min, t_max),
                        "v_start": v_a,
                        "v_end": v_b,
                        "t_start": t_min,
                        "t_end": t_max,
                        "closed": False,
                        "edge_key": edge_key,
                    })
            else:
                # Open polyline: trim to outermost vertices
                first_v, first_idx = sorted_verts[0]
                last_v, last_idx = sorted_verts[-1]
                trimmed = mod_poly[first_idx:last_idx + 1]

                # Recompute vertex indices relative to trimmed polyline
                trimmed_verts = []  # (local_idx, v_idx)
                for v_idx, mod_idx in sorted_verts:
                    local_idx = mod_idx - first_idx
                    trimmed_verts.append((local_idx, v_idx))

                # Split into sub-polylines between consecutive vertices
                for m in range(len(trimmed_verts) - 1):
                    idx_a, v_a = trimmed_verts[m]
                    idx_b, v_b = trimmed_verts[m + 1]
                    sub_poly = trimmed[idx_a:idx_b + 1]

                    if len(sub_poly) < 2:
                        continue

                    curve = _fit_bspline(sub_poly)
                    if curve is None:
                        continue

                    t_min = curve.FirstParameter()
                    t_max = curve.LastParameter()
                    arcs_for_edge.append({
                        "curve": Geom_TrimmedCurve(curve, t_min, t_max),
                        "v_start": v_a,
                        "v_end": v_b,
                        "t_start": t_min,
                        "t_end": t_max,
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
