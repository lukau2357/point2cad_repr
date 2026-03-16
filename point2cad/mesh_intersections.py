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
    Parse connectivity into ordered polyline point arrays.

    The intersection result contains line segments as VTK cells.  We build
    an adjacency graph from point indices, find connected components, and
    walk each component from an endpoint (degree-1 node) to produce an
    ordered (N, 3) array per polyline.
    """
    points = np.array(intersection_polydata.points)
    if intersection_polydata.n_cells == 0:
        return []

    lines = intersection_polydata.lines

    # Parse VTK connectivity: [npts, p0, p1, ..., npts, p0, p1, ...]
    graph = defaultdict(set)
    idx = 0
    while idx < len(lines):
        npts = lines[idx]
        seg = lines[idx + 1 : idx + 1 + npts]
        for k in range(len(seg) - 1):
            graph[seg[k]].add(seg[k + 1])
            graph[seg[k + 1]].add(seg[k])
        idx += 1 + npts

    # Find connected components and order each into a polyline
    visited = set()
    polylines = []

    for seed in graph:
        if seed in visited:
            continue

        # BFS to find connected component
        component = set()
        stack = [seed]
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(graph[node] - component)
        visited |= component

        # Start from an endpoint (degree 1) if one exists, else arbitrary
        endpoints = [n for n in component if len(graph[n]) == 1]
        start = endpoints[0] if endpoints else min(component)

        # Walk the chain
        ordered = [start]
        prev = None
        current = start
        while True:
            neighbors = graph[current] - {prev}
            if not neighbors:
                break
            next_node = min(neighbors)   # deterministic tie-breaking
            if next_node == ordered[0] and len(ordered) > 2:
                ordered.append(next_node)   # close the loop
                break
            if next_node in set(ordered):
                break
            ordered.append(next_node)
            prev = current
            current = next_node

        polylines.append(points[ordered])

    return polylines


# ---------------------------------------------------------------------------
# Curve fitting
# ---------------------------------------------------------------------------

_CLOSURE_TOL = 5e-3


def _fit_bspline(pts):
    """Fit a BSpline curve to an ordered point array via GeomAPI_PointsToBSpline.

    If the polyline is closed (first ≈ last point), the last point is snapped
    to exactly the first point before fitting so that the resulting BSpline
    has coincident endpoints and passes the downstream closure check.
    """
    # Snap closed polylines: replace last point with exact copy of first
    if len(pts) >= 3 and np.linalg.norm(pts[0] - pts[-1]) < _CLOSURE_TOL:
        pts = pts.copy()
        pts[-1] = pts[0]

    arr = TColgp_Array1OfPnt(1, len(pts))
    for k, p in enumerate(pts):
        arr.SetValue(k + 1, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))

    approx = GeomAPI_PointsToBSpline(arr, 3, 8, GeomAbs_C2, 1e-4)
    if not approx.IsDone():
        return None
    return approx.Curve()


# ---------------------------------------------------------------------------
# Public API — intersection computation
# ---------------------------------------------------------------------------

def compute_mesh_intersections(adj, surface_ids, results, fit_meshes,
                               tol=1e-6):
    """
    Compute intersections for every adjacent pair using mesh representations.

    Parameters
    ----------
    adj        : (n, n) bool adjacency matrix
    surface_ids: list of surface type ids (SURFACE_PLANE, etc.)
    results    : list of fitting result dicts (with "params" key)
    fit_meshes : list of Open3D TriangleMesh, one per cluster
    tol        : unused (kept for API compatibility with compute_all_intersections)

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

    for i, j in adjacency_pairs(adj):
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


# ---------------------------------------------------------------------------
# Public API — vertex detection from polyline endpoints
# ---------------------------------------------------------------------------

def compute_vertices_from_polylines(polyline_map, threshold=1e-3):
    """
    Find B-Rep vertices from polyline endpoint coincidence.

    A vertex exists where endpoints from 2+ different edges (surface pairs)
    cluster within `threshold` distance.  Each vertex is attributed to the
    set of edges whose endpoints contributed to the cluster.

    Parameters
    ----------
    polyline_map : dict (i, j) → list of (N, 3) arrays
    threshold    : clustering radius for endpoint matching

    Returns
    -------
    vertices     : (M, 3) float64 array
    vertex_edges : list[set] of length M; vertex_edges[v] is the set of
                   edge tuples (i, j) incident to vertex v
    """
    # Collect all polyline endpoints tagged with their edge key
    ep_positions = []
    ep_edge_keys = []

    for (i, j), polylines in polyline_map.items():
        for poly in polylines:
            if len(poly) < 2:
                continue
            ep_positions.append(poly[0])
            ep_edge_keys.append((i, j))
            # Add the other endpoint only if the polyline is open
            if np.linalg.norm(poly[0] - poly[-1]) > threshold:
                ep_positions.append(poly[-1])
                ep_edge_keys.append((i, j))

    if not ep_positions:
        print("[mesh-vertices] no polyline endpoints found")
        return np.empty((0, 3), dtype=np.float64), []

    positions = np.array(ep_positions, dtype=np.float64)

    print(f"[mesh-vertices] {len(positions)} polyline endpoints "
          f"from {len(polyline_map)} edges")

    # Greedy clustering
    used = np.zeros(len(positions), dtype=bool)
    vertices = []
    vertex_edges = []

    for idx in range(len(positions)):
        if used[idx]:
            continue
        dists = np.linalg.norm(positions - positions[idx], axis=1)
        close = (dists < threshold) & ~used
        close_indices = np.where(close)[0]

        # Collect distinct edge keys in this cluster
        edges = set()
        for ci in close_indices:
            edges.add(ep_edge_keys[ci])

        # A vertex must be incident to at least 2 distinct edges
        if len(edges) < 2:
            used[close] = True
            continue

        pos = positions[close].mean(axis=0)
        v_idx = len(vertices)
        print(f"  v{v_idx}: ({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f})  "
              f"edges={sorted(edges)}  merged={len(close_indices)} endpoints")
        vertices.append(pos)
        vertex_edges.append(edges)
        used[close] = True

    print(f"[mesh-vertices] {len(vertices)} vertices after clustering")
    return (np.array(vertices, dtype=np.float64) if vertices
            else np.empty((0, 3), dtype=np.float64)), vertex_edges


# ---------------------------------------------------------------------------
# Vertex detection from polyline proximity (edges sharing exactly one face)
# ---------------------------------------------------------------------------

def compute_vertices_from_polyline_proximity(polyline_map, threshold=1e-3):
    """
    Find B-Rep vertices where polylines from edges sharing exactly one
    cluster index intersect.

    Uses KDTree point-to-point proximity between polyline point arrays.

    Parameters
    ----------
    polyline_map : dict (i, j) → list of (N, 3) arrays
    threshold    : proximity radius for polyline intersection detection

    Returns
    -------
    vertices       : (M, 3) float64 array
    vertex_edges   : list[set] of length M
    """
    edge_keys = list(polyline_map.keys())

    candidates = []  # (position, edge_key_a, edge_key_b)

    for a_idx in range(len(edge_keys)):
        for b_idx in range(a_idx + 1, len(edge_keys)):
            edge_a = edge_keys[a_idx]
            edge_b = edge_keys[b_idx]
            shared = set(edge_a) & set(edge_b)
            if len(shared) != 1:
                continue

            polys_a = polyline_map[edge_a]
            polys_b = polyline_map[edge_b]
            for poly_a in polys_a:
                if len(poly_a) < 2:
                    continue
                for poly_b in polys_b:
                    if len(poly_b) < 2:
                        continue
                    tree_b = cKDTree(poly_b)
                    dists, indices = tree_b.query(poly_a)
                    min_dist = float(dists.min())
                    print(f"  [proximity] {edge_a}∩{edge_b}: "
                          f"min_dist={min_dist:.6e}  "
                          f"pts_a={len(poly_a)} pts_b={len(poly_b)}")
                    close_mask = dists < threshold
                    if not np.any(close_mask):
                        continue
                    best = np.argmin(dists)
                    pos = 0.5 * (poly_a[best] + poly_b[indices[best]])
                    candidates.append((pos, edge_a, edge_b))

    if not candidates:
        print("[polyline-vertices] no proximity candidates found")
        return np.empty((0, 3), dtype=np.float64), []

    # Greedy clustering
    positions = np.array([c[0] for c in candidates], dtype=np.float64)
    used = np.zeros(len(positions), dtype=bool)
    vertices = []
    vertex_edges = []

    print(f"[polyline-vertices] {len(candidates)} raw candidates "
          f"from {len(polyline_map)} edges")

    for idx in range(len(positions)):
        if used[idx]:
            continue
        dists_arr = np.linalg.norm(positions - positions[idx], axis=1)
        close = (dists_arr < threshold) & ~used
        close_indices = np.where(close)[0]

        edges = set()
        for ci in close_indices:
            _, ea, eb = candidates[ci]
            edges.add(ea)
            edges.add(eb)

        if len(edges) < 2:
            used[close] = True
            continue

        pos = positions[close].mean(axis=0)
        v_idx = len(vertices)
        print(f"  v{v_idx}: ({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f})  "
              f"edges={sorted(edges)}  merged={len(close_indices)} candidates")
        vertices.append(pos)
        vertex_edges.append(edges)
        used[close] = True

    print(f"[polyline-vertices] {len(vertices)} vertices after clustering")
    verts_arr = (np.array(vertices, dtype=np.float64) if vertices
                 else np.empty((0, 3), dtype=np.float64))
    return verts_arr, vertex_edges


# ---------------------------------------------------------------------------
# Vertex detection: tripoint (triple-face convergence)
# ---------------------------------------------------------------------------

def _get_edge_key(i, j):
    """Canonical edge key with i < j."""
    return (min(i, j), max(i, j))


def compute_vertices_tripoint(polyline_map, threshold=0.01,
                               cluster_radius=5e-3):
    """
    Find B-Rep vertices from face triples using polyline proximity.

    A vertex is where 3 faces (i, j, k) meet. For each such triple where
    all three edge polylines exist, we sweep points on polyline (i,j) and
    find the point that minimizes max(dist_to_(i,k), dist_to_(j,k)).
    We repeat using each of the 3 polylines as the base and keep the best.

    Each edge (i,j) may have multiple polylines; all combinations are tried.

    Parameters
    ----------
    polyline_map   : dict (i, j) → list of (N, 3) arrays
    threshold      : max acceptable min-max-distance for a vertex candidate
    cluster_radius : greedy deduplication radius
    """
    from itertools import combinations

    # Collect all face indices that appear in any edge
    faces = set()
    for (i, j) in polyline_map:
        faces.add(i)
        faces.add(j)
    faces = sorted(faces)

    # Build set of edges with non-empty polylines for fast lookup
    active_edges = set()
    for key, polys in polyline_map.items():
        if any(len(p) >= 2 for p in polys):
            active_edges.add(key)

    # Pre-build KDTrees for each (edge, polyline_index)
    kdtrees = {}  # (edge_key, poly_idx) → (cKDTree, poly_array)
    for key, polys in polyline_map.items():
        for pi, poly in enumerate(polys):
            if len(poly) >= 2:
                kdtrees[(key, pi)] = (cKDTree(poly), poly)

    candidates = []  # (position, score, edge_ij, edge_ik, edge_jk)

    for i, j, k in combinations(faces, 3):
        e_ij = _get_edge_key(i, j)
        e_ik = _get_edge_key(i, k)
        e_jk = _get_edge_key(j, k)

        if not (e_ij in active_edges and e_ik in active_edges
                and e_jk in active_edges):
            continue

        best_triple_score = float("inf")

        # Try each of the 3 edges as the base polyline
        for base_edge, other1, other2 in [
            (e_ij, e_ik, e_jk),
            (e_ik, e_ij, e_jk),
            (e_jk, e_ij, e_ik),
        ]:
            base_polys = polyline_map[base_edge]
            for bp_idx, base_poly in enumerate(base_polys):
                if len(base_poly) < 2:
                    continue

                # Query base points against all polylines of other1 and other2
                for o1_idx, o1_poly in enumerate(polyline_map[other1]):
                    if len(o1_poly) < 2:
                        continue
                    tree1, _ = kdtrees[(other1, o1_idx)]

                    for o2_idx, o2_poly in enumerate(polyline_map[other2]):
                        if len(o2_poly) < 2:
                            continue
                        tree2, _ = kdtrees[(other2, o2_idx)]

                        d1, _ = tree1.query(base_poly)
                        d2, _ = tree2.query(base_poly)
                        max_d = np.maximum(d1, d2)
                        best_idx = np.argmin(max_d)
                        score = float(max_d[best_idx])
                        best_triple_score = min(best_triple_score, score)

                        if score < threshold:
                            candidates.append((
                                base_poly[best_idx].copy(),
                                score, e_ij, e_ik, e_jk,
                            ))

        if best_triple_score >= threshold:
            print(f"  [tripoint] REJECTED triple ({i},{j},{k})  "
                  f"best_score={best_triple_score:.6e}  "
                  f"edges={e_ij},{e_ik},{e_jk}")

    if not candidates:
        print("[tripoint] no vertex candidates found")
        return np.empty((0, 3), dtype=np.float64), []

    # Sort by score (best first) for greedy clustering
    candidates.sort(key=lambda c: c[1])

    positions = np.array([c[0] for c in candidates], dtype=np.float64)
    used = np.zeros(len(positions), dtype=bool)
    vertices = []
    vertex_edges = []

    print(f"[tripoint] {len(candidates)} raw candidates "
          f"from {len(polyline_map)} edges")

    for idx in range(len(positions)):
        if used[idx]:
            continue
        dists_arr = np.linalg.norm(positions - positions[idx], axis=1)
        close = (dists_arr < cluster_radius) & ~used
        close_indices = np.where(close)[0]

        edges = set()
        for ci in close_indices:
            _, _, e1, e2, e3 = candidates[ci]
            edges.add(e1)
            edges.add(e2)
            edges.add(e3)

        if len(edges) < 2:
            used[close] = True
            continue

        pos = positions[close].mean(axis=0)
        best_score = min(candidates[ci][1] for ci in close_indices)
        v_idx = len(vertices)
        print(f"  v{v_idx}: ({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f})  "
              f"score={best_score:.6e}  "
              f"edges={sorted(edges)}  merged={len(close_indices)}")
        vertices.append(pos)
        vertex_edges.append(edges)
        used[close] = True

    print(f"[tripoint] {len(vertices)} vertices after clustering")
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

            # k >= 2: trim polyline to outermost vertices
            first_pi, first_vi = incident[0]
            last_pi, last_vi = incident[-1]

            trimmed = poly[first_pi:last_pi + 1].copy()
            if len(trimmed) < 2:
                continue

            # Snap endpoints to exact vertex positions
            trimmed[0] = vertices[first_vi]
            trimmed[-1] = vertices[last_vi]

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

            # k > 2: project interior vertices onto BSpline, split into arcs
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
