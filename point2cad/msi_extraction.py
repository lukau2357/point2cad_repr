"""
Topology extraction from fitted surface meshes using libigl.

Replaces the PyVista pairwise intersection + point-matching vertex detection
pipeline with a single global self-intersection resolution step.  After
resolution, intersection curves and B-Rep vertices are extracted purely from
mesh topology (face provenance + edge adjacency), with no distance thresholds.

Pipeline:
  1. Merge all surface meshes into one, tracking face → surface provenance
  2. Resolve self-intersections (libigl/CGAL)
  3. Deduplicate vertices via the IM map
  4. Find intersection edges (shared by faces from different surfaces)
  5. Identify junction vertices (vertices in edges from 2+ surface pairs)
  6. Chain edges into polylines per pair, merge nearby endpoints, filter short
  7. Build B-Rep vertices directly from junction_set (no endpoint re-detection)
  8. Split polylines at B-Rep vertices → arcs, fit B-spline per arc

Returns edge_arcs, vertices, vertex_edges — the same format as
build_arcs_from_polylines, so the output plugs directly into the oracle filter.
"""

import numpy as np
from collections import defaultdict

try:
    from igl.copyleft.cgal import remesh_self_intersections
    HAS_LIBIGL_CGAL = True
except ImportError:
    HAS_LIBIGL_CGAL = False

try:
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.TColgp import TColgp_Array1OfPnt
    from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
    from OCC.Core.GeomAbs import GeomAbs_C2
    from OCC.Core.Geom import Geom_TrimmedCurve
    HAS_OCC = True
except ImportError:
    HAS_OCC = False

_CLOSURE_TOL = 5e-3


def _fit_bspline(pts):
    """Fit a BSpline curve to an ordered point array."""
    if not HAS_OCC:
        raise ImportError("pythonocc-core is required for B-spline fitting")

    is_closed = len(pts) >= 3 and np.linalg.norm(pts[0] - pts[-1]) < _CLOSURE_TOL
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
            pass
    return curve


def o3d_mesh_to_numpy(o3d_mesh):
    """Convert an Open3D TriangleMesh to (V, F) numpy arrays.

    Filters double-sided duplicates: triangulate_and_mesh() creates 4 triangles
    per quad cell (indices 0,1 are original, 2,3 are reversed duplicates).
    We keep only the original-orientation triangles (i % 4 < 2).
    """
    vertices = np.asarray(o3d_mesh.vertices, dtype=np.float64)
    triangles = np.asarray(o3d_mesh.triangles, dtype=np.int64)
    mask = np.arange(len(triangles)) % 4 < 2
    triangles = triangles[mask]
    return vertices, triangles


def _merge_meshes(mesh_list):
    """Merge list of (V, F) tuples into a single mesh with face provenance.

    Returns
    -------
    V : (N, 3) float64 — merged vertices
    F : (M, 3) int64   — merged faces (vertex indices adjusted)
    face_sources : (M,) int32 — surface index for each face
    """
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


def _chain_edges_into_polylines(edges, vertices):
    """Chain unordered mesh edges into ordered polylines.

    Splits at degree-1 and degree-3+ nodes (natural graph terminators).
    Junction vertices are NOT split here — they are used later for arc splitting.

    Parameters
    ----------
    edges        : list of (v0, v1) vertex index pairs
    vertices     : (N, 3) array — vertex positions

    Returns
    -------
    polylines       : list of (K, 3) arrays — ordered point sequences
    polyline_vindices : list of list[int] — vertex indices for each polyline
    """
    if not edges:
        return [], []

    graph = defaultdict(set)
    for v0, v1 in edges:
        graph[v0].add(v1)
        graph[v1].add(v0)

    visited_edges = set()
    polylines = []
    polyline_vindices = []

    def _edge_key(a, b):
        return (min(a, b), max(a, b))

    # Start from degree-1 (endpoints) and degree-3+ nodes
    start_nodes = [v for v in graph if len(graph[v]) != 2]

    for start in start_nodes:
        for neighbor in list(graph[start]):
            ek = _edge_key(start, neighbor)
            if ek in visited_edges:
                continue

            chain = [start]
            current = start
            next_v = neighbor

            while True:
                ek = _edge_key(current, next_v)
                if ek in visited_edges:
                    break
                visited_edges.add(ek)
                chain.append(next_v)
                current = next_v

                if len(graph[current]) != 2:
                    break

                remaining = graph[current] - {chain[-2]}
                if not remaining:
                    break
                next_v = remaining.pop()

            if len(chain) >= 2:
                polylines.append(vertices[chain])
                polyline_vindices.append(chain)

    # Handle remaining closed loops (all degree-2 vertices)
    remaining_edges = set()
    for v0, v1 in edges:
        ek = _edge_key(v0, v1)
        if ek not in visited_edges:
            remaining_edges.add(ek)

    if remaining_edges:
        sub_graph = defaultdict(set)
        for v0, v1 in remaining_edges:
            sub_graph[v0].add(v1)
            sub_graph[v1].add(v0)

        visited_nodes = set()
        for start in sub_graph:
            if start in visited_nodes:
                continue
            chain = [start]
            visited_nodes.add(start)
            current = start
            while True:
                neighbors = sub_graph[current] - visited_nodes
                if not neighbors:
                    break
                current = neighbors.pop()
                visited_nodes.add(current)
                chain.append(current)
            chain.append(start)  # close the loop
            if len(chain) >= 3:
                polylines.append(vertices[chain])
                polyline_vindices.append(chain)

    return polylines, polyline_vindices


def _merge_nearby_polylines(polylines, polyline_vindices, merge_tol):
    """Merge polylines whose endpoints are within merge_tol of each other.

    Heals small gaps from mesh discretization artifacts.
    """
    if len(polylines) <= 1:
        return polylines, polyline_vindices

    merged = True
    while merged:
        merged = False
        for i in range(len(polylines)):
            for j in range(i + 1, len(polylines)):
                pi = polylines[i]
                pj = polylines[j]
                pairs = [
                    ("end_start",   pi[-1], pj[0]),
                    ("end_end",     pi[-1], pj[-1]),
                    ("start_start", pi[0],  pj[0]),
                    ("start_end",   pi[0],  pj[-1]),
                ]
                for tag, pa, pb in pairs:
                    if np.linalg.norm(pa - pb) < merge_tol:
                        vi = polyline_vindices[i]
                        vj = polyline_vindices[j]
                        if tag == "end_start":
                            combined = np.concatenate([pi, pj], axis=0)
                            combined_vi = vi + vj
                        elif tag == "end_end":
                            combined = np.concatenate([pi, pj[::-1]], axis=0)
                            combined_vi = vi + vj[::-1]
                        elif tag == "start_start":
                            combined = np.concatenate([pi[::-1], pj], axis=0)
                            combined_vi = vi[::-1] + vj
                        elif tag == "start_end":
                            combined = np.concatenate([pj, pi], axis=0)
                            combined_vi = vj + vi
                        polylines[i] = combined
                        polyline_vindices[i] = combined_vi
                        polylines.pop(j)
                        polyline_vindices.pop(j)
                        merged = True
                        break
                if merged:
                    break
            if merged:
                break

    return polylines, polyline_vindices



def _split_polylines_into_arcs(polyline_map, polyline_vindices_map,
                                junction_map, vertices, vertex_edges, SV):
    """Split polylines at junction vertices and fit B-splines → edge_arcs.

    Parameters
    ----------
    polyline_map          : dict (i,j) -> list of (N,3) arrays
    polyline_vindices_map : dict (i,j) -> list of list[int]
    junction_map          : dict resolved_vert_idx -> brep_vertex_idx
    vertices              : (M, 3) B-Rep vertex positions
    vertex_edges          : list[set] — edge sets per vertex
    SV                    : resolved mesh vertices

    Returns
    -------
    edge_arcs : dict (i,j) -> list[arc_dict]
    """
    edge_arcs = {}

    for edge_key, polylines in polyline_map.items():
        vindices_list = polyline_vindices_map[edge_key]
        arcs_for_edge = []

        for poly_idx, (poly, vindices) in enumerate(
                zip(polylines, vindices_list)):
            if len(poly) < 2:
                continue

            is_closed = (len(poly) >= 3 and
                         np.linalg.norm(poly[0] - poly[-1]) < _CLOSURE_TOL)

            # Find junction vertices that lie on this polyline
            # (they can be at endpoints or interior if polyline passes through)
            junction_positions = []  # (position_in_vindices, brep_idx)
            for pos_in_chain, vi in enumerate(vindices):
                if vi in junction_map:
                    junction_positions.append((pos_in_chain, junction_map[vi]))

            if is_closed and len(junction_positions) == 0:
                # Closed polyline with no junctions → single closed arc
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

            if not is_closed and len(junction_positions) < 2:
                # Open polyline with 0 or 1 junctions — not enough to bound an arc
                continue

            # Sort by position along polyline
            junction_positions.sort(key=lambda x: x[0])

            if is_closed:
                # Closed polyline with junctions: split at each junction,
                # including wrap-around arc from last junction to first
                n_jv = len(junction_positions)
                for m in range(n_jv):
                    pos_a, v_a = junction_positions[m]
                    pos_b, v_b = junction_positions[(m + 1) % n_jv]

                    if pos_b > pos_a:
                        sub_poly = poly[pos_a:pos_b + 1]
                    else:
                        # Wrap-around: skip duplicated closure point
                        sub_poly = np.concatenate([
                            poly[pos_a:],
                            poly[1:pos_b + 1],
                        ], axis=0)

                    # Snap endpoints to exact vertex positions
                    sub_poly = sub_poly.copy()
                    sub_poly[0] = vertices[v_a]
                    sub_poly[-1] = vertices[v_b]

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
                # Open polyline: trim to outermost junctions, split between them
                first_pos, first_v = junction_positions[0]
                last_pos, last_v = junction_positions[-1]

                for m in range(len(junction_positions) - 1):
                    pos_a, v_a = junction_positions[m]
                    pos_b, v_b = junction_positions[m + 1]
                    sub_poly = poly[pos_a:pos_b + 1].copy()

                    # Snap endpoints to exact vertex positions
                    sub_poly[0] = vertices[v_a]
                    sub_poly[-1] = vertices[v_b]

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

    return edge_arcs


def extract_topology_msi(mesh_list, min_polyline_points=3, merge_tol=1e-3,
                          adj=None):
    """
    Extract B-Rep topology from fitted surface meshes and produce arcs directly.

    Parameters
    ----------
    mesh_list : list of (V, F) tuples — one per surface/cluster
    min_polyline_points : int — discard polylines shorter than this
    merge_tol : float — static tolerance for merging polyline endpoints
    adj : (n, n) bool array, optional — adjacency matrix; if provided,
          only edges between adjacent surfaces are kept

    Returns
    -------
    edge_arcs     : dict (i,j) -> list[arc_dict]
    vertices      : (M, 3) float64 array — B-Rep vertex positions
    vertex_edges  : list[set] of length M — set of edge_key tuples per vertex
    polyline_map  : dict (i,j) -> list of (N,3) arrays — for visualization
    """
    if not HAS_LIBIGL_CGAL:
        raise ImportError("libigl with CGAL copyleft module is required. "
                          "Install with: pip install libigl (built with "
                          "-DLIBIGL_COPYLEFT_CGAL=ON)")

    # --- Step 1: Merge all meshes ---
    V, F, face_sources = _merge_meshes(mesh_list)
    n_surfaces = len(mesh_list)
    print(f"[msi] Merged {n_surfaces} meshes: "
          f"{len(V)} vertices, {len(F)} faces")

    # --- Step 2: Resolve self-intersections ---
    print("[msi] Resolving self-intersections (CGAL)...")
    SV, SF, IF, J, IM = remesh_self_intersections(V, F)
    print(f"[msi] Resolved: {len(SV)} vertices, {len(SF)} faces")

    # --- Step 3: Deduplicate vertices via IM ---
    SF = IM[SF]
    used_verts, inv = np.unique(SF.ravel(), return_inverse=True)
    SF = inv.reshape(-1, 3)
    SV = SV[used_verts]

    # Remap IM to compacted indices for vertex index tracking
    im_remap = np.full(IM.max() + 1, -1, dtype=np.int64)
    im_remap[used_verts] = np.arange(len(used_verts))

    # Face provenance: resolved face → original face → surface
    resolved_face_sources = face_sources[J]
    print(f"[msi] After dedup: {len(SV)} vertices, {len(SF)} faces")

    # --- Step 4: Find intersection edges ---
    edge_to_faces = defaultdict(list)
    for fi in range(len(SF)):
        f = SF[fi]
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            e = (min(int(f[a]), int(f[b])), max(int(f[a]), int(f[b])))
            edge_to_faces[e].append(fi)

    intersection_edges_by_pair = defaultdict(list)
    junction_edges = []  # edges touching 3+ surfaces (junction artifacts)

    for edge, face_list in edge_to_faces.items():
        surfaces = set(int(resolved_face_sources[fi]) for fi in face_list)
        if len(surfaces) < 2:
            continue

        if len(surfaces) >= 3:
            # Non-manifold junction edge — don't assign to any pair's polyline,
            # but record for junction vertex detection
            junction_edges.append((edge, surfaces))
            continue

        sorted_surfaces = sorted(int(s) for s in surfaces)
        pair = (sorted_surfaces[0], sorted_surfaces[1])
        if adj is not None and not adj[pair[0], pair[1]]:
            continue
        intersection_edges_by_pair[pair].append(edge)

    n_int_edges = sum(len(e) for e in intersection_edges_by_pair.values())
    print(f"[msi] {n_int_edges} intersection edges across "
          f"{len(intersection_edges_by_pair)} surface pairs"
          f" ({len(junction_edges)} junction edges excluded)")

    # --- Step 5: Identify junction vertices from edge topology ---
    # A vertex appearing in intersection edges from 2+ surface pairs
    # is a junction where chains must be split.
    # Also include vertices from junction edges (touching 3+ surfaces).
    vertex_pairs = defaultdict(set)
    for pair, edges in intersection_edges_by_pair.items():
        for v0, v1 in edges:
            vertex_pairs[v0].add(pair)
            vertex_pairs[v1].add(pair)
    for (v0, v1), surfaces in junction_edges:
        sorted_s = sorted(surfaces)
        for ai in range(len(sorted_s)):
            for bi in range(ai + 1, len(sorted_s)):
                vertex_pairs[v0].add((sorted_s[ai], sorted_s[bi]))
                vertex_pairs[v1].add((sorted_s[ai], sorted_s[bi]))
    junction_set = {v for v, pairs in vertex_pairs.items() if len(pairs) >= 2}
    print(f"[msi] {len(junction_set)} junction vertices (edges from 2+ pairs)")

    # --- Step 6: Chain edges into polylines per pair ---
    polyline_map = {}
    polyline_vindices_map = {}

    for pair, edges in intersection_edges_by_pair.items():
        polylines, vindices = _chain_edges_into_polylines(edges, SV)

        # Merge nearby endpoints (heal discretization gaps)
        polylines, vindices = _merge_nearby_polylines(
            polylines, vindices, merge_tol)

        # Filter short polylines
        keep = [i for i, p in enumerate(polylines)
                if len(p) >= min_polyline_points]
        polylines = [polylines[i] for i in keep]
        vindices = [vindices[i] for i in keep]

        if polylines:
            polyline_map[pair] = polylines
            polyline_vindices_map[pair] = vindices
            print(f"  edge {pair}: {len(polylines)} polyline(s), "
                  f"lengths={[len(p) for p in polylines]}")

    print(f"[msi] {len(polyline_map)} edges in polyline_map")

    # --- Step 7: Build B-Rep vertices directly from junction_set ---
    # vertex_pairs already maps resolved_vert_idx → set of pairs.
    # Only keep junctions that appear on surviving polylines.
    polyline_verts = set()
    for vindices_list in polyline_vindices_map.values():
        for vindices in vindices_list:
            polyline_verts.update(vindices)

    junction_map = {}
    vertices = []
    vertex_edges = []
    for vi in sorted(junction_set):
        if vi not in polyline_verts:
            continue
        brep_idx = len(vertices)
        junction_map[vi] = brep_idx
        vertices.append(SV[vi])
        vertex_edges.append(vertex_pairs[vi])
    vertices = (np.array(vertices, dtype=np.float64)
                if vertices
                else np.empty((0, 3), dtype=np.float64))

    print(f"[msi] {len(vertices)} B-Rep vertices")
    for vi in range(len(vertices)):
        print(f"  v{vi}: ({vertices[vi][0]:.6f}, "
              f"{vertices[vi][1]:.6f}, "
              f"{vertices[vi][2]:.6f})  "
              f"edges={sorted(vertex_edges[vi])}")

    # --- Step 8: Split polylines at junctions → arcs ---
    edge_arcs = _split_polylines_into_arcs(
        polyline_map, polyline_vindices_map,
        junction_map, vertices, vertex_edges, SV)

    return edge_arcs, vertices, vertex_edges, polyline_map
