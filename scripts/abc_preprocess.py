import argparse
import glob
import os
import sys
import yaml
import numpy as np
import json
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from point2cad.color_config import get_surface_color

try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopExp      import TopExp_Explorer
    from OCC.Core.TopAbs      import TopAbs_FACE
    from OCC.Core             import topods
    from OCC.Core.BRep        import BRep_Tool
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    _OCC_AVAILABLE = True
except ImportError:
    _OCC_AVAILABLE = False

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_model_files(abc_dir, model_id):
    obj_dir  = os.path.join(abc_dir, "obj",  model_id)
    feat_dir = os.path.join(abc_dir, "feat", model_id)

    obj_files  = sorted(glob.glob(os.path.join(obj_dir,  "*_trimesh_*.obj")))
    feat_files = sorted(glob.glob(os.path.join(feat_dir, "*_features_*.yml")))

    if len(obj_files) == 0 or len(feat_files) == 0:
        return None, None
    return obj_files[0], feat_files[0]


def find_step_file(abc_dir, model_id):
    """Return the path to the STEP file for model_id, or None if not found."""
    # The STEP batch directory may be named abc_XXXX_step_vYY
    for entry in os.listdir(abc_dir):
        if "step" in entry and os.path.isdir(os.path.join(abc_dir, entry)):
            candidate = os.path.join(abc_dir, entry, model_id)
            if os.path.isdir(candidate):
                hits = sorted(glob.glob(os.path.join(candidate, "*_step_*.step")))
                if hits:
                    return hits[0]
    return None


# ---------------------------------------------------------------------------
# OBJ / FEAT loading
# ---------------------------------------------------------------------------

def load_obj(path):
    vertices = []
    faces    = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                # Handles f v//vn and f v/vt/vn and f v formats
                face_verts = [int(p.split("/")[0]) - 1 for p in parts]
                faces.append(face_verts)
    return np.array(vertices, dtype=np.float64), faces


def load_features(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("surfaces", []), data.get("curves", [])


def build_face_to_surface(surfaces, num_faces):
    """Map each OBJ face index to a surface ID. Returns array of length
    num_faces with -1 for unassigned faces."""
    face_labels = np.full(num_faces, -1, dtype=np.int32)
    for sid, surface in enumerate(surfaces):
        for fi in surface.get("face_indices", []):
            if fi < num_faces:
                face_labels[fi] = sid
    return face_labels


# ---------------------------------------------------------------------------
# OBJ-based sampling (area-weighted barycentric)
# ---------------------------------------------------------------------------

def _sample_surface_obj(vertices, faces, face_areas, face_labels, sid, n_pts, rng):
    """Sample n_pts from a single surface via area-weighted barycentric sampling."""
    mask  = face_labels == sid
    areas = face_areas * mask
    probs = areas / areas.sum()
    sampled_face_ids = rng.choice(len(faces), size=n_pts, p=probs)

    r1      = rng.random(n_pts)
    r2      = rng.random(n_pts)
    sqrt_r1 = np.sqrt(r1)
    w0      = 1 - sqrt_r1
    w1      = sqrt_r1 * (1 - r2)
    w2      = sqrt_r1 * r2

    pts = np.zeros((n_pts, 3))
    for i, fi in enumerate(sampled_face_ids):
        face   = faces[fi]
        pts[i] = w0[i] * vertices[face[0]] + w1[i] * vertices[face[1]] + w2[i] * vertices[face[2]]
    return pts


def sample_points_from_mesh(vertices, faces, face_labels, num_points,
                            min_points_per_surface, rng):
    """Sample num_points globally (area-weighted), then upsample surfaces
    below min_points_per_surface."""
    face_areas = np.zeros(len(faces))
    for i, face in enumerate(faces):
        if len(face) < 3:
            continue
        v0, v1, v2    = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        face_areas[i] = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

    valid             = face_labels >= 0
    face_areas_valid  = face_areas * valid
    total_area        = face_areas_valid.sum()
    if total_area == 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int32)

    # Global area-weighted sampling
    probs            = face_areas_valid / total_area
    sampled_face_ids = rng.choice(len(faces), size=num_points, p=probs)

    r1      = rng.random(num_points)
    r2      = rng.random(num_points)
    sqrt_r1 = np.sqrt(r1)
    w0      = 1 - sqrt_r1
    w1      = sqrt_r1 * (1 - r2)
    w2      = sqrt_r1 * r2

    points = np.zeros((num_points, 3))
    labels = np.zeros(num_points, dtype=np.int32)
    for i, fi in enumerate(sampled_face_ids):
        face      = faces[fi]
        points[i] = w0[i] * vertices[face[0]] + w1[i] * vertices[face[1]] + w2[i] * vertices[face[2]]
        labels[i] = face_labels[fi]

    # Upsample surfaces below minimum
    unique_labels, counts = np.unique(labels, return_counts=True)
    extra_points, extra_labels = [], []
    for sid, count in zip(unique_labels, counts):
        if count < min_points_per_surface:
            deficit = (min_points_per_surface - count
                       + rng.integers(0, min_points_per_surface // 5 + 1))
            pts = _sample_surface_obj(vertices, faces, face_areas,
                                      face_labels, sid, deficit, rng)
            extra_points.append(pts)
            extra_labels.append(np.full(deficit, sid, dtype=np.int32))

    if extra_points:
        points = np.concatenate([points] + extra_points)
        labels = np.concatenate([labels] + extra_labels)

    _print_cluster_stats(labels)
    return points, labels


# ---------------------------------------------------------------------------
# STEP-based sampling (exact parametric surface via OCC)
# ---------------------------------------------------------------------------

def _evaluate_face_grid(geom_surf, u0, u1, v0, v1, grid_res):
    """
    Evaluate geom_surf on a regular (grid_res+1) × (grid_res+1) UV grid.
    Returns an array of shape (grid_res+1, grid_res+1, 3).
    """
    us  = np.linspace(u0, u1, grid_res + 1)
    vs  = np.linspace(v0, v1, grid_res + 1)
    pts = np.zeros((grid_res + 1, grid_res + 1, 3))
    for i, u in enumerate(us):
        for j, v in enumerate(vs):
            p          = geom_surf.Value(u, v)
            pts[i, j]  = (p.X(), p.Y(), p.Z())
    return pts


def _face_triangles(pts3d):
    """
    Decompose the UV grid into two triangles per cell and return their 3D areas.

    For each grid cell (i, j) the four corners in 3D are:
      A = pts3d[i,   j  ]   bottom-left
      B = pts3d[i+1, j  ]   bottom-right
      C = pts3d[i,   j+1]   top-left
      D = pts3d[i+1, j+1]   top-right

    The cell is split along the B-C diagonal:
      Triangle 1: (A, B, C)  —  area = 0.5 * |AB × AC|
      Triangle 2: (B, D, C)  —  area = 0.5 * |BD × BC|

    Returns two (grid_res, grid_res) arrays: area1, area2.
    The factor 0.5 is the standard triangle-area formula (parallelogram / 2);
    it is a universal multiplicative constant and cancels in normalised
    probabilities, but is kept so the arrays carry true surface-area values.
    """
    A = pts3d[:-1, :-1]   # bottom-left
    B = pts3d[1:,  :-1]   # bottom-right
    C = pts3d[:-1, 1:]    # top-left
    D = pts3d[1:,  1:]    # top-right

    area1 = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=-1)
    area2 = 0.5 * np.linalg.norm(np.cross(D - B, C - B), axis=-1)
    return area1, area2


def _sample_from_grid(pts3d, area1, area2, n_pts, rng):
    """
    Sample n_pts points uniformly (area-weighted) from a triangulated UV grid.

    All triangles from both types are pooled into a single flat distribution;
    no explicit 0.5 factor for choosing between the two triangle types per cell
    is needed — their relative areas already determine selection probability.

    Returns (n_pts, 3) float array, or (0, 3) for degenerate faces.
    """
    if n_pts == 0:
        return np.zeros((0, 3))

    all_areas = np.concatenate([area1.ravel(), area2.ravel()])
    total     = all_areas.sum()
    if total < 1e-12:
        return np.zeros((0, 3))

    probs    = all_areas / total
    n_cells  = area1.size                      # grid_res * grid_res
    tri_ids  = rng.choice(len(all_areas), size=n_pts, p=probs)
    is_type1 = tri_ids < n_cells
    cell_ids = np.where(is_type1, tri_ids, tri_ids - n_cells)
    ci       = cell_ids // area1.shape[1]
    cj       = cell_ids %  area1.shape[1]

    # Uniform barycentric sampling within the chosen triangle
    r1      = rng.random(n_pts)
    sqrt_r1 = np.sqrt(r1)
    r2      = rng.random(n_pts)
    w0      = 1.0 - sqrt_r1
    w1      = sqrt_r1 * (1.0 - r2)
    w2      = sqrt_r1 * r2

    # Vectorised vertex lookup — no Python loop over samples
    vA = pts3d[ci,     cj    ]   # bottom-left
    vB = pts3d[ci + 1, cj    ]   # bottom-right
    vC = pts3d[ci,     cj + 1]   # top-left  (shared by both triangle types)
    vD = pts3d[ci + 1, cj + 1]   # top-right

    # Triangle 1: (A, B, C)   Triangle 2: (B, D, C)
    mask  = is_type1[:, None]
    vert0 = np.where(mask, vA, vB)
    vert1 = np.where(mask, vB, vD)
    vert2 = vC                     # C is the shared vertex in both triangles

    return w0[:, None] * vert0 + w1[:, None] * vert1 + w2[:, None] * vert2


def sample_points_from_step(step_path, num_points, min_points_per_surface,
                            rng, grid_res=50):
    """
    Sample num_points from a STEP B-Rep model using exact parametric surfaces.

    Each TopoDS_Face is treated as one surface cluster (cluster ID = face index).
    Points are sampled globally area-weighted via UV-grid triangulation, then
    clusters below min_points_per_surface are upsampled.

    Requires pythonocc-core (OCC). Run inside the Docker container.

    Returns:
        points : (N, 3) float32 array
        labels : (N,)   int32 array  (face index = cluster ID)
    """
    if not _OCC_AVAILABLE:
        print("ERROR: pythonocc-core is not installed. "
              "The STEP sampler requires OCC — run inside the Docker container.")
        sys.exit(1)

    reader = STEPControl_Reader()
    if reader.ReadFile(step_path) != 1:
        print(f"  STEP read error for {step_path}")
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int32)
    reader.TransferRoots()
    shape = reader.OneShape()

    # Collect all faces in a stable order
    exp        = TopExp_Explorer(shape, TopAbs_FACE)
    face_list  = []
    while exp.More():
        face_list.append(topods.Face(exp.Current()))
        exp.Next()

    if not face_list:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int32)

    # Build UV grids and compute triangle areas for every face
    face_grids  = []
    face_area1s = []
    face_area2s = []
    face_totals = []

    for face in face_list:
        adaptor   = BRepAdaptor_Surface(face)
        u0, u1    = adaptor.FirstUParameter(), adaptor.LastUParameter()
        v0, v1    = adaptor.FirstVParameter(), adaptor.LastVParameter()
        geom_surf = BRep_Tool.Surface(face)

        pts3d     = _evaluate_face_grid(geom_surf, u0, u1, v0, v1, grid_res)
        a1, a2    = _face_triangles(pts3d)

        face_grids.append(pts3d)
        face_area1s.append(a1)
        face_area2s.append(a2)
        face_totals.append(a1.sum() + a2.sum())

    face_totals = np.array(face_totals)
    total_area  = face_totals.sum()
    if total_area < 1e-12:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int32)

    # Global area-weighted allocation: draw face counts from a multinomial
    face_probs  = face_totals / total_area
    face_counts = rng.multinomial(num_points, face_probs)

    all_points, all_labels = [], []
    for fid, (pts3d, a1, a2, n) in enumerate(
            zip(face_grids, face_area1s, face_area2s, face_counts)):
        pts = _sample_from_grid(pts3d, a1, a2, int(n), rng)
        if len(pts):
            all_points.append(pts)
            all_labels.append(np.full(len(pts), fid, dtype=np.int32))

    # Upsample faces below minimum
    counts_per_face = face_counts.copy()
    for fid in range(len(face_list)):
        if counts_per_face[fid] < min_points_per_surface:
            deficit = (min_points_per_surface - counts_per_face[fid]
                       + rng.integers(0, min_points_per_surface // 5 + 1))
            pts = _sample_from_grid(face_grids[fid], face_area1s[fid],
                                    face_area2s[fid], int(deficit), rng)
            if len(pts):
                all_points.append(pts)
                all_labels.append(np.full(len(pts), fid, dtype=np.int32))

    if not all_points:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int32)

    points = np.concatenate(all_points).astype(np.float32)
    labels = np.concatenate(all_labels)
    _print_cluster_stats(labels)
    return points, labels


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _print_cluster_stats(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"  Cluster stats: {len(unique_labels)} surfaces, "
          f"min={counts.min()}, max={counts.max()}, "
          f"mean={counts.mean():.0f}, total={len(labels)} points")


def export_xyzc(points, labels, output_path):
    data = np.column_stack([points, labels.astype(np.float64)])
    np.savetxt(output_path, data)
    print(f"  Saved {len(points)} points to {output_path}")


# ---------------------------------------------------------------------------
# Analysis / visualisation helpers (OBJ+feat only)
# ---------------------------------------------------------------------------

ABC_TYPE_TO_COLOR_KEY = {
    "Plane":    "plane",
    "Sphere":   "sphere",
    "Cylinder": "cylinder",
    "Cone":     "cone",
}


def surface_color_for_label(surfaces, label):
    abc_type  = surfaces[label]["type"]
    color_key = ABC_TYPE_TO_COLOR_KEY.get(abc_type, "inr")
    return get_surface_color(color_key)


def analyze_model(abc_dir, model_id):
    obj_path, feat_path = find_model_files(abc_dir, model_id)
    if obj_path is None:
        print(f"Model {model_id}: OBJ or features file not found.")
        return None

    vertices, faces   = load_obj(obj_path)
    surfaces, curves  = load_features(feat_path)
    surface_types     = Counter(s["type"] for s in surfaces)
    curve_types       = Counter(c["type"] for c in curves)
    face_labels       = build_face_to_surface(surfaces, len(faces))
    unlabeled         = int((face_labels == -1).sum())

    print(f"Model: {model_id}")
    print(f"  OBJ: {os.path.basename(obj_path)}")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Faces: {len(faces)}")
    print(f"  Surfaces: {len(surfaces)}")
    for stype, count in surface_types.most_common():
        print(f"    {stype}: {count}")
    print(f"  Curves: {len(curves)}")
    for ctype, count in curve_types.most_common():
        print(f"    {ctype}: {count}")
    print(f"  Unlabeled faces: {unlabeled}/{len(faces)}")

    return vertices, faces, surfaces, curves, face_labels


def visualize_model(vertices, faces, face_labels, surfaces, points, labels):
    import open3d as o3d
    import time

    colors = np.array([surface_color_for_label(surfaces, label) for label in labels])
    pcd    = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    mesh      = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(faces))

    vertex_labels = np.full(len(vertices), -1, dtype=np.int32)
    for fi, face in enumerate(faces):
        label = face_labels[fi]
        if label < 0:
            continue
        for vi in face:
            if vertex_labels[vi] < 0:
                vertex_labels[vi] = label

    vertex_colors = np.ones((len(vertices), 3)) * 0.7
    for vi in range(len(vertices)):
        if vertex_labels[vi] >= 0:
            vertex_colors[vi] = surface_color_for_label(surfaces, vertex_labels[vi])
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    mesh.compute_vertex_normals()

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window("Sampled point cloud", width=640, height=720, left=0,   top=50)
    vis1.add_geometry(pcd)
    vis1.get_render_option().point_size = 2.0

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window("Original mesh",       width=640, height=720, left=640, top=50)
    vis2.add_geometry(mesh)
    vis2.get_render_option().mesh_show_back_face = True

    running1 = running2 = True
    while running1 and running2:
        running1 = vis1.poll_events(); vis1.update_renderer()
        running2 = vis2.poll_events(); vis2.update_renderer()
        import time as _t; _t.sleep(0.01)

    vis1.destroy_window()
    vis2.destroy_window()


def list_available_models(abc_dir, sampler="step", max_display=20):
    if sampler == "step":
        # Enumerate model IDs from the STEP batch directory
        step_batch = None
        for entry in os.listdir(abc_dir):
            if "step" in entry and os.path.isdir(os.path.join(abc_dir, entry)):
                step_batch = os.path.join(abc_dir, entry)
                break
        if step_batch is None:
            print("No STEP batch directory found.")
            return []
        complete = sorted(
            d for d in os.listdir(step_batch)
            if os.path.isdir(os.path.join(step_batch, d))
            and glob.glob(os.path.join(step_batch, d, "*_step_*.step"))
        )
        print(f"Models with STEP files: {len(complete)}")
    else:
        obj_dir  = os.path.join(abc_dir, "obj")
        feat_dir = os.path.join(abc_dir, "feat")
        obj_ids  = set(d for d in os.listdir(obj_dir)
                       if os.path.isdir(os.path.join(obj_dir, d)))
        feat_ids = set(d for d in os.listdir(feat_dir)
                       if os.path.isdir(os.path.join(feat_dir, d)))
        complete = sorted(
            mid for mid in obj_ids & feat_ids
            if find_model_files(abc_dir, mid)[0] is not None
        )
        print(f"Models with both OBJ and features: {len(complete)}")

    print(f"  IDs: {complete[:max_display]}{'...' if len(complete) > max_display else ''}")

    os.makedirs("../output", exist_ok=True)
    with open("../output/tmp_models.json", "w", encoding="utf-8") as f:
        json.dump(complete, f, indent=4)
    return complete


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ABC dataset preprocessing for Point2CAD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--abc_dir",   type=str, required=True,
                        help="Path to ABC dataset root")
    parser.add_argument("--model_id",  type=str, default=None,
                        help="Analyze / export a single model")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for .xyzc files")
    parser.add_argument("--num_points", type=int, default=10000,
                        help="Total number of points to sample (area-weighted)")
    parser.add_argument("--min_points_per_surface", type=int, default=500,
                        help="Minimum points per surface; undersized clusters are upsampled")
    parser.add_argument("--sampler", type=str, default="step",
                        choices=["step", "obj"],
                        help="Sampling back-end: 'step' uses exact OCC parametric surfaces "
                             "(requires pythonocc, all 10k models); "
                             "'obj' uses OBJ mesh barycentric sampling "
                             "(no OCC needed, ~7.1k models with feat annotations)")
    parser.add_argument("--grid_res", type=int, default=50,
                        help="UV grid resolution for STEP sampler "
                             "(higher = more accurate area estimates, slower)")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the sampled point cloud (obj sampler only)")
    parser.add_argument("--list",  action="store_true",
                        help="List available models")
    parser.add_argument("--batch", action="store_true",
                        help="Batch-convert all available models")
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.list:
        list_available_models(args.abc_dir, sampler=args.sampler)

    elif args.model_id:
        if args.sampler == "step":
            step_path = find_step_file(args.abc_dir, args.model_id)
            if step_path is None:
                print(f"Model {args.model_id}: STEP file not found.")
                sys.exit(1)
            print(f"Model: {args.model_id}  STEP: {os.path.basename(step_path)}")
            points, labels = sample_points_from_step(
                step_path, args.num_points, args.min_points_per_surface,
                rng, grid_res=args.grid_res,
            )
        else:
            result = analyze_model(args.abc_dir, args.model_id)
            if result is None:
                sys.exit(1)
            vertices, faces, surfaces, curves, face_labels = result
            points, labels = sample_points_from_mesh(
                vertices, faces, face_labels,
                args.num_points, args.min_points_per_surface, rng,
            )
            if args.visualize:
                visualize_model(vertices, faces, face_labels, surfaces, points, labels)

        if args.output_dir and len(points):
            os.makedirs(args.output_dir, exist_ok=True)
            out_path = os.path.join(args.output_dir, f"abc_{args.model_id}.xyzc")
            export_xyzc(points, labels, out_path)

    elif args.batch:
        if args.output_dir is None:
            print("--output_dir required for batch mode")
            sys.exit(1)
        os.makedirs(args.output_dir, exist_ok=True)
        model_ids = list_available_models(args.abc_dir, sampler=args.sampler)

        for i, mid in enumerate(model_ids):
            if args.sampler == "step":
                step_path = find_step_file(args.abc_dir, mid)
                if step_path is None:
                    print(f"  [{i+1}/{len(model_ids)}] {mid}: STEP not found, skipping")
                    continue
                points, labels = sample_points_from_step(
                    step_path, args.num_points, args.min_points_per_surface,
                    rng, grid_res=args.grid_res,
                )
            else:
                obj_path, feat_path = find_model_files(args.abc_dir, mid)
                vertices, faces     = load_obj(obj_path)
                surfaces, _         = load_features(feat_path)
                face_labels         = build_face_to_surface(surfaces, len(faces))
                points, labels      = sample_points_from_mesh(
                    vertices, faces, face_labels,
                    args.num_points, args.min_points_per_surface, rng,
                )

            if len(points) == 0:
                print(f"  [{i+1}/{len(model_ids)}] {mid}: skipped (no valid geometry)")
                continue

            out_path = os.path.join(args.output_dir, f"abc_{mid}.xyzc")
            export_xyzc(points, labels, out_path)
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{len(model_ids)}")

    else:
        parser.print_help()
