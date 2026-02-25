import argparse
import glob
import json
import os
import sys
import yaml
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from point2cad.color_config import get_surface_color

try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopExp      import TopExp_Explorer
    from OCC.Core.TopAbs      import (TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX,
                                       TopAbs_SOLID, TopAbs_SHELL,
                                       TopAbs_COMPOUND, TopAbs_COMPSOLID)
    from OCC.Core.TopoDS             import topods
    from OCC.Core.BRep        import BRep_Tool
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
    from OCC.Core.GeomAbs     import (
        GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere,
        GeomAbs_Torus, GeomAbs_BezierSurface, GeomAbs_BSplineSurface,
        GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion,
        GeomAbs_OffsetSurface, GeomAbs_OtherSurface,
        GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse,
        GeomAbs_Hyperbola, GeomAbs_Parabola,
        GeomAbs_BezierCurve, GeomAbs_BSplineCurve,
        GeomAbs_OffsetCurve, GeomAbs_OtherCurve,
    )
    from OCC.Core.Bnd        import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.GProp      import GProp_GProps
    from OCC.Core.BRepGProp  import brepgprop

    _GEOMABS_SURF_NAMES = {
        GeomAbs_Plane:               "Plane",
        GeomAbs_Cylinder:            "Cylinder",
        GeomAbs_Cone:                "Cone",
        GeomAbs_Sphere:              "Sphere",
        GeomAbs_Torus:               "Torus",
        GeomAbs_BezierSurface:       "Bezier",
        GeomAbs_BSplineSurface:      "BSpline",
        GeomAbs_SurfaceOfRevolution: "Revolution",
        GeomAbs_SurfaceOfExtrusion:  "Extrusion",
        GeomAbs_OffsetSurface:       "Offset",
        GeomAbs_OtherSurface:        "Other",
    }
    _GEOMABS_CURVE_NAMES = {
        GeomAbs_Line:        "Line",
        GeomAbs_Circle:      "Circle",
        GeomAbs_Ellipse:     "Ellipse",
        GeomAbs_Hyperbola:   "Hyperbola",
        GeomAbs_Parabola:    "Parabola",
        GeomAbs_BezierCurve: "Bezier",
        GeomAbs_BSplineCurve:"BSpline",
        GeomAbs_OffsetCurve: "Offset",
        GeomAbs_OtherCurve:  "Other",
    }
    _TOPABS_SHAPE_NAMES = {
        TopAbs_SOLID:     "Solid",
        TopAbs_COMPSOLID: "CompSolid",
        TopAbs_SHELL:     "Shell",
        TopAbs_COMPOUND:  "Compound",
    }
    _OCC_AVAILABLE = True
except ImportError as err:
    print(err)
    _OCC_AVAILABLE = False
    _GEOMABS_SURF_NAMES  = {}
    _GEOMABS_CURVE_NAMES = {}
    _TOPABS_SHAPE_NAMES  = {}

# Surface-type string → color_config key (used for both STEP and OBJ visualisation)
_STEP_SURF_TO_COLOR_KEY = {
    "Plane":      "plane",
    "Cylinder":   "cylinder",
    "Cone":       "cone",
    "Sphere":     "sphere",
    "Torus":      "inr",
    "Bezier":     "inr",
    "BSpline":    "inr",
    "Revolution": "inr",
    "Extrusion":  "inr",
    "Offset":     "inr",
    "Other":      "inr",
}


def _require_occ():
    if not _OCC_AVAILABLE:
        print("ERROR: pythonocc-core is not installed. "
              "The STEP sampler requires OCC — run inside the Docker container.")
        sys.exit(1)


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


def _find_step_batches(abc_dir):
    """Return sorted list of all subdirectories whose name contains 'step'."""
    return sorted(
        os.path.join(abc_dir, e)
        for e in os.listdir(abc_dir)
        if "step" in e and os.path.isdir(os.path.join(abc_dir, e))
    )


def find_step_file(abc_dir, model_id):
    """Return the path to the STEP file for model_id across all batches, or None."""
    for batch_dir in _find_step_batches(abc_dir):
        candidate = os.path.join(batch_dir, model_id)
        if os.path.isdir(candidate):
            hits = sorted(glob.glob(os.path.join(candidate, "*.step")))
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
                face_verts = [int(p.split("/")[0]) - 1 for p in parts]
                faces.append(face_verts)
    return np.array(vertices, dtype=np.float64), faces


def load_features(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("surfaces", []), data.get("curves", [])


def build_face_to_surface(surfaces, num_faces):
    """Map each OBJ face index to a surface ID (-1 = unassigned)."""
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
    w0, w1, w2 = 1 - sqrt_r1, sqrt_r1 * (1 - r2), sqrt_r1 * r2

    pts = np.zeros((n_pts, 3))
    for i, fi in enumerate(sampled_face_ids):
        face   = faces[fi]
        pts[i] = w0[i]*vertices[face[0]] + w1[i]*vertices[face[1]] + w2[i]*vertices[face[2]]
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

    valid            = face_labels >= 0
    face_areas_valid = face_areas * valid
    total_area       = face_areas_valid.sum()
    if total_area == 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int32)

    probs            = face_areas_valid / total_area
    sampled_face_ids = rng.choice(len(faces), size=num_points, p=probs)

    r1      = rng.random(num_points)
    r2      = rng.random(num_points)
    sqrt_r1 = np.sqrt(r1)
    w0, w1, w2 = 1 - sqrt_r1, sqrt_r1 * (1 - r2), sqrt_r1 * r2

    points = np.zeros((num_points, 3))
    labels = np.zeros(num_points, dtype=np.int32)
    for i, fi in enumerate(sampled_face_ids):
        face      = faces[fi]
        points[i] = w0[i]*vertices[face[0]] + w1[i]*vertices[face[1]] + w2[i]*vertices[face[2]]
        labels[i] = face_labels[fi]

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
# STEP loading
# ---------------------------------------------------------------------------

def load_step(step_path):
    """
    Load a STEP file and return (shape, face_list).
    Returns (None, []) on read error.
    """
    _require_occ()
    reader = STEPControl_Reader()
    if reader.ReadFile(step_path) != 1:
        print(f"  STEP read error: {step_path}")
        return None, []
    reader.TransferRoots()
    shape = reader.OneShape()

    exp       = TopExp_Explorer(shape, TopAbs_FACE)
    face_list = []
    while exp.More():
        face_list.append(topods.Face(exp.Current()))
        exp.Next()
    return shape, face_list


# ---------------------------------------------------------------------------
# STEP statistics extraction
# ---------------------------------------------------------------------------

def extract_step_stats(model_id, step_path, shape, face_list):
    """
    Extract comprehensive statistics from a loaded STEP model.

    Returns a dict suitable for JSON serialisation and per-dataset aggregation.
    The 'face_types' list (one entry per face, indexed by cluster ID) is included
    so the caller can colour-code visualisations without re-querying OCC.
    """
    # --- Surface types and per-face areas ---
    surface_type_counts   = Counter()
    surface_areas_by_type = Counter()
    face_types            = []
    props                 = GProp_GProps()

    for face in face_list:
        adaptor   = BRepAdaptor_Surface(face)
        type_name = _GEOMABS_SURF_NAMES.get(adaptor.GetType(), "Other")
        surface_type_counts[type_name] += 1
        face_types.append(type_name)

        brepgprop.SurfaceProperties(face, props)
        surface_areas_by_type[type_name] += props.Mass()

    total_surface_area = sum(surface_areas_by_type.values())

    # --- Edge / curve types ---
    curve_type_counts = Counter()
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        edge = topods.Edge(exp.Current())
        try:
            adaptor   = BRepAdaptor_Curve(edge)
            type_name = _GEOMABS_CURVE_NAMES.get(adaptor.GetType(), "Other")
        except Exception:
            type_name = "Other"
        curve_type_counts[type_name] += 1
        exp.Next()

    # --- Vertex count ---
    n_vertices = 0
    exp = TopExp_Explorer(shape, TopAbs_VERTEX)
    while exp.More():
        n_vertices += 1
        exp.Next()

    # --- Bounding box ---
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

    # --- Shape type ---
    shape_type = _TOPABS_SHAPE_NAMES.get(shape.ShapeType(), "Other")

    return {
        "model_id":               model_id,
        "step_file":              os.path.basename(step_path),
        "shape_type":             shape_type,
        "n_faces":                len(face_list),
        "n_edges":                sum(curve_type_counts.values()),
        "n_vertices":             n_vertices,
        "surface_types":          dict(surface_type_counts),
        "surface_areas_by_type":  {k: float(v) for k, v in surface_areas_by_type.items()},
        "total_surface_area":     float(total_surface_area),
        "curve_types":            dict(curve_type_counts),
        "bounding_box": {
            "xmin": float(xmin), "xmax": float(xmax),
            "ymin": float(ymin), "ymax": float(ymax),
            "zmin": float(zmin), "zmax": float(zmax),
        },
        "bounding_box_extents":   [float(xmax-xmin), float(ymax-ymin), float(zmax-zmin)],
        "face_types":             face_types,
        # filled in by caller after sampling:
        "n_sampled_points":       0,
        "n_clusters":             0,
    }


def print_step_stats(stats):
    print(f"  Shape type : {stats['shape_type']}")
    print(f"  Faces      : {stats['n_faces']}   "
          f"Edges: {stats['n_edges']}   Vertices: {stats['n_vertices']}")
    ext = stats["bounding_box_extents"]
    print(f"  BBox extents: [{ext[0]:.4f}, {ext[1]:.4f}, {ext[2]:.4f}]")
    print(f"  Surface types (count / area):")
    for stype, count in sorted(stats["surface_types"].items(), key=lambda x: -x[1]):
        area = stats["surface_areas_by_type"].get(stype, 0.0)
        print(f"    {stype:<12} {count:>4}   area={area:.6f}")
    print(f"  Total surface area: {stats['total_surface_area']:.6f}")
    print(f"  Curve types:")
    for ctype, count in sorted(stats["curve_types"].items(), key=lambda x: -x[1]):
        print(f"    {ctype:<12} {count:>4}")


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
            p         = geom_surf.Value(u, v)
            pts[i, j] = (p.X(), p.Y(), p.Z())
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
    it cancels in normalised probabilities but keeps the arrays in true area units.
    """
    A = pts3d[:-1, :-1]
    B = pts3d[1:,  :-1]
    C = pts3d[:-1, 1:]
    D = pts3d[1:,  1:]

    area1 = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=-1)
    area2 = 0.5 * np.linalg.norm(np.cross(D - B, C - B), axis=-1)
    return area1, area2


def _sample_from_grid(pts3d, area1, area2, n_pts, rng):
    """
    Sample n_pts points (area-weighted) from a triangulated UV grid.

    All triangles from both types are pooled into a single flat distribution.
    Returns (n_pts, 3) float array, or (0, 3) for degenerate faces.
    """
    if n_pts == 0:
        return np.zeros((0, 3))

    all_areas = np.concatenate([area1.ravel(), area2.ravel()])
    total     = all_areas.sum()
    if total < 1e-12:
        return np.zeros((0, 3))

    probs    = all_areas / total
    n_cells  = area1.size
    tri_ids  = rng.choice(len(all_areas), size=n_pts, p=probs)
    is_type1 = tri_ids < n_cells
    cell_ids = np.where(is_type1, tri_ids, tri_ids - n_cells)
    ci       = cell_ids // area1.shape[1]
    cj       = cell_ids %  area1.shape[1]

    r1      = rng.random(n_pts)
    sqrt_r1 = np.sqrt(r1)
    r2      = rng.random(n_pts)
    w0      = 1.0 - sqrt_r1
    w1      = sqrt_r1 * (1.0 - r2)
    w2      = sqrt_r1 * r2

    vA = pts3d[ci,     cj    ]
    vB = pts3d[ci + 1, cj    ]
    vC = pts3d[ci,     cj + 1]   # shared by both triangle types
    vD = pts3d[ci + 1, cj + 1]

    # Triangle 1: (A, B, C)   Triangle 2: (B, D, C)
    mask  = is_type1[:, None]
    vert0 = np.where(mask, vA, vB)
    vert1 = np.where(mask, vB, vD)
    vert2 = vC

    return w0[:, None] * vert0 + w1[:, None] * vert1 + w2[:, None] * vert2


def _sample_faces(face_list, num_points, min_points_per_surface, rng, grid_res=50):
    """
    Sample num_points from a pre-loaded list of TopoDS_Face objects.
    Each face is one cluster (cluster ID = face index in face_list).
    """
    if not face_list:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int32)

    face_grids, face_area1s, face_area2s, face_totals = [], [], [], []

    for face in face_list:
        adaptor   = BRepAdaptor_Surface(face)
        u0, u1    = adaptor.FirstUParameter(), adaptor.LastUParameter()
        v0, v1    = adaptor.FirstVParameter(), adaptor.LastVParameter()
        geom_surf = BRep_Tool.Surface(face)

        pts3d  = _evaluate_face_grid(geom_surf, u0, u1, v0, v1, grid_res)
        a1, a2 = _face_triangles(pts3d)

        face_grids.append(pts3d)
        face_area1s.append(a1)
        face_area2s.append(a2)
        face_totals.append(a1.sum() + a2.sum())

    face_totals = np.array(face_totals)
    total_area  = face_totals.sum()
    if total_area < 1e-12:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int32)

    face_probs  = face_totals / total_area
    face_counts = rng.multinomial(num_points, face_probs)

    all_points, all_labels = [], []
    for fid, (pts3d, a1, a2, n) in enumerate(
            zip(face_grids, face_area1s, face_area2s, face_counts)):
        pts = _sample_from_grid(pts3d, a1, a2, int(n), rng)
        if len(pts):
            all_points.append(pts)
            all_labels.append(np.full(len(pts), fid, dtype=np.int32))

    for fid in range(len(face_list)):
        face_min = max(min_points_per_surface, int(face_probs[fid] * num_points))
        if face_counts[fid] < face_min:
            deficit = face_min - face_counts[fid]
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


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def export_xyzc(points, labels, output_path):
    data = np.column_stack([points, labels.astype(np.float64)])
    np.savetxt(output_path, data)
    print(f"  Saved {len(points)} points → {output_path}")


def export_stats_json(stats, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, cls=_NumpyEncoder)
    print(f"  Saved stats      → {output_path}")


# ---------------------------------------------------------------------------
# Visualisation
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


def _step_color_for_face(face_type):
    color_key = _STEP_SURF_TO_COLOR_KEY.get(face_type, "inr")
    return get_surface_color(color_key)


def visualize_step_model(points, labels, face_types):
    """Visualize a STEP-sampled point cloud, colored by surface type."""
    import open3d as o3d
    import time

    colors = np.array([_step_color_for_face(face_types[lbl]) for lbl in labels])
    pcd    = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window("STEP sampled point cloud", width=900, height=720)
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 2.0

    while vis.poll_events():
        vis.update_renderer()
        time.sleep(0.01)
    vis.destroy_window()


def analyze_model(abc_dir, model_id):
    obj_path, feat_path = find_model_files(abc_dir, model_id)
    if obj_path is None:
        print(f"Model {model_id}: OBJ or features file not found.")
        return None

    vertices, faces  = load_obj(obj_path)
    surfaces, curves = load_features(feat_path)
    surface_types    = Counter(s["type"] for s in surfaces)
    curve_types      = Counter(c["type"] for c in curves)
    face_labels      = build_face_to_surface(surfaces, len(faces))
    unlabeled        = int((face_labels == -1).sum())

    print(f"Model: {model_id}")
    print(f"  OBJ: {os.path.basename(obj_path)}")
    print(f"  Vertices: {len(vertices)}  Faces: {len(faces)}")
    print(f"  Surfaces: {len(surfaces)}")
    for stype, count in surface_types.most_common():
        print(f"    {stype}: {count}")
    print(f"  Curves: {len(curves)}")
    for ctype, count in curve_types.most_common():
        print(f"    {ctype}: {count}")
    print(f"  Unlabeled faces: {unlabeled}/{len(faces)}")
    return vertices, faces, surfaces, curves, face_labels


def visualize_obj_model(vertices, faces, face_labels, surfaces, points, labels):
    import open3d as o3d
    import time

    colors = np.array([surface_color_for_label(surfaces, lbl) for lbl in labels])
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
        time.sleep(0.01)

    vis1.destroy_window()
    vis2.destroy_window()


# ---------------------------------------------------------------------------
# Model listing and dataset integrity
# ---------------------------------------------------------------------------

def check_model_id_uniqueness(abc_dir):
    """
    Check whether model IDs are unique across all STEP batches.
    Returns a dict mapping duplicated model IDs to the list of batch names they
    appear in (empty dict means all IDs are unique).
    """
    batches = _find_step_batches(abc_dir)
    if not batches:
        print("No STEP batch directories found.")
        return {}

    id_to_batches = {}
    for batch_dir in batches:
        batch_name = os.path.basename(batch_dir)
        for entry in sorted(os.listdir(batch_dir)):
            if os.path.isdir(os.path.join(batch_dir, entry)):
                id_to_batches.setdefault(entry, []).append(batch_name)

    duplicates = {mid: bs for mid, bs in id_to_batches.items() if len(bs) > 1}
    print(f"Scanned {len(batches)} STEP batch(es), {len(id_to_batches)} distinct model IDs.")
    if duplicates:
        print(f"DUPLICATES: {len(duplicates)} model ID(s) appear in multiple batches:")
        for mid, bs in sorted(duplicates.items()):
            print(f"  {mid}: {bs}")
    else:
        print("All model IDs are unique across batches.")
    return duplicates


def list_available_models(abc_dir, sampler="step", max_display=20):
    if sampler == "step":
        batches = _find_step_batches(abc_dir)
        if not batches:
            print("No STEP batch directories found.")
            return []
        complete = sorted(
            d
            for batch_dir in batches
            for d in os.listdir(batch_dir)
            if os.path.isdir(os.path.join(batch_dir, d))
            and glob.glob(os.path.join(batch_dir, d, "*.step"))
        )
        print(f"Found {len(batches)} STEP batch(es), {len(complete)} models total.")
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

    print(f"  First {min(max_display, len(complete))} IDs: "
          f"{complete[:max_display]}{'...' if len(complete) > max_display else ''}")
    return complete


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ABC dataset preprocessing for Point2CAD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--abc_dir",    type=str, required=True,
                        help="Path to ABC dataset root")
    parser.add_argument("--model_id",   type=str, default=None,
                        help="Analyse / export a single model")
    parser.add_argument("--output_dir", type=str, default="../sample_clouds",
                        help="Output directory for .xyzc point cloud files")
    parser.add_argument("--stats_dir",  type=str, default="../sample_clouds_stats",
                        help="Output directory for _stats.json files.")
    parser.add_argument("--num_points", type=int, default=10000,
                        help="Total number of points to sample (area-weighted)")
    parser.add_argument("--min_points_per_surface", type=int, default=300,
                        help="Minimum points per surface; undersized clusters are upsampled")
    parser.add_argument("--sampler", type=str, default="step",
                        choices=["step", "obj"],
                        help="'step': exact OCC parametric surfaces, all 10k models; "
                             "'obj': OBJ mesh barycentric, ~7.1k models with feat annotations")
    parser.add_argument("--grid_res", type=int, default=50,
                        help="UV grid resolution for STEP sampler")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the sampled point cloud")
    parser.add_argument("--list",  action="store_true",
                        help="List available models")
    parser.add_argument("--check_uniqueness", action="store_true",
                        help="Check that model IDs are unique across all STEP batches")
    parser.add_argument("--batch", action="store_true",
                        help="Batch-convert all available models")
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.stats_dir is None:
        args.stats_dir = args.output_dir

    # ------------------------------------------------------------------
    if args.check_uniqueness:
        check_model_id_uniqueness(args.abc_dir)

    # ------------------------------------------------------------------
    elif args.list:
        list_available_models(args.abc_dir, sampler=args.sampler)

    # ------------------------------------------------------------------
    elif args.model_id:
        if args.sampler == "step":
            _require_occ()
            step_path = find_step_file(args.abc_dir, args.model_id)
            if step_path is None:
                print(f"Model {args.model_id}: STEP file not found.")
                sys.exit(1)

            print(f"Model: {args.model_id}  STEP: {os.path.basename(step_path)}")
            shape, face_list = load_step(step_path)
            if shape is None:
                sys.exit(1)

            stats = extract_step_stats(args.model_id, step_path, shape, face_list)
            print_step_stats(stats)

            points, labels = _sample_faces(
                face_list, args.num_points, args.min_points_per_surface,
                rng, grid_res=args.grid_res,
            )
            stats["n_sampled_points"] = int(len(points))
            stats["n_clusters"]       = int(len(np.unique(labels)))

            if args.visualize and len(points):
                visualize_step_model(points, labels, stats["face_types"])

            if len(points):
                stem = f"abc_{args.model_id}"
                if args.output_dir:
                    os.makedirs(args.output_dir, exist_ok=True)
                    export_xyzc(points, labels,
                                os.path.join(args.output_dir, f"{stem}.xyzc"))
                if args.stats_dir:
                    os.makedirs(args.stats_dir, exist_ok=True)
                    export_stats_json(stats,
                                      os.path.join(args.stats_dir, f"{stem}_stats.json"))

        else:  # obj sampler
            result = analyze_model(args.abc_dir, args.model_id)
            if result is None:
                sys.exit(1)
            vertices, faces, surfaces, curves, face_labels = result
            points, labels = sample_points_from_mesh(
                vertices, faces, face_labels,
                args.num_points, args.min_points_per_surface, rng,
            )

            if args.visualize and len(points):
                visualize_obj_model(vertices, faces, face_labels,
                                    surfaces, points, labels)

            if args.output_dir and len(points):
                os.makedirs(args.output_dir, exist_ok=True)
                stem = f"abc_{args.model_id}"
                export_xyzc(points, labels,
                             os.path.join(args.output_dir, f"{stem}.xyzc"))

    # ------------------------------------------------------------------
    elif args.batch:
        if args.output_dir is None:
            print("--output_dir required for batch mode")
            sys.exit(1)
        os.makedirs(args.output_dir, exist_ok=True)
        if args.stats_dir:
            os.makedirs(args.stats_dir, exist_ok=True)
        model_ids = list_available_models(args.abc_dir, sampler=args.sampler)

        for i, mid in enumerate(model_ids):
            stem = f"abc_{mid}"
            if args.sampler == "step":
                _require_occ()
                step_path = find_step_file(args.abc_dir, mid)
                if step_path is None:
                    print(f"  [{i+1}/{len(model_ids)}] {mid}: STEP not found, skipping")
                    continue
                shape, face_list = load_step(step_path)
                if shape is None:
                    continue
                stats          = extract_step_stats(mid, step_path, shape, face_list)
                points, labels = _sample_faces(
                    face_list, args.num_points, args.min_points_per_surface,
                    rng, grid_res=args.grid_res,
                )
                if len(points):
                    stats["n_sampled_points"] = int(len(points))
                    stats["n_clusters"]       = int(len(np.unique(labels)))
                    if args.stats_dir:
                        export_stats_json(
                            stats, os.path.join(args.stats_dir, f"{stem}_stats.json"))
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

            export_xyzc(points, labels, os.path.join(args.output_dir, f"{stem}.xyzc"))
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{len(model_ids)}")

    # ------------------------------------------------------------------
    else:
        parser.print_help()
