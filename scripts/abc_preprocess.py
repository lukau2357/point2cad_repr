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
    from OCC.Core.STEPControl import (STEPControl_Reader, STEPControl_Writer,
                                       STEPControl_AsIs)
    from OCC.Core.TopExp      import TopExp_Explorer
    from OCC.Core.TopAbs      import (TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX,
                                       TopAbs_SOLID, TopAbs_SHELL,
                                       TopAbs_COMPOUND, TopAbs_COMPSOLID,
                                       TopAbs_IN, TopAbs_ON)
    from OCC.Core.TopoDS      import topods, TopoDS_Iterator, TopoDS_Compound
    from OCC.Core.BRep        import BRep_Tool, BRep_Builder
    from OCC.Core.BRepClass   import BRepClass_FaceClassifier
    from OCC.Core.gp          import gp_Pnt2d
    from OCC.Core.Message     import Message_ProgressRange
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

# OCC surface type → pipeline surface type.
# Primitives keep their identity; all non-primitive types are mapped to
# "BSpline" following the ParSeNet convention (a BSpline can approximate
# Torus, Bezier, Revolution, Extrusion, Offset, and Other surfaces).
_MAPPED_SURF_TYPE = {
    "Plane":      "Plane",
    "Cylinder":   "Cylinder",
    "Cone":       "Cone",
    "Sphere":     "Sphere",
    "Torus":      "BSpline",
    "Bezier":     "BSpline",
    "BSpline":    "BSpline",
    "Revolution": "BSpline",
    "Extrusion":  "BSpline",
    "Offset":     "BSpline",
    "Other":      "BSpline",
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
# Part decomposition helpers
# ---------------------------------------------------------------------------

def extract_top_level_parts(shape):
    """
    Return the immediate sub-shapes (parts) of a Compound / CompSolid.
    For any other shape type (Solid, Shell, …) returns [shape] so that
    callers can always iterate uniformly over parts.
    """
    if shape.ShapeType() not in (TopAbs_COMPOUND, TopAbs_COMPSOLID):
        return [shape]
    parts = []
    it = TopoDS_Iterator(shape)
    while it.More():
        parts.append(it.Value())
        it.Next()
    return parts if parts else [shape]


def faces_of_shape(shape):
    """Return all TopoDS_Face objects contained anywhere within shape."""
    exp   = TopExp_Explorer(shape, TopAbs_FACE)
    faces = []
    while exp.More():
        faces.append(topods.Face(exp.Current()))
        exp.Next()
    return faces


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
    surface_type_counts        = Counter()   # original OCC types
    surface_areas_by_type      = Counter()
    mapped_type_counts         = Counter()   # pipeline types (primitives + BSpline)
    mapped_areas_by_type       = Counter()
    face_types                 = []          # per-face original OCC type
    face_types_mapped          = []          # per-face pipeline type
    props                      = GProp_GProps()

    for face in face_list:
        adaptor     = BRepAdaptor_Surface(face)
        type_name   = _GEOMABS_SURF_NAMES.get(adaptor.GetType(), "Other")
        mapped_name = _MAPPED_SURF_TYPE.get(type_name, "BSpline")
        surface_type_counts[type_name] += 1
        mapped_type_counts[mapped_name] += 1
        face_types.append(type_name)
        face_types_mapped.append(mapped_name)

        brepgprop.SurfaceProperties(face, props)
        area = props.Mass()
        surface_areas_by_type[type_name] += area
        mapped_areas_by_type[mapped_name] += area

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
        "mapped_surface_types":       dict(mapped_type_counts),
        "mapped_areas_by_type":       {k: float(v) for k, v in mapped_areas_by_type.items()},
        "total_surface_area":     float(total_surface_area),
        "curve_types":            dict(curve_type_counts),
        "bounding_box": {
            "xmin": float(xmin), "xmax": float(xmax),
            "ymin": float(ymin), "ymax": float(ymax),
            "zmin": float(zmin), "zmax": float(zmax),
        },
        "bounding_box_extents":   [float(xmax-xmin), float(ymax-ymin), float(zmax-zmin)],
        "face_types":             face_types,
        "face_types_mapped":      face_types_mapped,
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
    print(f"  Surface types (original OCC → pipeline mapped):")
    for stype, count in sorted(stats["surface_types"].items(), key=lambda x: -x[1]):
        area   = stats["surface_areas_by_type"].get(stype, 0.0)
        mapped = _MAPPED_SURF_TYPE.get(stype, "BSpline")
        suffix = f"  → {mapped}" if mapped != stype else ""
        print(f"    {stype:<12} {count:>4}   area={area:.6f}{suffix}")
    mapped_types = stats.get("mapped_surface_types", {})
    if mapped_types:
        print(f"  Pipeline surface types:")
        for mtype, count in sorted(mapped_types.items(), key=lambda x: -x[1]):
            area = stats.get("mapped_areas_by_type", {}).get(mtype, 0.0)
            print(f"    {mtype:<12} {count:>4}   area={area:.6f}")
    print(f"  Total surface area: {stats['total_surface_area']:.6f}")
    print(f"  Curve types:")
    for ctype, count in sorted(stats["curve_types"].items(), key=lambda x: -x[1]):
        print(f"    {ctype:<12} {count:>4}")


# ---------------------------------------------------------------------------
# STEP-based sampling (exact parametric surface via OCC)
# ---------------------------------------------------------------------------

def _evaluate_face_grid(geom_surf, us, vs):
    """
    Evaluate geom_surf on the grid defined by 1-D arrays us and vs.
    Returns an array of shape (len(us), len(vs), 3).
    """
    pts = np.zeros((len(us), len(vs), 3))
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


def _make_cell_mask(face, us, vs, tol=1e-7):
    """
    Return a bool array of shape (len(us)-1, len(vs)-1).
    Cell (i, j) is True when its UV centroid lies inside or on the face wire,
    as determined by BRepClass_FaceClassifier (2D ray-casting in UV space).
    """
    clf  = BRepClass_FaceClassifier()
    mask = np.zeros((len(us) - 1, len(vs) - 1), dtype=bool)
    for i in range(len(us) - 1):
        u_mid = 0.5 * (us[i] + us[i + 1])
        for j in range(len(vs) - 1):
            v_mid = 0.5 * (vs[j] + vs[j + 1])
            clf.Perform(face, gp_Pnt2d(u_mid, v_mid), tol)
            mask[i, j] = clf.State() in (TopAbs_IN, TopAbs_ON)
    return mask


def _sample_from_grid_with_uv(pts3d, us, vs, area1, area2, n_pts, rng):
    """
    Like _sample_from_grid but also returns the (u, v) coordinates of each
    sampled point (computed via the same barycentric weights applied to the
    UV grid, so no surface inversion is needed).

    Returns (pts3d_out, uv_out) where uv_out is (n_pts, 2).
    Returns empty arrays for degenerate faces.
    """
    if n_pts == 0:
        return np.zeros((0, 3)), np.zeros((0, 2))

    all_areas = np.concatenate([area1.ravel(), area2.ravel()])
    total     = all_areas.sum()
    if total < 1e-12:
        return np.zeros((0, 3)), np.zeros((0, 2))

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

    # 3D vertices of each triangle
    vA = pts3d[ci,     cj    ]
    vB = pts3d[ci + 1, cj    ]
    vC = pts3d[ci,     cj + 1]
    vD = pts3d[ci + 1, cj + 1]
    mask3  = is_type1[:, None]
    vert0  = np.where(mask3, vA, vB)
    vert1  = np.where(mask3, vB, vD)
    pts_out = w0[:, None] * vert0 + w1[:, None] * vert1 + w2[:, None] * vC

    # UV coordinates via the same barycentric weights
    uA = us[ci];     vA_uv = vs[cj]
    uB = us[ci + 1]; vB_uv = vs[cj]
    uC = us[ci];     vC_uv = vs[cj + 1]
    uD = us[ci + 1]; vD_uv = vs[cj + 1]

    u0_tri = np.where(is_type1, uA, uB)
    u1_tri = np.where(is_type1, uB, uD)
    v0_tri = np.where(is_type1, vA_uv, vB_uv)
    v1_tri = np.where(is_type1, vB_uv, vD_uv)

    u_out = w0 * u0_tri + w1 * u1_tri + w2 * uC
    v_out = w0 * v0_tri + w1 * v1_tri + w2 * vC_uv

    return pts_out, np.column_stack([u_out, v_out])


def _classify_points(face, uv_pts, tol=1e-7):
    """
    Return a bool mask of length len(uv_pts).
    True when the corresponding (u, v) lies inside or on the face wire.
    Uses BRepClass_FaceClassifier (2D ray-casting in UV parameter space).
    """
    clf  = BRepClass_FaceClassifier()
    keep = np.zeros(len(uv_pts), dtype=bool)
    for i, (u, v) in enumerate(uv_pts):
        clf.Perform(face, gp_Pnt2d(float(u), float(v)), tol)
        keep[i] = clf.State() in (TopAbs_IN, TopAbs_ON)
    return keep


def _sample_faces(face_list, min_points_per_surface, rng, grid_res=50, area_budget=None):
    """
    Sample points from a pre-loaded list of TopoDS_Face objects.
    Each face is one cluster (cluster ID = face index in face_list).

    Per-face target = min_points_per_surface + proportional share of area_budget.
    area_budget (total extra points distributed proportionally to face area)
    defaults to len(face_list) * min_points_per_surface when None, so the
    average target per face is 2 * min_points_per_surface.

    Sampling respects the trimmed face boundary:
      1. A cell mask zeros out UV cells whose centroid lies outside the face wire.
      2. Every sampled point is individually tested; outside points are rejected.
      3. Each iteration draws exactly the remaining deficit (no oversampling
         multiplier). Repeats up to 20 times until the quota is met.
    """
    if not face_list:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int32)

    face_data = []   # (pts3d, us, vs, a1_masked, a2_masked, total_masked, face)

    for face in face_list:
        adaptor   = BRepAdaptor_Surface(face)
        u0, u1    = adaptor.FirstUParameter(), adaptor.LastUParameter()
        v0, v1    = adaptor.FirstVParameter(), adaptor.LastVParameter()
        geom_surf = BRep_Tool.Surface(face)

        us    = np.linspace(u0, u1, grid_res + 1)
        vs    = np.linspace(v0, v1, grid_res + 1)
        pts3d = _evaluate_face_grid(geom_surf, us, vs)
        a1, a2 = _face_triangles(pts3d)

        # Zero out cells outside the face boundary
        cell_mask = _make_cell_mask(face, us, vs)
        a1_m = a1 * cell_mask
        a2_m = a2 * cell_mask
        total = a1_m.sum() + a2_m.sum()

        # Degenerate fallback: if masking eliminates everything, use unmasked
        if total < 1e-12:
            a1_m, a2_m = a1, a2
            total = a1.sum() + a2.sum()

        face_data.append((pts3d, us, vs, a1_m, a2_m, total, face))

    face_totals = np.array([fd[5] for fd in face_data])
    A_total = face_totals.sum()
    if A_total < 1e-12:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int32)

    if area_budget is None:
        area_budget = min_points_per_surface * len(face_data)

    all_points, all_labels = [], []
    for fid, (pts3d, us, vs, a1, a2, total, face) in enumerate(face_data):
        n_target = min_points_per_surface + int(round(
            area_budget * face_totals[fid] / A_total
        ))

        if total < 1e-12:
            continue

        collected   = []
        n_collected = 0

        for _attempt in range(20):
            n_draw   = n_target - n_collected
            pts, uvs = _sample_from_grid_with_uv(pts3d, us, vs, a1, a2, n_draw, rng)
            if len(pts) == 0:
                break
            keep = _classify_points(face, uvs)
            pts  = pts[keep]
            if len(pts):
                collected.append(pts)
                n_collected += len(pts)
            if n_collected >= n_target:
                break

        if collected:
            pts = np.concatenate(collected)[:n_target]
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


def _visualize_step_from_disk(model_id, xyzc_dir, stats_dir, by_part):
    """Load saved .xyzc / _stats.json files and display without re-sampling."""
    import glob as _glob

    vis_points, vis_labels, vis_face_types = [], [], []
    label_offset = 0

    if by_part:
        xyzc_files = sorted(
            f for f in _glob.glob(os.path.join(xyzc_dir, "*.xyzc"))
            if os.path.basename(f).split(".")[0].isdigit()
        )
        if not xyzc_files:
            print(f"[visualize] No .xyzc files found in {xyzc_dir}")
            return
        for xyzc_path in xyzc_files:
            part_idx = int(os.path.basename(xyzc_path).split(".")[0])
            data = np.loadtxt(xyzc_path)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            pts = data[:, :3].astype(np.float32)
            lbl = data[:, 3].astype(np.int32)
            stats_path = os.path.join(stats_dir, f"{part_idx}_stats.json")
            if os.path.exists(stats_path):
                with open(stats_path) as f:
                    face_types = json.load(f).get("face_types", [])
            else:
                face_types = ["Other"] * (int(lbl.max()) + 1)
            vis_points.append(pts)
            vis_labels.append(lbl + label_offset)
            vis_face_types.extend(face_types)
            label_offset += len(face_types)
    else:
        xyzc_path = os.path.join(xyzc_dir, f"abc_{model_id}.xyzc")
        if not os.path.exists(xyzc_path):
            print(f"[visualize] File not found: {xyzc_path}")
            return
        data = np.loadtxt(xyzc_path)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        pts = data[:, :3].astype(np.float32)
        lbl = data[:, 3].astype(np.int32)
        stats_path = os.path.join(stats_dir, f"abc_{model_id}_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                face_types = json.load(f).get("face_types", [])
        else:
            face_types = ["Other"] * (int(lbl.max()) + 1)
        vis_points.append(pts)
        vis_labels.append(lbl)
        vis_face_types.extend(face_types)

    if vis_points:
        visualize_step_model(
            np.concatenate(vis_points),
            np.concatenate(vis_labels),
            vis_face_types,
        )


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
    parser.add_argument("--abc_dir",    type=str, default=None,
                        help="Path to ABC dataset root (not required for --visualize)")
    parser.add_argument("--model_id",   type=str, default=None,
                        help="Analyse / export a single model")
    parser.add_argument("--output_dir", type=str, default="../sample_clouds",
                        help="Output directory for .xyzc point cloud files")
    parser.add_argument("--stats_dir",  type=str, default="../sample_clouds_stats",
                        help="Output directory for _stats.json files.")
    parser.add_argument("--min_points_per_surface", type=int, default=300,
                        help="Guaranteed minimum points per surface")
    parser.add_argument("--area_budget", type=int, default=None,
                        help="Total extra points distributed proportionally to surface area "
                             "(default: min_points_per_surface × n_surfaces per part)")
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
    parser.add_argument("--no_by_part", action="store_true", default=False,
                        help="Disable part-level decomposition and sample the whole "
                             "STEP as one point cloud.  By default each top-level "
                             "sub-shape produces a separate "
                             "abc_{id}_part_{idx:03d}.xyzc + _stats.json file.")
    args   = parser.parse_args()
    by_part = not args.no_by_part

    # --visualize with --model_id doesn't need abc_dir (loads from disk)
    is_visualize_only = args.visualize and args.model_id and args.sampler == "step"
    if not is_visualize_only and args.abc_dir is None:
        parser.error("--abc_dir is required (unless using --visualize with --model_id)")

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
            if args.visualize:
                xyzc_dir  = os.path.join(args.output_dir, f"abc_{args.model_id}")
                stats_dir = os.path.join(args.stats_dir,  f"abc_{args.model_id}")
                _visualize_step_from_disk(args.model_id, xyzc_dir, stats_dir, by_part)
                sys.exit(0)

            _require_occ()
            step_path = find_step_file(args.abc_dir, args.model_id)
            if step_path is None:
                print(f"Model {args.model_id}: STEP file not found.")
                sys.exit(1)

            print(f"Model: {args.model_id}  STEP: {os.path.basename(step_path)}")
            shape, face_list = load_step(step_path)
            if shape is None:
                sys.exit(1)

            if by_part:
                parts = extract_top_level_parts(shape)
                print(f"  Found {len(parts)} top-level part(s)")
                xyzc_dir  = os.path.join(args.output_dir, f"abc_{args.model_id}") if args.output_dir else None
                stats_dir = os.path.join(args.stats_dir,  f"abc_{args.model_id}") if args.stats_dir  else None
                if xyzc_dir:
                    os.makedirs(xyzc_dir, exist_ok=True)
                if stats_dir:
                    os.makedirs(stats_dir, exist_ok=True)
                for part_idx, part in enumerate(parts):
                    part_faces = faces_of_shape(part)
                    if not part_faces:
                        print(f"  Part {part_idx}: no faces, skipping")
                        continue
                    part_id = f"{args.model_id}_part_{part_idx}"
                    print(f"  Part {part_idx}: {len(part_faces)} face(s)")
                    stats = extract_step_stats(part_id, step_path, part, part_faces)
                    print_step_stats(stats)
                    points, labels = _sample_faces(
                        part_faces, args.min_points_per_surface,
                        rng, grid_res=args.grid_res, area_budget=args.area_budget,
                    )
                    if not len(points):
                        continue
                    stats["n_sampled_points"] = int(len(points))
                    stats["n_clusters"]       = int(len(np.unique(labels)))
                    if xyzc_dir:
                        export_xyzc(points, labels,
                                    os.path.join(xyzc_dir, f"{part_idx}.xyzc"))
                    if stats_dir:
                        export_stats_json(
                            stats, os.path.join(stats_dir, f"{part_idx}_stats.json"))
            else:
                xyzc_dir  = os.path.join(args.output_dir, f"abc_{args.model_id}") if args.output_dir else None
                stats_dir = os.path.join(args.stats_dir,  f"abc_{args.model_id}") if args.stats_dir  else None
                if xyzc_dir:
                    os.makedirs(xyzc_dir, exist_ok=True)
                if stats_dir:
                    os.makedirs(stats_dir, exist_ok=True)
                stats = extract_step_stats(args.model_id, step_path, shape, face_list)
                print_step_stats(stats)
                points, labels = _sample_faces(
                    face_list, args.min_points_per_surface,
                    rng, grid_res=args.grid_res, area_budget=args.area_budget,
                )
                if len(points):
                    stats["n_sampled_points"] = int(len(points))
                    stats["n_clusters"]       = int(len(np.unique(labels)))
                    stem = f"abc_{args.model_id}"
                    if xyzc_dir:
                        export_xyzc(points, labels,
                                    os.path.join(xyzc_dir, f"{stem}.xyzc"))
                    if stats_dir:
                        export_stats_json(
                            stats, os.path.join(stats_dir, f"{stem}_stats.json"))

        else:  # obj sampler
            result = analyze_model(args.abc_dir, args.model_id)
            if result is None:
                sys.exit(1)
            vertices, faces, surfaces, curves, face_labels = result
            n_pts = (args.area_budget if args.area_budget is not None
                     else len(surfaces) * args.min_points_per_surface)
            points, labels = sample_points_from_mesh(
                vertices, faces, face_labels,
                n_pts, args.min_points_per_surface, rng,
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

                if by_part:
                    parts     = extract_top_level_parts(shape)
                    xyzc_dir  = os.path.join(args.output_dir, f"abc_{mid}")
                    stats_dir = os.path.join(args.stats_dir, f"abc_{mid}") if args.stats_dir else None
                    os.makedirs(xyzc_dir, exist_ok=True)
                    if stats_dir:
                        os.makedirs(stats_dir, exist_ok=True)
                    for part_idx, part in enumerate(parts):
                        part_faces = faces_of_shape(part)
                        if not part_faces:
                            continue
                        part_id        = f"{mid}_part_{part_idx}"
                        stats          = extract_step_stats(part_id, step_path,
                                                            part, part_faces)
                        points, labels = _sample_faces(
                            part_faces, args.min_points_per_surface,
                            rng, grid_res=args.grid_res, area_budget=args.area_budget,
                        )
                        if not len(points):
                            continue
                        stats["n_sampled_points"] = int(len(points))
                        stats["n_clusters"]       = int(len(np.unique(labels)))
                        export_xyzc(points, labels,
                                    os.path.join(xyzc_dir, f"{part_idx}.xyzc"))
                        if stats_dir:
                            export_stats_json(
                                stats,
                                os.path.join(stats_dir, f"{part_idx}_stats.json"))
                else:
                    xyzc_dir  = os.path.join(args.output_dir, f"abc_{mid}")
                    stats_dir = os.path.join(args.stats_dir, f"abc_{mid}") if args.stats_dir else None
                    os.makedirs(xyzc_dir, exist_ok=True)
                    if stats_dir:
                        os.makedirs(stats_dir, exist_ok=True)
                    stats          = extract_step_stats(mid, step_path, shape, face_list)
                    points, labels = _sample_faces(
                        face_list, args.min_points_per_surface,
                        rng, grid_res=args.grid_res, area_budget=args.area_budget,
                    )
                    if len(points):
                        stats["n_sampled_points"] = int(len(points))
                        stats["n_clusters"]       = int(len(np.unique(labels)))
                        export_xyzc(points, labels,
                                    os.path.join(xyzc_dir, f"abc_{mid}.xyzc"))
                        if stats_dir:
                            export_stats_json(
                                stats,
                                os.path.join(stats_dir, f"abc_{mid}_stats.json"))
            else:  # obj sampler
                obj_path, feat_path = find_model_files(args.abc_dir, mid)
                vertices, faces     = load_obj(obj_path)
                surfaces, _         = load_features(feat_path)
                face_labels         = build_face_to_surface(surfaces, len(faces))
                n_pts = (args.area_budget if args.area_budget is not None
                         else len(surfaces) * args.min_points_per_surface)
                points, labels      = sample_points_from_mesh(
                    vertices, faces, face_labels,
                    n_pts, args.min_points_per_surface, rng,
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
