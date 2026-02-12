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

def find_model_files(abc_dir, model_id):
    obj_dir = os.path.join(abc_dir, "obj", model_id)
    feat_dir = os.path.join(abc_dir, "feat", model_id)

    obj_files = sorted(glob.glob(os.path.join(obj_dir, "*_trimesh_*.obj")))
    feat_files = sorted(glob.glob(os.path.join(feat_dir, "*_features_*.yml")))

    if len(obj_files) == 0 or len(feat_files) == 0:
        return None, None
    return obj_files[0], feat_files[0]

def load_obj(path):
    vertices = []
    faces = []
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
    return np.array(vertices, dtype = np.float64), faces

def load_features(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("surfaces", []), data.get("curves", [])

def build_face_to_surface(surfaces, num_faces):
    """Map each face index to a surface ID. Returns array of length num_faces, -1 for unassigned."""
    face_labels = np.full(num_faces, -1, dtype = np.int32)
    for sid, surface in enumerate(surfaces):
        for fi in surface.get("face_indices", []):
            if fi < num_faces:
                face_labels[fi] = sid
    return face_labels

def _sample_surface(vertices, faces, face_areas, face_labels, sid, n_pts, rng):
    """Sample n_pts points from a single surface via area-weighted barycentric sampling."""
    mask = face_labels == sid
    areas = face_areas * mask
    probs = areas / areas.sum()
    sampled_face_ids = rng.choice(len(faces), size = n_pts, p = probs)

    r1 = rng.random(n_pts)
    r2 = rng.random(n_pts)
    sqrt_r1 = np.sqrt(r1)
    w0 = 1 - sqrt_r1
    w1 = sqrt_r1 * (1 - r2)
    w2 = sqrt_r1 * r2

    pts = np.zeros((n_pts, 3))
    for i, fi in enumerate(sampled_face_ids):
        face = faces[fi]
        pts[i] = w0[i] * vertices[face[0]] + w1[i] * vertices[face[1]] + w2[i] * vertices[face[2]]
    return pts

def sample_points_from_mesh(vertices, faces, face_labels, num_points, min_points_per_surface, rng):
    """Sample num_points globally (area-weighted), then upsample surfaces below min_points_per_surface."""
    face_areas = np.zeros(len(faces))
    for i, face in enumerate(faces):
        if len(face) < 3:
            continue
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        face_areas[i] = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

    valid = face_labels >= 0
    face_areas_valid = face_areas * valid
    total_area = face_areas_valid.sum()
    if total_area == 0:
        return np.zeros((0, 3)), np.zeros(0, dtype = np.int32)

    # Global area-weighted sampling
    probs = face_areas_valid / total_area
    sampled_face_ids = rng.choice(len(faces), size = num_points, p = probs)

    r1 = rng.random(num_points)
    r2 = rng.random(num_points)
    sqrt_r1 = np.sqrt(r1)
    w0 = 1 - sqrt_r1
    w1 = sqrt_r1 * (1 - r2)
    w2 = sqrt_r1 * r2

    points = np.zeros((num_points, 3))
    labels = np.zeros(num_points, dtype = np.int32)
    for i, fi in enumerate(sampled_face_ids):
        face = faces[fi]
        points[i] = w0[i] * vertices[face[0]] + w1[i] * vertices[face[1]] + w2[i] * vertices[face[2]]
        labels[i] = face_labels[fi]

    # Upsample surfaces below minimum
    unique_labels, counts = np.unique(labels, return_counts = True)
    extra_points = []
    extra_labels = []
    for sid, count in zip(unique_labels, counts):
        if count < min_points_per_surface:
            deficit = min_points_per_surface - count + rng.integers(0, min_points_per_surface // 5 + 1)
            pts = _sample_surface(vertices, faces, face_areas, face_labels, sid, deficit, rng)
            extra_points.append(pts)
            extra_labels.append(np.full(deficit, sid, dtype = np.int32))

    if len(extra_points) > 0:
        points = np.concatenate([points] + extra_points)
        labels = np.concatenate([labels] + extra_labels)

    # Print cluster statistics
    unique_labels, counts = np.unique(labels, return_counts = True)
    print(f"  Cluster stats: {len(unique_labels)} surfaces, "
          f"min={counts.min()}, max={counts.max()}, mean={counts.mean():.0f}, "
          f"total={len(labels)} points")

    return points, labels

def analyze_model(abc_dir, model_id):
    obj_path, feat_path = find_model_files(abc_dir, model_id)
    if obj_path is None:
        print(f"Model {model_id}: OBJ or features file not found.")
        return None

    vertices, faces = load_obj(obj_path)
    surfaces, curves = load_features(feat_path)

    surface_types = Counter(s["type"] for s in surfaces)
    curve_types = Counter(c["type"] for c in curves)

    face_labels = build_face_to_surface(surfaces, len(faces))
    unlabeled = int((face_labels == -1).sum())

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

ABC_TYPE_TO_COLOR_KEY = {
    "Plane": "plane",
    "Sphere": "sphere",
    "Cylinder": "cylinder",
    "Cone": "cone",
}

def surface_color_for_label(surfaces, label):
    abc_type = surfaces[label]["type"]
    color_key = ABC_TYPE_TO_COLOR_KEY.get(abc_type, "inr")
    return get_surface_color(color_key)

def visualize_model(vertices, faces, face_labels, surfaces, points, labels):
    import open3d as o3d
    import time

    # Point cloud colored by surface type
    colors = np.array([surface_color_for_label(surfaces, label) for label in labels])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Triangle mesh colored by surface type (per-vertex)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(faces))
    vertex_colors = np.ones((len(vertices), 3)) * 0.7
    for vi in range(len(vertices)):
        # Find any face containing this vertex to determine its surface label
        pass
    # Color faces via vertex colors: assign each vertex the color of its surface
    vertex_labels = np.full(len(vertices), -1, dtype = np.int32)
    for fi, face in enumerate(faces):
        label = face_labels[fi]
        if label < 0:
            continue
        for vi in face:
            if vertex_labels[vi] < 0:
                vertex_labels[vi] = label
    for vi in range(len(vertices)):
        if vertex_labels[vi] >= 0:
            vertex_colors[vi] = surface_color_for_label(surfaces, vertex_labels[vi])
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    mesh.compute_vertex_normals()

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name = "Sampled point cloud", width = 640, height = 720, left = 0, top = 50)
    vis1.add_geometry(pcd)
    vis1.get_render_option().point_size = 2.0

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name = "Original mesh", width = 640, height = 720, left = 640, top = 50)
    vis2.add_geometry(mesh)
    vis2.get_render_option().mesh_show_back_face = True

    running1, running2 = True, True
    while running1 and running2:
        running1 = vis1.poll_events()
        vis1.update_renderer()
        running2 = vis2.poll_events()
        vis2.update_renderer()
        time.sleep(0.01)

    vis1.destroy_window()
    vis2.destroy_window()

def list_available_models(abc_dir, max_display = 20):
    obj_dir = os.path.join(abc_dir, "obj")
    feat_dir = os.path.join(abc_dir, "feat")

    obj_ids = set(d for d in os.listdir(obj_dir) if os.path.isdir(os.path.join(obj_dir, d)))
    feat_ids = set(d for d in os.listdir(feat_dir) if os.path.isdir(os.path.join(feat_dir, d)))

    # Models with both OBJ and features
    valid_ids = sorted(obj_ids & feat_ids)

    # Filter to those that actually have files
    complete = []
    for mid in valid_ids:
        obj_path, feat_path = find_model_files(abc_dir, mid)
        if obj_path is not None:
            complete.append(mid)

    print(f"Models with both OBJ and features: {len(complete)}")
    if len(complete) > max_display:
        print(f"  First {max_display}: {complete[:max_display]}")
    else:
        print(f"  IDs: {complete}")

    with open("../output/tmp_models.json", "w+", encoding = "utf-8") as f:
        json.dump(complete, f, indent = 4)
    
    return complete

def export_xyzc(points, labels, output_path):
    data = np.column_stack([points, labels.astype(np.float64)])
    np.savetxt(output_path, data)
    print(f"Saved {len(points)} points to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "ABC dataset preprocessing for Point2CAD")
    parser.add_argument("--abc_dir", type = str, required = True, help = "Path to ABC dataset root")
    parser.add_argument("--model_id", type = str, default = None, help = "Analyze/export a single model")
    parser.add_argument("--output_dir", type = str, default = None, help = "Output directory for .xyzc files")
    parser.add_argument("--num_points", type = int, default = 10000, help = "Total number of points to sample (area-weighted)")
    parser.add_argument("--min_points_per_surface", type = int, default = 500, help = "Minimum points per surface, undersized clusters are upsampled")
    parser.add_argument("--visualize", action = "store_true", help = "Visualize the sampled point cloud")
    parser.add_argument("--list", action = "store_true", help = "List available models")
    parser.add_argument("--batch", action = "store_true", help = "Batch convert all available models")
    parser.add_argument("--seed", type = int, default = 42)
    args = parser.parse_args()

    if args.list:
        list_available_models(args.abc_dir)

    elif args.model_id:
        result = analyze_model(args.abc_dir, args.model_id)
        if result is None:
            exit(1)
        vertices, faces, surfaces, curves, face_labels = result

        if args.visualize or args.output_dir:
            rng = np.random.default_rng(args.seed)
            points, labels = sample_points_from_mesh(vertices, faces, face_labels, args.num_points, args.min_points_per_surface, rng)

            if args.visualize:
                visualize_model(vertices, faces, face_labels, surfaces, points, labels)

            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok = True)
                out_path = os.path.join(args.output_dir, f"abc_{args.model_id}.xyzc")
                export_xyzc(points, labels, out_path)

    elif args.batch:
        if args.output_dir is None:
            print("--output_dir required for batch mode")
            exit(1)
        os.makedirs(args.output_dir, exist_ok = True)
        model_ids = list_available_models(args.abc_dir)
        rng = np.random.default_rng(args.seed)

        for i, mid in enumerate(model_ids):
            obj_path, feat_path = find_model_files(args.abc_dir, mid)
            vertices, faces = load_obj(obj_path)
            surfaces, _ = load_features(feat_path)
            face_labels = build_face_to_surface(surfaces, len(faces))
            points, labels = sample_points_from_mesh(vertices, faces, face_labels, args.num_points, args.min_points_per_surface, rng)

            if len(points) == 0:
                print(f"  [{i+1}/{len(model_ids)}] {mid}: skipped (no valid faces)")
                continue

            out_path = os.path.join(args.output_dir, f"abc_{mid}.xyzc")
            export_xyzc(points, labels, out_path)
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{len(model_ids)}")

    else:
        parser.print_help()
