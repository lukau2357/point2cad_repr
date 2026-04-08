"""Test STEP orientation flips on closed-surface faces.

Produces modified STEP files where sphere face orientation flags are flipped.
Import original + modified in FreeCAD to check which produces correct hemisphere.
"""
import re
import os
import shutil

STEP_PATH = "output_brep/test_sphere_2planes/part_0/part_0.step"
OUT_DIR = "output_brep/test_sphere_2planes/part_0"

with open(STEP_PATH) as f:
    text = f.read()

# --- Identify sphere-backed ADVANCED_FACEs ---
# Pattern: ADVANCED_FACE('',(...),#surface_ref,.T./.F.)
af_re = re.compile(r"(#\d+)\s*=\s*ADVANCED_FACE\s*\(\s*'[^']*'\s*,\s*\([^)]*\)\s*,\s*(#\d+)\s*,\s*\.(T|F)\.\s*\)\s*;", re.DOTALL)
# Pattern: SPHERICAL_SURFACE
sphere_re = re.compile(r"(#\d+)\s*=\s*SPHERICAL_SURFACE\s*\(")

# Find all spherical surface entity IDs
sphere_ids = set(m.group(1) for m in sphere_re.finditer(text))
print(f"Spherical surfaces: {sphere_ids}")

# Find ADVANCED_FACEs referencing spherical surfaces
sphere_faces = []
for m in af_re.finditer(text):
    af_id, surf_ref, sense = m.group(1), m.group(2), m.group(3)
    if surf_ref in sphere_ids:
        sphere_faces.append((af_id, surf_ref, sense, m.span()))
        print(f"  {af_id} = ADVANCED_FACE on {surf_ref}, same_sense=.{sense}.")

# Find FACE_BOUND entries referenced by sphere ADVANCED_FACEs
fb_re = re.compile(r"(#\d+)\s*=\s*FACE_BOUND\s*\(\s*'[^']*'\s*,\s*#\d+\s*,\s*\.(T|F)\.\s*\)\s*;")
face_bounds = {m.group(1): (m.group(2), m.span()) for m in fb_re.finditer(text)}

# --- Variant 1: Flip ADVANCED_FACE same_sense for sphere faces ---
text_v1 = text
for af_id, surf_ref, sense, span in reversed(sphere_faces):
    old = text_v1[span[0]:span[1]]
    flipped = ".F." if sense == "T" else ".T."
    original = f".{sense}."
    # Replace the last occurrence of .T./.F. in the match (that's same_sense)
    idx = old.rfind(original)
    new = old[:idx] + flipped + old[idx+len(original):]
    text_v1 = text_v1[:span[0]] + new + text_v1[span[1]:]
    print(f"V1: {af_id} same_sense .{sense}. -> {flipped}")

out_v1 = os.path.join(OUT_DIR, "part_0_flip_samesense.step")
with open(out_v1, "w") as f:
    f.write(text_v1)
print(f"Wrote {out_v1}")

# --- Variant 2: Flip FACE_BOUND orientation for sphere faces ---
# Need to find which FACE_BOUND IDs are inside sphere ADVANCED_FACEs
fb_ref_re = re.compile(r"ADVANCED_FACE\s*\(\s*'[^']*'\s*,\s*\(([^)]*)\)")
text_v2 = text
for af_id, surf_ref, sense, span in sphere_faces:
    af_text = text[span[0]:span[1]]
    fb_refs = re.findall(r"#\d+", re.search(r"\(([^)]*)\)", af_text.split("ADVANCED_FACE")[1]).group(1))
    for fb_id in fb_refs:
        if fb_id in face_bounds:
            fb_sense, fb_span = face_bounds[fb_id]
            old_fb = text_v2[fb_span[0]:fb_span[1]]
            flipped = ".F." if fb_sense == "T" else ".T."
            original = f".{fb_sense}."
            idx = old_fb.rfind(original)
            new_fb = old_fb[:idx] + flipped + old_fb[idx+len(original):]
            text_v2 = text_v2[:fb_span[0]] + new_fb + text_v2[fb_span[1]:]
            print(f"V2: {fb_id} FACE_BOUND orientation .{fb_sense}. -> {flipped}")

out_v2 = os.path.join(OUT_DIR, "part_0_flip_facebound.step")
with open(out_v2, "w") as f:
    f.write(text_v2)
print(f"Wrote {out_v2}")

# --- Variant 3: Both flips ---
text_v3 = text
# Flip same_sense
for af_id, surf_ref, sense, span in reversed(sphere_faces):
    old = text_v3[span[0]:span[1]]
    flipped = ".F." if sense == "T" else ".T."
    original = f".{sense}."
    idx = old.rfind(original)
    new = old[:idx] + flipped + old[idx+len(original):]
    text_v3 = text_v3[:span[0]] + new + text_v3[span[1]:]
# Flip face_bound (re-parse since offsets changed)
fb_matches_v3 = {m.group(1): (m.group(2), m.span()) for m in fb_re.finditer(text_v3)}
for af_id, surf_ref, sense, _ in sphere_faces:
    af_match = re.search(re.escape(af_id) + r"\s*=\s*ADVANCED_FACE\s*\(\s*'[^']*'\s*,\s*\(([^)]*)\)", text_v3)
    if af_match:
        fb_refs = re.findall(r"#\d+", af_match.group(1))
        for fb_id in fb_refs:
            if fb_id in fb_matches_v3:
                fb_sense, fb_span = fb_matches_v3[fb_id]
                old_fb = text_v3[fb_span[0]:fb_span[1]]
                flipped = ".F." if fb_sense == "T" else ".T."
                original = f".{fb_sense}."
                idx = old_fb.rfind(original)
                new_fb = old_fb[:idx] + flipped + old_fb[idx+len(original):]
                text_v3 = text_v3[:fb_span[0]] + new_fb + text_v3[fb_span[1]:]
                print(f"V3: {fb_id} FACE_BOUND orientation .{fb_sense}. -> {flipped}")

out_v3 = os.path.join(OUT_DIR, "part_0_flip_both.step")
with open(out_v3, "w") as f:
    f.write(text_v3)
print(f"Wrote {out_v3}")

print("\n--- Summary ---")
print(f"Original:           {STEP_PATH}")
print(f"V1 (same_sense):    {out_v1}")
print(f"V2 (face_bound):    {out_v2}")
print(f"V3 (both flipped):  {out_v3}")
print("\nImport all four in FreeCAD and compare which sphere hemisphere is shown.")
