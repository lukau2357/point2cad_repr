"""Diagnose: is the STEP crash caused by the transform or the compound?

Test 1: Export the compound WITHOUT any transform (normalized space)
Test 2: Export each face individually WITH the transform
Test 3: Export the full compound WITH the transform (what the pipeline does)

Open each in FreeCAD to isolate the problem.
"""
from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods, TopoDS_Compound
from OCC.Core.BRep import BRep_Builder
from OCC.Core.Message import Message_ProgressRange
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                               GeomAbs_Sphere, GeomAbs_BSplineSurface)

import os
os.makedirs("diag", exist_ok=True)

# Read the STEP file produced by the pipeline
reader = STEPControl_Reader()
status = reader.ReadFile("output_boundary/abc_00000078/part_0/part_0.step")
print(f"Read status: {status}")
reader.TransferRoots()
shape = reader.OneShape()

analyzer = BRepCheck_Analyzer(shape, True)
print(f"Overall shape valid: {analyzer.IsValid()}")

surf_type_names = {
    GeomAbs_Plane: "Plane", GeomAbs_Cylinder: "Cylinder",
    GeomAbs_Cone: "Cone", GeomAbs_Sphere: "Sphere",
    GeomAbs_BSplineSurface: "BSpline",
}

# Check each face
exp = TopExp_Explorer(shape, TopAbs_FACE)
faces = []
idx = 0
while exp.More():
    face = topods.Face(exp.Current())
    adaptor = BRepAdaptor_Surface(face)
    stype = adaptor.GetType()
    sname = surf_type_names.get(stype, f"unknown({stype})")
    valid = BRepCheck_Analyzer(face, True).IsValid()
    print(f"  Face {idx}: {sname}  valid={valid}")
    faces.append(face)
    idx += 1
    exp.Next()

print(f"\nTotal faces: {idx}")

# Test: re-export the compound as-is (read back → write)
print("\n--- Test: re-export compound as-is ---")
writer = STEPControl_Writer()
writer.Transfer(shape, STEPControl_AsIs, True, Message_ProgressRange())
ok = writer.Write("diag/reexport.step")
print(f"  Re-export: {'ok' if ok == IFSelect_RetDone else 'FAILED'}")

# Test: build a NEW compound from the individual faces and export
print("\n--- Test: new compound from individual faces ---")
builder = BRep_Builder()
compound2 = TopoDS_Compound()
builder.MakeCompound(compound2)
for f in faces:
    builder.Add(compound2, f)

analyzer2 = BRepCheck_Analyzer(compound2, True)
print(f"  New compound valid: {analyzer2.IsValid()}")

writer2 = STEPControl_Writer()
writer2.Transfer(compound2, STEPControl_AsIs, True, Message_ProgressRange())
ok2 = writer2.Write("diag/new_compound.step")
print(f"  New compound export: {'ok' if ok2 == IFSelect_RetDone else 'FAILED'}")

# Test: export first 5 faces as a compound (smaller test)
print("\n--- Test: small compound (first 5 faces) ---")
compound3 = TopoDS_Compound()
builder.MakeCompound(compound3)
for f in faces[:5]:
    builder.Add(compound3, f)
writer3 = STEPControl_Writer()
writer3.Transfer(compound3, STEPControl_AsIs, True, Message_ProgressRange())
ok3 = writer3.Write("diag/first_5.step")
print(f"  First 5 faces export: {'ok' if ok3 == IFSelect_RetDone else 'FAILED'}")

# Test: export each face individually
print("\n--- Test: individual face exports ---")
for i, f in enumerate(faces):
    w = STEPControl_Writer()
    w.Transfer(f, STEPControl_AsIs, True, Message_ProgressRange())
    ok_i = w.Write(f"diag/face_{i}.step")
    print(f"  Face {i}: {'ok' if ok_i == IFSelect_RetDone else 'FAILED'}")

print("\nDone! Try opening diag/reexport.step, diag/first_5.step, and")
print("individual diag/face_*.step files in FreeCAD to isolate the crash.")
