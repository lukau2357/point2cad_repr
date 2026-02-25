import os, glob

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp      import TopExp_Explorer
from OCC.Core.TopAbs      import TopAbs_FACE
from OCC.Core.TopoDS           import topods
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface

step_id = "00000000"
step_path = "../abc_dataset/abc_0000_step_v00"
step_path = os.path.join(step_path, step_id)
steps = glob.glob(os.path.join(step_path, "*.step"))

if not steps:
    print("No step file found for given ID")
    exit(0)

step = steps[0]
reader = STEPControl_Reader()
reader.ReadFile(step)
reader.TransferRoots()
shape = reader.OneShape()

exp = TopExp_Explorer(shape, TopAbs_FACE)
while exp.More():
    face    = topods.Face(exp.Current())
    adaptor = BRepAdaptor_Surface(face)
    stype   = adaptor.GetType()   # GeomAbs_Plane, GeomAbs_Cylinder, etc.
    print(stype)
    exp.Next()