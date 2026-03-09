from OCC.Core.gp import gp_Ax3, gp_Pnt, gp_Dir                                                                                                                                        
from OCC.Core.Geom import Geom_CylindricalSurface                                                                                                                                     
from OCC.Core.GeomAPI import GeomAPI_IntSS                                                                                                                                            
                                                                                                                                                                                    
# Near-coaxial: axes slightly off
ax1 = gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
ax2 = gp_Ax3(gp_Pnt(0.001, 0, 0), gp_Dir(0.001, 0, 1))
cyl1 = Geom_CylindricalSurface(ax1, 1.0)
cyl2 = Geom_CylindricalSurface(ax2, 2.0)

try:
    inter = GeomAPI_IntSS(cyl1, cyl2, 1e-6)
    print(f"IsDone: {inter.IsDone()}, NbLines: {inter.NbLines()}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")
