"""
Surface type integer constants — single source of truth for the entire pipeline.

Rules for adding a new primitive (e.g. torus):
  1. Append its constant here (keep values contiguous starting from 0).
  2. Update SURFACE_NAMES with its display name.
  3. In surface_fitter.py, add its fitter to PRIMITIVE_FITTERS and its mesh
     generator to PRIMITIVE_MESHERS using the same key.

IMPORTANT: The integer values must stay contiguous starting from 0 so that
surface_fitter.fit_surface can build an error array indexed directly by
surface ID via sorted(PRIMITIVE_FITTERS).
"""

SURFACE_PLANE    = 0
SURFACE_SPHERE   = 1
SURFACE_CYLINDER = 2
SURFACE_CONE     = 3
SURFACE_INR      = 4

SURFACE_NAMES = {
    SURFACE_PLANE:    "plane",
    SURFACE_SPHERE:   "sphere",
    SURFACE_CYLINDER: "cylinder",
    SURFACE_CONE:     "cone",
    SURFACE_INR:      "inr",
}
