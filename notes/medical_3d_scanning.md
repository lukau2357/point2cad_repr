# Medical 3D Scanning & Marching Cubes

## Scanning pipeline

Medical scanners (CT, MRI) produce **volumetric data** — a 3D voxel grid, not a point cloud.
A voxel grid is a 3D tensor of shape $(D, H, W)$ where each entry holds a scalar intensity/density value.
Abstractly, a voxel grid generalizes a grayscale image by adding one more spatial index dimension:
- Grayscale image: $(H, W) \to \text{scalar}$
- Voxel grid: $(D, H, W) \to \text{scalar}$

For CT, voxel values are radiodensity in **Hounsfield Units (HU)**, calibrated against water (0 HU) and air (-1000 HU).

## Surface extraction: Marching Cubes

**Marching Cubes** extracts the **isosurface** from a voxel grid — the surface where the scalar field equals a given threshold.

Why not just threshold voxels directly? The threshold almost never lands exactly on a grid point.
If voxel A = 950 and neighbor B = 1050 with threshold = 1000, the isosurface passes *between* them.
Marching Cubes:
1. **Sub-voxel interpolation** — places vertices at interpolated positions along grid edges, not at grid points
2. **Connectivity** — connects crossing points into a consistent, watertight triangle mesh via a lookup table (256 cases, reduced to 15 by symmetry)

Simple thresholding yields blocky staircases (Minecraft-like). Marching Cubes yields smooth surfaces.

The 2D counterpart is **Marching Squares**. The algorithm is closer in spirit to contour extraction (`plt.contour` in 2D) than to Delaunay triangulation. Delaunay takes *points* and connects them; Marching Cubes takes a *scalar field* and extracts a boundary.

The resulting triangle mesh is the final geometric representation used by doctors for surgical planning, 3D printing, implant design, etc.

## Threshold selection

CT thresholds are standardized from physics (Hounsfield Units):
- Bone: ~+300 to +1900
- Soft tissue: ~+10 to +40
- Fat: ~-100 to -50
- Lung/air: ~-1000

Software (3D Slicer, Mimics) offers presets; doctors adjust a slider and see the surface update in real-time.
For low-contrast boundaries (e.g. tumors), manual or ML-based segmentation produces a binary mask first, then Marching Cubes meshes that mask.

## External scanning (structured light, photogrammetry)

For surface scanning (e.g. scanning a face externally), the result *is* a point cloud, meshed by standard algorithms like **Screened Poisson** or **Ball Pivoting (BPA)**.

## Where parametric fitting (Point2CAD-style) fits

Medical imaging does not benefit from parametric surface decomposition — triangle meshes are the end product.

The right market is **mechanical/industrial engineering**: reverse engineering manufactured parts, quality inspection, digital twin creation. The value is going from a triangle mesh to a structured CAD model with semantic surfaces (this face is a plane, that hole is a cylinder, etc.), enabling re-manufacture, tolerance analysis, and CAD software integration (STEP/IGES).
