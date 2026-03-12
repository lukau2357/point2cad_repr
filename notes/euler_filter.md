# Progressive Euler Filter (removed)

The progressive Euler filter (`progressive_euler_filter`) has been removed
from `topology.py`.  Its role — removing spurious vertices and arcs to achieve
Eulerian face wire graphs — is now handled by the **greedy oracle filter**
(see `notes/brep_construction.md`, Section 0d).

The key difference: the progressive Euler filter attempted to fix the wire
graph heuristically by scoring and removing vertices/arcs until all faces were
Eulerian.  The greedy oracle filter instead **tries building the actual BRep
via OCC** after each removal, using `BRepCheck_Analyzer` as the validity
oracle.  This is more robust because:

1. The Eulerian condition is necessary but not sufficient — a model can be
   Eulerian yet fail `BRepCheck_Analyzer` due to geometric issues.
2. OCC's sewing can resolve some near-duplicate edges (e.g. nearly-closed
   BSplines from cone∩cylinder intersections) that would cause issues for
   the Euler filter's graph-based reasoning.
3. The oracle approach avoids the cascade risk inherent in vertex removal
   (dropping a vertex drops all incident arcs, potentially making
   previously-Eulerian faces non-Eulerian).

The scoring helpers (`_score_vertex`, `_score_arc`) and the Euler check
(`_non_eulerian_faces_direct`) are retained in `topology.py` as they are
used by the oracle filter.

## Removed functions

- `progressive_euler_filter`
- `_merge_arcs_at_vertex`
- `_make_merged_arc`
- `_vertex_faces`
- `_non_eulerian_faces` (the version with `removed_vertices`/`removed_arcs` params)
- `_candidate_faces`
- `filter_vertices_by_proximity`
- `filter_curves_by_proximity`
- `filter_arcs_by_proximity`
- `_arc_inside_bboxes`
- `_passes_nn`
