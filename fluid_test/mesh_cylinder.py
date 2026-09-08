"""
2D mesh: flow around a cylinder.

Domain  : rectangle [0, L] x [0, H]
Cylinder: disk centred at (cx, cy) with radius r

Boundary physical tags
  1 – Left     (x = 0)
  2 – Right    (x = L)
  3 – Bottom   (y = 0)
  4 – Top      (y = H)
  5 – Cylinder (circle)

Surface physical tag
  1 – Fluid
"""

import sys
import gmsh
import argparse

parser = argparse.ArgumentParser(description="make cylinder mesh")
parser.add_argument("--nref", type=int, default=1, dest="nref")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Geometry parameters
# ---------------------------------------------------------------------------
L, H = 2.2, 0.41         # domain length and height
cx, cy, r = 0.2, 0.2, 0.05    # cylinder centre and radius (DFG benchmark)

# Mesh-size parameters
h_far  = 0.05 / args.nref   # element size far from cylinder  (scales with refinement)
h_cyl  = 0.02 / args.nref   # element size on cylinder surface (scales with refinement)
d_min  = 0.05               # distance below which h = h_cyl  (geometry, fixed)
d_max  = 0.20               # distance above which h = h_far  (geometry, fixed)

# ---------------------------------------------------------------------------
# Build geometry with OpenCASCADE kernel
# ---------------------------------------------------------------------------
gmsh.initialize()
gmsh.model.add("flow_around_cylinder")

occ = gmsh.model.occ

rect_tag = occ.addRectangle(0, 0, 0, L, H)
disk_tag = occ.addDisk(cx, cy, 0, r, r)

# Remove disk from rectangle (keep_tool=False discards the disk surface)
occ.cut([(2, rect_tag)], [(2, disk_tag)], removeObject=True, removeTool=True)
occ.synchronize()

# ---------------------------------------------------------------------------
# Identify boundary curves from their bounding boxes
# ---------------------------------------------------------------------------
tol = 1e-6

left_tags     = []
right_tags    = []
bottom_tags   = []
top_tags      = []
cylinder_tags = []

for dim, tag in gmsh.model.getEntities(dim=1):
    xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dim, tag)
    dx = xmax - xmin
    dy = ymax - ymin

    if dx < tol and abs(xmin) < tol:           # vertical line at x = 0
        left_tags.append(tag)
    elif dx < tol and abs(xmin - L) < tol:     # vertical line at x = L
        right_tags.append(tag)
    elif dy < tol and abs(ymin) < tol:          # horizontal line at y = 0
        bottom_tags.append(tag)
    elif dy < tol and abs(ymin - H) < tol:     # horizontal line at y = H
        top_tags.append(tag)
    else:                                        # curved: the cylinder
        cylinder_tags.append(tag)

# ---------------------------------------------------------------------------
# Physical groups
# ---------------------------------------------------------------------------
# Boundaries (dim = 1)
gmsh.model.addPhysicalGroup(1, left_tags,     tag=1, name="Left")
gmsh.model.addPhysicalGroup(1, right_tags,    tag=2, name="Right")
gmsh.model.addPhysicalGroup(1, bottom_tags,   tag=3, name="Bottom")
gmsh.model.addPhysicalGroup(1, top_tags,      tag=4, name="Top")
gmsh.model.addPhysicalGroup(1, cylinder_tags, tag=5, name="Cylinder")

# Domain surface (dim = 2)
fluid_tags = [tag for _, tag in gmsh.model.getEntities(dim=2)]
gmsh.model.addPhysicalGroup(2, fluid_tags, tag=1, name="Fluid")

# ---------------------------------------------------------------------------
# Mesh-size fields – refinement near the cylinder
# ---------------------------------------------------------------------------
gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "CurvesList", cylinder_tags)
gmsh.model.mesh.field.setNumber(1, "Sampling", 200)

gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField",  1)
gmsh.model.mesh.field.setNumber(2, "SizeMin",  h_cyl)
gmsh.model.mesh.field.setNumber(2, "SizeMax",  h_far)
gmsh.model.mesh.field.setNumber(2, "DistMin",  d_min)
gmsh.model.mesh.field.setNumber(2, "DistMax",  d_max)

gmsh.model.mesh.field.setAsBackgroundMesh(2)

# Disable the automatic global size bounds so the field drives everything
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

# ---------------------------------------------------------------------------
# Generate, optimise, and save
# ---------------------------------------------------------------------------
gmsh.model.mesh.generate(2)
gmsh.model.mesh.optimize("Netgen")

gmsh.write("cylinder_flow.msh")
print("Mesh written to cylinder_flow.msh")
print()
print("Boundary physical tags:")
print("  1 – Left     (x = 0)")
print("  2 – Right    (x = L = {})".format(L))
print("  3 – Bottom   (y = 0)")
print("  4 – Top      (y = H = {})".format(H))
print("  5 – Cylinder (centre ({}, {}), r = {})".format(cx, cy, r))
print()
print("Surface physical tag:")
print("  1 – Fluid")

# if "-nopopup" not in sys.argv:
#     gmsh.fltk.run()

gmsh.finalize()
