"""Generate a unit-cube mesh refined around x + y + z = 5/2."""

import argparse
import itertools
import sys
import tempfile
from pathlib import Path

import numpy as np


N = 5
REFINEMENT_LEVELS = 3
LAYER_WIDTH = 0.01
PLANE_OFFSET = 5.0 / 2.0

OUTPUT_DIR = Path(__file__).resolve().parent
MSH_OUTPUT = OUTPUT_DIR / "plaplace_3d_mesh.msh"
VTU_OUTPUT = OUTPUT_DIR / "plaplace_3d_mesh.vtu"

BOUNDARY_NAMES = {
    1: "left",
    2: "right",
    3: "front",
    4: "back",
    5: "bottom",
    6: "top",
}
OUTWARD_NORMALS = {
    1: np.asarray((-1.0, 0.0, 0.0)),
    2: np.asarray((1.0, 0.0, 0.0)),
    3: np.asarray((0.0, -1.0, 0.0)),
    4: np.asarray((0.0, 1.0, 0.0)),
    5: np.asarray((0.0, 0.0, -1.0)),
    6: np.asarray((0.0, 0.0, 1.0)),
}


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def nonnegative_float(value):
    value = float(value)
    if value < 0.0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Refine a unit-cube mesh around x + y + z = 5/2."
    )
    parser.add_argument(
        "--N",
        type=positive_int,
        default=N,
        help=f"coarse subdivisions in each direction (default: {N})",
    )
    parser.add_argument(
        "--levels",
        type=nonnegative_int,
        default=REFINEMENT_LEVELS,
        help=f"number of local refinement passes (default: {REFINEMENT_LEVELS})",
    )
    parser.add_argument(
        "--layer-width",
        type=nonnegative_float,
        default=LAYER_WIDTH,
        help=(
            "half-width of the refined slab measured perpendicular to the "
            f"plane (default: {LAYER_WIDTH:g})"
        ),
    )
    return parser.parse_args(argv)


def unit_cube_arrays(mesh):
    """Return consistently oriented coordinates and tetrahedra."""
    coordinates = np.asarray(
        mesh.coordinates.dat.data_ro, dtype=np.float64
    ).copy()
    cells = np.asarray(
        mesh.coordinates.cell_node_map().values, dtype=np.int32
    ).copy()

    for cell in cells:
        points = coordinates[cell]
        jacobian = np.stack(
            (
                points[1] - points[0],
                points[2] - points[0],
                points[3] - points[0],
            ),
            axis=1,
        )
        if np.linalg.det(jacobian) < 0.0:
            cell[0], cell[1] = cell[1], cell[0]

    return coordinates, cells


def boundary_tag(points, tolerance):
    boundaries = (
        (1, 0, 0.0),
        (2, 0, 1.0),
        (3, 1, 0.0),
        (4, 1, 1.0),
        (5, 2, 0.0),
        (6, 2, 1.0),
    )
    matches = [
        tag
        for tag, axis, value in boundaries
        if np.all(np.abs(points[:, axis] - value) <= tolerance)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"could not identify boundary triangle {points.tolist()}")
    return matches[0]


def boundary_triangles(coordinates, cells):
    """Return outwardly oriented exterior triangles grouped by cube face."""
    face_counts = {}
    for cell in cells:
        for face in itertools.combinations(cell.tolist(), 3):
            face = tuple(sorted(face))
            face_counts[face] = face_counts.get(face, 0) + 1

    tolerance = 128.0 * np.finfo(coordinates.dtype).eps
    triangles = {tag: [] for tag in BOUNDARY_NAMES}
    for face, count in face_counts.items():
        if count != 1:
            continue

        triangle = list(face)
        points = coordinates[triangle]
        tag = boundary_tag(points, tolerance)
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        if np.dot(normal, OUTWARD_NORMALS[tag]) < 0.0:
            triangle[1], triangle[2] = triangle[2], triangle[1]
        triangles[tag].append(triangle)

    return {
        tag: np.asarray(faces, dtype=np.int32)
        for tag, faces in triangles.items()
    }


def make_netgen_seed(netgen_meshing, coordinates, cells):
    """Copy the coarse UnitCubeMesh into Netgen for marked refinement."""
    mesh = netgen_meshing.Mesh(dim=3)
    mesh.AddPoints(coordinates)
    mesh.SetMaterial(1, "domain")
    mesh.AddElements(dim=3, index=1, data=cells, base=0)

    for tag, faces in boundary_triangles(coordinates, cells).items():
        descriptor = netgen_meshing.FaceDescriptor(
            surfnr=tag,
            domin=1,
            domout=0,
            bc=tag,
        )
        descriptor_index = mesh.Add(descriptor)
        mesh.SetBCName(tag - 1, BOUNDARY_NAMES[tag])
        mesh.AddElements(
            dim=2,
            index=descriptor_index,
            data=faces,
            base=0,
        )

    mesh.Compress()
    return mesh


def mark_layer_cells(fd, mesh, layer_width):
    coordinates = np.asarray(mesh.coordinates.dat.data_ro)
    cells = np.asarray(mesh.coordinates.cell_node_map().values)
    vertex_sums = coordinates[cells].sum(axis=2)

    sum_half_width = np.sqrt(3.0) * layer_width
    lower = PLANE_OFFSET - sum_half_width
    upper = PLANE_OFFSET + sum_half_width
    tolerance = 128.0 * np.finfo(coordinates.dtype).eps
    marked = np.logical_and(
        vertex_sums.min(axis=1) <= upper + tolerance,
        vertex_sums.max(axis=1) >= lower - tolerance,
    )

    marker = fd.Function(fd.FunctionSpace(mesh, "DG", 0))
    marker.dat.data[:] = marked
    return marker, int(marked.sum())


def mesh_arrays(mesh):
    coordinates = np.asarray(
        mesh.coordinates.dat.data_ro, dtype=np.float64
    ).copy()
    cells = np.asarray(
        mesh.coordinates.cell_node_map().values, dtype=np.int32
    ).copy()
    return coordinates, cells


def write_msh(mesh):
    import gmsh

    MSH_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary_dir:
        netgen_output = Path(temporary_dir) / "mesh.msh"
        mesh.netgen_mesh.Export(str(netgen_output), "Gmsh2 Format")

        gmsh.initialize()
        try:
            gmsh.open(str(netgen_output))
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.write(str(MSH_OUTPUT))
        finally:
            gmsh.finalize()


def write_vtu(coordinates, cells):
    import vtk

    points = vtk.vtkPoints()
    for point in coordinates:
        points.InsertNextPoint(*point)

    mesh = vtk.vtkUnstructuredGrid()
    mesh.SetPoints(points)
    for cell in cells:
        mesh.InsertNextCell(vtk.VTK_TETRA, 4, cell.tolist())

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(VTU_OUTPUT))
    writer.SetInputData(mesh)
    writer.SetDataModeToBinary()
    if writer.Write() != 1:
        raise RuntimeError(f"could not write {VTU_OUTPUT}")


def generate_mesh(n, refinement_levels, layer_width):
    try:
        import firedrake as fd
        from netgen import meshing as netgen_meshing
    except ImportError as exc:
        raise RuntimeError(
            "mesh generation requires Firedrake, Netgen, Gmsh, and VTK"
        ) from exc

    if fd.COMM_WORLD.size != 1:
        raise RuntimeError("mesh generation must use one MPI process")

    coarse_mesh = fd.UnitCubeMesh(
        n,
        n,
        n,
        comm=fd.COMM_SELF,
        reorder=False,
    )
    coordinates, cells = unit_cube_arrays(coarse_mesh)
    mesh = fd.Mesh(
        make_netgen_seed(netgen_meshing, coordinates, cells),
        comm=fd.COMM_SELF,
        reorder=False,
    )

    print(
        f"Coarse mesh: {len(coordinates)} vertices, "
        f"{len(cells)} tetrahedra"
    )
    for level in range(1, refinement_levels + 1):
        marker, marked_cells = mark_layer_cells(fd, mesh, layer_width)
        cells_before = mesh.num_cells()
        mesh = mesh.refine_marked_elements(marker)
        print(
            f"Refinement {level}: marked {marked_cells}/{cells_before} cells; "
            f"result has {mesh.num_cells()} tetrahedra"
        )

    write_msh(mesh)
    coordinates, cells = mesh_arrays(mesh)
    write_vtu(coordinates, cells)

    print(f"Gmsh mesh written to {MSH_OUTPUT}")
    print(f"ParaView mesh written to {VTU_OUTPUT}")
    print(f"Final mesh: {len(coordinates)} vertices, {len(cells)} tetrahedra")
    print(
        f"N={n}, levels={refinement_levels}, layer_width={layer_width:g}, "
        "plane=x+y+z=5/2"
    )


def main(argv=None):
    args = parse_args(argv)
    sys.argv = [sys.argv[0]]
    generate_mesh(args.N, args.levels, args.layer_width)


if __name__ == "__main__":
    main()
