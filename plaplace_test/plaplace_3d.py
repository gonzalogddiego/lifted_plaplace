import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


P_VALUES = (1.6, 1.1, 1.01)
ELEMENTS = ("CG1", "CG2")
METHOD_NAMES = ("standard", "lifted", "picard", "pnls")
STEP_NAMES = tuple(name for name in METHOD_NAMES if name != "pnls")
if "pnls" in METHOD_NAMES:
    STEP_NAMES += ("pnls_picard", "pnls_newton")
PLANE_OFFSET = 5.0 / 2.0

TEST_DIR = Path(__file__).resolve().parent
DEFAULT_MESH = TEST_DIR / "plaplace_3d_mesh.msh"
MESH_GENERATOR = TEST_DIR / "generate_plaplace_3d_adaptive_mesh.py"
DATA_ROOT = TEST_DIR / "data/plaplace_3d"


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def p_tag(p):
    return f"p{int(p * 100)}"


def solution_name(element, p):
    return f"plaplace_3d_{element}_{p_tag(p)}"


def make_mesh(fd, args):
    if args.mesh_type == "uniform":
        mesh = fd.UnitCubeMesh(args.N, args.N, args.N)
        mesh_name = f"uniform_N{args.N}"
    else:
        mesh_path = args.mesh_file.expanduser().resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(
                f"Adaptive mesh not found: {mesh_path}\n"
                f"Generate it with: {sys.executable} {MESH_GENERATOR}"
            )
        mesh = fd.Mesh(str(mesh_path))
        mesh_name = mesh_path.stem

    print(f"Using {mesh_name} ({mesh.num_cells()} tetrahedra on this rank)")
    return mesh, mesh_name


def make_problem(fd, mesh):
    x, y, z = fd.SpatialCoordinate(mesh)
    boundary_value = fd.conditional(
        fd.ge(x + y + z, PLANE_OFFSET - 1.0e-4),
        1.0,
        0.0,
    )
    return fd.Constant(0.0), boundary_value


def solve_problem(method, mesh, p, f_rhs, boundary_value, element, args):
    solver = method(
        mesh,
        p,
        args.delta_min,
        f_rhs,
        boundary_value,
        rtol=args.rtol,
        max_iter=args.max_iter,
        max_iter_ls=args.max_iter_ls,
        quad_degree=args.quad_degree,
        element_family="CG",
        element_degree=int(element[2:]),
    )
    success, data = solver.solve(return_data=True)
    return solver, success, data


def save_solution(fd, solution, element, p, data_dir):
    name = solution_name(element, p)
    saved_solution = fd.Function(solution.function_space(), name=name)
    saved_solution.assign(solution)

    fd.VTKFile(str(data_dir / f"{name}.pvd")).write(saved_solution)
    with fd.CheckpointFile(str(data_dir / f"{name}.h5"), "w") as checkpoint:
        checkpoint.save_mesh(solution.function_space().mesh())
        checkpoint.save_function(saved_solution, name=name)


def extract_gradients(solve_data):
    gradients = {
        name: np.asarray(solve_data[name]["gradients"], dtype=float)
        for name in METHOD_NAMES
        if name != "pnls"
    }

    if "pnls" not in solve_data:
        return gradients

    pnls_data = solve_data["pnls"]
    pnls_gradients = [pnls_data["gradients"][0]]
    for final_gradient, diagnostics in zip(
        pnls_data["gradients"][1:],
        pnls_data["diagnostics"],
    ):
        pnls_gradients.extend(
            (diagnostics.get("predictor_gradient", np.nan), final_gradient)
        )

    gradients["pnls"] = np.asarray(pnls_gradients, dtype=float)
    return gradients


def extract_step_lengths(solve_data):
    step_lengths = {
        name: np.asarray(solve_data[name]["step_lengths"], dtype=float)
        for name in METHOD_NAMES
        if name != "pnls"
    }
    if "pnls" not in solve_data:
        return step_lengths

    diagnostics = solve_data["pnls"]["diagnostics"]
    step_lengths["pnls_picard"] = np.asarray(
        [item.get("picard_step_length", np.nan) for item in diagnostics],
        dtype=float,
    )
    step_lengths["pnls_newton"] = np.asarray(
        [item.get("newton_step_length", np.nan) for item in diagnostics],
        dtype=float,
    )
    return step_lengths


def extract_iterations(solve_data):
    iterations = {
        name: int(solve_data[name]["iterations"])
        for name in METHOD_NAMES
    }
    if "pnls" in iterations:
        iterations["pnls"] *= 2
    return iterations


def save_table(path, records, series_names, first_iteration, relative=False):
    nrows = max(
        len(record[name])
        for _, record in records
        for name in series_names
    )
    data = np.full((nrows, 1 + len(records) * len(series_names)), np.nan)
    data[:, 0] = np.arange(first_iteration, first_iteration + nrows)
    header = ["iteration"]

    column = 1
    for p, record in records:
        for name in series_names:
            values = record[name]
            if relative and values.size:
                initial_value = values[0]
                if np.isfinite(initial_value) and initial_value != 0.0:
                    values = values / initial_value

            data[: len(values), column] = values
            header.append(f"{name}_{p_tag(p)}")
            column += 1

    np.savetxt(
        path,
        data,
        header=" ".join(header),
        fmt=["%d"] + ["%.16e"] * (data.shape[1] - 1),
        comments="",
    )


def save_iteration_table(path, records):
    lines = ["method " + " ".join(p_tag(p) for p, _ in records)]
    for name in METHOD_NAMES:
        counts = " ".join(str(record[name]) for _, record in records)
        lines.append(f"{name} {counts}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_element(
    fd,
    methods,
    mesh,
    element,
    f_rhs,
    boundary_value,
    data_dir,
    args,
):
    gradient_records = []
    step_records = []
    iteration_records = []

    for p in P_VALUES:
        print(f"\nSolving the 3D p-Laplace problem for p = {p:g}, {element}")
        solve_data = {}

        for name in METHOD_NAMES:
            solver, success, solve_data[name] = solve_problem(
                methods[name],
                mesh,
                p,
                f_rhs,
                boundary_value,
                element,
                args,
            )
            print(f"  {name}: {'converged' if success else 'did not converge'}")
            if name == "lifted":
                save_solution(fd, solver.u, element, p, data_dir)

        gradient_records.append((p, extract_gradients(solve_data)))
        step_records.append((p, extract_step_lengths(solve_data)))
        iteration_records.append((p, extract_iterations(solve_data)))

    save_table(
        data_dir / f"rel_grad_norms_plaplace_3d_{element}.txt",
        gradient_records,
        METHOD_NAMES,
        first_iteration=0,
        relative=True,
    )
    save_table(
        data_dir / f"step_lengths_plaplace_3d_{element}.txt",
        step_records,
        STEP_NAMES,
        first_iteration=1,
    )
    save_iteration_table(
        data_dir / f"iterations_plaplace_3d_{element}.txt",
        iteration_records,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare Newton variants on the three-dimensional p-Laplace problem."
    )
    parser.add_argument(
        "--mesh-type",
        choices=("adaptive", "uniform"),
        default="adaptive",
    )
    parser.add_argument("--mesh-file", type=Path, default=DEFAULT_MESH)
    parser.add_argument(
        "--N",
        type=positive_int,
        default=10,
        help="subdivisions per direction for a uniform mesh (default: 10)",
    )
    parser.add_argument(
        "--elements",
        nargs="+",
        choices=ELEMENTS,
        default=ELEMENTS,
    )
    parser.add_argument("--delta-min", type=float, default=1e-6, dest="delta_min")
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--max-iter", type=positive_int, default=500)
    parser.add_argument("--max-iter-ls", type=positive_int, default=20)
    parser.add_argument("--quad-degree", type=positive_int, default=5)
    parser.add_argument(
        "--suffix",
        default="",
        help="suffix appended to the data output directory name",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.elements = tuple(dict.fromkeys(args.elements))
    sys.argv = [sys.argv[0]]

    import firedrake as fd
    from plaplace import PLaplace, PLaplaceLifted, PLaplacePicard, PLaplacePNLS

    methods = {
        "standard": PLaplace,
        "lifted": PLaplaceLifted,
        "picard": PLaplacePicard,
        "pnls": PLaplacePNLS,
    }
    try:
        mesh, mesh_name = make_mesh(fd, args)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    data_root = DATA_ROOT.with_name(DATA_ROOT.name + args.suffix)
    data_dir = data_root / mesh_name
    data_dir.mkdir(parents=True, exist_ok=True)
    f_rhs, boundary_value = make_problem(fd, mesh)

    for element in args.elements:
        run_element(
            fd,
            methods,
            mesh,
            element,
            f_rhs,
            boundary_value,
            data_dir,
            args,
        )


if __name__ == "__main__":
    main()
