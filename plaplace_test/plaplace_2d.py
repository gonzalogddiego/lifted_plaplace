import argparse
import sys
from functools import partial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from plaplace_test.plaplace_plot import plot_solutions, solution_name


P_VALUES = (1.6, 1.3, 1.1, 1.05, 1.01)
METHOD_NAMES = (
    "standard",
    "lifted",
    "picard",
    "pnls"
)
STEP_NAMES = tuple(name for name in METHOD_NAMES if name != "pnls")
if "pnls" in METHOD_NAMES:
    STEP_NAMES += ("pnls_picard", "pnls_newton")

TEST_DIR = Path(__file__).resolve().parent
DATA_DIR = TEST_DIR / "data/plaplace_2d"
FIGS_DIR = TEST_DIR / "figs/plaplace_2d"


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def p_tag(p):
    return f"p{int(p * 100)}"


def make_problem(fd, n):
    mesh = fd.UnitSquareMesh(n, n, diagonal="crossed")
    x, y = fd.SpatialCoordinate(mesh)

    eps = fd.Constant(1e-12)
    left_strip = fd.And(
        fd.lt(abs(x), eps),
        fd.And(fd.ge(y, 0.25), fd.le(y, 0.75)),
    )
    upper_right = fd.And(
        fd.ge(x, 0.6),
        fd.And(fd.ge(y, 0.25), fd.le(y, 1)),
    )
    boundary_value = fd.conditional(fd.Or(left_strip, upper_right), 1, 0)
    return mesh, fd.Constant(0), boundary_value


def solve_problem(method, mesh, p, f_rhs, boundary_value, args):
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
    )
    success, data = solver.solve(return_data=True)
    return solver, success, data


def save_solution(fd, solution, p, data_dir):
    name = solution_name(p)
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

    pnls_diagnostics = solve_data["pnls"]["diagnostics"]
    step_lengths["pnls_picard"] = np.asarray(
        [item.get("picard_step_length", np.nan) for item in pnls_diagnostics],
        dtype=float,
    )
    step_lengths["pnls_newton"] = np.asarray(
        [item.get("newton_step_length", np.nan) for item in pnls_diagnostics],
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Newton variants on the two-dimensional p-Laplace problem."
    )
    parser.add_argument("--N", type=positive_int, default=50)
    parser.add_argument("--delta-min", type=float, default=1e-6, dest="delta_min")
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--max-iter", type=positive_int, default=500)
    parser.add_argument("--max-iter-ls", type=positive_int, default=20)
    parser.add_argument("--quad-degree", type=positive_int, default=5)
    parser.add_argument("--projection-mode", choices=("lhs", "state"), default="lhs")
    parser.add_argument(
        "--suffix",
        default="",
        help="suffix appended to the data and figure output directory names",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sys.argv = [sys.argv[0]]

    data_dir = DATA_DIR.with_name(DATA_DIR.name + args.suffix)
    figs_dir = FIGS_DIR.with_name(FIGS_DIR.name + args.suffix)
    data_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    import firedrake as fd
    from plaplace import PLaplace, PLaplaceLifted, PLaplacePicard, PLaplacePNLS

    methods = {
        "standard": PLaplace,
        "lifted": partial(PLaplaceLifted, lambda_projection_mode=args.projection_mode),
        "picard": PLaplacePicard,
        "pnls": PLaplacePNLS,
    }
    mesh, f_rhs, boundary_value = make_problem(fd, args.N)
    gradient_records = []
    step_records = []
    iteration_records = []

    for p in P_VALUES:
        print(f"\nSolving the p-Laplace problem for p = {p:g}")
        solve_data = {}

        for name in METHOD_NAMES:
            solver, success, solve_data[name] = solve_problem(
                methods[name],
                mesh,
                p,
                f_rhs,
                boundary_value,
                args,
            )
            print(f"  {name}: {'converged' if success else 'did not converge'}")
            if name == "lifted":
                save_solution(fd, solver.u, p, data_dir)

        gradient_records.append((p, extract_gradients(solve_data)))
        step_records.append((p, extract_step_lengths(solve_data)))
        iteration_records.append((p, extract_iterations(solve_data)))

    save_table(
        data_dir / "rel_grad_norms_plaplace_2d.txt",
        gradient_records,
        METHOD_NAMES,
        first_iteration=0,
        relative=True,
    )
    save_table(
        data_dir / "step_lengths_plaplace_2d.txt",
        step_records,
        STEP_NAMES,
        first_iteration=1,
    )
    save_iteration_table(
        data_dir / "iterations_plaplace_2d.txt",
        iteration_records,
    )
    plot_solutions(fd, P_VALUES, data_dir, figs_dir)


if __name__ == "__main__":
    main()
