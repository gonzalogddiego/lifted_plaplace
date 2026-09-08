import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

from plot_utils import linewidth, pgf_with_latex


P_VALUES = (1.6, 1.3, 1.1, 1.05, 1.01)
COLORMAP = "viridis"

TEST_DIR = Path(__file__).resolve().parent
DATA_DIR = TEST_DIR / "data/plaplace_2d"
FIGS_DIR = TEST_DIR / "figs/plaplace_2d"


def solution_name(p):
    return f"plaplace_2d_p{int(p * 100)}"


def load_solution(fd, data_dir, p):
    name = solution_name(p)
    with fd.CheckpointFile(str(data_dir / f"{name}.h5"), "r") as checkpoint:
        mesh = checkpoint.load_mesh()
        return checkpoint.load_function(mesh, name=name)


def plot_solutions(fd, p_values, data_dir=DATA_DIR, figs_dir=FIGS_DIR):
    from matplotlib import pyplot as plt
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize

    p_values = tuple(p_values)
    if not p_values:
        raise ValueError("p_values must not be empty")

    solutions = [(p, load_solution(fd, data_dir, p)) for p in p_values]
    vmin = min(float(solution.dat.data_ro.min()) for _, solution in solutions)
    vmax = max(float(solution.dat.data_ro.max()) for _, solution in solutions)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps[COLORMAP]
    panel_size = linewidth / len(p_values)

    figs_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    with matplotlib.rc_context(
        pgf_with_latex(nplots=len(p_values), hscale=0.72)
    ):
        for p, solution in solutions:
            fig, ax = plt.subplots(
                figsize=(panel_size, panel_size),
                constrained_layout=True,
            )
            fd.tripcolor(solution, axes=ax, cmap=cmap, norm=norm)
            ax.set_aspect("equal")
            ax.set_axis_off()

            output_path = figs_dir / f"{solution_name(p)}.pdf"
            fig.savefig(output_path)
            plt.close(fig)
            output_paths.append(output_path)

        colorbar_path = figs_dir / "plaplace_2d_colorbar.pdf"
        colorbar_fig, colorbar_ax = plt.subplots(
            figsize=(0.075 * linewidth, panel_size),
            constrained_layout=True,
        )
        ColorbarBase(
            colorbar_ax,
            cmap=cmap,
            norm=norm,
            orientation="vertical",
        )
        colorbar_fig.savefig(colorbar_path)
        plt.close(colorbar_fig)
        output_paths.append(colorbar_path)

    print("Saved plots:")
    for output_path in output_paths:
        print(f"  {output_path}")

    return output_paths


def main():
    sys.argv = [sys.argv[0]]

    import firedrake as fd

    plot_solutions(fd, P_VALUES)


if __name__ == "__main__":
    main()
