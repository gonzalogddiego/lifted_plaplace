#!/usr/bin/env python3
"""Compare PStokes and PStokesLifted lid-driven cavity solutions (tikz-friendly outputs).

Reads newton_grad.npy and velocity_pressure.h5 from runs produced by
pstokes.py (lid-driven cavity section) and produces, for each of the two
p-values in --p_fig:
  - a standalone velocity-contour + streamlines plot  →  plots/lid_contour_p<PP>.png
  - a plain-text Newton gradient-norm table            →  plots/lid_iteration_p<PP>.txt
    (columns: iteration, PStokes, PStokesLifted; ready for a pgfplots \\addplot table)

Also writes the same LaTeX Newton-iteration-count table as compare_pstokes_lid.py
  →  plots/pstokes_lid_newton_its.txt

No helper functions are defined; everything is inlined at module scope.
"""

import argparse
import os
import sys
import numpy as np
from scipy.interpolate import griddata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(
    description="Compare PStokes and PStokesLifted lid-driven cavity runs (tikz outputs)"
)
parser.add_argument("--p",     type=float, nargs="+", default=[1.5, 1.1, 1.05, 1.01],
                    help="p values for the table (default: 1.5 1.1 1.05 1.01)")
parser.add_argument("--p_fig", type=float, nargs=2,   default=[1.5, 1.01],
                    help="two p values shown in the separate figures/txt files (default: 1.5 1.01)")
parser.add_argument("--N",     type=int,   nargs="+", default=[25, 50, 100],
                    help="mesh sizes for the table (default: 25 50 100); largest N used for figures")
args = parser.parse_args()

from firedrake import *
from firedrake.pyplot import tripcolor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_utils import linewidth, pgf_with_latex
matplotlib.rcParams.update(pgf_with_latex(1))

METHODS = ["PStokes", "PStokesLifted"]
method_labels = {"PStokes": "standard", "PStokesLifted": "lifted"}
colors = {"PStokes": "tab:blue", "PStokesLifted": "tab:orange"}

os.makedirs("plots", exist_ok=True)

all_p = sorted(set(args.p) | set(args.p_fig))
N_list = sorted(args.N)
N_fig  = max(N_list)


# ──────────────────────────────── data collection ─────────────────────────────
# run_data[N][p][method] = {"grads": ..., "n_iters": ...} or None

print("=" * 65)
print("Newton gradient data  (lid-driven cavity)")
print("=" * 65)

run_data = {}
for N in N_list:
    run_data[N] = {}
    for p in all_p:
        run_data[N][p] = {}
        for method in METHODS:
            stem = f"plots/lid_{method}_p{int(p * 100)}_N{N}"
            grad_path = os.path.join(stem, "newton_grad.npy")
            if not os.path.exists(grad_path):
                print(f"  [{method}  p={p}  N={N}]  newton_grad.npy not found – skipping")
                run_data[N][p][method] = None
                continue
            grads = np.load(grad_path)
            n_iters = len(grads) - 1   # grads[0] is the pre-step gradient
            run_data[N][p][method] = {"grads": grads, "n_iters": n_iters}
            print(f"  [{method}  p={p}  N={N}]  {n_iters} Newton iterations")


# ──────────────────────────────── LaTeX table ─────────────────────────────────
# booktabs style: rows = (N, method), columns = p values; \addlinespace between
# N groups. "Cells" is the number of mesh cells (4*N^2), shown as a\times10^b.

method_short = {"PStokes": "SN", "PStokesLifted": "LN"}


def format_cell_count(n):
    exp = int(np.floor(np.log10(n) + 1e-9))
    mantissa = n / 10.0 ** exp
    mantissa = int(round(mantissa)) if abs(mantissa - round(mantissa)) < 1e-6 else round(mantissa, 2)
    exp_str = str(exp) if exp < 10 else f"{{{exp}}}"
    if mantissa == 1:
        return f"$N=10^{exp_str}$"
    return rf"$N={mantissa}\times10^{exp_str}$"


p_cols = args.p

cells_strs = {N: format_cell_count(4 * N ** 2) for N in N_list}
cells_w = max(len(s) for s in cells_strs.values())

rows = [
    r"\begin{tabular}{ll" + "r" * len(p_cols) + "}",
    r"\toprule",
    "Cells & Method" + "".join(f" & $p={p:g}$" for p in p_cols) + r" \\",
    r"\midrule",
]
for i, N in enumerate(N_list):
    if i > 0:
        rows.append(r"\addlinespace")
    for method in METHODS:
        prefix = cells_strs[N] if method == METHODS[0] else ""
        row = f"{prefix.ljust(cells_w)} & {method_short[method]}"
        for p in p_cols:
            d = run_data[N].get(p, {}).get(method)
            row += f" & {d['n_iters']}" if d is not None else " & --"
        row += r" \\"
        rows.append(row)
rows += [r"\bottomrule", r"\end{tabular}"]

table_path = "plots/pstokes_lid_newton_its.txt"
with open(table_path, "w") as fh:
    fh.write("\n".join(rows) + "\n")
print(f"\nLaTeX table  →  {table_path}")
print("\n".join(rows))


# ───────────────────── separate contour+streamlines figures ───────────────────
# and separate Newton-gradient txt files, one pair per p in args.p_fig

p_a, p_b = args.p_fig

# ---- pass 1: load velocity/speed for both p values, find shared colour limits ----
vel_store = {}
vmin, vmax = np.inf, -np.inf
for p_val in [p_a, p_b]:
    h5 = os.path.join(f"plots/lid_PStokesLifted_p{int(p_val * 100)}_N{N_fig}", "velocity_pressure.h5")
    if not os.path.exists(h5):
        print(f"  Warning: {h5} not found – skipping contour plot for p={p_val}")
        vel_store[p_val] = None
        continue
    with CheckpointFile(h5, "r") as f:
        m = f.load_mesh()
        u = f.load_function(m, "velocity")
    V_s = FunctionSpace(m, "CG", 1)
    spd = Function(V_s, name="speed")
    spd.interpolate(sqrt(inner(u, u)))
    vel_store[p_val] = (m, u, spd)
    vmin = min(vmin, float(spd.dat.data_ro.min()))
    vmax = max(vmax, float(spd.dat.data_ro.max()))

if np.isinf(vmin):
    vmin, vmax = 0.0, 1.0

# ---- pass 2: contour+streamlines figures, no colorbar, shared vmin/vmax ----
for p_val in [p_a, p_b]:
    tag = f"p{int(p_val * 100)}"

    vd = vel_store.get(p_val)
    if vd is None:
        pass
    else:
        m, u, spd = vd

        fig, ax = plt.subplots(figsize=(0.5 * linewidth, 0.5 * linewidth))

        coll = tripcolor(spd, axes=ax, cmap="viridis", vmin=vmin, vmax=vmax)

        V_vec = VectorFunctionSpace(m, "CG", 1)
        u_cg1 = Function(V_vec)
        u_cg1.interpolate(u)
        coords = m.coordinates.dat.data_ro
        vals = u_cg1.dat.data_ro
        xi = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 120)
        yi = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 120)
        Xi, Yi = np.meshgrid(xi, yi)
        Ui = griddata(coords, vals[:, 0], (Xi, Yi), method="linear")
        Vi = griddata(coords, vals[:, 1], (Xi, Yi), method="linear")

        n_pnts = 7
        start_points_x = 0.5 * np.ones(n_pnts)
        start_points_y = np.linspace(0.1, 0.6, n_pnts)
        start_points = np.array([start_points_x, start_points_y]).T

        ax.streamplot(Xi, Yi, Ui, Vi, color="white", linewidth=0.3,
                      broken_streamlines=False, start_points=start_points,
                      density=0.3, arrowsize=0.3)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        contour_path = f"plots/lid_contour_{tag}.png"
        plt.savefig(contour_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Contour plot  →  {contour_path}")

    # ---- Newton gradient-norm data, both methods, for a pgfplots table ----
    grads_by_method = {}
    for method in METHODS:
        d = run_data[N_fig].get(p_val, {}).get(method)
        grads_by_method[method] = d["grads"] if d is not None else np.array([])

    max_len = max(len(grads_by_method[m]) for m in METHODS)

    iteration_path = f"plots/lid_iteration_{tag}.txt"
    with open(iteration_path, "w") as fh:
        fh.write("iteration " + " ".join(METHODS) + "\n")
        for i in range(max_len):
            entries = [str(i)]
            for method in METHODS:
                g = grads_by_method[method]
                entries.append(f"{g[i]:.16e}" if i < len(g) else "nan")
            fh.write(" ".join(entries) + "\n")
    print(f"Iteration data  →  {iteration_path}")

# ───────────────────── standalone colorbar figure ──────────────────────────
# Shares vmin/vmax with the two contour plots above.

sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax),
                                   cmap="viridis")
sm.set_array([])

fig_cb, ax_cb = plt.subplots(figsize=(0.08 * linewidth, 0.5 * linewidth))
cb = fig_cb.colorbar(sm, cax=ax_cb, orientation="vertical")
cb.ax.tick_params(labelsize=7)

colorbar_path = "plots/lid_colorbar.png"
plt.savefig(colorbar_path, bbox_inches="tight", dpi=150)
plt.close(fig_cb)
print(f"Colorbar plot  →  {colorbar_path}")

print("\nDone.")
