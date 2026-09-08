#!/usr/bin/env python3
"""Compare PStokes and PStokesLifted lid-driven cavity solutions.

Reads newton_grad.npy and velocity_pressure.h5 from runs produced by
pstokes.py (lid-driven cavity section) and produces:
  - LaTeX table of Newton iteration counts      →  plots/pstokes_lid_newton_its.txt
  - Combined plot (velocity+quiver | grad norm) →  plots/pstokes_lid_combined.{pdf,png}
"""

import argparse
import os
import sys
import numpy as np
from scipy.interpolate import griddata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(
    description="Compare PStokes and PStokesLifted lid-driven cavity runs"
)
parser.add_argument("--p",     type=float, nargs="+", default=[1.5, 1.1, 1.05, 1.01], help="p values for the table (default: 1.5 1.1 1.05 1.01)")
parser.add_argument("--p_fig", type=float, nargs=2,   default=[1.5, 1.01], help="two p values shown in the figure (default: 1.5 1.01)")
parser.add_argument("--N",     type=int,   nargs="+", default=[25, 50, 100], help="mesh sizes for the table (default: 25 50 100); largest N used for plots")
parser.add_argument('--save',  action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

from firedrake import *
from firedrake.pyplot import tripcolor
try:
    from firedrake.pyplot import quiver as fd_quiver
    _has_fd_quiver = True
except ImportError:
    _has_fd_quiver = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from plot_utils import linewidth, pgf_with_latex
matplotlib.rcParams.update(pgf_with_latex(1))

METHODS = ["PStokes", "PStokesLifted"]
method_labels = {"PStokes": "standard", "PStokesLifted": "lifted"}
colors = {"PStokes": "tab:blue", "PStokesLifted": "tab:orange"}


# ─────────────────────────────────── helpers ──────────────────────────────────

def run_stem(method, p, N):
    return f"plots/lid_{method}_p{int(p * 100)}_N{N}"


def load_grads(method, p, N):
    path = os.path.join(run_stem(method, p, N), "newton_grad.npy")
    return np.load(path) if os.path.exists(path) else None


def load_velocity(method, p, N):
    h5 = os.path.join(run_stem(method, p, N), "velocity_pressure.h5")
    if not os.path.exists(h5):
        return None, None
    try:
        with CheckpointFile(h5, "r") as f:
            m = f.load_mesh()
            u = f.load_function(m, "velocity")
        return m, u
    except Exception as exc:
        print(f"  Warning: could not load {method} p={p} N={N}: {exc}")
        return None, None


def _draw_quiver(ax, m, u):
    """Overlay a subsampled quiver of u on ax (white arrows)."""
    if _has_fd_quiver:
        try:
            fd_quiver(u, axes=ax, color="white", alpha=0.7)
            return
        except Exception as exc:
            print(f"  fd_quiver failed ({exc}), using manual quiver")
    # Fallback: CG1 interpolation + manual subsampling
    V_vec = VectorFunctionSpace(m, "CG", 1)
    u_cg1 = Function(V_vec)
    u_cg1.interpolate(u)
    coords = m.coordinates.dat.data_ro          # (nvert, 2)
    vals   = u_cg1.dat.data_ro                  # (nvert, 2)
    stride = max(1, len(coords) // 300)
    ax.quiver(coords[::stride, 0], coords[::stride, 1],
              vals[::stride, 0],   vals[::stride, 1],
              color="white", alpha=0.7, headwidth=3)


# ──────────────────────────────── data collection ─────────────────────────────

print("=" * 65)
print("Newton gradient data  (lid-driven cavity)")
print("=" * 65)

all_p = sorted(set(args.p) | set(args.p_fig))
N_list = sorted(args.N)
N_fig  = max(N_list)

# run_data[N][p][method]
run_data = {}
for N in N_list:
    run_data[N] = {}
    for p in all_p:
        run_data[N][p] = {}
        for method in METHODS:
            grads = load_grads(method, p, N)
            if grads is None:
                print(f"  [{method}  p={p}  N={N}]  newton_grad.npy not found – skipping")
                run_data[N][p][method] = None
            else:
                n_iters = len(grads) - 1   # grads[0] is the pre-step gradient
                run_data[N][p][method] = {"grads": grads, "n_iters": n_iters}
                print(f"  [{method}  p={p}  N={N}]  {n_iters} Newton iterations")


# ──────────────────────────────── LaTeX table ─────────────────────────────────
# Layout: rows = (N, method), columns = p values; horizontal rule between N groups

p_cols = args.p

rows = [
    r"\begin{tabular}{cc|" + "c" * len(p_cols) + "}",
    r"$N$ & Method" + "".join(f" & $p = {p}$" for p in p_cols) + r" \\",
    r"\midrule",
]
for i, N in enumerate(N_list):
    if i > 0:
        rows.append(r"\midrule")
    for method in METHODS:
        row = f"$N={4 * N**2}$ & {method_labels[method]}" if method == METHODS[0] else f"       & {method_labels[method]}"
        for p in p_cols:
            d = run_data[N].get(p, {}).get(method)
            row += f" & {d['n_iters']}" if d is not None else " & --"
        row += r" \\"
        rows.append(row)
rows += [r"\end{tabular}"]

os.makedirs("plots", exist_ok=True)
table_path = "plots/pstokes_lid_newton_its.txt"
with open(table_path, "w") as fh:
    fh.write("\n".join(rows) + "\n")
print(f"\nLaTeX table  →  {table_path}")
print("\n".join(rows))


# ───────────── Combined figure: 1 row × [vel_A | nit_A | vel_B | nit_B] ───────
# Horizontal colorbar centred under all panels.

p_a, p_b = args.p_fig

# Load PStokesLifted velocity for both figure p values (largest N); find shared colour limits
vel_store = {}
vmin, vmax = np.inf, -np.inf
for p_val in [p_a, p_b]:
    m, u = load_velocity("PStokesLifted", p_val, N_fig)
    if u is None:
        vel_store[p_val] = None
        continue
    V_s = FunctionSpace(m, "CG", 1)
    spd = Function(V_s, name="speed")
    spd.interpolate(sqrt(inner(u, u)))
    vel_store[p_val] = (m, u, spd)
    vmin = min(vmin, float(spd.dat.data_ro.min()))
    vmax = max(vmax, float(spd.dat.data_ro.max()))

if np.isinf(vmin):
    vmin, vmax = 0.0, 1.0

# Nested-GridSpec horizontal layout:
#   outer col 0: group A  [nit_a | vel_a]  (tight inner wspace)
#   outer col 1: group B  [nit_b | vel_b]  (tight inner wspace)
#   groups separated by a larger outer wspace to keep y-tick labels clear
#   colorbar: thin strip placed manually to the right

fig_w = linewidth
fig_h = linewidth * 0.28   # single row → keep it flat

fig = plt.figure(figsize=(fig_w, fig_h))

# Outer: two equal-width group slots; right=0.91 leaves room for the cbar
outer = GridSpec(1, 2, figure=fig,
                 left=0.08, right=0.91, top=0.88, bottom=0.1,
                 wspace=0.25)   # gap between groups → enough for y-tick labels

# Inner grids: each group is [nit | vel] packed tightly
inner_a = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 0], wspace=0.10)
inner_b = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 1], wspace=0.10)

ax_nit_a = fig.add_subplot(inner_a[0, 0])
ax_vel_a = fig.add_subplot(inner_a[0, 1])
ax_nit_b = fig.add_subplot(inner_b[0, 0])
ax_vel_b = fig.add_subplot(inner_b[0, 1])

for ax, title in [
    (ax_nit_a, f"gradient norm, $p={p_a}$"),
    (ax_vel_a, f"velocity, $p={p_a}$"),
    (ax_nit_b, f"gradient norm, $p={p_b}$"),
    (ax_vel_b, f"velocity, $p={p_b}$"),
]:
    ax.set_title(title, fontsize=5, pad=2)

# ---- velocity colour + quiver ----
last_coll = None
for ax_vel, p_val in [(ax_vel_a, p_a), (ax_vel_b, p_b)]:
    vd = vel_store.get(p_val)
    if vd is None:
        ax_vel.text(0.5, 0.5, "No data", transform=ax_vel.transAxes,
                    ha="center", va="center")
        ax_vel.axis("off")
        continue
    m, u, spd = vd
    coll = tripcolor(spd, axes=ax_vel, cmap="viridis", vmin=vmin, vmax=vmax)
    last_coll = coll
    # _draw_quiver(ax_vel, m, u)

    # ---- streamlines (interpolate CG1 velocity onto a regular grid) ----
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

    ax_vel.streamplot(Xi, Yi, Ui, Vi, color="white", linewidth=0.3, broken_streamlines=False, start_points=start_points,
                       density=0.3, arrowsize=0.3)
    ax_vel.set_aspect("equal")
    ax_vel.set_xticks([])
    ax_vel.set_yticks([])
    for spine in ax_vel.spines.values():
        spine.set_visible(False)

# ---- gradient norm plots (largest N) ----
max_iters = 0
for ax_nit, p_val in [(ax_nit_a, p_a), (ax_nit_b, p_b)]:
    ax_nit.set_box_aspect(1)
    any_data = False
    for method in METHODS:
        d = run_data[N_fig].get(p_val, {}).get(method)
        if d is None:
            continue
        grads = d["grads"]
        max_iters = max(max_iters, len(grads) - 1)
        ax_nit.semilogy(np.arange(len(grads)), grads,
                        label=method_labels[method],
                        color=colors[method], linewidth=0.8)
        any_data = True
    ax_nit.set_xlim([0, max_iters+1])  
    if not any_data:
        ax_nit.text(0.5, 0.5, "No data", transform=ax_nit.transAxes,
                    ha="center", va="center")
    ax_nit.set_xlabel("Newton iteration")
    ax_nit.set_ylabel(r"$\|\nabla J\|$")

ax_nit_a.legend(fontsize=4, loc="upper right")

# ---- thin vertical colorbar to the right ----
# Spans the full panel height (bottom=0.16 … top=0.88) and sits just past right=0.91
if last_coll is not None:
    cbar_ax = fig.add_axes([0.93, 0.30, 0.011, 0.42])
    cb = fig.colorbar(last_coll, cax=cbar_ax, orientation="vertical")
    cb.set_label("")
    cb.ax.set_title(r"$|\mathbf{u}|$", fontsize=5, pad=3)
    cb.ax.tick_params(labelsize=4)

combined_plot = "plots/pstokes_lid_driven"
plt.savefig(combined_plot + ".pdf", bbox_inches="tight", dpi=150)
plt.savefig(combined_plot + ".png", bbox_inches="tight", dpi=150)
print(f"Combined plot  →  {combined_plot}.{{pdf,png}}")

if args.save:
    os.makedirs("report/figs", exist_ok=True)
    report_plot = "report/figs/pstokes_lid.pdf"
    plt.savefig(report_plot, bbox_inches="tight", dpi=150)
    print(f"Report plot  →  {report_plot}")

plt.close()
