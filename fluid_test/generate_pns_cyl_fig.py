#!/usr/bin/env python3
"""Compare PStokes and PStokesLifted cylinder flow solutions.

Reads newton_iters.npy and velocity_pressure.h5 from runs produced by
pstokes_time_dep_cyl.py and produces:
  - printed L2-norm comparison of final-timestep solutions
  - LaTeX table of Newton iteration statistics  →  plots/pstokes_cyl_newton_its.txt
  - Combined plot (velocity | Newton iters)     →  plots/pstokes_cyl_combined.{pdf,png}
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(
    description="Compare PStokes and PStokesLifted cylinder runs"
)
parser.add_argument("--p",      type=float, nargs="+", default=[1.25, 1.1, 1.01],
                    help="list of p values to compare")
parser.add_argument("--nref",   type=int,   default=2,    help="mesh refinement level")
parser.add_argument("--nsteps", type=int,   default=3392, help="number of time steps")
parser.add_argument("--T",      type=float, default=8, help="total time")
parser.add_argument("--nsave",  type=int,   default=250,  help="number of saved snapshots")
parser.add_argument('--save',   action=argparse.BooleanOptionalAction, default=False)
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


# ─────────────────────────────────── helpers ──────────────────────────────────

def run_stem(method, p):
    return f"plots/NEW_cylinder_{method}_p{int(p * 100)}_n{args.nref}_nt{args.nsteps}"


def load_iters(method, p):
    path = os.path.join(run_stem(method, p), "newton_iters.npy")
    return np.load(path) if os.path.exists(path) else None


def last_snapshot_idx(n_completed):
    """Zero-based index of the last saved snapshot, given n_completed steps."""
    save_every = max(1, args.nsteps // args.nsave)
    count = sum(1 for k in range(n_completed) if (k + 1) % save_every == 0)
    return count - 1  # returns -1 when no snapshot was saved


def load_velocity_own_mesh(method, p, iters):
    """Load the final velocity from its checkpoint onto its own mesh.

    Returns (mesh, Function) or (None, None) on failure.
    """
    h5 = os.path.join(run_stem(method, p), "velocity_pressure.h5")
    if not os.path.exists(h5):
        return None, None
    idx = last_snapshot_idx(len(iters))
    if idx < 0:
        return None, None
    try:
        with CheckpointFile(h5, "r") as f:
            m = f.load_mesh()
            u = f.load_function(m, "velocity", idx=idx)
        return m, u
    except Exception as exc:
        print(f"  Warning: could not load {method} p={p} velocity: {exc}")
        return None, None


# ──────────────────────────────── data collection ─────────────────────────────

print("=" * 65)
print("Newton iteration data")
print("=" * 65)

run_data = {}
for p in args.p:
    run_data[p] = {}
    for method in METHODS:
        iters = load_iters(method, p)
        if iters is None:
            print(f"  [{method}  p={p}]  newton_iters.npy not found – skipping")
            run_data[p][method] = None
        else:
            completed = len(iters) == args.nsteps
            d = {
                "iters":     iters,
                "completed": completed,
                "max":       int(np.max(iters)),
                "mean":      float(np.mean(iters)),
            }
            run_data[p][method] = d
            status = "COMPLETE" if completed else f"INCOMPLETE ({len(iters)}/{args.nsteps})"
            print(f"  [{method}  p={p}]  {status}  "
                  f"max={d['max']}  mean={d['mean']:.2f}")


# ──────────────────────────────── L2 comparison ───────────────────────────────

print("\n" + "=" * 65)
print("L2 norm comparison  (final saved time step)")
print("=" * 65)

for p in args.p:
    d_ps  = run_data[p]["PStokes"]
    d_psl = run_data[p]["PStokesLifted"]

    m_ps,  u_ps  = (load_velocity_own_mesh("PStokes",       p, d_ps["iters"])
                    if d_ps  is not None else (None, None))
    m_psl, u_psl = (load_velocity_own_mesh("PStokesLifted", p, d_psl["iters"])
                    if d_psl is not None else (None, None))

    if u_ps is None and u_psl is None:
        print(f"  p={p:.2f}:  no velocity data available")
        continue

    # if u_ps is not None:
    #     print(f"  p={p:.2f}:  ||u_PS||_L2        = {norm(u_ps):.6e}")
    # if u_psl is not None:
    #     print(f"  p={p:.2f}:  ||u_PSL||_L2       = {norm(u_psl):.6e}")

    if u_ps is not None and u_psl is not None:
        try:
            # cross-mesh interpolation: bring u_psl onto mesh_ps
            V_ps = u_ps.function_space()
            u_psl_on_ps = Function(V_ps)
            u_psl_on_ps.interpolate(u_psl)
            diff = Function(V_ps)
            diff.assign(u_ps - u_psl_on_ps)
            print(f"  p={p:.2f}:  ||u_PS-u_PSL||_L2  = {norm(diff):.6e}")
        except Exception as exc:
            print(f"  p={p:.2f}:  cross-mesh diff failed ({exc})")


# ──────────────────────────────── LaTeX table ─────────────────────────────────

def _cell(d, key, fmt=None):
    if d is None:
        return "--"
    val = d[key]
    return (f"{val:{fmt}}" if fmt else str(val))

def _done(d):
    if d is None:
        return "--"
    return r"\checkmark" if d["completed"] else r"\texttimes"

rows = [
    r"\begin{tabular}{c|rr|rr|cc}",
    r"\hline",
    (r"$p$"
     r" & \multicolumn{2}{c|}{PStokes}"
     r" & \multicolumn{2}{c|}{PStokesLifted}"
     r" & \multicolumn{2}{c}{Completed} \\"),
    r"    & max & mean & max & mean & PS & PSL \\",
    r"\hline",
]
for p in args.p:
    d_ps  = run_data[p]["PStokes"]
    d_psl = run_data[p]["PStokesLifted"]
    rows.append(
        f"${p}$"
        f" & {_cell(d_ps,  'max')}"
        f" & {_cell(d_ps,  'mean', '.2f')}"
        f" & {_cell(d_psl, 'max')}"
        f" & {_cell(d_psl, 'mean', '.2f')}"
        f" & {_done(d_ps)}"
        f" & {_done(d_psl)}"
        r" \\"
    )
rows += [r"\hline", r"\end{tabular}"]

os.makedirs("plots", exist_ok=True)
table_path = "plots/pstokes_cyl_newton_its.txt"
with open(table_path, "w") as fh:
    fh.write("\n".join(rows) + "\n")
print(f"\nLaTeX table  →  {table_path}")
print("\n".join(rows))


# ──────────────────── Combined figure: velocity | Newton iters ────────────────
# n_p rows × 2 columns; left = velocity magnitude, right = Newton iterations
# shared horizontal colorbar below the velocity column

n_p = len(args.p)
colors = {"PStokes": "tab:blue", "PStokesLifted": "tab:orange"}
t = np.linspace(0, args.T, args.nsteps + 1)


# First pass: load all velocity data and find global vmin/vmax
speeds = []  # list of (spd, mesh) or None
vmin, vmax = np.inf, -np.inf
for p in args.p:
    d = run_data[p].get("PStokesLifted")
    if d is None:
        speeds.append(None)
        continue
    m, u = load_velocity_own_mesh("PStokesLifted", p, d["iters"])
    if u is None:
        speeds.append(None)
        continue
    V_cg1 = FunctionSpace(m, "CG", 1)
    spd = Function(V_cg1, name="speed")
    spd.interpolate(sqrt(inner(u, u)))
    speeds.append((spd, m))
    vmin = min(vmin, spd.dat.data_ro.min())
    vmax = max(vmax, spd.dat.data_ro.max())

# Second pass: build the combined figure
fig, axes = plt.subplots(n_p, 2, figsize=(linewidth, 0.8 * n_p), constrained_layout=True, sharex="col")

if n_p == 1:
    axes = axes[np.newaxis, :]

last_coll = None
for i, p in enumerate(args.p):
    ax_vel   = axes[i, 0]
    ax_iters = axes[i, 1]

    if i == 0:
        ax_vel.set_title("velocity magnitude $|\mathbf{u}|$", fontsize=5)
        ax_iters.set_title("Newton iterations", fontsize=5)

    # left column – velocity magnitude colour plot
    spd_data = speeds[i] if i < len(speeds) else None
    if spd_data is not None:
        spd, m = spd_data
        coll = tripcolor(spd, axes=ax_vel, cmap="turbo", vmin=vmin, vmax=vmax)
        last_coll = coll
        # ax_vel.set_aspect("equal")
        coords = m.coordinates.dat.data_ro
        ax_vel.set_xlim(coords[:, 0].min(), coords[:, 0].max())
        ax_vel.set_ylim(coords[:, 1].min(), coords[:, 1].max())
        ax_vel.set_xticks([])
        ax_vel.set_yticks([])
    else:
        ax_vel.text(0.5, 0.5, "No data", transform=ax_vel.transAxes,
                    ha="center", va="center")
        ax_vel.axis("off")
    ax_vel.set_ylabel(f"$p = {p}$")

    # right column – Newton iterations vs time
    any_data = False
    max_iter = 0
    for method in METHODS:
        d = run_data[p].get(method)
        if d is None:
            continue
        max_iter = max(max_iter, d["max"])
        ax_iters.plot(t[1:], d["iters"], label=method_labels[method], color=colors[method],
                      linewidth=0.8, alpha=0.9)
        ax_iters.set_xlim(0, t[-1])
        any_data = True
    if not any_data:
        ax_iters.text(0.5, 0.5, "No data", transform=ax_iters.transAxes,
                      ha="center", va="center")
    if i == n_p - 1:
        ax_iters.set_xlabel("time step")
    # ax_iters.set_ylabel("Newton iterations")
    ax_iters.set_ylim(bottom=0)
    ytop = int(max_iter)
    ax_iters.set_ylim(top=ytop)
    ax_iters.set_yticks([0, ytop // 2, ytop])
    ax_iters.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: str(int(v))))
    if i == 0:
        ax_iters.legend(fontsize=5)

if last_coll is not None:
    fig.colorbar(last_coll, ax=axes[:, 0], 
                 orientation="horizontal", location="bottom", shrink=0.8, aspect=40, pad=-0.075)

combined_plot = "plots/pstokes_cyl_combined"
plt.savefig(combined_plot + ".pdf", bbox_inches="tight", dpi=150)
plt.savefig(combined_plot + ".png", bbox_inches="tight", dpi=150)
print(f"Combined plot  →  {combined_plot}.{{pdf,png}}")

if args.save:
    report_plot = "report/figs/pstokes_cyl.pdf"
    plt.savefig(report_plot, bbox_inches="tight", dpi=150)
    print(f"report plot  →  {report_plot}")

plt.close()
