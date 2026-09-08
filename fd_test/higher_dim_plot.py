import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 5,
})

# Parse arguments
parser = argparse.ArgumentParser(description="Project lifted vs standard Newton trajectories onto (u_1, u_iplot) planes")
parser.add_argument("--p", type=float, default=1.01)
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--iplot", type=int, nargs="+", default=[2, 3, 5, 10])
parser.add_argument("--eps", type=float, default=1e-12)
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()

# Load the corresponding lifted and standard Newton trajectory data (saved by two_dim_pLaplace.py)
tag = f"p{int(args.p * 1000)}_eps{int(-np.log10(args.eps))}_n{args.n}_seed{args.seed}"
data_lifted = np.load(f"data/lifted_{tag}_trajectory_data.npz")
data_std = np.load(f"data/standard_{tag}_trajectory_data.npz")

iter_lifted = data_lifted["iterates"]
iter_std = data_std["iterates"]
A = data_lifted["A"]
f = data_lifted["f"]

print("Number of iterations:\n\tLifted:   ", iter_lifted.shape[0]-1, "\n\tStandard: ", iter_std.shape[0]-1)


# Energy functional (as in two_dim_pLaplace.py), vectorized over rows of x
energy = lambda x: np.sum(np.sqrt((A @ x.T) ** 2 + args.eps) ** args.p, axis=0) / args.p - x @ f

# Set up a 1 x len(iplot) figure, sized to fill approximately \linewidth
linewidth = 4.78
n_panels = len(args.iplot)
fig, axes = plt.subplots(1, n_panels, figsize=(linewidth, linewidth / n_panels))
if n_panels == 1:
    axes = [axes]

# For each requested coordinate u_iplot[i], project trajectories and energy onto (u_1, u_iplot[i])
for j, k in enumerate(args.iplot):
    ax = axes[j]

    u1_lifted, uk_lifted = iter_lifted[:, 1], iter_lifted[:, k]
    u1_std, uk_std = iter_std[:, 1], iter_std[:, k]

    # Build a grid covering all iterations of both trajectories; all other components fixed at the converged solution
    base = iter_lifted[-1].copy()
    u1_all = np.concatenate([u1_lifted, u1_std])
    uk_all = np.concatenate([uk_lifted, uk_std])
    margin = max(np.ptp(u1_all), np.ptp(uk_all)) * 0.1 + 1e-4
    g1 = np.linspace(u1_all.min() - margin, u1_all.max() + margin, 200)
    g2 = np.linspace(uk_all.min() - margin, uk_all.max() + margin, 200)
    G1, G2 = np.meshgrid(g1, g2)

    X_grid = np.tile(base, (G1.size, 1))
    X_grid[:, 1] = G1.ravel()
    X_grid[:, k] = G2.ravel()
    E = energy(X_grid).reshape(G1.shape)

    # Plot energy contours
    ax.contourf(G1, G2, E, levels=30, cmap="RdYlBu_r", alpha=0.6, zorder=0)
    ax.contour(G1, G2, E, levels=30, colors="k", linewidths=0.1, alpha=0.4, zorder=0)

    # Overlay lifted and standard Newton trajectories, with start/end markers
    ax.plot(u1_lifted, uk_lifted, color="tab:blue", marker="o", markersize=1.2, linewidth=0.8, zorder=3, label="lifted")
    ax.plot(u1_std, uk_std, color="tab:red", marker="o", markersize=1.2, linewidth=0.4, zorder=2, label="standard")
    ax.plot(u1_lifted[0], uk_lifted[0], "x", color="green", markersize=3, markeredgewidth=0.8, zorder=3)
    ax.plot(u1_lifted[-1], uk_lifted[-1], "*", color="black", markersize=4, zorder=3)

    ax.set_xlim(g1[0], g1[-1])
    ax.set_ylim(g2[0], g2[-1])

    ax.set_title(rf"$(u_1, u_{{{k}}})$ projection")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
    if j == 0:
        marker_handles = [
            Line2D([], [], marker="x", color="green", linestyle="None", markersize=3, markeredgewidth=0.8, label="initial guess"),
            Line2D([], [], marker="*", color="black", linestyle="None", markersize=4, label="solution"),
        ]
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles + marker_handles, fontsize=4, handlelength=1.5, borderpad=0.3, labelspacing=0.3, loc = "upper right")

# Save and show figure
plt.tight_layout(pad=0.2)
fig.subplots_adjust(wspace=0.025)
os.makedirs("plots", exist_ok=True)
plt.savefig(f"plots/higher_dim_{tag}.pdf", bbox_inches="tight", pad_inches=0.01)
plt.show()
