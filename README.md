# lifted_plaplace
Implementation of the standard and lifted Newton methods, and the scripts used to
reproduce the results in the paper ''Transform before linearizing: robust Newton methods for singular $p$-Laplace and p-Stokes equations''

## Library
- `newtons_method.py` — base Newton solver.
- `plaplace.py` — standard/lifted/Picard solvers for the $p$-Laplace equation.
- `utils.py`, `plot_utils.py` — shared solver/plotting settings.
- `plaplace_test/` — scripts for computing $p$-Laplace equation.
- `fluid_test/` — scripts for computing $p$-Stokes/NS equations.
- `fd_test/` — scripts for computing one-dimensional finite difference example.

## Generating the plots

| Result | Script |
|---|---|
| 2D $p$-Laplace convergence & solution plots | `plaplace_test/plaplace_2d.py` |
| 3D $p$-Laplace convergence & solution plots | `plaplace_test/plaplace_3d.py` (mesh via `plaplace_test/generate_plaplace_3d_adaptive_mesh.py`) |
| Lid-driven cavity ($p$-Stokes) | run `fluid_test/pstokes_lid_driven_cavity.py` for each method/`p`, then `fluid_test/generate_pstokes_lid_fig.py` |
| Flow around a cylinder ($p$-Navier-Stokes) | run `fluid_test/pns_time_dep_cyl.py` for each method/`p`, then `fluid_test/generate_pns_cyl_fig.py` (mesh via `fluid_test/mesh_cylinder.py`) |
| 1D finite-difference example | `fd_test/two_dim_pLaplace.py`, then `fd_test/higher_dim_plot.py` |

Each script exposes its parameters (e.g. `--p`, `--nref`, `--N`) via `--help`.
Run the `fluid_test/` and `fd_test/` scripts from inside their own folder (`cd fluid_test`, etc.) so their relative
output paths (`plots/`, mesh files) resolve correctly.
