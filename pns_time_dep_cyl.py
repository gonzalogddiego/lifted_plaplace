import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import inspect
from firedrake import *
from newtons_method import NewtonMethod
from utils import solver_params

# --- helpers ---
def abs_grad(u, delta_min):
    return sqrt(delta_min**2 + inner(sym(grad(u)), sym(grad(u))))

def pstokes_residual(z, zold, f, params, dt, dX,
                     g_D=None, ds_D=None, ds_N=None, dS_int=None,
                     viscous_stress = True):
    """
    Residual for the (p,δ)-Navier-Stokes system (eq. 12, Margenberg & Mehlmann 2026).

    Base terms (always active):
      time derivative + convection + p-viscous stress + pressure coupling − f

    Optional Nitsche weak BC terms on ds_D (eqs. 7, 10):
      Requires g_D (Dirichlet lifting to Ω) and ds_D (boundary measure for Γ_D).
      gamma_1, gamma_2 are the tangential / normal Nitsche penalty parameters.
      Viscous coefficients in the boundary terms are frozen at Dg_D for bilinearity.

    Optional CIP stabilisation on interior facets dS_int (eq. 9):
      Convection-aligned jump penalty weighted by |u·n_F|.
    """
    u, pres = split(z)
    uold, presold = split(zold)
    z_test = TestFunction(z.function_space())
    v, q = split(z_test)
    mesh = z.function_space().mesh()
    n = FacetNormal(mesh)

    # Load parameters
    p = Constant(params.p)
    nu = Constant(params.nu)
    delta_min = Constant(params.delta_min)
    gamma_1 = Constant(params.nitsche1)
    gamma_2 = Constant(params.nitsche2)
    gamma_CIP = Constant(params.cip)

    # Standard IBP form: a(u)(v) = ⟨S(Du), Dv⟩ (volume only).
    # Do-nothing on Γ_out is the natural BC; Nitsche handles Γ_D explicitly.
    residual = (
        - pres * div(v) * dX
        - dot(f, v) * dX
        - q * div(u) * dX
    )

    if params.stokes:
        print("Not adding inertial term - Stokes flow")
    else:
        print("Adding inertial term")
        
        # divergence form of convection: c(u)(z) = −⟨u⊗u, ∇z⟩ + ⟨(u·n)u, z⟩_Γ  (eq. 8)
        # boundary integral runs over the full ∂Ω; B^c (below) corrects the Γ_D part
        convection = - inner(outer(u, u), grad(v)) * dX
        if ds_N is not None:
            convection += dot(dot(u, n) * u, v) * ds(ds_N)        
        residual += dot((u-uold)/dt , v) * dX + convection
        
    # Viscous stress tensor
    tau_tensor = nu * abs_grad(u, delta_min)**(p-2) * sym(grad(u))
    if viscous_stress:
        residual += inner(tau_tensor, sym(grad(v))) * dX

    # ------------------------------------------------------------------
    # Nitsche terms  (eqs. 7, 10a–10d)
    # g_D and ds_D may be single values or equal-length lists, one entry
    # per Dirichlet boundary subregion (e.g. g_D=[u_in, zero], ds_D=[1, 5])
    # ------------------------------------------------------------------
    if g_D is not None and ds_D is not None:
        print("Adding Nitsche weak BCs with gamma_1 =", float(gamma_1), "and gamma_2 =", float(gamma_2))
        # normalise to lists so the loop handles both single and multi cases
        if not isinstance(g_D, (list, tuple)):
            g_D,  ds_D = [g_D],  [ds_D]

        h   = CellDiameter(mesh)
        dim = mesh.geometric_dimension()

        # quantities depending only on the current solution (computed once)
        eta_u      = abs_grad(u, delta_min)**(p - 2)
        un_minus_u = 0.5 * (abs(dot(u, n)) - dot(u, n))
        Dv         = sym(grad(v))

        for g_Di, tag_i in zip(g_D, ds_D):
            # accept either an integer boundary tag or a ready-made measure
            ds_i = ds(tag_i) if (isinstance(tag_i, int) or isinstance(tag_i, tuple)) else tag_i

            # viscous coefficients frozen at Dg_Di for bilinearity (§3.2)
            eta_gDi = abs_grad(g_Di, delta_min)**(p - 2)
            abs_gDi = abs_grad(g_Di, delta_min)
            A_gDi   = sym(grad(g_Di))

            # DS(Dg_Di)(Dv)  –  Fréchet derivative of S at Dg_Di  (eq. 17)
            #   DS(A)(B) = η(A) B + ν(p−2) η_noNu/|A|² (A:B) A,  η(A) = ν·|A|^{p-2}
            DS_gDi_Dv = nu * (
                eta_gDi * Dv
                + (p - 2) * (eta_gDi / abs_gDi**2) * inner(A_gDi, Dv) * A_gDi
            )

            # (i) stress-flux consistency: −⟨(S(Du)−pres·Id)n, v⟩_{Γ_Di}  (eq. 7)
            #   S(Du) = tau_tensor  (includes ν)
            nitsche_flux = (
                - inner(dot(tau_tensor, n), v) * ds_i
                + pres * dot(n, v) * ds_i
            )

            # (ii) B^c(u,w) − B^c(g_Di,w)  (eq. 10b): inflow convective correction
            un_minus_gDi = 0.5 * (abs(dot(g_Di, n)) - dot(g_Di, n))
            B_c = (
                - un_minus_u   * inner(u,    v) * ds_i
                + un_minus_gDi * inner(g_Di, v) * ds_i
            )

            # (iii) B^s(u,w) − B^s(g_Di,w)  (eq. 10c): adjoint consistency
            B_s = (
                - inner(u - g_Di,
                        dot(DS_gDi_Dv + q * Identity(dim), n)) * ds_i
            )

            # (iv) B^γ_γ(u,w) − B^γ_γ(g_Di,w)  (eq. 10d): Nitsche penalty
            #   η(Dg_Di) = ν·eta_gDi  (includes ν)
            B_pen = (
                  gamma_1 / h * (nu * eta_gDi) * inner(u - g_Di, v) * ds_i
                + gamma_2 / h * dot(u - g_Di, n) * dot(v, n) * ds_i
            )

            residual += nitsche_flux + B_c + B_s + B_pen

    # ------------------------------------------------------------------
    # CIP stabilisation on interior skeleton  (eq. 9)
    # ------------------------------------------------------------------
    if (dS_int is not None) and (not params.stokes):
        # s_CIP(û; v_h, z_h) = Σ_F γ_CIP h_F² |û·n_F| ⟨[[∇v·n_F]], [[∇z·n_F]]⟩_F
        # û is the advecting velocity, lagged at the current iterate
        print("Adding CIP stabilisation with γ_CIP =", float(gamma_CIP))
        h_F = avg(CellDiameter(mesh))
        residual += (
            gamma_CIP * h_F**2 * abs(dot(u('+'), n('+')))
            * inner(jump(grad(u), n), jump(grad(v), n)) * dS_int
        )

    return residual


# --- Newton subclasses ---
class PStokesTimeBase(NewtonMethod):
    """Base class for time-dependent p-Stokes solvers.

    Parameters
    ----------
    params : namespace
        Must expose .p, .nu, .delta_min, .nitsche1, .nitsche2, .cip
        (e.g. the argparse namespace from __main__).
    bc : DirichletBC or list thereof, optional
        Strong Dirichlet BCs.  Pass None when using Nitsche weak BCs.
    """

    def __init__(self, mesh, Z, params, f, dt,
                 g_D=None, ds_D=None, ds_N = None, dS_int=None,
                 bc=None, nullspace=False,
                 rtol=1e-8, atol = 1e-10, stol = 1e-12, max_iter=1000, max_iter_ls=30, quad_degree=6):
        super().__init__(rtol, atol = atol, stol = stol, max_iter=max_iter, max_iter_ls =max_iter_ls)
        self.params    = params
        self.p         = params.p
        self.delta_min = Constant(params.delta_min)
        self.nu        = Constant(params.nu)
        self.dt        = dt if isinstance(dt, Constant) else Constant(dt)
        self.f         = f
        self.dS_int    = dS_int
        self.g_D       = g_D
        self.ds_D      = ds_D
        self.ds_N      = ds_N

        self.mesh        = mesh
        self.Z           = Z
        self.bc          = bc if bc is not None else []
        self.dx          = dx(degree=quad_degree)
        self.quad_degree = quad_degree

        self.z    = Function(self.Z)
        self.zold = Function(self.Z)
        self.znew = Function(self.Z)
        self.dz   = Function(self.Z)

        self.F = pstokes_residual(
            self.z, self.zold, self.f, self.params, self.dt, self.dx,
            g_D=g_D, ds_D=ds_D, ds_N = ds_N, dS_int=dS_int,
        )

        if nullspace:
            print("Setting up nullspace for pressure indeterminacy")
            self.nullspace = MixedVectorSpaceBasis(
                self.Z, [self.Z.sub(0), VectorSpaceBasis(constant=True)]
            )

    def apply_bcs(self):
        print("Applying boundary conditions to the current state")
        for bc_i in (self.bc if isinstance(self.bc, list) else [self.bc]):
            bc_i.apply(self.z)

    def step(self, return_data=False):
        """Solve one time step to Newton convergence, then advance zold."""
        data = self.solve(return_data=return_data)
        self.zold.assign(self.z)
        return data

    def objective(self):
        return norm(assemble(self.F, bcs=self.bc).riesz_representation(riesz_map="l2"))

    def objective_at(self, s):
        # temporarily set z = z + s·dz to evaluate F there, then restore
        self.znew.assign(self.z)              # save current z
        self.z.assign(self.z + s * self.dz)  # advance z
        val = norm(assemble(self.F, bcs=self.bc).riesz_representation(riesz_map="l2"))
        self.z.assign(self.znew)              # restore z
        return val

    def gradient_norm(self):
        return norm(assemble(self.F, bcs=self.bc).riesz_representation(riesz_map="l2"))
    
    def residual(self):
        self.apply_bcs()
        return self.Flin, -self.F, self.bc, self.dz    


class PStokes(PStokesTimeBase):
    """Exact Newton for time-dependent p-Stokes.

    Jacobian = full Gâteaux derivative of the residual (Nitsche and CIP included).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"PStokes (exact Newton), p = {self.p}")
        self.Flin = derivative(self.F, self.z)

    def accept(self, s):
        self.z.assign(self.z + s * self.dz)



class PStokesLifted(PStokesTimeBase):
    """Lifted Newton for time-dependent p-Stokes (designed for p < 2).

    The viscous Jacobian block is replaced by a symmetrized rank-one surrogate
    acting on a dual variable λ ≈ |Du|^{p-2} Du.
    Note: boundary Nitsche/CIP terms are not included in the Jacobian.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"PStokesLifted (lifted Newton), p = {self.p}")

        quad_FE = FiniteElement("Quadrature", degree=self.quad_degree, quad_scheme="default")
        Q_quad  = TensorFunctionSpace(self.mesh, quad_FE, self.quad_degree)

        self.lamda  = Function(Q_quad)
        self.dlamda = Function(Q_quad)

        u,   _       = split(self.z)
        du,  _       = split(self.dz)
        utr, pres_tr = TrialFunctions(self.Z)
        v,   q       = TestFunctions(self.Z)

        abs_gradu = abs_grad(u, self.delta_min)
        eps_u   = sym(grad(u))
        eps_du  = sym(grad(du))
        eps_utr = sym(grad(utr))
        eps_v   = sym(grad(v))

        # Symmetrized rank-4 tensor surrogate:
        #   T(A) = A − ½(2−p)|Du|^{−p}(⟨Du,A⟩λ + ⟨λ,A⟩Du)
        self.tensor_op = lambda A: (
            A - (2 - self.p) * abs_gradu**(-self.p) * Constant(0.5) * (
                inner(eps_u, A) * self.lamda + inner(self.lamda, A) * eps_u
            )
        )

        F_without_lamda = pstokes_residual(
            self.z, self.zold, self.f, self.params, self.dt, self.dx,
            g_D=self.g_D, ds_D=self.ds_D, ds_N=self.ds_N, dS_int=self.dS_int, viscous_stress=False
        )

        Flin_without_lamda = derivative(F_without_lamda, self.z)

        self.Flin = (
            Flin_without_lamda 
            + self.nu * abs_gradu**(self.p - 2) * inner(self.tensor_op(eps_utr), eps_v) * self.dx
        )

        self.norm_gradu_p = abs_gradu**(self.p - 1) / (2 - self.p + 1e-12)
        self.norm_lamda   = sqrt(inner(self.lamda, self.lamda))
        self.dlamda_form  = (
            - self.lamda
            + abs_gradu**(self.p - 2) * eps_u
            + abs_gradu**(self.p - 2) * self.tensor_op(eps_du)
        )


    def dual_variable(self):
        self.dlamda.interpolate(self.dlamda_form)

    def accept(self, s):
        self.z.assign(self.z + s * self.dz)
        self.lamda.assign(self.lamda + s * self.dlamda)

        condition = conditional(
            gt(self.norm_lamda, self.norm_gradu_p),
            self.lamda * self.norm_gradu_p / self.norm_lamda,
            self.lamda,
        )
        self.lamda.interpolate(condition)


# PStokesTimeBase is abstract; exclude it from the selectable method list
method_map = {name: cls
               for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
               if issubclass(cls, NewtonMethod)
               and cls not in (NewtonMethod, PStokesTimeBase)}
method_list = sorted(method_map)

# --- entry point ---

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="p-Stokes time-dependent cylinder flow")
    parser.add_argument("--method", choices=method_list + ["firedrake"], default="firedrake",
                        help="firedrake = raw Firedrake solve(); others use the Newton class")
    parser.add_argument("--p",         type=float, default=1.25)
    parser.add_argument("--vmean",     type=float, default=1.0)
    parser.add_argument("--T",         type=float, default=8.0)
    parser.add_argument("--nref",      type=int,   default=2)
    parser.add_argument("--ntsteps",   type=int,   default=3392)
    parser.add_argument("--nsave",     type=int,   default=250,
                        help="number of time steps written to VTK and HDF5 (evenly spaced)")
    parser.add_argument("--nu",        type=float, default=1e-3)
    parser.add_argument("--delta-min", type=float, default=1e-10, dest="delta_min")
    parser.add_argument("--cip",       type=float, default=1.0)
    parser.add_argument("--nitsche1",  type=float, default=1e7)
    parser.add_argument("--nitsche2",  type=float, default=1e7)
    parser.add_argument('--stokes', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--nitsche', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    # ------------------------------------------------------------------ mesh
    import subprocess
    subprocess.run(["python", "mesh_cylinder.py", "--nref", str(args.nref)], check=True)
    mesh = Mesh("cylinder_flow.msh")

    # --------------------------------------------------------- function spaces
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    Z = MixedFunctionSpace([V, Q])

    # ----------------------------------------- time-dependent Dirichlet data
    t = Constant(0.0)
    x, y = SpatialCoordinate(mesh)
    H = mesh.coordinates.dat.data[:, 1].max() - mesh.coordinates.dat.data[:, 1].min()
    omega     = conditional(le(t, 1), 0.5 * (1 - cos(pi * t)), 1.0)

    # expp = (args.p/(args.p - 1))
    # u_in      = omega * args.vmean * (
    #     (0.5 * H)**expp
    #     - (abs(0.5 * H - y))**expp
    #     )/ (0.5 * H)**expp
    
    u_in      = omega * args.vmean * (
        (0.5 * H)**2
        - (abs(0.5 * H - y))**2
        )/ (0.5 * H)**2    
    
    u_bc_in   = Function(V).interpolate(as_vector([u_in, 0]))
    u_bc_zero = Function(V).interpolate(as_vector([0, 0]))

    # BCs: tag 1 = inflow, tags (3,4,5) = walls + cylinder
    if args.nitsche:
        g_D  = [u_bc_in, u_bc_zero]
        ds_D = [1, (3, 4, 5)]
        ds_N = 2
        bcs = []
    else:
        print("Using strong Dirichlet BCs")
        g_D  = None
        ds_D = None
        ds_N = 2

        bcs = [
            DirichletBC(Z.sub(0), as_vector([u_in, 0]), 1), 
            DirichletBC(Z.sub(0), u_bc_zero, (3, 4, 5))
        ]

    # set time step size
    dt = Constant(args.T / args.ntsteps)

    # save every k-th step so that at most nsave snapshots are written
    save_every = max(1, args.ntsteps // args.nsave)

    # output folder — one per run, named by its key parameters
    import numpy as np
    stem = f"plots/NEW_cylinder_{args.method}_p{int(args.p * 100)}_n{args.nref}_nt{args.ntsteps}"
    if not args.nitsche:
        stem += "_strongBC"
    if args.stokes:
        stem += "_stokes"    
    os.makedirs(stem, exist_ok=True)
    outfile = VTKFile(f"{stem}/velocity_pressure.pvd")
    h5name  = f"{stem}/velocity_pressure.h5"

    # ================================================================
    # METHOD A: raw Firedrake solve()   (--method firedrake)
    # ================================================================
    if args.method == "firedrake":

        z    = Function(Z)

        t.assign(dt)
        z.sub(0).interpolate(as_vector([u_in, 0]))
        t.assign(0)
        
        zold = Function(Z)
        F    = pstokes_residual(z, zold, Constant((0, 0)), args, dt, dx,
                                g_D=g_D, ds_D=ds_D, ds_N=ds_N, dS_int=dS)

        # use NonlinearVariationalSolver so we can read the SNES iteration count
        solver_params["snes_monitor"] = None  # print to console
        problem  = NonlinearVariationalProblem(F, z, bcs=bcs)
        nlsolver = NonlinearVariationalSolver(problem, solver_parameters=solver_params)

        u_out, p_out = z.sub(0), z.sub(1)
        u_out.rename("velocity")
        p_out.rename("pressure")

        newton_iters = []
        with CheckpointFile(h5name, "w") as h5file:
            h5file.save_mesh(mesh)
            save_idx = 0
            for tstep in range(args.ntsteps):
                t.assign(float(t) + float(dt))
                u_bc_in.interpolate(as_vector([u_in, 0]))

                print(f"[firedrake] step {tstep+1}/{args.ntsteps}, t = {float(t):.3f}")
                nlsolver.solve()
                newton_iters.append(nlsolver.snes.getIterationNumber())

                u_bc_diff = (
                    assemble(inner(u_out - u_bc_in, u_out - u_bc_in) * ds(1))
                               / u_bc_in.dat.data.max()**2
                    + assemble(inner(u_out, u_out) * ds((3, 4, 5)))
                )
                print(f"  Newton iters          = {newton_iters[-1]}")
                print(f"  L2 error on vel BC    = {u_bc_diff:.3e}")
                print(f"  velocity L2 norm      = {norm(u_out):.3e}")

                if (tstep + 1) % save_every == 0:
                    outfile.write(u_out, p_out, time=float(t))
                    h5file.save_function(u_out, name="velocity", idx=save_idx)
                    h5file.save_function(p_out, name="pressure", idx=save_idx)
                    save_idx += 1

                zold.assign(z)

        np.save(f"{stem}/newton_iters.npy", np.array(newton_iters))
        print(f"Saved {len(newton_iters)} iteration counts to {stem}/newton_iters.npy")

    # ================================================================
    # METHOD B: Newton class   (--method PStokes | PStokesLifted)
    # ================================================================
    else:

        solver = method_map[args.method](
            mesh, Z, args, Constant((0, 0)), dt, bc = bcs,
            g_D=g_D, ds_D=ds_D, ds_N =ds_N, dS_int=dS,
        )

        # t.assign(dt)
        # solver.z.sub(0).interpolate(as_vector([u_in, 0]))
        # t.assign(0)       

        u_out = solver.z.sub(0)
        p_out = solver.z.sub(1)
        u_out.rename("velocity")
        p_out.rename("pressure")

        newton_iters = []
        with CheckpointFile(h5name, "w") as h5file:
            h5file.save_mesh(mesh)
            save_idx = 0
            for tstep in range(args.ntsteps):
                print("Updating time")
                t.assign(float(t) + float(dt))
                u_bc_in.interpolate(as_vector([u_in, 0]))            

                print(f"[{args.method}] step {tstep+1}/{args.ntsteps}, t = {float(t):.3f}")
                newton_success, data = solver.step(return_data=True)
                newton_iters.append(data["iterations"] if data is not None else 0)

                u_bc_diff = (
                    assemble(inner(u_out - u_bc_in, u_out - u_bc_in) * ds(1))
                               / u_bc_in.dat.data.max()**2
                    + assemble(inner(u_out, u_out) * ds((3, 4, 5)))
                )
                
                print(f"  Newton iters          = {newton_iters[-1]}")
                print(f"  L2 error on vel BC    = {u_bc_diff:.3e}")
                print(f"  velocity L2 norm      = {norm(u_out):.3e}")

                if (tstep + 1) % save_every == 0:
                    outfile.write(u_out, p_out, time=float(t))
                    h5file.save_function(u_out, name="velocity", idx=save_idx)
                    h5file.save_function(p_out, name="pressure", idx=save_idx)
                    save_idx += 1
            
                if not newton_success:
                    print("Newton failed to converge at this time step, aborting time-stepping loop")
                    break                    

        np.save(f"{stem}/newton_iters.npy", np.array(newton_iters))
        print(f"Saved {len(newton_iters)} iteration counts to {stem}/newton_iters.npy")

