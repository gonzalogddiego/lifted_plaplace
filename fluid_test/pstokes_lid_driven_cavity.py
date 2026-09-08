import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import inspect
from firedrake import *
import finat
from newtons_method import NewtonMethod


# --- helpers ---
def abs_grad(u, delta_min):
    return sqrt(delta_min**2 + inner(sym(grad(u)), sym(grad(u))))

def pstokes_energy(z, f, p, delta_min, dX):
    u, pres = split(z)
    energy = 1.0 / p * abs_grad(u, delta_min)**p * dX - dot(f , u) * dX
    return energy

def pstokes_laplacian(z, f, p, delta_min, dX):
    u, pres = split(z)
    return pstokes_energy(z, f, p, delta_min, dX) - pres * div(u) * dX


# --- Newton subclasses ---
class PStokesBase(NewtonMethod):

    def __init__(self, mesh, Z, bc, p, delta_min, f,
                 nullspace = False,
                 rtol=1e-4, max_iter=1000, max_iter_ls=20, quad_degree=6):
        super().__init__(rtol, max_iter, max_iter_ls)
        self.p = p
        self.f = f
        self.delta_min = Constant(delta_min)

        self.mesh = mesh
        self.Z = Z
        self.bc = bc
        self.bc_hom = homogenize(bc)
        self.dx = dx(degree=quad_degree)
        self.quad_degree = quad_degree

        self.z = Function(self.Z) 
        if type(self.bc) == list:
            for bc_i in self.bc:
                    bc_i.apply(self.z)  # apply BCs to initial guess
        else:
            self.bc.apply(self.z)  # apply BCs to initial guess
        self.znew = Function(self.Z) 
        self.dz = Function(self.Z)
        self.F = derivative(self.J(self.z), self.z)

        if nullspace:
            print("Setting up nullspace for P-Stokes problem")
            self.nullspace = MixedVectorSpaceBasis(self.Z, [self.Z.sub(0), VectorSpaceBasis(constant=True)])

    def apply_bcs(self):
        print("\tApplying boundary conditions to the current state")
        for bc_i in (self.bc if isinstance(self.bc, list) else [self.bc]):
            bc_i.apply(self.z)

    def residual(self): 
        self.apply_bcs()
        return self.Flin, -self.F, self.bc, self.dz

    def J(self, z):
        return pstokes_laplacian(z, self.f, self.p, self.delta_min, self.dx)     

    def objective(self): 
        # return assemble(self.J(self.z))
        return assemble(pstokes_energy(self.z, self.f, self.p, self.delta_min, self.dx), bcs=self.bc_hom)
    
    def objective_at(self, s):
        self.znew.assign(self.z + s * self.dz)
        return assemble(pstokes_energy(self.znew, self.f, self.p, self.delta_min, self.dx), bcs=self.bc_hom) 
    
    def gradient_norm(self):
        g = assemble(self.F, bcs=self.bc_hom)
        return norm(g.riesz_representation(riesz_map="H1"), norm_type="H1")


class PStokes(PStokesBase):
    """Standard Newton for p-Stokes. Uses weighted (p-Stokes Hessian) preconditioner
    by default; pass weighted=False for the full Jacobian."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Setting up Standard Newton for p-Stokes with p =", self.p)
        self.Flin = derivative(self.F, self.z)
    
    def accept(self, s): 
        self.z.assign(self.z + s * self.dz)


class PStokesLifted(PStokesBase):
    """Reduced-form lifted Newton for p-Stokes (designed for p < 2): u in CG1, lambda
    in vector DG1. The primal Newton step du is solved from a symmetrized projected
    tensor operator; the dual step dlamda is then computed algebraically. Lambda is
    projected onto the ball of radius |grad u|^(p-1)/(2-p) before forming the system."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Setting up Lifted Newton for p-Stokes with p =", self.p)

        # Create function space of quadrature points
        quad_FE = FiniteElement("Quadrature", degree=self.quad_degree, quad_scheme="default")
        Q_quad = TensorFunctionSpace(mesh, quad_FE, self.quad_degree)
        # Q_quad = TensorFunctionSpace(self.mesh, "DQ", 1)

        # Define dual variable and its step
        self.lamda = Function(Q_quad)
        self.dlamda = Function(Q_quad)

        # Define Newton Jacobian form
        u, pres = split(self.z)
        du, dpres = split(self.dz)
        utr, pres_tr = TrialFunctions(self.Z)
        v, q = TestFunctions(self.Z)

        abs_gradu = abs_grad(u, self.delta_min)
        eps_u = sym(grad(u))
        eps_du = sym(grad(du))
        eps_utr = sym(grad(utr))
        eps_v = sym(grad(v))

        # Symmetrized rank-4 tensor action on symmetric 2-tensors
        # Analogous to PLaplaceLifted.tensor_op acting on vectors via dot:
        #   T(A) = A - (2-p)|eps_u|^{-p} * 1/2 * (inner(eps_u, A)*lamda + inner(lamda, A)*eps_u)
        self.tensor_op = lambda A: (
            A - (2 - self.p) * abs_gradu**(-self.p) * Constant(0.5) * (
                inner(eps_u, A) * self.lamda + inner(self.lamda, A) * eps_u
            )
        )    

        self.Flin = (
            abs_gradu**(self.p - 2) * inner(self.tensor_op(eps_utr), eps_v) * self.dx
            - pres_tr * div(v) * self.dx
            - q * div(utr) * self.dx
        )

        # Define dual variable update form
        beta = 1-1e-4 # ensure the coercivity of the operator
        self.norm_gradu_p = beta * abs_gradu**(self.p-1)/(2 - self.p + 1e-12)
        self.norm_lamda = sqrt(inner(self.lamda, self.lamda))
        self.dlamda_form = (
            -self.lamda
            + abs_gradu**(self.p-2) * eps_u
            + abs_gradu**(self.p-2) * self.tensor_op(eps_du)
        )
    
    def dual_variable(self): 
        self.dlamda.interpolate(self.dlamda_form)

    def accept(self, s):

        # Update step
        self.z.assign(self.z + s * self.dz)
        self.lamda.assign(self.lamda + s * self.dlamda)

        # Project lamda to proper norm
        condition = conditional(gt(self.norm_lamda,self.norm_gradu_p), self.lamda * self.norm_gradu_p/self.norm_lamda, self.lamda)
        self.lamda.interpolate(condition)        



method_map = {name: cls
               for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
               if issubclass(cls, NewtonMethod) and cls is not NewtonMethod}
method_list = sorted(method_map)

# --- entry point ---

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="p-Stokes Newton variants")
    parser.add_argument("--method", choices=method_list, default=method_list[0])
    parser.add_argument("--p", type=float, default=1.1)
    parser.add_argument("--delta-min", type=float, default=1e-6, dest="delta_min")
    parser.add_argument("--N", type=int, default=50)
    args = parser.parse_args()

    # Lid-driven cavity flow
    mesh = UnitSquareMesh(args.N, args.N, quadrilateral=False, diagonal="crossed")
    f_rhs = Constant([0,0])
    x, y = SpatialCoordinate(mesh)
    u_bc = as_vector([0.5 * (1 - cos(2 * pi * x)), 0])
    f_bc = conditional(ge(y, 0.9), u_bc, Constant([0,0]))
    nullspace = True
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    Z = MixedFunctionSpace([V, Q])
    bc = DirichletBC(Z.sub(0), f_bc, "on_boundary")
    stem = f"plots/lid_{args.method}_p{int(args.p * 100)}_N{int(args.N)}"

    # Initialize with p = 2!
    if args.p != 2.0:
        print("Initializing with p=2 solution")
        solver_newtonian = method_map[args.method](mesh, Z, bc, 2.0, args.delta_min, f_rhs, nullspace = nullspace, rtol = 1e-6, quad_degree=10)
        solver_newtonian.solve()
        z_newtonian = solver_newtonian.z.copy(deepcopy=True)

    # Solve with our Newton solver 
    solver = method_map[args.method](mesh, Z, bc, args.p, args.delta_min, f_rhs, nullspace = nullspace, rtol = 1e-6, quad_degree=10)
    solver.z.assign(z_newtonian)  # set initial guess to BCs
    newton_success, newton_data = solver.solve(return_data=True)

    if not newton_success:
        print("Newton failed to converge")

    print("Final velocity norm = %.3e" % norm(solver.z.sub(0)))
    print("Max velocity        = %.3e" % solver.z.sub(0).dat.data[:,0].max())
    print("Number of Newton iterations = %d" % (len(newton_data["gradients"])-1))
    print("Number of cells in mesh = %d" % mesh.num_cells())

    # Save to VTK
    os.makedirs(stem, exist_ok=True)
    vtk_file_name = f"plots/{args.method}.pvd"
    file = VTKFile(f"{stem}/velocity_pressure.pvd")
    u = solver.z.sub(0)  # extract velocity component
    p = solver.z.sub(1)  # extract pressure component
    u.rename("velocity")
    p.rename("pressure")
    try:
        uexact = z.sub(0)
        uexact.rename("velocity_direct")
        pexact = z.sub(1)
        pexact.rename("pressure_direct")
        file.write(u, p, uexact, pexact)
    except:
        file.write(u, p)

    # Save h5
    h5name  = f"{stem}/velocity_pressure.h5"
    with CheckpointFile(h5name, "w") as h5file:
        h5file.save_mesh(mesh)
        h5file.save_function(u, name="velocity")
        h5file.save_function(p, name="pressure")

    # Save newton its
    np.save(f"{stem}/newton_grad.npy", np.array(newton_data["gradients"]))
    print(f"Saved solution to {stem}")