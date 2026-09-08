import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import inspect
from firedrake import *
from matplotlib import pylab as plt
from newtons_method import NewtonMethod, IterationResult


# --- helpers ---
def abs_grad(u, delta_min):
    return sqrt(delta_min**2 + dot(grad(u), grad(u)))


def plaplace_energy(u, f, p, delta_min, dX):
    return 1.0 / p * abs_grad(u, delta_min)**p * dX - f * u * dX


# --- Newton subclasses ---
class PLaplaceBase(NewtonMethod):

    def __init__(self, mesh, p, delta_min, f, fbc,
                 rtol=1e-4, max_iter=1000, max_iter_ls=20, quad_degree=5, element_family="CG", element_degree=1):
        super().__init__(rtol, max_iter, max_iter_ls, stol=0.0)
        self.mesh = mesh
        self.p = p
        self.f = f
        self.delta_min = Constant(delta_min)
        self.element_family = element_family
        self.element_degree = element_degree
        self.V = FunctionSpace(mesh, element_family, element_degree)
        self.bc = DirichletBC(self.V, fbc, "on_boundary")
        self.bc_hom = DirichletBC(self.V, Constant(0), "on_boundary")   
        self.dx = dx(degree=quad_degree)
        self.quad_degree = quad_degree

        self.u = Function(self.V)  
        self.bc.apply(self.u)  # apply BCs to initial guess
        self.unew = Function(self.V) 
        self.du = Function(self.V)
        self.F = derivative(self.J(self.u), self.u)

    def J(self, u):
        return plaplace_energy(u, self.f, self.p, self.delta_min, self.dx)     

    def objective(self): 
        return assemble(self.J(self.u))       
    
    def objective_at(self, s):
        self.unew.assign(self.u + s * self.du)
        return assemble(self.J(self.unew))    
    
    def gradient_norm(self):
        g = assemble(self.F, bcs=self.bc_hom)
        return norm(g.riesz_representation(riesz_map="H1"), norm_type="H1")    


class PLaplace(PLaplaceBase):
    """Standard Newton for p-Laplace. Uses weighted (p-Laplace Hessian) preconditioner
    by default; pass weighted=False for the full Jacobian."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Setting up Standard Newton for p-Laplace with p =", self.p)
        self.Flin = derivative(self.F, self.u)

    def residual(self): 
        return self.Flin, -self.F, self.bc_hom, self.du
    
    def accept(self, s): 
        self.u.assign(self.u + s * self.du)


class PLaplaceLifted(PLaplaceBase):
    """Reduced-form lifted Newton for p-Laplace (designed for p < 2): u in CG1, lambda
    in vector DG1. The primal Newton step du is solved from a symmetrized projected
    tensor operator; the dual step dlamda is then computed algebraically.

    ``lambda_projection_mode="state"`` projects the stored lambda after each
    accepted update. ``lambda_projection_mode="lhs"`` keeps the stored lambda raw
    and projects a copy used to form the reduced-system left-hand side.
    """

    _VALID_LAMBDA_PROJECTION_MODES = {"state", "lhs"}

    def __init__(self, *args, **kwargs):
        self.lambda_projection_mode = kwargs.pop("lambda_projection_mode", "lhs")
        if self.lambda_projection_mode not in self._VALID_LAMBDA_PROJECTION_MODES:
            modes = ", ".join(sorted(self._VALID_LAMBDA_PROJECTION_MODES))
            raise ValueError(
                f"lambda_projection_mode must be one of {{{modes}}}, "
                f"got {self.lambda_projection_mode!r}"
            )

        super().__init__(*args, **kwargs)
        print("Setting up Lifted Newton for p-Laplace with p =", self.p)

        # Create function space of quadrature points
        quad_FE = FiniteElement("Quadrature", degree=self.quad_degree, quad_scheme="default")
        Q = VectorFunctionSpace(self.mesh, quad_FE, self.quad_degree)

        # Define dual variable and its step
        self.lamda = Function(Q)
        self.lamda_lhs = Function(Q)
        self.dlamda = Function(Q)

        # Define Newton Jacobian form
        utr = TrialFunction(self.V)
        v = TestFunction(self.V)
        abs_gradu = abs_grad(self.u, self.delta_min)
        dim = self.mesh.geometric_dimension
        if callable(dim):
            dim = dim()
        lamda_for_lhs = (
            self.lamda
            if self.lambda_projection_mode == "state"
            else self.lamda_lhs
        )
        self.tensor_op = Identity(dim) - (2-self.p) * abs_gradu**(-self.p) * 0.5 * (outer(lamda_for_lhs, grad(self.u)) + outer(grad(self.u), lamda_for_lhs))
        # self.tensor_op = Identity(dim) - (2-self.p) * abs_gradu**(-self.p) * outer(self.lamda, grad(self.u)) # Non-symmetric version
        self.Flin = abs_gradu**(self.p - 2) * inner(dot(self.tensor_op, grad(utr)), grad(v)) * self.dx        

        # Define dual variable update form
        beta = 1-1e-4 # ensure the coercivity of the operator
        self.norm_gradu_p = beta*abs_gradu**(self.p-1)/(2-self.p + 1e-12)
        self.norm_lamda = sqrt(dot(self.lamda, self.lamda))
        self.lamda_projection = conditional(gt(self.norm_lamda,self.norm_gradu_p), self.lamda * self.norm_gradu_p/self.norm_lamda, self.lamda)
        self.dlamda_form = abs_gradu**(self.p-2) * dot(self.tensor_op, grad(self.du)) - self.lamda + abs_gradu**(self.p-2) * grad(self.u)        

    def residual(self):
        if self.lambda_projection_mode == "lhs":
            self.lamda_lhs.interpolate(self.lamda_projection)
        return self.Flin, -self.F, self.bc_hom, self.du
    
    def dual_variable(self): 
        self.dlamda.interpolate(self.dlamda_form)

    def accept(self, s):

        # Update step
        self.u.assign(self.u + s * self.du)
        self.lamda.assign(self.lamda + s * self.dlamda)

        if self.lambda_projection_mode == "state":
            # Project lamda to proper norm
            self.lamda.interpolate(self.lamda_projection)

class PLaplacePicard(PLaplaceBase):
    """Picard iterations for p-Laplace."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Setting up Picard iterations for p-Laplace with p =", self.p)
        
        # Define the Picard bilinear form
        utr = TrialFunction(self.V)
        v = TestFunction(self.V)
        abs_gradu = abs_grad(self.u, self.delta_min)
        
        # Frozen picard coefficient based on current state 'u'
        picard_coeff = abs_gradu**(self.p - 2)

        # lhs operator: A(u_k) * du
        self.A_picard = picard_coeff * inner(grad(utr), grad(v)) * self.dx

    def residual(self): 
        return self.A_picard, -self.F, self.bc_hom, self.du
    
    def accept(self, s): 
        self.u.assign(self.u + s * self.du)

class PLaplacePNLS(PLaplaceBase):
    """Picard-Newton iterations with line search on both stages."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Setting up PNLS iterations for p-Laplace with p =", self.p)

        # Define the Picard bilinear form
        utr = TrialFunction(self.V)
        v = TestFunction(self.V)
        abs_gradu = abs_grad(self.u, self.delta_min)

        # Frozen picard coefficient based on current state 'u'
        picard_coeff = abs_gradu**(self.p - 2)

        # lhs operator: A(u_k) * du
        self.A_picard = picard_coeff * inner(grad(utr), grad(v)) * self.dx
        self.Flin = derivative(self.F, self.u)
        self.u_old = Function(self.V)

    def take_iteration(self, objective_value):
        self.u_old.assign(self.u)
        diagnostics = {}
        try:
            self._linear_solve(self.A_picard, -self.F, self.bc_hom, self.du)
            picard_step_norm = norm(self.du)
            diagnostics["picard_step_norm"] = picard_step_norm

            picard_line_search = self._line_search(objective_value)
            if picard_line_search.accepted:
                self.u.assign(self.u + picard_line_search.step_length * self.du)
                diagnostics["picard_step_length"] = picard_line_search.step_length
                diagnostics["picard_objective"] = picard_line_search.objective
            else:
                self.u.assign(self.u_old)
                diagnostics["picard_step_length"] = picard_line_search.step_length
                diagnostics["picard_objective"] = picard_line_search.objective
                diagnostics["line_search_failed"] = True
                diagnostics["failed_stage"] = "picard"
                return IterationResult(
                    accepted=False,
                    objective=objective_value,
                    step_norm=0.0,
                    step_length=0.0,
                    diagnostics=diagnostics,
                )

            predictor_objective = self.objective()
            predictor_gradient = self.gradient_norm()
            diagnostics["predictor_objective"] = predictor_objective
            diagnostics["predictor_gradient"] = predictor_gradient

            self._linear_solve(self.Flin, -self.F, self.bc_hom, self.du)
            newton_step_norm = norm(self.du)
            diagnostics["newton_step_norm"] = newton_step_norm

            newton_line_search = self._line_search(predictor_objective)
            if newton_line_search.accepted:
                self.u.assign(self.u + newton_line_search.step_length * self.du)
                diagnostics["newton_step_length"] = newton_line_search.step_length
                diagnostics["line_search_objective"] = newton_line_search.objective
            else:
                self.u.assign(self.u_old)
                diagnostics["newton_step_length"] = newton_line_search.step_length
                diagnostics["line_search_objective"] = newton_line_search.objective
                diagnostics["line_search_failed"] = True
                diagnostics["failed_stage"] = "newton"
                return IterationResult(
                    accepted=False,
                    objective=objective_value,
                    step_norm=0.0,
                    step_length=0.0,
                    diagnostics=diagnostics,
                )

            self.unew.assign(self.u - self.u_old)
            return IterationResult(
                accepted=True,
                objective=self.objective(),
                step_norm=norm(self.unew),
                step_length=newton_line_search.step_length,
                diagnostics=diagnostics,
            )
        except Exception:
            self.u.assign(self.u_old)
            raise

method_map = {name: cls
               for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
               if issubclass(cls, NewtonMethod) and cls is not NewtonMethod}
method_list = sorted(method_map)

# --- entry point ---

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="p-Laplace Newton variants")
    parser.add_argument("--method", choices=method_list, default=method_list[0])
    parser.add_argument("--p", type=float, default=1.1)
    parser.add_argument("--delta-min", type=float, default=1e-6, dest="delta_min")
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--projection-mode", choices=sorted(PLaplaceLifted._VALID_LAMBDA_PROJECTION_MODES), default="lhs")
    args = parser.parse_args()

    # 2D example in Loisel's paper
    mesh = UnitSquareMesh(args.N, args.N, diagonal="crossed")
    x, y = SpatialCoordinate(mesh)
    eps = Constant(1e-12)
    left_strip = And(lt(abs(x), eps), And(ge(y, 0.25), le(y, 0.75)))
    upper_right = And(ge(x, 0.6), And(ge(y, 0.25), le(y, 1)))
    f_bc = conditional(Or(left_strip, upper_right), 1, 0)    
    f_rhs = Constant(0)

    method = method_map[args.method]
    method_kwargs = {"lambda_projection_mode": args.projection_mode} if issubclass(method, PLaplaceLifted) else {}
    solver = method(mesh, args.p, args.delta_min, f_rhs, f_bc, args.rtol, **method_kwargs)
    _, newton_data = solver.solve(return_data=True)

    # Save data in folder
    stem = f"plots/plaplace_{args.method}_p{int(args.p * 100)}"

    # Save to VTK
    VTKFile(f"{stem}/solution.pvd").write(solver.u)

    # Make a plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    col = tripcolor(solver.u, axes=ax[0])
    fig.colorbar(col, ax=ax[0], location="left", shrink=0.5)
    ax[0].set_title("Solution u")
    ax[0].set_aspect("equal")
    ax[0].set_axis_off()
    ax[1].set_title("Gradient norm")
    # ax[1].semilogy(newton_data["objectives"], marker="o", color = "blue", label = "objective", markersize=4)
    ax[1].semilogy(newton_data["gradients"], marker="o", color = "blue", label = "gradient", markersize=4)
    # ax[1].legend()
    fig.savefig(f"{stem}/plaplace.png", dpi=300)


    # Save h5
    u_sol = solver.u  # extract velocity component
    u_sol.rename("u")
    h5name  = f"{stem}/plaplace.h5"
    with CheckpointFile(h5name, "w") as h5file:
        h5file.save_mesh(mesh)
        h5file.save_function(u_sol, name="solution")

    # Save newton its
    np.save(f"{stem}/newton_grad.npy", np.array(newton_data["gradients"]))
    print(f"Saved solution to {stem}")
