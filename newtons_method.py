from dataclasses import dataclass, field
from sys import float_info

from firedrake import *


@dataclass
class LineSearchResult:
    accepted: bool
    objective: float
    step_length: float


@dataclass
class IterationResult:
    accepted: bool
    objective: float
    step_norm: float
    step_length: float
    # linear_solves: int
    diagnostics: dict = field(default_factory=dict)


class NewtonMethod:
    """Base class for Newton's method with backtracking line search.

    Subclasses must implement:
        residual()      — linear system (lhs, rhs, bcs, step) for each Newton iteration
        objective()     — current objective value (for line search)
        objective_at()  — objective at state + step_length * step (without modifying state)
        accept()        — update state by step_length * step
        gradient_norm() — primal gradient norm for convergence

    Override dual_variable() if an algebraic dual variable step is needed after
    each linear solve (default: no-op).
    """

    def __init__(self, rtol=1e-6, max_iter=200, max_iter_ls=15, atol = 1e-10, stol = 1e-12):
        self.rtol = rtol
        self.atol = atol
        self.stol = stol
        self.max_iter = max_iter
        self.max_iter_ls = max_iter_ls
        self.solver_params = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "mat_type": "aij",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_14": 200,
        }
        self.nullspace = None  # set in subclass if needed

    def residual(self):
        """Return (lhs, rhs, bcs, step): the linear system lhs == rhs with BCs,
        whose solution is stored in the Function step."""
        raise NotImplementedError

    def dual_variable(self):
        """Compute dual variable step after the linear solve. No-op by default."""
        pass

    def objective(self):
        """Return current objective value."""
        raise NotImplementedError

    def objective_at(self, step_length):
        """Return objective at current state + step_length * step, without modifying state."""
        raise NotImplementedError

    def accept(self, step_length):
        """Update state by step_length * step."""
        raise NotImplementedError

    def gradient_norm(self):
        """Return gradient norm for convergence check."""
        raise NotImplementedError
    
    # def apply_bcs(self):
    #     '''Apply boundary conditions to the current state. Call this before the first iteration if needed.'''
    #     raise NotImplementedError

    def _linear_solve(self, lhs, rhs, bcs, step):
        bcs_hom = homogenize(bcs) if bcs else []
        solve(lhs == rhs, step, bcs=bcs_hom, solver_parameters=self.solver_params, nullspace=self.nullspace)

    def _line_search(self, objective_value):
        step_length = 1.0
        obj_new = objective_value
        objective_tolerance = 16.0 * float_info.epsilon * max(1.0, abs(objective_value))
        for _ in range(self.max_iter_ls):
            obj_new = self.objective_at(step_length)
            # print(f"  Line search: step_length = {step_length:.2e}, objective = {obj_new:.12e}")
            if obj_new <= objective_value + objective_tolerance:
                return LineSearchResult(True, obj_new, step_length)
            step_length *= 0.5
        return LineSearchResult(False, obj_new, step_length)

    def take_iteration(self, objective_value):
        lhs, rhs, bcs, step = self.residual()
        self._linear_solve(lhs, rhs, bcs, step)
        self.dual_variable()

        line_search = self._line_search(objective_value)
        if line_search.accepted:
            self.accept(line_search.step_length)
            return IterationResult(
                accepted=True,
                objective=self.objective(),
                step_norm=norm(line_search.step_length * step),
                step_length=line_search.step_length,
                # linear_solves=1,
            )

        return IterationResult(
            accepted=False,
            objective=objective_value,
            step_norm=norm(line_search.step_length * step),
            step_length=line_search.step_length,
            # linear_solves=1,
        )

    def solve(self, return_data=False):
        obj_val = self.objective()
        g_init = self.gradient_norm()
        gnorm_list = [g_init]
        objective_list = [obj_val]
        # linear_solves = []
        step_norms = []
        step_lengths = []
        diagnostics = []
        iterations = 0
        newton_success = False    

        # self.apply_bcs()        

        print(f"Objective before Newton: {obj_val:.5e}")
        print(f"Gradient norm before Newton: {g_init:.5e}")

        for i in range(self.max_iter):
            result = self.take_iteration(obj_val)
            iterations = i + 1
            # linear_solves.append(result.linear_solves)
            step_norms.append(result.step_norm)
            step_lengths.append(result.step_length)
            diagnostics.append(result.diagnostics)

            if not result.accepted:
                print("Line search failed")
                break

            obj_val = result.objective
            g_norm = self.gradient_norm()
            gnorm_list.append(g_norm)
            objective_list.append(obj_val)
            res_change = result.step_norm
            res_abs = g_norm
            rel_res = g_norm / g_init if g_init != 0.0 else (0.0 if g_norm == 0.0 else float("inf"))
            print(f"i = {i} :: {obj_val:.6e} :: step = {result.step_length:.2e} :: ||grad|| = {g_norm:.4e} :: rel res = {rel_res:.4e}")

            if rel_res < self.rtol:
                print("Relative tol reached")
                newton_success = True
                break
            elif res_abs < self.atol:
                print("Absolute tol reached")
                newton_success = True
                break
            elif res_change < self.stol:
                print("Stol reached")
                newton_success = True
                break

        if return_data:
            newton_data = {
                "iterations": iterations,
                "gradients": gnorm_list,
                "objectives": objective_list,
                # "linear_solves": linear_solves,
                "step_norms": step_norms,
                "step_lengths": step_lengths,
                "diagnostics": diagnostics,
            }
            return newton_success, newton_data
        else:
            return newton_success
