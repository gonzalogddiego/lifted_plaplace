import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

parser = argparse.ArgumentParser(description="p-Laplace Newton variants")
parser.add_argument("--method", choices=["standard", "lifted"], default="standard")
parser.add_argument("--p", type=float, default=1.01)
parser.add_argument("--eps", type=float, default=1e-12)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--beta", type=float, default=0.99)
args = parser.parse_args()
np.random.seed(args.seed)

# Define initial condition
n = args.n + 2
delta_init = 0.05
fac1 = 2 * delta_init
x0 = fac1 * np.random.rand(n-2) + (1 - delta_init) # Draw initial condition from U([0.95,1.05])
x = np.concatenate(([0], x0, [0]))
x_coords = np.linspace(0, 1, n)
h_mesh = x_coords[1] - x_coords[0]

# Define operators
h = 1.0 / (n - 1)
A = np.zeros((n-1, n))
for i in range(n-1):
    A[i, i] = -1.0 / h
    A[i, i+1] = 1.0 / h
Phi = lambda x: np.diag(np.sqrt(x**2 + args.eps)**(args.p - 2))

# RHS
u_exact = np.concatenate(([0], np.ones(n-2), [0]))
y_exact = A @ u_exact
f = A.T @ Phi(y_exact) @ y_exact
f[0] = 0
f[-1] = 0

# Define energy
energy = lambda x: np.sum(np.sqrt((A @ x)**2 + args.eps)**args.p) / args.p - f @ x

# Start Newton iteration
max_iter_ls = 15
max_it = 1000
tol = 1e-5
beta_correction = args.beta  # correction parameter for lifted method (eq. 2.12)

coords = np.linspace(0, 1, n)
iterates = [x.copy()]
res_all = []
D_eig_all = []
step_size = []
angle = []

# Initialise dual variable for lifted method: λ = Φ(Ax)Ax
if args.method == "lifted":
    y0 = A @ x
    lam = np.zeros(y0.shape)#Phi(A@x) @ (A@x)
    dual_var = [lam.copy()]

    def proj_lam(lam_in, x_in, return_mask=False):
        y_in = A @ x_in
        norm_y_in = np.sqrt(y_in**2 + args.eps)
        bound = beta_correction * norm_y_in**args.p / ((2 - args.p) * np.abs(y_in) + 1e-15)
        mask = np.abs(lam_in) > bound
        lam_proj = lam_in.copy()
        lam_proj[mask] = bound[mask] * np.sign(lam_in[mask])
        if return_mask:
            return lam_proj, mask
        else:
            return lam_proj

    lam_proj = proj_lam(lam, x)        

for i in range(max_it):

    # Build residual and check convergence (standard primal residual for both methods)
    res = A.T @ Phi(A @ x) @ (A @ x) - f
    res[0] = 0
    res[-1] = 0

    res_norm_abs = np.linalg.norm(res)
    res_norm = res_norm_abs
    res_all.append(res_norm)
    if res_norm < tol:
        print(f"Converged at i = {i} :: ||res|| = {res_norm:.4e}")
        break

    # Build linear system
    y = A @ x
    norm_y_eps = np.sqrt(y**2 + args.eps)
    
    if args.method == "standard":
        d = norm_y_eps**(args.p -2) * ((args.p - 2) * y**2/norm_y_eps**2 + 1)
    else:
        d = norm_y_eps**(args.p - 2) * (1 + (args.p - 2) * lam_proj * y / norm_y_eps**args.p) 

    D = np.diag(d)
    A_lin = A.T @ D @ A

    # Set rows to enforce boundary conditions
    A_lin[0, :] = 0;  A_lin[0, 0]   = 1
    A_lin[-1, :] = 0; A_lin[-1, -1] = 1

    # Solve linear system
    dx = np.linalg.solve(A_lin, -res)

    # Linesearch (energy decrease on primal)
    obj_val = energy(x)
    step_length = 1.0
    step_fail = True
    for _ in range(max_iter_ls):
        if energy(x + step_length * dx) < obj_val:
            step_fail = False
            break
        step_length *= 0.5

    if step_fail:
        print("Line search failed")
        break

    step_size.append(step_length)
    angle.append(-dx @ res/(np.linalg.norm(res) * np.linalg.norm(dx)))

    x += step_length * dx

    # Update and correct dual variable
    if args.method == "lifted":
        dlam = D @ (A @ dx) - lam + norm_y_eps**(args.p - 2) * y # Dual update direction (eq. 2.10): λ̂ = D(Aû) - λ + Φ(y)y
        lam += step_length * dlam
        dual_var.append(lam.copy())

        # Correction (eq. 2.12): clip |λ_i| to β|y_i|_ε^{p-1}/(2-p) for p < 2
        if args.p < 2:
            lam_proj, mask = proj_lam(lam, x, return_mask=True)
        if n - 2 == 2:
            print("\tlambda = ", lam, " :: proj_comps = ", mask)
            dual_primal = Phi(A@x) @ (A@x)
            print("\tdual primal = ", dual_primal)
            

    print(f"i = {i} :: energy = {energy(x):.6e} :: step = {step_length:.2e} :: ||res|| = {res_norm:.4e} :: angle = {angle[-1]:.4f}")

    iterates.append(x.copy())

# Save data
run_tag = f"{args.method}_p{int(args.p * 1000)}_eps{int(-np.log10(args.eps))}_n{int(len(x) - 2)}_seed{args.seed}"
os.makedirs("data", exist_ok=True)
data_name = f"data/{run_tag}_trajectory_data.npz"
print(f"Saving data to {data_name}...")
iterates_arr = np.array(iterates)
np.savez(data_name, iterates=iterates_arr, f=f, A = A, step_size=step_size, dual_var=dual_var if args.method == "lifted" else None)