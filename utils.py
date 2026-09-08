from firedrake import *
import numpy as np

solver_params = {
    # "snes_monitor": None,
    "snes_linesearch_type": "bt",
    "snes_max_it": 500,
    "snes_rtol": 1.0e-8,
    "snes_atol": 1.0e-10,
    "snes_stol": 1e-15,        
    "ksp_type": "preonly",
    "pc_type": "lu",
    "mat_type": "aij",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_14": 200,
}