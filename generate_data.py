import json
import numpy as np
import os
from moralhazard import MoralHazardProblem
from moralhazard.config_maker import make_utility_cfg, make_distribution_cfg

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------
# Configuration derived from the original demo
CASES = {
    'log-gaussian': {
        'w0': 50, 'target_a': 100.0, 'a_min': 0.0, 'a_max': 180.0,
        'utility': 'log', 'theta_mult': 1.0,
        'dist_name': 'gaussian', 'dist_kwargs': {'sigma': 50.0},
        'comp_kwargs': {'distribution_type': 'continuous', 'y_min': -300.0, 'y_max': 480.0, 'n': 201},
        'res_wage_range': (-1.0, 15.0)
    },
    'cara-gaussian': {
        'w0': 50, 'target_a': 100.0, 'a_min': 0.0, 'a_max': 180.0,
        'utility': 'cara', 'theta_mult': 10.0,
        'dist_name': 'gaussian', 'dist_kwargs': {'sigma': 50.0},
        'comp_kwargs': {'distribution_type': 'continuous', 'y_min': -300.0, 'y_max': 480.0, 'n': 201},
        'res_wage_range': (-1.0, 15.0)
    },
    'log-poisson': {
        'w0': 50, 'target_a': 7.0, 'a_min': 0.0, 'a_max': 10.0,
        'utility': 'log', 'theta_mult': 1.0,
        'dist_name': 'poisson', 'dist_kwargs': {},
        'comp_kwargs': {'distribution_type': 'discrete', 'y_min': 0.0, 'y_max': 28.0, 'step_size': 1.0},
        'res_wage_range': (-1.0, 5.0)
    },
    'log-exponential': {
        'w0': 50, 'target_a': 100.0, 'a_min': 10.0, 'a_max': 180.0,
        'utility': 'log', 'theta_mult': 1.0,
        'dist_name': 'exponential', 'dist_kwargs': {},
        'comp_kwargs': {'distribution_type': 'continuous', 'y_min': 0.01, 'y_max': 260.0, 'n': 201},
        'res_wage_range': (-1.0, 15.0)
    },
    'log-t': {
        'w0': 50, 'target_a': 100.0, 'a_min': 0.0, 'a_max': 180.0,
        'utility': 'log', 'theta_mult': 1.0,
        'dist_name': 'Student_t', 'dist_kwargs': {'sigma': 20.0, 'nu': 1.15},
        'comp_kwargs': {'distribution_type': 'continuous', 'y_min': -500.0, 'y_max': 680.0, 'n': 201},
        'res_wage_range': (-1.0, 100.0)
    }
}

# -----------------------------------------------------------------------------
# 2. SOLVER ENGINE
# -----------------------------------------------------------------------------
STEPS = 50  # Resolution of the slider (higher = smoother but larger file)
export_data = {}

print(f"Starting data generation... (Steps: {STEPS})")

for case_name, params in CASES.items():
    print(f"Processing case: {case_name}...")
    
    # Setup cost function
    theta_base = 1.0 / params['target_a'] / (params['target_a'] + params['w0'])
    theta = theta_base * params['theta_mult']
    
    def C(a): return theta * a ** 2 / 2
    def Cprime(a): return theta * a

    # Setup Utility
    if params['utility'] == 'cara':
        utility_cfg = make_utility_cfg("cara", w0=params['w0'], alpha=1.0/params['w0'])
    else:
        utility_cfg = make_utility_cfg("log", w0=params['w0'])

    # Setup Distribution
    dist_cfg = make_distribution_cfg(params['dist_name'], **params['dist_kwargs'])
    
    cfg = {
        "problem_params": {**utility_cfg, **dist_cfg, "C": C, "Cprime": Cprime},
        "computational_params": params['comp_kwargs']
    }
    
    mhp = MoralHazardProblem(cfg)
    
    # 1. Action Grid for Plotting (Static per case)
    action_grid_plot = np.linspace(params['a_min'], params['a_max'], 100)
    
    # 2. Reservation Wage Grid (The Slider)
    rw_min, rw_max = params['res_wage_range']
    res_wage_grid = np.linspace(rw_min, rw_max, STEPS)
    
    min_wage_all, max_wage_all = float('inf'), float('-inf')
    min_util_all, max_util_all = float('inf'), float('-inf')

    case_frames = []
    
    for rw in res_wage_grid:
        ru = utility_cfg["u"](rw)
        try:
            sol = mhp.solve_principal_problem(
                revenue_function=lambda a: a,
                reservation_utility=ru,
                a_min=params['a_min'], a_max=params['a_max'],
                a_ic_lb=params['a_min'], a_ic_ub=params['a_max'],
                n_a_iterations=100
            )
            
            contract = sol.cmp_result.optimal_contract
            wage_curve = mhp.k(contract)
            utility_curve = mhp.U(contract, action_grid_plot)
            density_curve = mhp.f(mhp._y_grid, sol.optimal_action) 
            # Normalize density curve to be visible alongside the wage curve
            # We want max(density_curve) to be proportional to max(wage_curve)
            if np.max(density_curve) > 1e-9: # Avoid division by zero for flat distributions
                density_curve = density_curve / np.max(density_curve) * np.max(wage_curve) * 0.75
            else:
                density_curve = np.zeros_like(density_curve) # If flat, make it all zeros for consistency

            min_wage_all = min(min_wage_all, np.min(wage_curve))
            max_wage_all = max(max_wage_all, np.max(wage_curve))
            min_util_all = min(min_util_all, np.min(utility_curve), mhp.U(contract, sol.optimal_action))
            max_util_all = max(max_util_all, np.max(utility_curve), mhp.U(contract, sol.optimal_action))
            
            case_frames.append({
                "rw": float(rw),
                "wage_curve": wage_curve.tolist(),
                "u_curve": utility_curve.tolist(),
                "density_curve": density_curve.tolist(), # Add normalized density curve
                "opt_a": float(sol.optimal_action),
                "opt_u": float(mhp.U(contract, sol.optimal_action)),
                "foa_holds": bool(sol.cmp_result.first_order_approach_holds)
            })
        except Exception as e:
            print(f"  Error at rw={rw:.2f}: {e}")
            case_frames.append(None)
    
    # Add some padding to the axis limits
    wage_padding = (max_wage_all - min_wage_all) * 0.1
    util_padding = (max_util_all - min_util_all) * 0.1

    export_data[case_name] = {
        "y_grid": mhp._y_grid.tolist(),
        "a_grid": action_grid_plot.tolist(),
        "frames": case_frames,
        "wage_axis_range": [float(min_wage_all - wage_padding), float(max_wage_all + wage_padding)],
        "utility_axis_range": [float(min_util_all - util_padding), float(max_util_all + util_padding)]
    }

# -----------------------------------------------------------------------------
# 3. EXPORT TO JS
# -----------------------------------------------------------------------------
# We write to a .js file to allow local loading without CORS errors
js_content = f"window.MORAL_HAZARD_DATA = {json.dumps(export_data, allow_nan=False)};"

output_path = os.path.join(os.path.dirname(__file__), "model_data.js")
with open(output_path, "w") as f:
    f.write(js_content)

print(f"Done! Data saved to {output_path}")
