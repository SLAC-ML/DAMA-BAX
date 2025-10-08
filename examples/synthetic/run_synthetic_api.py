"""
Synthetic BAX Example with Grid Expansion - Using High-Level API

This demonstrates the simplified run_bax_optimization() API for Pattern B
(grid/ensemble evaluation with expansion).

The API automatically handles:
- Initial data generation with LHS sampling
- Grid expansion for each objective
- Normalization setup
- BAXOpt configuration

User provides: oracles, objectives, algorithm, and expansion functions!
"""

import os
import sys
import numpy as np
import torch

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../core'))

from bax_core import run_bax_optimization, pareto_front


# ============================================================================
# Configuration
# ============================================================================

N_RADIAL_GRID = 10  # Grid points for objective 1
N_ANGULAR_GRID = 8  # Grid points for objective 2


# ============================================================================
# STEP 1: Define Grid Expansion Functions
# ============================================================================

def expand_for_obj1(x):
    """
    Expand base configs to radial grid for objective 1.

    Parameters:
    -----------
    x : np.ndarray, shape (n, 2)
        Base configurations

    Returns:
    --------
    X0 : np.ndarray, shape (n×n_radial, 3)
        Expanded grid: [x1, x2, radius]
    """
    n = x.shape[0]

    # Create radial grid [0.1, 0.2, ..., 1.0]
    radii = np.linspace(0.1, 1.0, N_RADIAL_GRID)

    # Expand: repeat each config n_radial times
    x_repeated = np.repeat(x, N_RADIAL_GRID, axis=0)  # (n×n_radial, 2)
    radii_tiled = np.tile(radii, n).reshape(-1, 1)     # (n×n_radial, 1)

    # Combine
    X0 = np.hstack([x_repeated, radii_tiled])  # (n×n_radial, 3)

    return X0


def expand_for_obj2(x):
    """
    Expand base configs to angular grid for objective 2.

    Parameters:
    -----------
    x : np.ndarray, shape (n, 2)
        Base configurations

    Returns:
    --------
    X1 : np.ndarray, shape (n×n_angular, 3)
        Expanded grid: [x1, x2, angle]
    """
    n = x.shape[0]

    # Create angular grid [0, π/4, π/2, ..., 2π)
    angles = np.linspace(0, 2*np.pi, N_ANGULAR_GRID, endpoint=False)

    # Expand: repeat each config n_angular times
    x_repeated = np.repeat(x, N_ANGULAR_GRID, axis=0)  # (n×n_angular, 2)
    angles_tiled = np.tile(angles, n).reshape(-1, 1)    # (n×n_angular, 1)

    # Combine
    X1 = np.hstack([x_repeated, angles_tiled])  # (n×n_angular, 3)

    return X1


# ============================================================================
# STEP 2: Define Oracle Functions
# ============================================================================

def oracle_obj1(X0):
    """
    Oracle for objective 1: Sphere function evaluated on radial grid.

    INPUT IS EXPANDED: X0 includes grid information.

    Parameters:
    -----------
    X0 : np.ndarray, shape (n×n_radial, 3)
        Expanded input: [x1, x2, radius]

    Returns:
    --------
    Y0 : np.ndarray, shape (n×n_radial, 1)
        Function values at each grid point
    """
    x1 = X0[:, 0]
    x2 = X0[:, 1]
    radius = X0[:, 2]

    # Sphere function scaled by radius
    Y0 = ((x1**2 + x2**2) * radius).reshape(-1, 1)

    # Add noise
    Y0 += 0.01 * np.random.randn(X0.shape[0], 1)

    return Y0


def oracle_obj2(X1):
    """
    Oracle for objective 2: Rosenbrock function evaluated on angular grid.

    INPUT IS EXPANDED: X1 includes grid information.

    Parameters:
    -----------
    X1 : np.ndarray, shape (n×n_angular, 3)
        Expanded input: [x1, x2, angle]

    Returns:
    --------
    Y1 : np.ndarray, shape (n×n_angular, 1)
        Function values at each grid point
    """
    x1 = X1[:, 0]
    x2 = X1[:, 1]
    angle = X1[:, 2]

    # Rotate coordinates by angle
    x1_rot = x1 * np.cos(angle) - x2 * np.sin(angle)
    x2_rot = x1 * np.sin(angle) + x2 * np.cos(angle)

    # Rosenbrock function on rotated coordinates
    Y1 = ((1 - x1_rot)**2 + 100 * (x2_rot - x1_rot**2)**2).reshape(-1, 1)

    # Add noise
    Y1 += 0.1 * np.random.randn(X1.shape[0], 1)

    return Y1


# ============================================================================
# STEP 3: Define Objective Functions
# ============================================================================

def objective_obj1(x, fn_model):
    """
    Objective 1: Mean sphere value over radial grid.

    Parameters:
    -----------
    x : np.ndarray, shape (n, 2)
        Base configuration parameters
    fn_model : function
        Surrogate model

    Returns:
    --------
    obj : np.ndarray, shape (n, 1)
        Objective values (mean over grid)
    """
    n = x.shape[0]

    # Expand to grid
    X0 = expand_for_obj1(x)

    # Predict with surrogate
    Y0_pred = fn_model(X0)  # (1, n×n_radial)

    # Reshape and aggregate
    Y0_pred = Y0_pred.T.reshape(n, N_RADIAL_GRID)  # (n, n_radial)
    obj = Y0_pred.mean(axis=1, keepdims=True)  # (n, 1)

    return obj


def objective_obj2(x, fn_model):
    """
    Objective 2: Mean Rosenbrock value over angular grid.

    Parameters:
    -----------
    x : np.ndarray, shape (n, 2)
        Base configuration parameters
    fn_model : function
        Surrogate model

    Returns:
    --------
    obj : np.ndarray, shape (n, 1)
        Objective values (mean over grid)
    """
    n = x.shape[0]

    # Expand to grid
    X1 = expand_for_obj2(x)

    # Predict with surrogate
    Y1_pred = fn_model(X1)  # (1, n×n_angular)

    # Reshape and aggregate
    Y1_pred = Y1_pred.T.reshape(n, N_ANGULAR_GRID)  # (n, n_angular)
    obj = Y1_pred.mean(axis=1, keepdims=True)  # (n, 1)

    return obj


# ============================================================================
# STEP 4: Define Algorithm Function
# ============================================================================

def make_algo():
    """
    Random sampling acquisition with objective-based selection.
    """
    def algo(fn_model_list):
        """
        Select next candidates using random sampling.

        Returns BASE configs (not expanded!)
        """
        fn_model1, fn_model2 = fn_model_list

        # Generate random BASE candidates
        n_candidates = 300
        X_candidates = np.random.rand(n_candidates, 2)

        # Evaluate with objective functions (they handle expansion)
        obj1_vals = objective_obj1(X_candidates, fn_model1)
        obj2_vals = objective_obj2(X_candidates, fn_model2)

        # Select diverse points
        n_select = 15

        # Sort by each objective
        idx1 = np.argsort(obj1_vals.flatten())
        idx2 = np.argsort(obj2_vals.flatten())

        # Take spread of points
        select1 = idx1[::len(idx1)//n_select][:n_select]
        select2 = idx2[::len(idx2)//n_select][:n_select]

        # Return BASE configs
        X_selected1 = X_candidates[select1]
        X_selected2 = X_candidates[select2]

        # IMPORTANT: Expand here before returning to BAXOpt
        X0_selected = expand_for_obj1(X_selected1)
        X1_selected = expand_for_obj2(X_selected2)

        return X0_selected, X1_selected

    return algo


# ============================================================================
# STEP 5: Run BAX Optimization
# ============================================================================

def main():
    print("=" * 70)
    print("Synthetic BAX Example with Grid Expansion - High-Level API")
    print("=" * 70)
    print()

    # That's it! Just call run_bax_optimization!
    opt, results = run_bax_optimization(
        # Required: The 3 core functions
        oracles=[oracle_obj1, oracle_obj2],
        objectives=[objective_obj1, objective_obj2],
        algorithm=make_algo(),

        # Pattern B specific: Expansion functions
        expansion_funcs=[expand_for_obj1, expand_for_obj2],

        # Optional: Configuration
        n_init=30,
        max_iterations=3,
        n_sampling=15,
        model_root='./models_grid_api/',
        nn_config={'n_neur': 150, 'lr': 1e-3, 'epochs': 30, 'iter_epochs': 10},
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        snapshot=True,
        verbose=True,
        seed=42,
    )

    # ========================================================================
    # Evaluate final Pareto front
    # ========================================================================
    print("\nEvaluating final Pareto front...")

    # Generate test set (BASE configs)
    X_test_base = np.random.rand(200, 2)

    # Evaluate using objective functions (they handle expansion)
    obj1_test = objective_obj1(X_test_base, opt.fn[0])
    obj2_test = objective_obj2(X_test_base, opt.fn[1])

    # Find Pareto front
    objectives = np.column_stack([obj1_test.flatten(), obj2_test.flatten()])
    pf = pareto_front(objectives)

    print(f"  Test set size: {len(X_test_base)} base configs")
    print(f"  Pareto front points: {len(pf)}")
    print(f"  Obj1 range: [{obj1_test.min():.4f}, {obj1_test.max():.4f}]")
    print(f"  Obj2 range: [{obj2_test.min():.4f}, {obj2_test.max():.4f}]")
    print()
    print("Pareto front (first 5 points):")
    for i in range(min(5, len(pf))):
        print(f"  {i+1}. Obj1={pf[i,0]:.4f}, Obj2={pf[i,1]:.4f}")

    print()
    print("=" * 70)
    print("DONE!")
    print("=" * 70)
    print()
    print("Compare this file (run_synthetic_api.py) with run_synthetic.py:")
    print("  - run_synthetic.py: ~470 lines (manual setup)")
    print("  - run_synthetic_api.py: ~340 lines (automatic setup)")
    print("  - Key difference: No manual data generation, normalization, or BAXOpt config!")
    print()
    print("Key takeaway:")
    print("  - Oracles received EXPANDED inputs (X0, X1)")
    print("  - Neural networks learned X0→Y0 and X1→Y1")
    print("  - Objectives aggregated predictions to compute final values")
    print("  - run_bax_optimization() handled all the boilerplate!")
    print()


if __name__ == '__main__':
    main()
