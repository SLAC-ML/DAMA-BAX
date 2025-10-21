"""
Synthetic BAX Example with Grid Expansion - New Unified API

This demonstrates how to handle grid expansion with the new API:
- Expansion logic is INSIDE objective functions (not exposed as parameter)
- BAX parameters controlled by run.py CLI
- Only 3 functions required: oracles, objectives, algorithm

Key insight: The oracles receive expanded inputs, the objectives handle
the expansion internally. This is an implementation detail, not something
the BAX framework needs to know about.

Usage:
  python run.py --case examples/synthetic --n-init 30 --max-iter 5

Or standalone:
  python run_synthetic_api_v2.py
"""

import os
import sys
import numpy as np

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../core'))


# ============================================================================
# Configuration
# ============================================================================

N_RADIAL_GRID = 10  # Grid points for objective 1
N_ANGULAR_GRID = 8  # Grid points for objective 2


# ============================================================================
# STEP 1: Define Oracle Functions (receive expanded inputs)
# ============================================================================

def oracle_obj1(X0):
    """
    Oracle for objective 1: Sphere function evaluated on radial grid.

    INPUT IS EXPANDED: X0 includes grid information [x1, x2, radius].

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
    Y0 = (x1**2 + x2**2) * radius

    # Add noise
    Y0 += 0.01 * np.random.randn(X0.shape[0])

    return Y0.reshape(-1, 1)  # Return (n×n_radial, 1) shape


def oracle_obj2(X1):
    """
    Oracle for objective 2: Rosenbrock function evaluated on angular grid.

    INPUT IS EXPANDED: X1 includes grid information [x1, x2, angle].

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
    Y1 = (1 - x1_rot)**2 + 100 * (x2_rot - x1_rot**2)**2

    # Add noise
    Y1 += 0.1 * np.random.randn(X1.shape[0])

    return Y1.reshape(-1, 1)  # Return (n×n_angular, 1) shape


# ============================================================================
# STEP 2: Define Objective Functions (handle expansion internally)
# ============================================================================

def objective_obj1(x, fn_model):
    """
    Objective 1: Mean sphere value over radial grid.

    EXPANSION HAPPENS HERE - this is hidden from the BAX framework.

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

    # INTERNAL HELPER: Expand to radial grid
    # Create radial grid [0.1, 0.2, ..., 1.0]
    radii = np.linspace(0.1, 1.0, N_RADIAL_GRID)

    # Expand: repeat each config n_radial times
    x_repeated = np.repeat(x, N_RADIAL_GRID, axis=0)  # (n×n_radial, 2)
    radii_tiled = np.tile(radii, n).reshape(-1, 1)     # (n×n_radial, 1)

    # Combine to get expanded input
    X0 = np.hstack([x_repeated, radii_tiled])  # (n×n_radial, 3)

    # Predict with surrogate
    Y0_pred = fn_model(X0)  # (n×n_radial, 1)

    # Reshape and aggregate
    Y0_pred = Y0_pred.reshape(n, N_RADIAL_GRID)  # (n, n_radial)
    obj = Y0_pred.mean(axis=1, keepdims=True)  # (n, 1)

    return obj


def objective_obj2(x, fn_model):
    """
    Objective 2: Mean Rosenbrock value over angular grid.

    EXPANSION HAPPENS HERE - this is hidden from the BAX framework.

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

    # INTERNAL HELPER: Expand to angular grid
    # Create angular grid [0, π/4, π/2, ..., 2π)
    angles = np.linspace(0, 2*np.pi, N_ANGULAR_GRID, endpoint=False)

    # Expand: repeat each config n_angular times
    x_repeated = np.repeat(x, N_ANGULAR_GRID, axis=0)  # (n×n_angular, 2)
    angles_tiled = np.tile(angles, n).reshape(-1, 1)    # (n×n_angular, 1)

    # Combine to get expanded input
    X1 = np.hstack([x_repeated, angles_tiled])  # (n×n_angular, 3)

    # Predict with surrogate
    Y1_pred = fn_model(X1)  # (n×n_angular, 1)

    # Reshape and aggregate
    Y1_pred = Y1_pred.reshape(n, N_ANGULAR_GRID)  # (n, n_angular)
    obj = Y1_pred.mean(axis=1, keepdims=True)  # (n, 1)

    return obj


# ============================================================================
# STEP 3: Define Algorithm Function
# ============================================================================

def make_algo():
    """
    Random sampling acquisition with objective-based selection.
    """
    # Helper functions for expansion (same as used in objectives and init_sampler)
    def expand_radial(x):
        """Expand base configs to radial grid."""
        n = x.shape[0]
        radii = np.linspace(0.1, 1.0, N_RADIAL_GRID)
        x_repeated = np.repeat(x, N_RADIAL_GRID, axis=0)
        radii_tiled = np.tile(radii, n).reshape(-1, 1)
        return np.hstack([x_repeated, radii_tiled])

    def expand_angular(x):
        """Expand base configs to angular grid."""
        n = x.shape[0]
        angles = np.linspace(0, 2*np.pi, N_ANGULAR_GRID, endpoint=False)
        x_repeated = np.repeat(x, N_ANGULAR_GRID, axis=0)
        angles_tiled = np.tile(angles, n).reshape(-1, 1)
        return np.hstack([x_repeated, angles_tiled])

    def algo(fn_model_list):
        """
        Select next candidates using random sampling.

        Returns EXPANDED configs (oracles expect expanded inputs!)
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

        # Get BASE configs
        X_selected1 = X_candidates[select1]
        X_selected2 = X_candidates[select2]

        # IMPORTANT: Expand before returning (oracles expect expanded inputs!)
        X0_selected = expand_radial(X_selected1)
        X1_selected = expand_angular(X_selected2)

        return X0_selected, X1_selected

    return algo


# ============================================================================
# STEP 4: Custom Initialization (since oracles expect expanded inputs)
# ============================================================================

def make_init_sampler():
    """
    Custom initialization that generates expanded data for the oracles.

    This is necessary because our oracles expect expanded inputs, but the
    optimization happens in the base 2D space.
    """
    def init_sampler():
        """
        Generate initial training data with expansion.

        Returns:
        --------
        X_list : list of [X0, X1]
            X0: (n_configs × n_radial, 3) for objective 1
            X1: (n_configs × n_angular, 3) for objective 2
        Y_list : list of [Y0, Y1]
            Y0: (n_configs × n_radial,)
            Y1: (n_configs × n_angular,)
        """
        from pyDOE import lhs

        # Generate base configurations
        n_configs = 30
        X_base = lhs(2, samples=n_configs, criterion='maximin')

        # Expand for objective 1 (radial grid)
        radii = np.linspace(0.1, 1.0, N_RADIAL_GRID)
        x_repeated = np.repeat(X_base, N_RADIAL_GRID, axis=0)
        radii_tiled = np.tile(radii, n_configs).reshape(-1, 1)
        X0 = np.hstack([x_repeated, radii_tiled])

        # Expand for objective 2 (angular grid)
        angles = np.linspace(0, 2*np.pi, N_ANGULAR_GRID, endpoint=False)
        x_repeated = np.repeat(X_base, N_ANGULAR_GRID, axis=0)
        angles_tiled = np.tile(angles, n_configs).reshape(-1, 1)
        X1 = np.hstack([x_repeated, angles_tiled])

        print(f"Generating {n_configs} base configurations...")
        print(f"  Objective 0: Evaluating {X0.shape[0]} oracle calls...")
        Y0 = oracle_obj1(X0)

        print(f"  Objective 1: Evaluating {X1.shape[0]} oracle calls...")
        Y1 = oracle_obj2(X1)

        return [X0, X1], [Y0, Y1]

    return init_sampler


# ============================================================================
# STEP 5: Entry Point for Unified Runner
# ============================================================================

def get_bax_config(args):
    """
    Called by run.py to get case configuration.

    All BAX parameters (n_init, max_iter, nn_neurons, etc.) are handled by
    run.py CLI arguments. This function only returns problem-specific config.

    Parameters:
    -----------
    args : argparse.Namespace
        Arguments from run.py

    Returns:
    --------
    config : dict
        Configuration dictionary
    """
    # Determine model directory based on run_id if provided
    if hasattr(args, 'run_id') and args.run_id is not None:
        model_root = f'./models_grid_run_{args.run_id}/'
    else:
        model_root = './models_grid/'

    return {
        'oracles': [oracle_obj1, oracle_obj2],
        'objectives': [objective_obj1, objective_obj2],
        'algorithm': make_algo(),
        'init_sampler': make_init_sampler(),  # Required because oracles expect expanded inputs
        'model_root': model_root,
        # Optional: Provide defaults (overridden by CLI)
        # 'n_init': 30,
        # 'max_iterations': 3,
        # 'n_sampling': 15,
    }


# ============================================================================
# Optional: Standalone Execution
# ============================================================================

if __name__ == '__main__':
    """
    Can still run standalone, but recommended to use:
      python run.py --case examples/synthetic
    """
    from bax_core import run_bax_optimization, pareto_front
    import torch

    print("=" * 70)
    print("Synthetic BAX Example with Grid Expansion - Standalone Mode")
    print("=" * 70)
    print("TIP: Use unified runner for more control:")
    print("  python run.py --case examples/synthetic --max-iter 10 --nn-neurons 200")
    print("=" * 70)
    print()

    # Create dummy args
    class Args:
        run_id = None

    config = get_bax_config(Args())

    # Run with hardcoded defaults
    opt, results = run_bax_optimization(
        oracles=config['oracles'],
        objectives=config['objectives'],
        algorithm=config['algorithm'],
        init_sampler=config['init_sampler'],
        model_root=config['model_root'],
        max_iterations=3,
        n_sampling=15,
        nn_config={'n_neur': 150, 'lr': 1e-3, 'epochs': 30, 'iter_epochs': 10},
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        snapshot=True,
        verbose=True,
        seed=42,
    )

    # Evaluate final Pareto front
    print("\nEvaluating final Pareto front...")
    X_test_base = np.random.rand(200, 2)
    obj1_test = objective_obj1(X_test_base, opt.fn[0])
    obj2_test = objective_obj2(X_test_base, opt.fn[1])

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
    print("Key takeaway:")
    print("  - Oracles received EXPANDED inputs (X0, X1)")
    print("  - Neural networks learned X0→Y0 and X1→Y1")
    print("  - Objectives handled expansion internally (hidden from BAX)")
    print("  - Algorithm works in BASE space (2D)")
    print()
