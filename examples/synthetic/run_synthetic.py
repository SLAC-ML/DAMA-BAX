"""
Synthetic Example with Grid Expansion

This example demonstrates the COMPLETE BAX pattern similar to DAMA:
- Oracle functions receive EXPANDED inputs (X0 for obj0, X1 for obj1)
- X → X0/X1 expansion happens in objective functions
- Different expansion for each objective (like DA vs MA in DAMA)
- Aggregation step to compute final objectives

This example optimizes two functions over different evaluation grids:
- Objective 1: Mean of sphere function over radial grid
- Objective 2: Mean of Rosenbrock over angular grid

Key Concept:
- Base configs: X (n, 2) - the configurations we optimize
- Expanded for obj1: X0 (n×n_radial, 3) - [x1, x2, radius]
- Expanded for obj2: X1 (n×n_angular, 3) - [x1, x2, angle]
- Neural networks model: X0→Y0 and X1→Y1
- Objectives aggregate: Y0→obj0 and Y1→obj1
"""

import os
import sys
import numpy as np
import torch

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../core'))

from bax_core import BAXOpt, pareto_front
import da_NN as dann


# ============================================================================
# Configuration
# ============================================================================

N_RADIAL_GRID = 10  # Grid points for objective 1
N_ANGULAR_GRID = 8  # Grid points for objective 2


# ============================================================================
# STEP 1: Define oracle functions (expensive "simulations")
# ============================================================================

def oracle_obj1(X0):
    """
    Oracle for objective 1: Sphere function evaluated on radial grid.

    INPUT IS EXPANDED: X0 includes grid information.

    Parameters:
    -----------
    X0 : np.ndarray, shape (n×n_radial, 3)
        Expanded input: [x1, x2, radius]
        Each base config (x1, x2) is evaluated at n_radial radius values

    Returns:
    --------
    Y0 : np.ndarray, shape (n×n_radial, 1)
        Function values at each grid point
    """
    # Extract coordinates
    x1 = X0[:, 0]
    x2 = X0[:, 1]
    radius = X0[:, 2]

    # Evaluate sphere function scaled by radius
    # This simulates evaluating at different distances from origin
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
        Each base config is evaluated at n_angular angle values

    Returns:
    --------
    Y1 : np.ndarray, shape (n×n_angular, 1)
        Function values at each grid point
    """
    # Extract coordinates and angle
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
# STEP 2: Grid expansion functions
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
# STEP 3: Define objective functions (predictions → objectives)
# ============================================================================

def objective_obj1(x, fn_model):
    """
    Objective 1: Mean sphere value over radial grid.

    This demonstrates the complete flow:
    x (base) → X0 (expanded) → Y0_pred (predict) → obj (aggregate)

    Parameters:
    -----------
    x : np.ndarray, shape (n, 2)
        Base configuration parameters
    fn_model : function
        Surrogate model that predicts Y0 from X0

    Returns:
    --------
    obj : np.ndarray, shape (n, 1)
        Objective values (mean over grid)
    """
    n = x.shape[0]

    # Step 1: Expand to grid
    X0 = expand_for_obj1(x)  # (n×n_radial, 3)

    # Step 2: Predict with surrogate
    Y0_pred = fn_model(X0)  # (n×n_radial, 1) in transposed form

    # Step 3: Reshape and aggregate
    Y0_pred = Y0_pred.T  # (n×n_radial, 1)
    Y0_reshaped = Y0_pred.reshape(n, N_RADIAL_GRID)  # (n, n_radial)

    # Aggregate: mean over grid points
    obj = Y0_reshaped.mean(axis=1, keepdims=True)  # (n, 1)

    return obj


def objective_obj2(x, fn_model):
    """
    Objective 2: Mean Rosenbrock value over angular grid.

    Parameters:
    -----------
    x : np.ndarray, shape (n, 2)
        Base configuration parameters
    fn_model : function
        Surrogate model that predicts Y1 from X1

    Returns:
    --------
    obj : np.ndarray, shape (n, 1)
        Objective values (mean over grid)
    """
    n = x.shape[0]

    # Step 1: Expand to grid
    X1 = expand_for_obj2(x)  # (n×n_angular, 3)

    # Step 2: Predict with surrogate
    Y1_pred = fn_model(X1)  # (n×n_angular, 1) in transposed form

    # Step 3: Reshape and aggregate
    Y1_pred = Y1_pred.T  # (n×n_angular, 1)
    Y1_reshaped = Y1_pred.reshape(n, N_ANGULAR_GRID)  # (n, n_angular)

    # Aggregate: mean over grid points
    obj = Y1_reshaped.mean(axis=1, keepdims=True)  # (n, 1)

    return obj


# ============================================================================
# STEP 4: Define algorithm function (acquisition strategy)
# ============================================================================

def make_algo():
    """
    Random sampling acquisition with objective-based selection.
    """
    def algo(fn_model_list):
        """
        Select next candidates using random sampling.

        Parameters:
        -----------
        fn_model_list : list of functions
            [fn_model1, fn_model2] - surrogate models

        Returns:
        --------
        X_obj1 : np.ndarray, shape (n_select, 2)
            Base candidates for objective 1
        X_obj2 : np.ndarray, shape (n_select, 2)
            Base candidates for objective 2

        NOTE: Returns BASE configs (not expanded!)
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

        # Return BASE configs (BAXOpt will pass these to oracles)
        # But oracles expect EXPANDED inputs!
        # So we need to return expanded versions
        X_selected1 = X_candidates[select1]
        X_selected2 = X_candidates[select2]

        # IMPORTANT: Expand here before returning to BAXOpt
        X0_selected = expand_for_obj1(X_selected1)
        X1_selected = expand_for_obj2(X_selected2)

        return X0_selected, X1_selected

    return algo


# ============================================================================
# STEP 5: Main function
# ============================================================================

def main():
    print("=" * 70)
    print("Synthetic BAX Example with Grid Expansion")
    print("=" * 70)
    print()

    # Setup
    n_dims_base = 2  # Base dimensionality
    n_dims_obj1 = 3  # After expansion for obj1
    n_dims_obj2 = 3  # After expansion for obj2
    n_init_base = 30  # Initial base configs
    n_iters = 3  # BAX iterations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Configuration:")
    print(f"  Base dimensions: {n_dims_base}")
    print(f"  Obj1 expanded dims: {n_dims_obj1} (radial grid: {N_RADIAL_GRID} points)")
    print(f"  Obj2 expanded dims: {n_dims_obj2} (angular grid: {N_ANGULAR_GRID} points)")
    print(f"  Initial base configs: {n_init_base}")
    print(f"  BAX iterations: {n_iters}")
    print(f"  Device: {device}")
    print()

    # ========================================================================
    # Generate initial data
    # ========================================================================
    print("Step 1: Generating initial training data...")

    # Generate base configs
    X_init_base = np.random.rand(n_init_base, n_dims_base)

    # Expand for each objective
    X0_init = expand_for_obj1(X_init_base)  # (n_init×n_radial, 3)
    X1_init = expand_for_obj2(X_init_base)  # (n_init×n_angular, 3)

    # Evaluate with oracles (EXPANDED inputs)
    Y0_init = oracle_obj1(X0_init)  # (n_init×n_radial, 1)
    Y1_init = oracle_obj2(X1_init)  # (n_init×n_angular, 1)

    print(f"  X_init_base shape: {X_init_base.shape}")
    print(f"  X0_init shape (expanded): {X0_init.shape}")
    print(f"  X1_init shape (expanded): {X1_init.shape}")
    print(f"  Y0_init shape: {Y0_init.shape}")
    print(f"  Y1_init shape: {Y1_init.shape}")
    print()

    # ========================================================================
    # Setup normalization (for EXPANDED dimensions)
    # ========================================================================
    print("Step 2: Setting up normalization...")

    # Normalization for obj1 (3D expanded space)
    X0_mu, X0_std = dann.get_norm(X0_init)
    def norm0(X):
        return dann.normalize(X.copy(), X0_mu, X0_std)

    # Normalization for obj2 (3D expanded space)
    X1_mu, X1_std = dann.get_norm(X1_init)
    def norm1(X):
        return dann.normalize(X.copy(), X1_mu, X1_std)

    # Init functions
    def init_obj1():
        return X0_init, Y0_init.flatten()

    def init_obj2():
        return X1_init, Y1_init.flatten()

    print("  Normalization ready (separate for each objective)")
    print()

    # ========================================================================
    # Create BAX optimizer
    # ========================================================================
    print("Step 3: Creating BAX optimizer...")
    algo = make_algo()

    opt = BAXOpt(
        algo=algo,
        fn_oracle=[oracle_obj1, oracle_obj2],
        norm=[norm0, norm1],  # Different normalization!
        init=[init_obj1, init_obj2],
        device=device,
        snapshot=True,
        model_root='./models_grid/'
    )

    # Configure
    opt.n_sampling = 15  # Points per iteration (will be expanded)
    opt.n_feat = n_dims_obj1  # IMPORTANT: Set to expanded dimensionality!
    opt.epochs = 30  # Initial training epochs
    opt.iter_epochs = 10  # Per-iteration training
    opt.n_neur = 150  # Network size
    opt.lr = 1e-3
    opt.dropout = 0.1

    print(f"  Network: {opt.n_neur} neurons, {opt.epochs} initial epochs")
    print(f"  Sampling: {opt.n_sampling} base configs/iteration")
    print()

    # ========================================================================
    # Run BAX optimization
    # ========================================================================
    print("Step 4: Running BAX optimization...")
    print()

    opt.run_acquisition(n_iters=n_iters, verbose=True)

    print()

    # ========================================================================
    # Evaluate final Pareto front
    # ========================================================================
    print("Step 5: Evaluating final Pareto front...")

    # Generate test set (BASE configs)
    X_test_base = np.random.rand(200, n_dims_base)

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
    print(f"Pareto front (first 5 points):")
    for i in range(min(5, len(pf))):
        print(f"  {i+1}. Obj1={pf[i,0]:.4f}, Obj2={pf[i,1]:.4f}")

    print()
    print("=" * 70)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 70)
    print(f"Total oracle calls:")
    print(f"  Obj1: {n_init_base * N_RADIAL_GRID + n_iters * opt.n_sampling * N_RADIAL_GRID} evaluations")
    print(f"  Obj2: {n_init_base * N_ANGULAR_GRID + n_iters * opt.n_sampling * N_ANGULAR_GRID} evaluations")
    print(f"Models saved to: {opt.model_root}")
    print()
    print("Key takeaway:")
    print("  - Oracles received EXPANDED inputs (X0, X1)")
    print("  - Neural networks learned X0→Y0 and X1→Y1")
    print("  - Objectives aggregated predictions to compute final values")
    print()


if __name__ == '__main__':
    main()
