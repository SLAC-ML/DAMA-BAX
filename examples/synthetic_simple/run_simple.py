"""
Simple Synthetic Example - No Grid Expansion

This is the SIMPLEST possible BAX example demonstrating:
- Direct oracle evaluation (no grid expansion: X0 = X)
- Direct objective calculation (objective = prediction)
- Random sampling acquisition
- Full BAX loop: init data → pretrain → 3 BAX iterations

This example optimizes two simple functions:
- Objective 1: Sphere function (sum of squares)
- Objective 2: Rosenbrock function

Both objectives are evaluated on the SAME input space (2D configurations).
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
# STEP 1: Define oracle functions (expensive "simulations")
# ============================================================================

def oracle_obj1(X):
    """
    Oracle for objective 1: Sphere function.

    NO EXPANSION: Input X is used directly.

    Parameters:
    -----------
    X : np.ndarray, shape (n_samples, 2)
        Input configurations in [0, 1]^2

    Returns:
    --------
    Y : np.ndarray, shape (n_samples, 1)
        Function values
    """
    # Sphere function: sum of squares
    Y = np.sum(X**2, axis=1, keepdims=True)
    # Add small noise to simulate stochastic simulation
    Y += 0.01 * np.random.randn(X.shape[0], 1)
    return Y


def oracle_obj2(X):
    """
    Oracle for objective 2: Rosenbrock function.

    NO EXPANSION: Input X is used directly.

    Parameters:
    -----------
    X : np.ndarray, shape (n_samples, 2)
        Input configurations in [0, 1]^2

    Returns:
    --------
    Y : np.ndarray, shape (n_samples, 1)
        Function values
    """
    # Rosenbrock function
    x1, x2 = X[:, 0], X[:, 1]
    Y = ((1 - x1)**2 + 100 * (x2 - x1**2)**2).reshape(-1, 1)
    # Add small noise
    Y += 0.1 * np.random.randn(X.shape[0], 1)
    return Y


# ============================================================================
# STEP 2: Define objective functions (predictions → objectives)
# ============================================================================

def objective_obj1(x, fn_model):
    """
    Objective 1: Direct prediction (no transformation needed).

    For this simple example: objective = model prediction

    Parameters:
    -----------
    x : np.ndarray, shape (n, 2)
        Configuration parameters
    fn_model : function
        Surrogate model that predicts oracle output

    Returns:
    --------
    obj : np.ndarray, shape (n, 1)
        Objective values
    """
    # Direct prediction (no expansion, no aggregation)
    predictions = fn_model(x)  # (n, 1)
    return predictions  # (n, 1)


def objective_obj2(x, fn_model):
    """
    Objective 2: Direct prediction (no transformation needed).

    Parameters:
    -----------
    x : np.ndarray, shape (n, 2)
        Configuration parameters
    fn_model : function
        Surrogate model that predicts oracle output

    Returns:
    --------
    obj : np.ndarray, shape (n, 1)
        Objective values
    """
    predictions = fn_model(x)  # (n, 1)
    return predictions  # (n, 1)


# ============================================================================
# STEP 3: Define algorithm function (acquisition strategy)
# ============================================================================

def make_algo():
    """
    Simple random sampling acquisition.

    Returns diverse candidates from random samples.
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
            Candidates for objective 1
        X_obj2 : np.ndarray, shape (n_select, 2)
            Candidates for objective 2
        """
        fn_model1, fn_model2 = fn_model_list

        # Generate random candidates
        n_candidates = 200
        X_candidates = np.random.rand(n_candidates, 2)

        # Evaluate with objective functions
        obj1_vals = objective_obj1(X_candidates, fn_model1)
        obj2_vals = objective_obj2(X_candidates, fn_model2)

        # Select diverse points (simple strategy: quartiles)
        n_select = 10
        idx1 = np.linspace(0, len(X_candidates)-1, n_select, dtype=int)
        idx2 = np.linspace(0, len(X_candidates)-1, n_select, dtype=int)

        # For now, return same candidates for both objectives
        # (In DAMA, these could be different based on boundary sampling)
        X_selected = X_candidates[idx1]

        return X_selected, X_selected

    return algo


# ============================================================================
# STEP 4: Main function
# ============================================================================

def main():
    print("=" * 70)
    print("Simple Synthetic BAX Example (No Grid Expansion)")
    print("=" * 70)
    print()

    # Setup
    n_dims = 2
    n_init = 50  # Initial training samples
    n_iters = 3  # BAX iterations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Configuration:")
    print(f"  Dimensions: {n_dims}")
    print(f"  Initial samples: {n_init}")
    print(f"  BAX iterations: {n_iters}")
    print(f"  Device: {device}")
    print()

    # ========================================================================
    # Generate initial data
    # ========================================================================
    print("Step 1: Generating initial training data...")
    X_init = np.random.rand(n_init, n_dims)

    # Evaluate with oracles (NO EXPANSION: X0 = X, X1 = X)
    Y0_init = oracle_obj1(X_init)  # (n_init, 1)
    Y1_init = oracle_obj2(X_init)  # (n_init, 1)

    print(f"  X_init shape: {X_init.shape}")
    print(f"  Y0_init shape: {Y0_init.shape}")
    print(f"  Y1_init shape: {Y1_init.shape}")
    print()

    # ========================================================================
    # Setup normalization
    # ========================================================================
    print("Step 2: Setting up normalization...")
    X_mu, X_std = dann.get_norm(X_init)

    def norm(X):
        return dann.normalize(X.copy(), X_mu, X_std)

    # Init functions
    def init_obj1():
        return X_init, Y0_init.flatten()  # BAXOpt expects 1D

    def init_obj2():
        return X_init, Y1_init.flatten()

    print("  Normalization ready")
    print()

    # ========================================================================
    # Create BAX optimizer
    # ========================================================================
    print("Step 3: Creating BAX optimizer...")
    algo = make_algo()

    opt = BAXOpt(
        algo=algo,
        fn_oracle=[oracle_obj1, oracle_obj2],
        norm=[norm, norm],
        init=[init_obj1, init_obj2],
        device=device,
        snapshot=True,
        model_root='./models_simple/'
    )

    # Configure
    opt.n_sampling = 10  # Points per iteration
    opt.n_feat = n_dims  # Input dimensionality
    opt.epochs = 30  # Initial training epochs
    opt.iter_epochs = 10  # Per-iteration training
    opt.n_neur = 100  # Small network for simple problem
    opt.lr = 1e-3
    opt.dropout = 0.1

    print(f"  Network: {opt.n_neur} neurons, {opt.epochs} initial epochs")
    print(f"  Sampling: {opt.n_sampling} points/iteration")
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

    # Generate test set
    X_test = np.random.rand(500, n_dims)
    obj1_test = objective_obj1(X_test, opt.fn[0])
    obj2_test = objective_obj2(X_test, opt.fn[1])

    # Find Pareto front
    objectives = np.column_stack([obj1_test.flatten(), obj2_test.flatten()])
    pf = pareto_front(objectives)

    print(f"  Test set size: {len(X_test)}")
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
    print(f"Total oracle calls: {n_init + n_iters * opt.n_sampling} per objective")
    print(f"Models saved to: {opt.model_root}")
    print()


if __name__ == '__main__':
    main()
