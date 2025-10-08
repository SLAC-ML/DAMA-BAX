"""
Simple Synthetic BAX Example Using High-Level API

This demonstrates the simplified run_bax_optimization() API which automates:
- Initial data generation (automatic LHS sampling)
- Normalization setup
- BAXOpt configuration
- Model training and optimization loop

User only needs to provide 3 functions: oracles, objectives, and algorithm!
"""

import os
import sys
import numpy as np
import torch

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../core'))

from bax_core import run_bax_optimization, pareto_front


# ============================================================================
# STEP 1: Define Oracle Functions
# ============================================================================

def oracle_obj1(X):
    """
    Oracle for objective 1: Sphere function.

    Parameters:
    -----------
    X : np.ndarray, shape (n, 2)
        Input configurations

    Returns:
    --------
    Y : np.ndarray, shape (n, 1)
        Function values
    """
    Y = np.sum(X**2, axis=1, keepdims=True)
    Y += 0.01 * np.random.randn(X.shape[0], 1)  # Add noise
    return Y


def oracle_obj2(X):
    """
    Oracle for objective 2: Rosenbrock function.

    Parameters:
    -----------
    X : np.ndarray, shape (n, 2)
        Input configurations

    Returns:
    --------
    Y : np.ndarray, shape (n, 1)
        Function values
    """
    x1 = X[:, 0]
    x2 = X[:, 1]
    Y = ((1 - x1)**2 + 100 * (x2 - x1**2)**2).reshape(-1, 1)
    Y += 0.1 * np.random.randn(X.shape[0], 1)  # Add noise
    return Y


# ============================================================================
# STEP 2: Define Objective Functions
# ============================================================================

def objective_obj1(x, fn_model):
    """
    Objective 1: Just return model predictions.

    Parameters:
    -----------
    x : np.ndarray, shape (n, 2)
        Candidate configurations
    fn_model : function
        Surrogate model

    Returns:
    --------
    obj : np.ndarray, shape (n, 1)
        Objective values
    """
    predictions = fn_model(x)  # (1, n)
    return predictions.T  # (n, 1)


def objective_obj2(x, fn_model):
    """
    Objective 2: Just return model predictions.

    Parameters:
    -----------
    x : np.ndarray, shape (n, 2)
        Candidate configurations
    fn_model : function
        Surrogate model

    Returns:
    --------
    obj : np.ndarray, shape (n, 1)
        Objective values
    """
    predictions = fn_model(x)  # (1, n)
    return predictions.T  # (n, 1)


# ============================================================================
# STEP 3: Define Algorithm Function
# ============================================================================

def make_algo():
    """
    Create a simple random sampling acquisition algorithm.
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
        X_obj1, X_obj2 : np.ndarray
            Candidates for each objective
        """
        # Generate random candidates
        n_candidates = 200
        X_candidates = np.random.rand(n_candidates, 2)

        # Evaluate with objective functions
        obj1_vals = objective_obj1(X_candidates, fn_model_list[0])
        obj2_vals = objective_obj2(X_candidates, fn_model_list[1])

        # Select diverse points
        n_select = 10
        idx = np.linspace(0, len(X_candidates)-1, n_select, dtype=int)

        X_selected = X_candidates[idx]

        return X_selected, X_selected

    return algo


# ============================================================================
# STEP 4: Run BAX Optimization
# ============================================================================

def main():
    print("=" * 70)
    print("Simple Synthetic BAX Example - High-Level API")
    print("=" * 70)
    print()

    # That's it! Just call run_bax_optimization with 3 functions!
    opt, results = run_bax_optimization(
        # Required: The 3 core functions
        oracles=[oracle_obj1, oracle_obj2],
        objectives=[objective_obj1, objective_obj2],
        algorithm=make_algo(),

        # Optional: Configuration (all have sensible defaults)
        n_init=50,
        max_iterations=3,
        n_sampling=10,
        model_root='./models_simple_api/',
        nn_config={'n_neur': 100, 'lr': 1e-3, 'epochs': 30, 'iter_epochs': 10},
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        snapshot=True,
        verbose=True,
        seed=42,
    )

    # ========================================================================
    # Evaluate final Pareto front
    # ========================================================================
    print("\nEvaluating final Pareto front...")

    # Generate test set
    X_test = np.random.rand(200, 2)

    # Evaluate using objective functions
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
    print("Pareto front (first 5 points):")
    for i in range(min(5, len(pf))):
        print(f"  {i+1}. Obj1={pf[i,0]:.4f}, Obj2={pf[i,1]:.4f}")

    print()
    print("=" * 70)
    print("DONE!")
    print("=" * 70)
    print()
    print("Compare this file (run_simple_api.py) with run_simple.py:")
    print("  - run_simple.py: ~260 lines (manual setup)")
    print("  - run_simple_api.py: ~180 lines (automatic setup)")
    print("  - Key difference: No manual data generation, normalization, or BAXOpt config!")
    print()


if __name__ == '__main__':
    main()
