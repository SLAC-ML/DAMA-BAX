"""
Simple Synthetic BAX Example - New Unified API

This demonstrates the new get_bax_config() pattern where:
- BAX parameters (n_init, max_iter, etc.) are controlled by run.py CLI
- Problem-specific logic is provided via get_bax_config()
- Only 3 functions required: oracles, objectives, algorithm

Usage:
  python run.py --case examples/synthetic_simple --n-init 50 --max-iter 5

Or standalone:
  python run_simple_api_v2.py
"""

import os
import sys
import numpy as np

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../core'))


# ============================================================================
# STEP 1: Define Oracle Functions
# ============================================================================

def oracle_obj1(X):
    """
    Oracle for objective 1: Sphere function.

    IMPORTANT: Normalized to [0, 1] range to work with sigmoid output activation.
    Neural networks by default use sigmoid(x) which constrains outputs to [0, 1].

    Parameters:
    -----------
    X : np.ndarray, shape (n, 2)
        Input configurations in [0, 1]

    Returns:
    --------
    Y : np.ndarray, shape (n, 1)
        Function values normalized to ~[0, 1]
    """
    Y = np.sum(X**2, axis=1)
    Y = Y / 2.0  # Normalize: max is ~2 for X in [0,1], so divide by 2 → [0, 1]
    Y += 0.01 * np.random.randn(X.shape[0])  # Add noise
    return Y.reshape(-1, 1)  # Return (n, 1) shape


def oracle_obj2(X):
    """
    Oracle for objective 2: Rosenbrock function.

    IMPORTANT: Normalized to [0, 1] range to work with sigmoid output activation.
    Neural networks by default use sigmoid(x) which constrains outputs to [0, 1].

    Parameters:
    -----------
    X : np.ndarray, shape (n, 2)
        Input configurations in [0, 1]

    Returns:
    --------
    Y : np.ndarray, shape (n, 1)
        Function values normalized to ~[0, 1]
    """
    x1 = X[:, 0]
    x2 = X[:, 1]
    Y = (1 - x1)**2 + 100 * (x2 - x1**2)**2
    Y = Y / 100.0  # Normalize: max is ~100 for X in [0,1], so divide by 100 → [0, 1]
    Y += 0.01 * np.random.randn(X.shape[0])  # Add noise
    return Y.reshape(-1, 1)  # Return (n, 1) shape


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
        Surrogate model that returns (n, 1)

    Returns:
    --------
    obj : np.ndarray, shape (n, 1)
        Objective values
    """
    return fn_model(x)  # Already (n, 1)


def objective_obj2(x, fn_model):
    """
    Objective 2: Just return model predictions.

    Parameters:
    -----------
    x : np.ndarray, shape (n, 2)
        Candidate configurations
    fn_model : function
        Surrogate model that returns (n, 1)

    Returns:
    --------
    obj : np.ndarray, shape (n, 1)
        Objective values
    """
    return fn_model(x)  # Already (n, 1)


# ============================================================================
# STEP 3: Define Algorithm Function
# ============================================================================

def algo(fn_model_list):
    """
    Simple random sampling acquisition algorithm.

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


# ============================================================================
# STEP 4: Entry Point for Unified Runner
# ============================================================================

def get_bax_config(args):
    """
    Called by run.py to get case configuration.

    All BAX parameters (n_init, max_iter, nn_neurons, etc.) are handled by
    run.py CLI arguments. This function only returns problem-specific config.

    Parameters:
    -----------
    args : argparse.Namespace
        Arguments from run.py (e.g., args.run_id, args.max_iter)
        Note: These are for information only; BAX params are handled by run.py

    Returns:
    --------
    config : dict
        Configuration dictionary with:
        - 'oracles': list of oracle functions
        - 'objectives': list of objective functions
        - 'algorithm': acquisition function
        - 'model_root': optional model directory (can use args.run_id)
        - Other optional keys: init_sampler, model_names, bounds, etc.
    """
    # Determine model directory based on run_id if provided
    if hasattr(args, 'run_id') and args.run_id is not None:
        model_root = f'./models_simple_run_{args.run_id}/'
    else:
        model_root = './models_simple/'

    return {
        'oracles': [oracle_obj1, oracle_obj2],
        'objectives': [objective_obj1, objective_obj2],
        'algorithm': algo,
        'model_root': model_root,
        # Can optionally specify defaults for BAX params (overridden by CLI):
        # 'n_init': 50,
        # 'max_iterations': 3,
        # 'n_sampling': 10,
    }


# ============================================================================
# Optional: Standalone Execution
# ============================================================================

if __name__ == '__main__':
    """
    Can still run standalone, but recommended to use:
      python run.py --case examples/synthetic_simple
    """
    from bax_core import run_bax_optimization, pareto_front
    import torch

    print("=" * 70)
    print("Simple Synthetic BAX Example - Standalone Mode")
    print("=" * 70)
    print("TIP: Use unified runner for more control:")
    print("  python run.py --case examples/synthetic_simple --n-init 100 --max-iter 10")
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
        model_root=config['model_root'],
        n_init=50,
        max_iterations=3,
        n_sampling=10,
        nn_config={'n_neur': 100, 'lr': 1e-3, 'epochs': 30, 'iter_epochs': 10},
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        snapshot=True,
        verbose=True,
        seed=42,
    )

    # Evaluate final Pareto front
    print("\nEvaluating final Pareto front...")
    X_test = np.random.rand(200, 2)
    obj1_test = objective_obj1(X_test, opt.fn[0])
    obj2_test = objective_obj2(X_test, opt.fn[1])

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
