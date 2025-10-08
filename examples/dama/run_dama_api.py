"""
DAMA-BAX: Multi-Objective Optimization Using High-Level API

This is the SIMPLIFIED version using run_bax_optimization() API.
For full manual control, see run_dama.py.

This demonstrates Pattern B with grid expansion:
- 4D sextupole configs → expanded to 6D grids (DA: x,y; MA: spos,momentum)
- Automatic initialization, normalization, and training
- User provides: 3 functions + 2 expansion functions + optional init_sampler
"""

import os
import sys
import time
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from tqdm.auto import tqdm
from scipy.io import loadmat
from pyDOE import lhs

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../../core'))

# Import BAX framework
from bax_core import (
    run_bax_optimization, pareto_front, get_PF_region,
    convert_seconds_to_time,
)

# Import DAMA modules
from utils import da_virtual_opt as davo
from utils import get_mopso_run
from dama_oracles import make_DA_oracle, make_MA_oracle
from dama_objectives import make_DA_objective, make_MA_objective
from dama_algo import make_algo


# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# --- Run Configuration ---
RUN_ID = 3
MAX_ITERATIONS = 3200
N_SAMPLING = 50
N_INIT = 10000  # Initial LHS samples
SNAPSHOT = True

# --- Problem Parameters ---
DA_THRESH = 0.75
DA_METHOD = 1
MA_THRESH = 0.94
MA_METHOD = 2

# --- Genetic Algorithm Parameters ---
GA_POP_SIZE = 200
GA_N_GEN = 20
GA_SEL_SIZE = 100

# --- Acquisition Strategy Parameters ---
ACQ_METHOD = 2
DA_RANGE_LB = 0.4
DA_RANGE_UB = 0.75
MA_RANGE_LB = 0.85
MA_RANGE_UB = 0.95

# --- Neural Network Training Parameters ---
NN_N_NEURONS = 800
NN_DROPOUT = 0.1
NN_MODEL_TYPE = 'split'
NN_EPOCHS_INIT = 150
NN_EPOCHS_ITER = 10
NN_LR = 1e-4
NN_BATCH_SIZE = 1000
NN_WEIGHT_NEW = 10
NN_TEST_RATIO = 0.05
NN_EARLY_STOP = 10

# --- Simulation Parameters ---
SIM_BUFFER_SIZE = 200
SIM_SEEDS = list(range(1, 11))

# --- Pareto Front Region Parameters ---
PF_PERCENTILE = 80


# ============================================================================
# Custom Init Sampler for DAMA
# ============================================================================

def make_dama_init_sampler(problem, data_root, global_store, sim_seeds, buffer_size, n_init=10000):
    """
    Create custom init sampler that generates 6D data (4D configs + 2D grid coords).

    For DAMA, we need:
    - DA: 4D sextupole + x + y (angles and radii from problem)
    - MA: 4D sextupole + spos + momentum
    """
    from bax_core import gen_simulation_data

    def init_sampler():
        """
        Generate initial training data for both DA and MA.

        Returns:
        --------
        X_list : list of [X_DA, X_MA], each (n_init, 6)
        Y_list : list of [Y_DA, Y_MA], each (n_init, 1)
        """
        # Generate DA data
        print(f"Generating {n_init} DA initial samples...")
        t0 = time.time()
        n_conf = 100
        X_DA, Y_DA = gen_simulation_data(
            n_conf,
            n_init / n_conf,
            data_root,
            'pre_DA',
            problem,
            buffer_size=buffer_size,
            seed=42,
            target='DA'
        )
        t1 = time.time()
        print(f"  Generated {X_DA.shape[0]} DA samples in {convert_seconds_to_time(t1 - t0)}")

        # Generate MA data
        print(f"Generating {n_init} MA initial samples...")
        t0 = time.time()
        X_MA, Y_MA = gen_simulation_data(
            n_conf,
            n_init / n_conf,
            data_root,
            'pre_MA',
            problem,
            buffer_size=buffer_size,
            seed=42,
            target='MA'
        )
        t1 = time.time()
        print(f"  Generated {X_MA.shape[0]} MA samples in {convert_seconds_to_time(t1 - t0)}")

        return [X_DA[:, :6], X_MA[:, :6]], [Y_DA, Y_MA]

    return init_sampler


# ============================================================================
# Grid Expansion Functions (Pattern B)
# ============================================================================

def make_expand_DA(problem):
    """
    Expand 4D configs to 6D DA grid (4D sextupole + x + y).
    """
    def expand_DA(x):
        """
        x: (n, 4) sextupole configs
        returns: (n×nrays×npts, 6) with x,y grid
        """
        XX, _ = problem.gen_X_data(x, exact=True)
        # XX shape: (n, nrays, npts_per_ray, 6)
        # Flatten to (n×nrays×npts, 6)
        return XX.reshape(-1, 6)

    return expand_DA


def make_expand_MA(problem):
    """
    Expand 4D configs to 6D MA grid (4D sextupole + spos + momentum).
    """
    def expand_MA(x):
        """
        x: (n, 4) sextupole configs
        returns: (n×nspos×npts, 6) with spos,momentum grid
        """
        _, XX = problem.gen_X_data(x)
        # XX shape: (n, nspos, npts_per_spos, 6)
        # Flatten to (n×nspos×npts, 6)
        return XX.reshape(-1, 6)

    return expand_MA


# ============================================================================
# Main Function
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='DAMA-BAX: Multi-objective optimization with high-level API',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--run-id', type=int, default=RUN_ID,
                        help='Run identifier (used in data/model paths)')
    parser.add_argument('--max-iter', type=int, default=MAX_ITERATIONS,
                        help='Maximum number of BAX iterations')
    parser.add_argument('--n-sampling', type=int, default=N_SAMPLING,
                        help='Number of points sampled per iteration')
    parser.add_argument('--n-init', type=int, default=N_INIT,
                        help='Number of initial training samples')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for training')

    return parser.parse_args()


def setup_device(device_arg='auto'):
    """Setup compute device."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print(f'Using device: CUDA ({torch.cuda.get_device_name(0)})')
        else:
            device = torch.device("cpu")
            print('Using device: CPU')
    elif device_arg == 'cuda':
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print(f'Using device: CUDA ({torch.cuda.get_device_name(0)})')
        else:
            print('Warning: CUDA requested but not available, falling back to CPU')
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print('Using device: CPU')

    return device


def main():
    # Parse arguments
    args = parse_args()

    # Override config with command line args
    run_id = args.run_id
    max_iterations = args.max_iter
    n_sampling = args.n_sampling
    n_init = args.n_init

    # Setup device
    device = setup_device(args.device)

    # Setup paths for this run
    data_root = f'./data/run_{run_id}/'
    model_root = f'./models/run_{run_id}/'
    log_filename = f'bax_log_run_{run_id}.pkl'

    print("=" * 70)
    print("DAMA-BAX: Multi-Objective Optimization (High-Level API)")
    print("=" * 70)
    print(f"Run ID: {run_id}")
    print(f"Data directory: {data_root}")
    print(f"Model directory: {model_root}")
    print(f"Max iterations: {max_iterations}")
    print(f"Sampling per iteration: {n_sampling}")
    print(f"Initial samples: {n_init}")
    print("=" * 70)
    print()

    os.makedirs(data_root, exist_ok=True)
    os.makedirs(model_root, exist_ok=True)

    # Initialize global store for oracle call counting
    global_store = {
        'count_DA': 1,
        'count_MA': 1,
    }

    # ========================================================================
    # STEP 1: Define the problem
    # ========================================================================
    print('Defining problem...')
    problem = davo.VirtualDAMA(
        None, None,
        da_thresh=DA_THRESH, da_method=DA_METHOD,
        ma_thresh=MA_THRESH, ma_method=MA_METHOD,
        verbose=0, exact=True
    )
    print(f"  DA threshold: {DA_THRESH}, method: {DA_METHOD}")
    print(f"  MA threshold: {MA_THRESH}, method: {MA_METHOD}")
    print()

    # Load ground truth Pareto front
    print('Loading ground truth data...')
    matdat = loadmat(str(get_mopso_run()))
    SEXT_4D = matdat['g_dama'][:, 1:5]
    DAMA = matdat['g_dama'][:, 5:7]

    vrange4D = matdat['vrange4D']
    SEXT_4D = (SEXT_4D - vrange4D[:, 0]) / (vrange4D[:, 1] - vrange4D[:, 0])

    SEXT_unique = []
    for i in range(20):
        SEXT_unique.append(SEXT_4D[600 * i:600 * i + 60])
    SEXT_unique = np.vstack(SEXT_unique)

    DAMA_median = []
    for i in tqdm(range(20), desc="Processing GT data"):
        DAMA_i = DAMA[i * 600:(i + 1) * 600, :]
        for j in range(60):
            DAMA_ij = DAMA_i[j::60]
            DAMA_median.append(np.median(DAMA_ij, axis=0))
    DAMA_median = np.vstack(DAMA_median)

    pf_GT_median = pareto_front(DAMA_median)
    pf_region_GT, pf_region_idx_GT = get_PF_region(DAMA_median, percentile=PF_PERCENTILE, plot=False)
    ps_region_GT = SEXT_unique[pf_region_idx_GT]
    print(f"  Loaded {SEXT_unique.shape[0]} unique configurations")
    print(f"  GT Pareto front: {pf_GT_median.shape[0]} points")
    print(f"  GT PF region ({PF_PERCENTILE}th percentile): {pf_region_GT.shape[0]} points")
    print()

    # ========================================================================
    # STEP 2: Create the 3 core functions
    # ========================================================================

    # Oracles
    fn_DA_oracle = make_DA_oracle(data_root, global_store, SIM_SEEDS, SIM_BUFFER_SIZE)
    fn_MA_oracle = make_MA_oracle(data_root, global_store, SIM_SEEDS, SIM_BUFFER_SIZE)

    # Objectives
    fn_obj_DA = make_DA_objective(problem)
    fn_obj_MA = make_MA_objective(problem)

    # Set objective functions in problem
    problem.fn_obj_da = fn_obj_DA
    problem.fn_obj_ma = fn_obj_MA

    # Algorithm
    ga_params = {
        'pop_size': GA_POP_SIZE,
        'n_gen': GA_N_GEN,
        'sel_size': GA_SEL_SIZE,
    }

    acq_params = {
        'method': ACQ_METHOD,
        'DA_RANGE_LB': DA_RANGE_LB,
        'DA_RANGE_UB': DA_RANGE_UB,
        'MA_RANGE_LB': MA_RANGE_LB,
        'MA_RANGE_UB': MA_RANGE_UB,
    }

    pf_params = {
        'percentile': PF_PERCENTILE,
    }

    algo = make_algo(problem, ga_params, acq_params, pf_params, ps_region_GT, log_filename)

    # ========================================================================
    # STEP 3: Create expansion functions (Pattern B)
    # ========================================================================
    expand_DA = make_expand_DA(problem)
    expand_MA = make_expand_MA(problem)

    # ========================================================================
    # STEP 4: Create custom init sampler
    # ========================================================================
    init_sampler = make_dama_init_sampler(
        problem, data_root, global_store, SIM_SEEDS, SIM_BUFFER_SIZE, n_init=n_init
    )

    # ========================================================================
    # STEP 5: Run BAX optimization with high-level API
    # ========================================================================
    print("\nStarting BAX optimization with high-level API...")
    print()

    nn_config = {
        'n_neur': NN_N_NEURONS,
        'dropout': NN_DROPOUT,
        'model_type': NN_MODEL_TYPE,
        'lr': NN_LR,
        'batch_size': NN_BATCH_SIZE,
        'epochs': NN_EPOCHS_INIT,
        'iter_epochs': NN_EPOCHS_ITER,
        'weight_new': NN_WEIGHT_NEW,
        'early_stop_patience': NN_EARLY_STOP,
    }

    opt, results = run_bax_optimization(
        # Required: The 3 core functions
        oracles=[fn_DA_oracle, fn_MA_oracle],
        objectives=[fn_obj_DA, fn_obj_MA],
        algorithm=algo,

        # Pattern B: Grid expansion
        expansion_funcs=[expand_DA, expand_MA],

        # Custom initialization
        init_sampler=init_sampler,

        # Configuration
        max_iterations=max_iterations,
        n_sampling=n_sampling,
        model_root=model_root,
        model_names=['danet', 'manet'],
        nn_config=nn_config,
        test_ratio=NN_TEST_RATIO,
        device=device,
        snapshot=SNAPSHOT,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Models saved to: {model_root}")
    print(f"Data saved to: {data_root}")
    print(f"Log saved to: {log_filename}")
    print()


if __name__ == '__main__':
    main()
