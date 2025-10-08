"""
DAMA-BAX: Bayesian Algorithm Execution for Multi-Objective Optimization
of Dynamic and Momentum Aperture in Particle Accelerators

This is the MANUAL version showing full control over all steps.
For a simplified version using the high-level API, see run_dama_api.py.

This demonstrates how to use the generic BAX framework with problem-specific oracles,
objectives, and acquisition strategies, with full manual control over initialization,
normalization, and training.
"""

import os
import sys
import time
import argparse
from importlib import reload
import warnings
import pickle

import torch
warnings.filterwarnings('ignore')

import numpy as np
from tqdm.auto import tqdm, trange
from scipy.io import loadmat
from pyDOE import lhs

# Add local directory first, then core (so local modules take precedence)
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../../core'))

# Import BAX framework modules (from core)
import da_NN as dann; reload(dann)
from bax_core import (
    pareto_front, pareto_front_idx, get_PF_region,
    convert_seconds_to_time, load_pretrained_model, pretrain,
    get_unique, collect_ps_opt,
    calc_hv, gen_daR_mask, gen_maPM_mask, gen_mask_range,
    fn_factory, BAXOpt, get_curr_loop_num,
    gen_simulation_data,
)

# Import DAMA-specific modules (from utils)
from utils import da_ssrl as dass; reload(dass)
from utils import da_virtual_opt as davo; reload(davo)
from utils import evaluate_DA, evaluate_MA
from utils import get_mopso_run
from dama_oracles import make_DA_oracle, make_MA_oracle
from dama_objectives import make_DA_objective, make_MA_objective
from dama_algo import make_algo


# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# --- Run Configuration ---
RUN_ID = 3
USE_PRETRAINED = True

PRETRAIN_DATA_ROOT = './data/run_0'
PRETRAIN_MODEL_ROOT = './models/run_0'

# --- Pretraining Configuration ---
N_PRETRAIN_INIT = 10000
N_PRETRAIN_CONF = 100
PRETRAIN_EPOCHS = 150
PRETRAIN_LR = 1e-4
PRETRAIN_BATCH_SIZE = 1000
PRETRAIN_BUFFER_SIZE = 200
PRETRAIN_SEED = 42

# --- Problem Parameters ---
DA_THRESH = 0.75
DA_METHOD = 1
MA_THRESH = 0.94
MA_METHOD = 2

# --- BAX Optimization Parameters ---
MAX_ITERATIONS = 3200
N_SAMPLING = 50
SNAPSHOT = True

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
# END CONFIGURATION
# ============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='DAMA-BAX: Multi-objective optimization with BAX',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--run-id', type=int, default=RUN_ID,
                        help='Run identifier (used in data/model paths)')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Perform pretraining even if models exist')
    parser.add_argument('--pretrain-data', type=str, default=PRETRAIN_DATA_ROOT,
                        help='Directory with pretrained data')
    parser.add_argument('--pretrain-models', type=str, default=PRETRAIN_MODEL_ROOT,
                        help='Directory with pretrained models')
    parser.add_argument('--max-iter', type=int, default=MAX_ITERATIONS,
                        help='Maximum number of BAX iterations')
    parser.add_argument('--n-sampling', type=int, default=N_SAMPLING,
                        help='Number of points sampled per iteration')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for training')

    return parser.parse_args()


def gen_lhs_confs(n_conf=1):
    """Generate Latin Hypercube Sampling configurations."""
    sext = lhs(4, samples=n_conf)
    return sext


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


def check_pretrained_exists(data_root, model_root):
    """Check if pretrained models and data exist."""
    danet_path = os.path.join(model_root, 'danet_l0_f.pt')
    manet_path = os.path.join(model_root, 'manet_l0_f.pt')

    da_x_path = os.path.join(data_root, 'pre_DA_X.npy')
    ma_x_path = os.path.join(data_root, 'pre_MA_X.npy')

    models_exist = os.path.exists(danet_path) and os.path.exists(manet_path)
    data_exists = os.path.exists(da_x_path) and os.path.exists(ma_x_path)

    return models_exist and data_exists


def do_pretraining(problem, data_root, model_root, device):
    """Perform pretraining from scratch."""
    print("=" * 70)
    print("PRETRAINING MODE")
    print("=" * 70)
    print(f"Generating {N_PRETRAIN_INIT} initial samples...")
    print(f"Using {N_PRETRAIN_CONF} LHS configurations")
    print(f"Data will be saved to: {data_root}")
    print(f"Models will be saved to: {model_root}")
    print()

    os.makedirs(data_root, exist_ok=True)
    os.makedirs(model_root, exist_ok=True)

    # Generate DA data
    print("Generating DA training data...")
    t0 = time.time()
    X_DA, Y_DA = gen_simulation_data(
        N_PRETRAIN_CONF,
        N_PRETRAIN_INIT / N_PRETRAIN_CONF,
        data_root,
        'pre_DA',
        problem,
        buffer_size=PRETRAIN_BUFFER_SIZE,
        seed=PRETRAIN_SEED,
        target='DA'
    )
    t1 = time.time()
    print(f"  Generated {X_DA.shape[0]} DA samples in {convert_seconds_to_time(t1 - t0)}")
    print(f"  X_DA shape: {X_DA.shape}, Y_DA shape: {Y_DA.shape}")

    # Generate MA data
    print("\nGenerating MA training data...")
    t0 = time.time()
    X_MA, Y_MA = gen_simulation_data(
        N_PRETRAIN_CONF,
        N_PRETRAIN_INIT / N_PRETRAIN_CONF,
        data_root,
        'pre_MA',
        problem,
        buffer_size=PRETRAIN_BUFFER_SIZE,
        seed=PRETRAIN_SEED,
        target='MA'
    )
    t1 = time.time()
    print(f"  Generated {X_MA.shape[0]} MA samples in {convert_seconds_to_time(t1 - t0)}")
    print(f"  X_MA shape: {X_MA.shape}, Y_MA shape: {Y_MA.shape}")

    # Pretrain DA model
    print("\nPretraining DA model...")
    train_params_da = {
        'epochs': PRETRAIN_EPOCHS,
        'lr': PRETRAIN_LR,
        'batch_size': PRETRAIN_BATCH_SIZE,
        'n_neur': NN_N_NEURONS,
        'dropout': NN_DROPOUT,
        'model_type': NN_MODEL_TYPE,
        'early_stop_patience': NN_EARLY_STOP,
    }

    savefile_da = os.path.join(model_root, 'danet_l0_f.pt')
    final_savefile_da = os.path.join(model_root, 'danet_l0_final.pt')

    t0 = time.time()
    danet = pretrain(
        X_DA[:, :6], Y_DA,
        savefile=savefile_da,
        final_savefile=final_savefile_da,
        train_params=train_params_da,
        test_ratio=NN_TEST_RATIO,
        device=device,
        seed=PRETRAIN_SEED,
    )
    t1 = time.time()
    print(f"  DA model pretrained in {convert_seconds_to_time(t1 - t0)}")

    # Pretrain MA model
    print("\nPretraining MA model...")
    train_params_ma = train_params_da.copy()

    savefile_ma = os.path.join(model_root, 'manet_l0_f.pt')
    final_savefile_ma = os.path.join(model_root, 'manet_l0_final.pt')

    t0 = time.time()
    manet = pretrain(
        X_MA[:, :6], Y_MA,
        savefile=savefile_ma,
        final_savefile=final_savefile_ma,
        train_params=train_params_ma,
        test_ratio=NN_TEST_RATIO,
        device=device,
        seed=PRETRAIN_SEED,
    )
    t1 = time.time()
    print(f"  MA model pretrained in {convert_seconds_to_time(t1 - t0)}")

    print("\n" + "=" * 70)
    print("PRETRAINING COMPLETE")
    print("=" * 70)
    print(f"Models saved to: {model_root}")
    print(f"Data saved to: {data_root}")
    print()

    return X_DA, Y_DA, X_MA, Y_MA, danet, manet


def main():
    # Parse arguments
    args = parse_args()

    # Override config with command line args
    run_id = args.run_id
    use_pretrained = not args.no_pretrained and USE_PRETRAINED
    pretrain_data_root = args.pretrain_data
    pretrain_model_root = args.pretrain_models
    max_iterations = args.max_iter
    n_sampling = args.n_sampling

    # Setup device
    device = setup_device(args.device)

    # Setup paths for this run
    data_root = f'./data/run_{run_id}/'
    model_root = f'./models/run_{run_id}/'
    log_filename = f'bax_log_run_{run_id}.pkl'

    print("=" * 70)
    print("DAMA-BAX: Multi-Objective Optimization")
    print("=" * 70)
    print(f"Run ID: {run_id}")
    print(f"Data directory: {data_root}")
    print(f"Model directory: {model_root}")
    print(f"Max iterations: {max_iterations}")
    print(f"Sampling per iteration: {n_sampling}")
    print("=" * 70)
    print()

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
    for i in trange(20, desc="Processing GT data"):
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
    # STEP 2: Create oracle functions (expensive simulations)
    # ========================================================================
    fn_DA_oracle = make_DA_oracle(data_root, global_store, SIM_SEEDS, SIM_BUFFER_SIZE)
    fn_MA_oracle = make_MA_oracle(data_root, global_store, SIM_SEEDS, SIM_BUFFER_SIZE)

    # ========================================================================
    # STEP 3: Create objective functions (model predictions → objectives)
    # ========================================================================
    fn_obj_DA = make_DA_objective(problem)
    fn_obj_MA = make_MA_objective(problem)

    # Set objective functions in problem
    problem.fn_obj_da = fn_obj_DA
    problem.fn_obj_ma = fn_obj_MA

    # ========================================================================
    # STEP 4: Create algorithm function (acquisition strategy)
    # ========================================================================
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
    # STEP 5: Setup neural network models
    # ========================================================================
    if use_pretrained and check_pretrained_exists(pretrain_data_root, pretrain_model_root):
        print("Loading pretrained models...")
        print(f"  Data from: {pretrain_data_root}")
        print(f"  Models from: {pretrain_model_root}")

        # Load pretrained models
        reloadfile_da = os.path.join(pretrain_model_root, 'danet_l0_f.pt')
        X_init, _, Y_init = dass.load_XY_batch_sim(data_root=pretrain_data_root, prefix='pre_DA')
        danet, X_mu_DA, X_std_DA = load_pretrained_model(X_init[:, :6], Y_init,
                                                          reloadfile=reloadfile_da, device=device)

        reloadfile_ma = os.path.join(pretrain_model_root, 'manet_l0_f.pt')
        X_init, _, Y_init = dass.load_XY_batch_sim(data_root=pretrain_data_root, prefix='pre_MA')
        manet, X_mu_MA, X_std_MA = load_pretrained_model(X_init[:, :6], Y_init,
                                                          reloadfile=reloadfile_ma, device=device)

        # Load initial data for BAX
        X0_DA, _, Y0_DA = dass.load_XY_batch_sim(data_root=pretrain_data_root, prefix='pre_DA')
        X0_MA, _, Y0_MA = dass.load_XY_batch_sim(data_root=pretrain_data_root, prefix='pre_MA')

        print("  Pretrained models loaded successfully")
        print()
    else:
        if use_pretrained:
            print("Warning: Pretrained models not found, performing pretraining...")
            print()

        # Perform pretraining
        X_DA, Y_DA, X_MA, Y_MA, danet, manet = do_pretraining(
            problem, pretrain_data_root, pretrain_model_root, device
        )

        # Compute normalization
        X_mu_DA, X_std_DA = dann.get_norm(X_DA[:, :6])
        X_mu_MA, X_std_MA = dann.get_norm(X_MA[:, :6])

        # Set initial data
        X0_DA, Y0_DA = X_DA, Y_DA
        X0_MA, Y0_MA = X_MA, Y_MA

    # Define normalization functions
    def norm_DA(X):
        _X = X.copy()
        _X = dann.normalize(_X, X_mu_DA, X_std_DA)
        return _X

    def norm_MA(X):
        _X = X.copy()
        _X = dann.normalize(_X, X_mu_MA, X_std_MA)
        return _X

    # Define initialization functions
    def init_DA():
        return X0_DA[:, :6], Y0_DA

    def init_MA():
        return X0_MA[:, :6], Y0_MA

    # ========================================================================
    # STEP 6: Run BAX optimization
    # ========================================================================
    curr_loop = get_curr_loop_num(model_root, model_name='manet')

    print("Starting BAX optimization...")
    if curr_loop is None:
        # Start new run
        print("  Starting from scratch")
        fn_oracle = [fn_DA_oracle, fn_MA_oracle]
        norm = [norm_DA, norm_MA]
        init = [init_DA, init_MA]

        opt = BAXOpt(algo, fn_oracle, norm, init, device, snapshot=SNAPSHOT,
                     model_root=model_root, model_names=['danet', 'manet'])
        opt.n_sampling = n_sampling
        opt.model = [danet, manet]

        opt.run_acquisition(max_iterations)
    else:
        # Resume from checkpoint
        print(f"  Resuming from iteration {curr_loop}")

        # Reload models
        def reload_model(loop, suffix='f'):
            reloadfile = os.path.join(model_root, f'danet_l{loop}_{suffix}.pt')
            danet.load_state_dict(torch.load(reloadfile, map_location=device))

            reloadfile = os.path.join(model_root, f'manet_l{loop}_{suffix}.pt')
            manet.load_state_dict(torch.load(reloadfile, map_location=device))

        fn_oracle = [fn_DA_oracle, fn_MA_oracle]
        norm = [norm_DA, norm_MA]
        init = [init_DA, init_MA]

        opt = BAXOpt(algo, fn_oracle, norm, init, device, snapshot=SNAPSHOT,
                     model_root=model_root, model_names=['danet', 'manet'])
        opt.n_sampling = n_sampling
        opt.model = [danet, manet]

        n_obj = len(opt.fn_oracle)

        # Initialization
        X0 = []
        Y0 = []
        for i in range(n_obj):
            _X0, _Y0 = opt.init[i]()
            X0.append(_X0)
            Y0.append(_Y0)
        opt.update_acq_data(X0, Y0)
        opt.iter_idx += 1

        # Fill in previous data
        for i in trange(curr_loop, desc="Loading simulated data"):
            X_DA, _, Y_DA = dass.load_XY_batch_sim(data_root=data_root, prefix=f'DA_loop_{i + 1}')
            X_MA, _, Y_MA = dass.load_XY_batch_sim(data_root=data_root, prefix=f'MA_loop_{i + 1}')
            X = [X_DA[:, :6], X_MA[:, :6]]
            Y = [Y_DA, Y_MA]
            opt.update_acq_data(X, Y)
            global_store['count_DA'] += 1
            global_store['count_MA'] += 1
            opt.iter_idx += 1

        reload_model(curr_loop)
        opt.identify_subspace()
        opt.sampling()

        # Continue the run
        num_iter = max_iterations - curr_loop
        opt.run_acquisition(num_iter)

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
