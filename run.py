"""
DAMA-BAX: Bayesian Algorithm Execution for Multi-Objective Optimization
of Dynamic and Momentum Aperture in Particle Accelerators

This script supports both pretraining from scratch and resuming from existing models.
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

from matplotlib import pyplot as plt
import numpy as np
from tqdm.auto import tqdm, trange
from scipy.io import loadmat
from pyDOE import lhs

import da_NN as dann; reload(dann)
import da_ssrl as dass; reload(dass)
import da_virtual_opt as davo; reload(davo)
from dama_utils import evaluate_DA, evaluate_MA
from bax_core import (
    pareto_front, pareto_front_idx, get_PF_region,
    convert_seconds_to_time, load_pretrained_model, pretrain,
    get_unique, collect_ps_opt,
    calc_hv, gen_daR_mask, gen_maPM_mask, gen_mask_range,
    fn_factory, BAXOpt, get_curr_loop_num,
    gen_simulation_data,
)


# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# --- Run Configuration ---
RUN_ID = 3  # Identifier for this run (used in data/model paths)
USE_PRETRAINED = True  # If True, load pretrained models from PRETRAIN_DATA_ROOT/PRETRAIN_MODEL_ROOT
                        # If False, perform pretraining first

# Pretrained model paths (only used if USE_PRETRAINED=True)
PRETRAIN_DATA_ROOT = './data/run_0'   # Directory with pre_DA_* and pre_MA_* files
PRETRAIN_MODEL_ROOT = './models/run_0'  # Directory with danet_l0_f.pt and manet_l0_f.pt

# --- Pretraining Configuration (only used if USE_PRETRAINED=False) ---
N_PRETRAIN_INIT = 10000  # Number of initial samples for pretraining
N_PRETRAIN_CONF = 100    # Number of configurations to sample (LHS)
PRETRAIN_EPOCHS = 150    # Number of training epochs for pretraining
PRETRAIN_LR = 1e-4       # Learning rate for pretraining
PRETRAIN_BATCH_SIZE = 1000  # Batch size for pretraining
PRETRAIN_BUFFER_SIZE = 200  # Buffer size for saving simulation data
PRETRAIN_SEED = 42       # Random seed for reproducibility

# --- Problem Parameters ---
DA_THRESH = 0.75    # DA threshold (survival fraction defining aperture boundary)
DA_METHOD = 1       # DA calculation method (0=Daniel's, 1=Xiaobiao's)
MA_THRESH = 0.94    # MA threshold
MA_METHOD = 2       # MA calculation method (2=matches GT simulator)

# --- BAX Optimization Parameters ---
MAX_ITERATIONS = 3200   # Maximum number of BAX iterations
N_SAMPLING = 50         # Number of points sampled per iteration
SNAPSHOT = True         # Save model snapshot at each iteration

# --- Genetic Algorithm Parameters ---
GA_POP_SIZE = 200       # NSGA2 population size
GA_N_GEN = 20          # Number of generations
GA_SEL_SIZE = 100      # Selection size after optimization

# --- Acquisition Strategy Parameters ---
ACQ_METHOD = 2         # 0=around boundary, 1=at boundary, 2=within survival range
DA_RANGE_LB = 0.4      # Lower bound for DA survival range (method 2)
DA_RANGE_UB = 0.75     # Upper bound for DA survival range (method 2)
MA_RANGE_LB = 0.85     # Lower bound for MA survival range (method 2)
MA_RANGE_UB = 0.95     # Upper bound for MA survival range (method 2)

# --- Neural Network Training Parameters ---
NN_N_NEURONS = 800     # Number of neurons in hidden layers
NN_DROPOUT = 0.1       # Dropout rate
NN_MODEL_TYPE = 'split'  # Model type ('fc', 'split', 'sine')
NN_EPOCHS_INIT = 150   # Training epochs for initial model
NN_EPOCHS_ITER = 10    # Training epochs per BAX iteration
NN_LR = 1e-4          # Learning rate
NN_BATCH_SIZE = 1000  # Batch size for training
NN_WEIGHT_NEW = 10    # Weight multiplier for new data points
NN_TEST_RATIO = 0.05  # Test set ratio
NN_EARLY_STOP = 10    # Early stopping patience (None to disable)

# --- Simulation Parameters ---
SIM_BUFFER_SIZE = 200  # Buffer size for simulation data I/O
SIM_SEEDS = list(range(1, 11))  # Random seeds for stochastic simulations (1-10)

# --- Pareto Front Region Parameters ---
PF_PERCENTILE = 80     # Percentile for defining PF region

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


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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
    """
    Perform pretraining from scratch.

    Generates initial data using Latin Hypercube Sampling and trains
    DA and MA models from scratch.
    """
    print("=" * 70)
    print("PRETRAINING MODE")
    print("=" * 70)
    print(f"Generating {N_PRETRAIN_INIT} initial samples...")
    print(f"Using {N_PRETRAIN_CONF} LHS configurations")
    print(f"Data will be saved to: {data_root}")
    print(f"Models will be saved to: {model_root}")
    print()

    # Create directories
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


# ============================================================================
# MAIN SCRIPT
# ============================================================================

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

    # Define the problem
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
    # Use resource path if available, otherwise fall back to current directory
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'examples/dama'))
        from dama_resources import get_mopso_run
        matdat = loadmat(str(get_mopso_run()))
    except:
        # Fallback for backward compatibility
        matdat = loadmat('mopso_run.mat')
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

    # Define oracle functions
    def fn_DA_oracle(X):
        count = global_store['count_DA']
        seeds = np.random.choice(SIM_SEEDS, (X.shape[0], 1))
        _X = np.hstack([X, seeds])
        Y = dass.get_Y_batch_sim_4d(_X, evaluate_DA, data_root=data_root,
                                     prefix=f'DA_loop_{count}', format_data=False,
                                     buffer_size=SIM_BUFFER_SIZE)
        global_store['count_DA'] += 1
        return Y.reshape(-1, 1)

    def fn_MA_oracle(X):
        count = global_store['count_MA']
        seeds = np.random.choice(SIM_SEEDS, (X.shape[0], 1))
        _X = np.hstack([X, seeds])
        Y = dass.get_Y_batch_sim_4d(_X, evaluate_MA, data_root=data_root,
                                     prefix=f'MA_loop_{count}', format_data=False,
                                     buffer_size=SIM_BUFFER_SIZE)
        global_store['count_MA'] += 1
        return Y.reshape(-1, 1)

    # Define objective evaluation functions
    def fn_obj_DA(x, fn, strict=False):
        XX, _ = problem.gen_X_data(x, exact=True)
        X = dass.X_to_train_shape(XX)
        Y = fn(X)

        da_thresh_use = 1 if strict else problem._da_thresh
        obj_pred, _ = dass.pred_to_obj(
            Y.T, problem.rr_xiaobiao, problem.angles_xiaobiao, obj_scaled=None,
            da_thresh=da_thresh_use, method=problem._da_method)

        return obj_pred[:, None]

    def fn_obj_MA(x, fn, strict=False):
        _, XX = problem.gen_X_data(x)
        X = dass.X_to_train_shape(XX)
        Y = fn(X)

        ma_thresh_use = 1 if strict else problem._ma_thresh
        obj_pred, _ = dass.pred_to_ma(
            Y.T, problem._mm, problem._spos, obj_scaled=None,
            ma_thresh=ma_thresh_use, method=problem._ma_method)

        return obj_pred[:, None]

    # Define the algorithm
    def algo(fn_list, method=ACQ_METHOD, plot=False):
        fn_DA = fn_list[0]
        fn_MA = fn_list[1]
        problem.set_fn_da(fn_DA)
        problem.set_fn_ma(fn_MA)

        t0 = time.time()

        # Run GA optimization on models
        Y_inc_top, X_inc_top, res = davo.run_opt(problem, pop_size=GA_POP_SIZE,
                                                   n_gen=GA_N_GEN, sel_size=GA_SEL_SIZE)

        t1 = time.time()
        print(f'GA opt time cost: {convert_seconds_to_time(t1 - t0)}')

        # Concatenate all solutions in GA opt
        DAMA_opt = []
        SEXT_opt = []
        for i in range(len(res.history)):
            DAMA_opt.append(res.history[i].pop.get('F'))
            SEXT_opt.append(res.history[i].pop.get('X'))
        DAMA_opt = np.vstack(DAMA_opt)
        SEXT_opt = np.vstack(SEXT_opt)

        # Remove duplicates
        SEXT_opt_unique, DAMA_opt_unique = get_unique(SEXT_opt, DAMA_opt)
        print(f'Unique ratio: {DAMA_opt_unique.shape[0] / DAMA_opt.shape[0]:.2f}')
        DAMA_opt = DAMA_opt_unique
        SEXT_opt = SEXT_opt_unique

        # Load PF from previous loops
        try:
            with open(log_filename, 'rb') as f:
                pf_info = pickle.load(f)
        except:
            pf_info = []

        if pf_info:
            ps_opt_pre = collect_ps_opt(pf_info)
            DA_opt_pre = problem.fn_obj_da(ps_opt_pre)
            MA_opt_pre = problem.fn_obj_ma(ps_opt_pre)
            DAMA_opt_pre = np.hstack([DA_opt_pre, MA_opt_pre])

            pf_opt_pre, pf_opt_pre_idx = pareto_front_idx(DAMA_opt_pre)
            ps_opt_pre = ps_opt_pre[pf_opt_pre_idx]

            hv_curr = calc_hv(DAMA_opt)
            hv_pre = calc_hv(pf_opt_pre)

            # Combine previous with current
            SEXT_opt = np.vstack([SEXT_opt, ps_opt_pre])
            DAMA_opt = np.vstack([DAMA_opt, pf_opt_pre])
            SEXT_opt, DAMA_opt = get_unique(SEXT_opt, DAMA_opt)

        # Get predictions on GT Pareto set
        DA_GT_pred = problem.fn_obj_da(ps_region_GT)
        MA_GT_pred = problem.fn_obj_ma(ps_region_GT)

        t2 = time.time()
        print(f'GT PF region pred time cost: {convert_seconds_to_time(t2 - t1)}')

        # Get current PF region
        pf_region_opt, pf_region_idx_opt = get_PF_region(DAMA_opt, percentile=PF_PERCENTILE)
        ps_region_opt = SEXT_opt[pf_region_idx_opt, :]
        XX_pf_DA, XX_pf_MA = problem.gen_X_data(ps_region_opt, exact=True)

        # Get boundary points for DA
        X_pf_DA = dass.X_to_train_shape(XX_pf_DA)
        Y_DA = fn_DA(X_pf_DA)

        nq, nrays, npts_per_ray, _ = XX_pf_DA.shape
        turns_pred_top = dass.Y_to_qar_shape(Y_DA.T, nq, nrays, npts_per_ray)
        daX_pred, daY_pred, daR_idx_pred = dass.calc_daXY_ex(
            turns_pred_top, problem.rr_xiaobiao, problem.angles_xiaobiao,
            da_thresh=problem._da_thresh, method=problem._da_method)

        if method == 0:
            daR_mask = gen_daR_mask(daR_idx_pred)
        elif method == 1:
            daR_mask = gen_daR_mask(daR_idx_pred, n_ext=0)
        elif method == 2:
            daR_mask = gen_mask_range(turns_pred_top, lb=DA_RANGE_LB, ub=DA_RANGE_UB)

        X_bound_DA = XX_pf_DA[daR_mask]
        YY_DA = dass.Y_to_qar_shape(Y_DA, XX_pf_DA.shape[0], 19, 72)
        Y_bound_DA = YY_DA[daR_mask]

        # Get boundary points for MA
        X_pf_MA = dass.X_to_train_shape(XX_pf_MA)
        Y_MA = fn_MA(X_pf_MA)

        nq, nspos, npts_per_spos, _ = XX_pf_MA.shape
        turns_pred_top = dass.Y_to_qar_shape(Y_MA.T, nq, nspos, npts_per_spos)
        maP_pred, maM_pred, maPM_idx_pred = dass.calc_maPM_ex(
            turns_pred_top, problem._mm, problem._spos,
            ma_thresh=problem._ma_thresh, method=problem._ma_method)

        if method == 0:
            maPM_mask = gen_maPM_mask(maPM_idx_pred)
        elif method == 1:
            maPM_mask = gen_maPM_mask(maPM_idx_pred, n_ext=0)
        elif method == 2:
            maPM_mask = gen_mask_range(turns_pred_top, lb=MA_RANGE_LB, ub=MA_RANGE_UB)

        X_bound_MA = XX_pf_MA[maPM_mask]
        YY_MA = dass.Y_to_qar_shape(Y_MA, XX_pf_MA.shape[0], 21, 49)
        Y_bound_MA = YY_MA[maPM_mask]

        t3 = time.time()
        print(f'BAX candidates sampling time cost: {convert_seconds_to_time(t3 - t2)}')

        # Save intermediate info
        pf_region_GT_pred = np.hstack([DA_GT_pred, MA_GT_pred])
        _info = [pf_region_opt, pf_region_idx_opt, SEXT_opt, pf_region_GT_pred,
                 X_bound_DA, Y_bound_DA, X_bound_MA, Y_bound_MA]
        pf_info.append(_info)

        # Atomic file write
        tmp_filename = log_filename + '.tmp'
        with open(tmp_filename, 'wb') as f:
            pickle.dump(pf_info, f)
        os.replace(tmp_filename, log_filename)

        t4 = time.time()
        print(f'Total algo execution time: {convert_seconds_to_time(t4 - t0)}')

        return X_bound_DA, X_bound_MA

    # Check if we need to do pretraining
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

    # Check if resuming from checkpoint
    curr_loop = get_curr_loop_num(model_root)

    print("Starting BAX optimization...")
    if curr_loop is None:
        # Start new run
        print("  Starting from scratch")
        fn_oracle = [fn_DA_oracle, fn_MA_oracle]
        norm = [norm_DA, norm_MA]
        init = [init_DA, init_MA]

        opt = BAXOpt(algo, fn_oracle, norm, init, device, snapshot=SNAPSHOT, model_root=model_root)
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

        opt = BAXOpt(algo, fn_oracle, norm, init, device, snapshot=SNAPSHOT, model_root=model_root)
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
