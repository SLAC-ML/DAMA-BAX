"""
DAMA Algorithm Function

The algorithm function implements the acquisition strategy for selecting the next
batch of points to query. It uses genetic algorithm optimization on the surrogate
models and samples boundary points near survival thresholds.
"""

import os
import sys
import time
import pickle
import numpy as np

# Add local directory first, then core (so local modules take precedence)
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../../core'))

# Import DAMA-specific modules (from utils)
from utils import da_virtual_opt as davo
from utils import da_ssrl as dass

# Import BAX framework modules (from core)
from bax_core import (
    pareto_front, pareto_front_idx, get_PF_region,
    convert_seconds_to_time, get_unique, collect_ps_opt,
    calc_hv, gen_daR_mask, gen_maPM_mask, gen_mask_range,
)


def make_algo(problem, ga_params, acq_params, pf_params, ps_region_GT, log_filename):
    """
    Factory function to create the DAMA acquisition algorithm.

    Parameters:
    -----------
    problem : VirtualDAMA
        Problem instance with methods for generating evaluation grids
    ga_params : dict
        Genetic algorithm parameters:
        - pop_size: population size
        - n_gen: number of generations
        - sel_size: selection size
    acq_params : dict
        Acquisition strategy parameters:
        - method: 0=around boundary, 1=at boundary, 2=within range
        - DA_RANGE_LB, DA_RANGE_UB: DA survival range bounds
        - MA_RANGE_LB, MA_RANGE_UB: MA survival range bounds
    pf_params : dict
        Pareto front parameters:
        - percentile: percentile for PF region
    ps_region_GT : np.ndarray
        Ground truth Pareto set region for comparison
    log_filename : str
        Filename for logging Pareto front history

    Returns:
    --------
    algo : function
        Algorithm function that takes surrogate models and returns candidate points
    """
    def algo(fn_list, plot=False):
        """
        Run acquisition algorithm to select next batch of candidates.

        Parameters:
        -----------
        fn_list : list of functions
            List of surrogate model functions [fn_DA, fn_MA]
        plot : bool
            Whether to plot results (not implemented)

        Returns:
        --------
        X_bound_DA : np.ndarray
            Candidate points for DA objective
        X_bound_MA : np.ndarray
            Candidate points for MA objective
        """
        fn_DA = fn_list[0]
        fn_MA = fn_list[1]

        # Set surrogate models in problem
        problem.set_fn_da(fn_DA)
        problem.set_fn_ma(fn_MA)

        t0 = time.time()

        # Run GA optimization on surrogate models
        Y_inc_top, X_inc_top, res = davo.run_opt(
            problem,
            pop_size=ga_params['pop_size'],
            n_gen=ga_params['n_gen'],
            sel_size=ga_params['sel_size']
        )

        t1 = time.time()
        print(f'GA opt time cost: {convert_seconds_to_time(t1 - t0)}')

        # Concatenate all solutions from GA history
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

        # Load Pareto front from previous loops
        try:
            with open(log_filename, 'rb') as f:
                pf_info = pickle.load(f)
        except:
            pf_info = []

        if pf_info:
            # Combine with previous Pareto front
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

        # Get predictions on ground truth Pareto set
        DA_GT_pred = problem.fn_obj_da(ps_region_GT)
        MA_GT_pred = problem.fn_obj_ma(ps_region_GT)

        t2 = time.time()
        print(f'GT PF region pred time cost: {convert_seconds_to_time(t2 - t1)}')

        # Get current Pareto front region
        pf_region_opt, pf_region_idx_opt = get_PF_region(
            DAMA_opt, percentile=pf_params['percentile']
        )
        ps_region_opt = SEXT_opt[pf_region_idx_opt, :]
        XX_pf_DA, XX_pf_MA = problem.gen_X_data(ps_region_opt, exact=True)

        # === Sample boundary points for DA ===
        X_pf_DA = dass.X_to_train_shape(XX_pf_DA)
        Y_DA = fn_DA(X_pf_DA)

        nq, nrays, npts_per_ray, _ = XX_pf_DA.shape
        turns_pred_top = dass.Y_to_qar_shape(Y_DA.T, nq, nrays, npts_per_ray)
        daX_pred, daY_pred, daR_idx_pred = dass.calc_daXY_ex(
            turns_pred_top, problem.rr_xiaobiao, problem.angles_xiaobiao,
            da_thresh=problem._da_thresh, method=problem._da_method
        )

        # Generate acquisition mask based on strategy
        method = acq_params['method']
        if method == 0:
            daR_mask = gen_daR_mask(daR_idx_pred)
        elif method == 1:
            daR_mask = gen_daR_mask(daR_idx_pred, n_ext=0)
        elif method == 2:
            daR_mask = gen_mask_range(
                turns_pred_top,
                lb=acq_params['DA_RANGE_LB'],
                ub=acq_params['DA_RANGE_UB']
            )

        X_bound_DA = XX_pf_DA[daR_mask]
        YY_DA = dass.Y_to_qar_shape(Y_DA, XX_pf_DA.shape[0], 19, 72)
        Y_bound_DA = YY_DA[daR_mask]

        # === Sample boundary points for MA ===
        X_pf_MA = dass.X_to_train_shape(XX_pf_MA)
        Y_MA = fn_MA(X_pf_MA)

        nq, nspos, npts_per_spos, _ = XX_pf_MA.shape
        turns_pred_top = dass.Y_to_qar_shape(Y_MA.T, nq, nspos, npts_per_spos)
        maP_pred, maM_pred, maPM_idx_pred = dass.calc_maPM_ex(
            turns_pred_top, problem._mm, problem._spos,
            ma_thresh=problem._ma_thresh, method=problem._ma_method
        )

        # Generate acquisition mask based on strategy
        if method == 0:
            maPM_mask = gen_maPM_mask(maPM_idx_pred)
        elif method == 1:
            maPM_mask = gen_maPM_mask(maPM_idx_pred, n_ext=0)
        elif method == 2:
            maPM_mask = gen_mask_range(
                turns_pred_top,
                lb=acq_params['MA_RANGE_LB'],
                ub=acq_params['MA_RANGE_UB']
            )

        X_bound_MA = XX_pf_MA[maPM_mask]
        YY_MA = dass.Y_to_qar_shape(Y_MA, XX_pf_MA.shape[0], 21, 49)
        Y_bound_MA = YY_MA[maPM_mask]

        t3 = time.time()
        print(f'BAX candidates sampling time cost: {convert_seconds_to_time(t3 - t2)}')

        # Save intermediate info
        pf_region_GT_pred = np.hstack([DA_GT_pred, MA_GT_pred])
        _info = [
            pf_region_opt, pf_region_idx_opt, SEXT_opt, pf_region_GT_pred,
            X_bound_DA, Y_bound_DA, X_bound_MA, Y_bound_MA
        ]
        pf_info.append(_info)

        # Atomic file write
        tmp_filename = log_filename + '.tmp'
        with open(tmp_filename, 'wb') as f:
            pickle.dump(pf_info, f)
        os.replace(tmp_filename, log_filename)

        t4 = time.time()
        print(f'Total algo execution time: {convert_seconds_to_time(t4 - t0)}')

        return X_bound_DA, X_bound_MA

    return algo
