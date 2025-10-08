"""
DAMA Objective Functions

These functions convert model predictions (survival turns) into objective values
(DA and MA metrics). The objective functions use the surrogate models rather than
the expensive simulations.
"""

import os
import sys
import numpy as np

# Add local directory first, then core (so local modules take precedence)
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../../core'))

# Import DAMA-specific modules (from utils)
from utils import da_ssrl as dass


def make_DA_objective(problem):
    """
    Factory function to create a DA objective function.

    Parameters:
    -----------
    problem : VirtualDAMA
        Problem instance with DA-specific parameters and methods

    Returns:
    --------
    fn_obj_DA : function
        Objective function that takes configs and surrogate model,
        returns DA objective values
    """
    def fn_obj_DA(x, fn, strict=False):
        """
        Calculate DA objective from surrogate model predictions.

        Parameters:
        -----------
        x : np.ndarray, shape (n, 4)
            Configuration parameters (4D sextupole settings)
        fn : function
            Surrogate model function that predicts survival turns
        strict : bool
            If True, use threshold of 1.0 instead of problem threshold

        Returns:
        --------
        obj_pred : np.ndarray, shape (n, 1)
            DA objective values (negative area in x-y space)
        """
        # Generate evaluation grid for DA
        XX, _ = problem.gen_X_data(x, exact=True)
        X = dass.X_to_train_shape(XX)

        # Get predictions from surrogate model
        Y = fn(X)

        # Convert predictions to DA objective
        da_thresh_use = 1 if strict else problem._da_thresh
        obj_pred, _ = dass.pred_to_obj(
            Y.T, problem.rr_xiaobiao, problem.angles_xiaobiao,
            obj_scaled=None,
            da_thresh=da_thresh_use,
            method=problem._da_method
        )

        return obj_pred[:, None]

    return fn_obj_DA


def make_MA_objective(problem):
    """
    Factory function to create an MA objective function.

    Parameters:
    -----------
    problem : VirtualDAMA
        Problem instance with MA-specific parameters and methods

    Returns:
    --------
    fn_obj_MA : function
        Objective function that takes configs and surrogate model,
        returns MA objective values
    """
    def fn_obj_MA(x, fn, strict=False):
        """
        Calculate MA objective from surrogate model predictions.

        Parameters:
        -----------
        x : np.ndarray, shape (n, 4)
            Configuration parameters (4D sextupole settings)
        fn : function
            Surrogate model function that predicts survival turns
        strict : bool
            If True, use threshold of 1.0 instead of problem threshold

        Returns:
        --------
        obj_pred : np.ndarray, shape (n, 1)
            MA objective values (negative momentum aperture area)
        """
        # Generate evaluation grid for MA
        _, XX = problem.gen_X_data(x)
        X = dass.X_to_train_shape(XX)

        # Get predictions from surrogate model
        Y = fn(X)

        # Convert predictions to MA objective
        ma_thresh_use = 1 if strict else problem._ma_thresh
        obj_pred, _ = dass.pred_to_ma(
            Y.T, problem._mm, problem._spos,
            obj_scaled=None,
            ma_thresh=ma_thresh_use,
            method=problem._ma_method
        )

        return obj_pred[:, None]

    return fn_obj_MA
