"""
DAMA Oracle Functions

These functions interface with the expensive particle tracking simulations.
Oracle functions run the actual simulations and return intermediate results
(survival turns for particles).
"""

import os
import sys
import numpy as np

# Add local directory first, then core (so local modules take precedence)
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../../core'))

# Import DAMA-specific modules (from utils)
from utils import da_ssrl as dass
from utils import evaluate_DA, evaluate_MA


def make_DA_oracle(data_root, global_store, sim_seeds, buffer_size=200):
    """
    Factory function to create a DA oracle function.

    Parameters:
    -----------
    data_root : str
        Directory to save simulation data
    global_store : dict
        Dictionary with 'count_DA' key to track oracle calls
    sim_seeds : list
        List of random seeds for stochastic simulations
    buffer_size : int
        Buffer size for saving data

    Returns:
    --------
    fn_DA_oracle : function
        Oracle function that takes X (configs) and returns Y (survival turns)
    """
    def fn_DA_oracle(X):
        """
        Run DA (Dynamic Aperture) simulations.

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, 6)
            Input configurations (4 sextupole params + x + y coordinates)

        Returns:
        --------
        Y : np.ndarray, shape (n_samples, 1)
            Survival turns for each particle
        """
        count = global_store['count_DA']
        # Randomly assign seeds to each configuration
        seeds = np.random.choice(sim_seeds, (X.shape[0], 1))
        _X = np.hstack([X, seeds])

        # Run simulation
        Y = dass.get_Y_batch_sim_4d(
            _X, evaluate_DA,
            data_root=data_root,
            prefix=f'DA_loop_{count}',
            format_data=False,
            buffer_size=buffer_size
        )

        global_store['count_DA'] += 1
        return Y.reshape(-1, 1)

    return fn_DA_oracle


def make_MA_oracle(data_root, global_store, sim_seeds, buffer_size=200):
    """
    Factory function to create an MA oracle function.

    Parameters:
    -----------
    data_root : str
        Directory to save simulation data
    global_store : dict
        Dictionary with 'count_MA' key to track oracle calls
    sim_seeds : list
        List of random seeds for stochastic simulations
    buffer_size : int
        Buffer size for saving data

    Returns:
    --------
    fn_MA_oracle : function
        Oracle function that takes X (configs) and returns Y (survival turns)
    """
    def fn_MA_oracle(X):
        """
        Run MA (Momentum Aperture) simulations.

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, 6)
            Input configurations (4 sextupole params + spos + momentum)

        Returns:
        --------
        Y : np.ndarray, shape (n_samples, 1)
            Survival turns for each particle
        """
        count = global_store['count_MA']
        # Randomly assign seeds to each configuration
        seeds = np.random.choice(sim_seeds, (X.shape[0], 1))
        _X = np.hstack([X, seeds])

        # Run simulation
        Y = dass.get_Y_batch_sim_4d(
            _X, evaluate_MA,
            data_root=data_root,
            prefix=f'MA_loop_{count}',
            format_data=False,
            buffer_size=buffer_size
        )

        global_store['count_MA'] += 1
        return Y.reshape(-1, 1)

    return fn_MA_oracle
