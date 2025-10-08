"""
DAMA Utilities Package

This package contains utility modules for the DAMA example:
- da_ssrl: Data handling, grid generation, prediction-to-objective conversions
- dama_utils: Aperture calculation functions (evaluate_DA, evaluate_MA)
- da_virtual_opt: VirtualDAMA problem class and optimization utilities
- da_utils: Grid generation utilities (angles, rays, etc.)
- dama_resources: Resource path management
"""

# Export commonly used functions and classes
from .da_ssrl import (
    load_XY_batch_sim,
    get_Y_batch_sim_4d,
    X_to_train_shape,
    Y_to_qar_shape,
    pred_to_obj,
    pred_to_ma,
    calc_daXY_ex,
    calc_maPM_ex,
    get_srange,
)

from .dama_utils import (
    evaluate_DA,
    evaluate_MA,
)

from .da_virtual_opt import (
    VirtualDAMA,
    run_opt,
)

from .da_utils import (
    gen_ray_angles,
    gen_ray_grid_xy,
    gen_ray_grid_ar,
    gen_spos,
    gen_momentum,
)

from .dama_resources import (
    get_resource_path,
    get_matlab_data,
    get_ring_file,
    get_setup_file,
    get_mopso_run,
    get_ssrlx_data,
    get_dama_4d_data,
    get_sextupole_setup,
    get_init_config,
    get_ma_config,
    get_momentum_grid,
)

__all__ = [
    # da_ssrl
    'load_XY_batch_sim',
    'get_Y_batch_sim_4d',
    'X_to_train_shape',
    'Y_to_qar_shape',
    'pred_to_obj',
    'pred_to_ma',
    'calc_daXY_ex',
    'calc_maPM_ex',
    'get_srange',

    # dama_utils
    'evaluate_DA',
    'evaluate_MA',

    # da_virtual_opt
    'VirtualDAMA',
    'run_opt',

    # da_utils
    'gen_ray_angles',
    'gen_ray_grid_xy',
    'gen_ray_grid_ar',
    'gen_spos',
    'gen_momentum',

    # dama_resources
    'get_resource_path',
    'get_matlab_data',
    'get_ring_file',
    'get_setup_file',
    'get_mopso_run',
    'get_ssrlx_data',
    'get_dama_4d_data',
    'get_sextupole_setup',
    'get_init_config',
    'get_ma_config',
    'get_momentum_grid',
]
