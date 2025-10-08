"""
DAMA Resource Path Management

Provides centralized access to DAMA-specific resource files
(MATLAB data, ring configurations, etc.)
"""

import os
from pathlib import Path


# Get the directory where this module is located (utils/)
# Resources are in the parent directory (examples/dama/)
UTILS_DIR = Path(__file__).parent
DAMA_DIR = UTILS_DIR.parent

# Resource directories
RESOURCES_DIR = DAMA_DIR / 'resources'
MATLAB_DATA_DIR = RESOURCES_DIR / 'matlab_data'
RINGS_DIR = RESOURCES_DIR / 'rings'
SETUP_DIR = RESOURCES_DIR / 'setup'


def get_resource_path(relative_path):
    """
    Get absolute path to a resource file.

    Parameters:
    -----------
    relative_path : str or Path
        Path relative to resources directory

    Returns:
    --------
    Path
        Absolute path to the resource

    Example:
    --------
    >>> mat_file = get_resource_path('matlab_data/mopso_run.mat')
    """
    return RESOURCES_DIR / relative_path


def get_matlab_data(filename):
    """
    Get path to a MATLAB data file.

    Parameters:
    -----------
    filename : str
        Name of the .mat file

    Returns:
    --------
    Path
        Absolute path to the file
    """
    return MATLAB_DATA_DIR / filename


def get_ring_file(ring_id=0):
    """
    Get path to a ring configuration file.

    Parameters:
    -----------
    ring_id : int
        Ring ID (0 for base, 1-26 for variations)

    Returns:
    --------
    Path
        Absolute path to the ring file
    """
    if ring_id == 0:
        return RINGS_DIR / 'ring0.pkl'
    else:
        return RINGS_DIR / f'ring_s{ring_id}.pkl'


def get_setup_file():
    """
    Get path to the setup numpy archive.

    Returns:
    --------
    Path
        Absolute path to setup.npz
    """
    return SETUP_DIR / 'setup.npz'


# Convenience functions for common files
def get_mopso_run():
    """Ground truth Pareto front"""
    return get_matlab_data('mopso_run.mat')


def get_ssrlx_data():
    """SSRL baseline data"""
    return get_matlab_data('data_SSRLX_gen100.mat')


def get_dama_4d_data():
    """4D DAMA dataset"""
    return get_matlab_data('data_DAMA4D_gen100.mat')


def get_sextupole_setup():
    """Sextupole configuration"""
    return get_matlab_data('data_setup_H6BA_10b_6var.mat')


def get_init_config():
    """Initial configuration"""
    return get_matlab_data('init_config.mat')


def get_ma_config():
    """MA s-position configuration"""
    return get_matlab_data('ma_config.mat')


def get_momentum_grid():
    """Momentum grid (dpp0)"""
    return get_matlab_data('dpp0.mat')


def check_resources_exist():
    """
    Check if all required resource directories exist.

    Returns:
    --------
    tuple
        (all_exist: bool, missing: list)
    """
    required_dirs = [
        RESOURCES_DIR,
        MATLAB_DATA_DIR,
        RINGS_DIR,
        SETUP_DIR
    ]

    missing = []
    for dir_path in required_dirs:
        if not dir_path.exists():
            missing.append(str(dir_path))

    return len(missing) == 0, missing


def list_available_rings():
    """
    List all available ring configuration files.

    Returns:
    --------
    list
        List of available ring IDs
    """
    if not RINGS_DIR.exists():
        return []

    ring_ids = []

    # Check for base ring
    if (RINGS_DIR / 'ring0.pkl').exists():
        ring_ids.append(0)

    # Check for ring variations
    for i in range(1, 27):
        if (RINGS_DIR / f'ring_s{i}.pkl').exists():
            ring_ids.append(i)

    return ring_ids


if __name__ == '__main__':
    # Test resource paths
    print("DAMA Resource Paths")
    print("=" * 60)
    print(f"DAMA directory: {DAMA_DIR}")
    print(f"Resources directory: {RESOURCES_DIR}")
    print()

    # Check if resources exist
    all_exist, missing = check_resources_exist()
    if all_exist:
        print("✓ All resource directories found")
    else:
        print("✗ Missing directories:")
        for path in missing:
            print(f"  - {path}")
    print()

    # List available rings
    rings = list_available_rings()
    print(f"Available rings: {len(rings)}")
    if rings:
        print(f"  Ring IDs: {rings}")
    print()

    # Show some example paths
    print("Example resource paths:")
    print(f"  Ground truth PF: {get_mopso_run()}")
    print(f"  Ring 0: {get_ring_file(0)}")
    print(f"  Setup file: {get_setup_file()}")
