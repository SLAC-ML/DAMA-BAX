"""
Configuration module for DAMA-BAX resource paths.

This module provides centralized path management for all data resources.
Users can override paths via environment variables or by modifying this file.

NOTE: Only includes resources that are actually used when running run.py
"""

import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent
RESOURCE_DIR = PROJECT_ROOT / "resources"

# Allow override via environment variable
if "DAMA_RESOURCE_DIR" in os.environ:
    RESOURCE_DIR = Path(os.environ["DAMA_RESOURCE_DIR"])

# Resource subdirectories
MATLAB_DATA_DIR = RESOURCE_DIR / "matlab_data"
RINGS_DIR = RESOURCE_DIR / "rings"
PRETRAINED_MODELS_DIR = RESOURCE_DIR / "models" / "run_0"
INITIAL_DATA_DIR = RESOURCE_DIR / "data" / "run_0"

# Specific file paths - MATLAB data (ACTUALLY USED in run.py)
MOPSO_RUN_MAT = MATLAB_DATA_DIR / "mopso_run.mat"                    # run.py:46
DATA_SETUP_H6BA_MAT = MATLAB_DATA_DIR / "data_setup_H6BA_10b_6var.mat"  # dama_utils.py:897 (module load)
INIT_CONFIG_MAT = MATLAB_DATA_DIR / "init_config.mat"                # dama_utils.py:901 (module load)

# Ring files (ACTUALLY USED in dama_utils.py module load)
RING_BASE_PKL = RINGS_DIR / "ring0.pkl"


def get_ring_path(seed_index):
    """
    Get path to ring file for a given seed index.

    Parameters
    ----------
    seed_index : int
        Seed index (0 for base ring, 1-26 for variations)

    Returns
    -------
    Path
        Path to the ring pickle file
    """
    if seed_index == 0:
        return RING_BASE_PKL
    else:
        return RINGS_DIR / f"ring_s{seed_index}.pkl"


# Pretrained models (ACTUALLY USED in run.py:397-403)
PRETRAINED_DANET = PRETRAINED_MODELS_DIR / "danet_l0_f.pt"
PRETRAINED_MANET = PRETRAINED_MODELS_DIR / "manet_l0_f.pt"


def check_resources():
    """
    Check if all required resource files exist.

    Returns
    -------
    dict
        Dictionary with file paths as keys and existence status as values
    """
    files_to_check = {
        # MATLAB data files
        "mopso_run.mat": MOPSO_RUN_MAT,
        "data_setup_H6BA_10b_6var.mat": DATA_SETUP_H6BA_MAT,
        "init_config.mat": INIT_CONFIG_MAT,
        # Ring files (base + 26 variations)
        "ring0.pkl": RING_BASE_PKL,
        # Pretrained models
        "danet_l0_f.pt": PRETRAINED_DANET,
        "manet_l0_f.pt": PRETRAINED_MANET,
    }

    # Check ring variation files
    for i in range(1, 27):
        files_to_check[f"ring_s{i}.pkl"] = get_ring_path(i)

    status = {name: path.exists() for name, path in files_to_check.items()}

    return status


def check_initial_data():
    """
    Check if initial training data exists.

    Returns
    -------
    dict
        Dictionary with data type as keys and existence status as values
    """
    # Check for at least the _X.npy files and one _Y file for each
    status = {
        "pre_DA_X.npy": (INITIAL_DATA_DIR / "pre_DA_X.npy").exists(),
        "pre_DA_Y (at least one batch)": any((INITIAL_DATA_DIR).glob("pre_DA_Y_*.npy")),
        "pre_MA_X.npy": (INITIAL_DATA_DIR / "pre_MA_X.npy").exists(),
        "pre_MA_Y (at least one batch)": any((INITIAL_DATA_DIR).glob("pre_MA_Y_*.npy")),
    }

    return status


def print_resource_status():
    """
    Print the status of all required resource files.
    """
    status = check_resources()
    data_status = check_initial_data()

    print("=" * 70)
    print("DAMA-BAX Required Resource Files Status")
    print("=" * 70)
    print(f"Resource directory: {RESOURCE_DIR}")
    print()

    # Main resource files
    print("MATLAB Data Files (3 files):")
    print("-" * 70)
    missing = []
    for name in ["mopso_run.mat", "data_setup_H6BA_10b_6var.mat", "init_config.mat"]:
        exists = status[name]
        status_str = "✓ FOUND" if exists else "✗ MISSING"
        print(f"  {status_str:12} {name}")
        if not exists:
            missing.append(name)

    print()
    print("Ring Files (27 files: ring0.pkl + ring_s1.pkl through ring_s26.pkl):")
    print("-" * 70)
    ring_missing = []
    for i in range(27):
        name = "ring0.pkl" if i == 0 else f"ring_s{i}.pkl"
        exists = status[name]
        if not exists:
            ring_missing.append(name)

    if not ring_missing:
        print(f"  ✓ FOUND    All 27 ring files present")
    else:
        print(f"  ✗ MISSING  {len(ring_missing)} ring file(s)")
        for name in ring_missing[:5]:
            print(f"             - {name}")
        if len(ring_missing) > 5:
            print(f"             ... and {len(ring_missing) - 5} more")
        missing.extend(ring_missing)

    print()
    print("Pretrained Models (2 files):")
    print("-" * 70)
    for name in ["danet_l0_f.pt", "manet_l0_f.pt"]:
        exists = status[name]
        status_str = "✓ FOUND" if exists else "✗ MISSING"
        print(f"  {status_str:12} {name}")
        if not exists:
            missing.append(name)

    print()
    print("Initial Training Data:")
    print("-" * 70)
    data_missing = []
    for name, exists in data_status.items():
        status_str = "✓ FOUND" if exists else "✗ MISSING"
        print(f"  {status_str:12} {name}")
        if not exists:
            data_missing.append(name)

    # Summary
    total_files = len(status) + len(data_status)
    total_missing = len(missing) + len(data_missing)
    total_present = total_files - total_missing

    print()
    print("=" * 70)
    print(f"Summary: {total_present}/{total_files} resources available")
    print("=" * 70)

    if missing or data_missing:
        print()
        print("⚠ Missing resources - run.py will fail without these files!")
        print()
        print("Please copy the required files to:")
        print(f"  {RESOURCE_DIR}/")
        print()
        print("See RESOURCES.md for detailed information on each file.")
        print("Run 'python setup_resources.py' to create directory structure.")

    return len(missing) == 0 and len(data_missing) == 0


if __name__ == "__main__":
    # When run as a script, print resource status
    all_present = print_resource_status()
    exit(0 if all_present else 1)
