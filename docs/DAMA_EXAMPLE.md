# DAMA Example - Complete Guide

This document describes the DAMA (Dynamic and Momentum Aperture) example implementation, which demonstrates a full-featured application of the BAX framework for particle accelerator optimization.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [File Structure](#file-structure)
4. [Configuration](#configuration)
5. [Required Resources](#required-resources)
6. [How It Works](#how-it-works)
7. [Implementation Details](#implementation-details)
8. [Outputs](#outputs)

---

## Overview

### What is DAMA?

DAMA optimizes **two competing objectives** in particle accelerator design:

1. **Dynamic Aperture (DA)**: Spatial region where particles remain stable
2. **Momentum Aperture (MA)**: Momentum range where particles remain stable

Both are expensive to compute (requires particle tracking simulations), making this an ideal application for BAX.

### Key Features

- **Multi-objective**: Finds Pareto front of DA vs MA trade-offs
- **Expensive simulations**: PyAT-based particle tracking (~seconds per configuration)
- **Stochastic**: Uses random seeds for robust sampling
- **4D optimization**: Optimizes 4 sextupole magnet families
- **Advanced acquisition**: NSGA2 genetic algorithm + boundary sampling

---

## Quick Start

### Option 1: Use Pretrained Models (Recommended)

If you have pretrained models in `examples/dama/resources/`:

```bash
cd examples/dama
python run_dama.py --run-id 3 --max-iter 100
```

### Option 2: Start from Scratch (Advanced)

Generate initial data and train models from scratch:

```bash
cd examples/dama
python run_dama.py --no-pretrained --run-id 0
```

This will:
1. Generate 10,000 initial samples using Latin Hypercube Sampling
2. Run simulations to create training data
3. Train DA and MA neural networks (150 epochs each)
4. Save models to `./models/run_0/`
5. Save data to `./data/run_0/`
6. Begin BAX optimization

### Command-Line Options

```bash
python run_dama.py --help

Options:
  --run-id RUN_ID              Run identifier (default: 3)
  --no-pretrained              Pretrain from scratch
  --pretrain-data PATH         Pretrained data directory (default: ./data/run_0)
  --pretrain-models PATH       Pretrained models directory (default: ./models/run_0)
  --max-iter N                 Maximum BAX iterations (default: 3200)
  --n-sampling N               Points per iteration (default: 50)
  --device {auto,cuda,cpu}     Device for training (default: auto)
```

---

## File Structure

```
examples/dama/
├── run_dama.py            # Main entry point
├── dama_oracles.py        # Oracle functions (DA/MA simulations)
├── dama_objectives.py     # Objective functions (aperture calculations)
├── dama_algo.py           # Algorithm (NSGA2 + boundary sampling)
├── dama_resources.py      # Resource path management
├── dama_utils.py          # Accelerator physics utilities
├── da_virtual_opt.py      # Multi-objective problem definition
├── da_ssrl.py             # Data transformations
├── da_utils.py            # Grid generation utilities
└── resources/             # Required data files
    ├── matlab_data/       # 3 .mat files
    ├── rings/             # 27 .pkl ring files
    ├── models/run_0/      # Pretrained models (optional)
    └── data/run_0/        # Initial training data (optional)
```

---

## Configuration

All parameters are at the top of `run_dama.py` (lines 50-105):

### Essential Parameters

```python
RUN_ID = 3                    # Run identifier
USE_PRETRAINED = True         # Load existing models?
MAX_ITERATIONS = 3200         # Maximum BAX iterations
N_SAMPLING = 50               # Points sampled per iteration
```

### Pretraining (if USE_PRETRAINED=False)

```python
N_PRETRAIN_INIT = 10000       # Initial samples for pretraining
N_PRETRAIN_CONF = 100         # Number of LHS configurations
PRETRAIN_EPOCHS = 150         # Training epochs
PRETRAIN_LR = 1e-4            # Learning rate
```

### Problem Parameters

```python
DA_THRESH = 0.75              # DA survival threshold (75%)
DA_METHOD = 1                 # DA calculation method (Xiaobiao's)
MA_THRESH = 0.94              # MA survival threshold (94%)
MA_METHOD = 2                 # MA calculation method (matches GT)
```

### Neural Network

```python
NN_N_NEURONS = 800            # Hidden layer width
NN_DROPOUT = 0.1              # Dropout rate
NN_MODEL_TYPE = 'split'       # Architecture ('fc', 'split', 'sine')
NN_LR = 1e-4                  # Learning rate
NN_BATCH_SIZE = 1000          # Batch size
NN_EARLY_STOP = 10            # Early stopping patience
```

### Genetic Algorithm

```python
GA_POP_SIZE = 200             # NSGA2 population size
GA_N_GEN = 20                 # Number of generations
GA_SEL_SIZE = 100             # Selection size
```

### Acquisition Strategy

```python
ACQ_METHOD = 2                # 0=around boundary, 1=at boundary, 2=within range
DA_RANGE_LB = 0.4             # DA lower bound (method 2)
DA_RANGE_UB = 0.75            # DA upper bound (method 2)
MA_RANGE_LB = 0.85            # MA lower bound (method 2)
MA_RANGE_UB = 0.95            # MA upper bound (method 2)
```

---

## Required Resources

### Quick Summary

**Total: ~32+ files, ~300 MB to 1.5 GB**

| Category | Files | Location |
|----------|-------|----------|
| MATLAB data | 3 | `resources/matlab_data/` |
| Ring files | 27 | `resources/rings/` |
| Pretrained models | 2 | `resources/models/run_0/` (optional) |
| Initial data | Variable | `resources/data/run_0/` (optional) |

### 1. MATLAB Data Files (Required)

Place in `resources/matlab_data/`:

**mopso_run.mat**
- Ground truth Pareto front from multi-objective PSO
- Fields: `g_dama` (12000, 7), `vrange4D` (2, 4)

**data_setup_H6BA_10b_6var.mat**
- Sextupole family configuration
- Fields: `SextPara`, `g_dpp_index`

**init_config.mat**
- Initial configuration and parameter mapping
- Fields: `vrange4D` (2, 4), `S0`, `v`

### 2. Ring Files (Required)

Place in `resources/rings/`:

- `ring0.pkl` - Base ring configuration
- `ring_s1.pkl` through `ring_s26.pkl` - Ring variations (26 files)

**Total: 27 .pkl files** (~130 MB)

These are PyAT Lattice objects. Generate from MATLAB rings using:

```python
from dama_utils import convert_ring
import pickle

ring = convert_ring('THERING_seed0.mat')
with open('resources/rings/ring0.pkl', 'wb') as f:
    pickle.dump(ring, f)
```

### 3. Pretrained Models (Optional)

If USE_PRETRAINED=True, place in `resources/models/run_0/`:

- `danet_l0_f.pt` - DA neural network weights
- `manet_l0_f.pt` - MA neural network weights

**Total: 2 .pt files** (~20-40 MB)

### 4. Initial Training Data (Optional)

If USE_PRETRAINED=True, place in `resources/data/run_0/`:

**DA data:**
- `pre_DA_X.npy` - Input features (N, 7)
- `pre_DA_Y_0.npy`, `pre_DA_Y_1.npy`, ... - Output batches

**MA data:**
- `pre_MA_X.npy` - Input features (N, 7)
- `pre_MA_Y_0.npy`, `pre_MA_Y_1.npy`, ... - Output batches

**Total: Variable** (~100-1000 MB)

---

## How It Works

### The BAX Loop for DAMA

1. **Surrogate Prediction**: DA/MA neural networks predict particle survival
2. **GA Optimization**: NSGA2 runs on surrogates to find Pareto front
3. **Boundary Sampling**: Extract points near survival thresholds (informative!)
4. **Oracle Query**: Run actual PyAT simulations on selected points
5. **Model Update**: Retrain surrogates with new data (10x weight on new points)
6. **Repeat**: Continue until convergence or max iterations

### Why Boundary Sampling?

Points near survival thresholds (e.g., 75% survival) are most informative for learning the aperture boundary. The acquisition strategy (method 2) samples points with predicted survival in ranges:
- DA: 40-75% survival turns
- MA: 85-95% survival turns

### Stochastic Simulations

Each configuration is evaluated with random seeds 1-10, providing:
- Robustness against initial condition sensitivity
- Better training data diversity
- More accurate surrogate models

---

## Implementation Details

### Oracle Functions (`dama_oracles.py`)

```python
def make_DA_oracle(config):
    """Factory for DA oracle."""
    def oracle(X):
        # X: (n, 4) sextupole configurations
        # Returns: (n*100*10, 1) survival turns for all (x,y) grid points × seeds
        return evaluate_DA(X, seeds=range(1,11))
    return oracle
```

**Key points:**
- Augments X with spatial grid (100 x-y points) and seeds (10 values)
- Returns survival turns for each particle
- Expensive: ~10 seconds per configuration

### Objective Functions (`dama_objectives.py`)

```python
def make_DA_objective(config):
    """Factory for DA objective."""
    def objective(x, fn_model):
        # Generate spatial grid for each config
        grid = generate_da_grid(x)  # (n*100, 6)

        # Predict survival with surrogate
        predictions = fn_model(grid)  # (n*100, 1)

        # Calculate aperture area
        aperture = calculate_area(predictions, threshold=0.75)
        return aperture
    return objective
```

**Key points:**
- Uses cheap surrogate model, not expensive oracle
- Calculates aperture area from survival predictions
- Called thousands of times during GA optimization

### Algorithm Function (`dama_algo.py`)

```python
def make_algo(config):
    """Factory for DAMA acquisition algorithm."""
    def algo(fn_model_list):
        # 1. Run NSGA2 on surrogates
        res = run_nsga2(fn_model_list, pop_size=200, n_gen=20)

        # 2. Extract Pareto front
        pf_configs = res.X  # Pareto front configurations

        # 3. Sample points near boundaries (method 2)
        da_candidates = sample_survival_range(
            pf_configs, fn_model_list[0],
            lb=0.4, ub=0.75
        )
        ma_candidates = sample_survival_range(
            pf_configs, fn_model_list[1],
            lb=0.85, ub=0.95
        )

        return da_candidates, ma_candidates
    return algo
```

**Key points:**
- NSGA2 finds Pareto front on surrogates (cheap!)
- Boundary sampling selects informative points
- Returns different candidates for DA and MA

### Data Transformations (`da_ssrl.py`)

DAMA uses two data representations:

**Train shape**: Flattened (N, features) for neural network
- DA: (n_configs × 100_points × 10_seeds, 6) → [s1, s2, s3, s4, x, y]
- MA: (n_configs × 50_spos × 10_momenta × 10_seeds, 6) → [s1, s2, s3, s4, s_idx, momentum]

**QAR shape**: Structured (queries, angles/spos, radius/momentum)
- DA: (n_configs, 100_angles, 20_radii)
- MA: (n_configs, 50_spos, 10_momenta)

Functions handle conversion between representations automatically.

---

## Outputs

### Directory Structure

After running, you'll have:

```
examples/dama/
├── data/run_3/
│   ├── DA_loop_0_X.npy        # DA inputs (iteration 0)
│   ├── DA_loop_0_Y_0.npy      # DA outputs (batch 0)
│   ├── MA_loop_0_X.npy        # MA inputs
│   ├── MA_loop_0_Y_0.npy      # MA outputs
│   └── ...                     # More iterations
├── models/run_3/
│   ├── danet_l0_f.pt          # DA model checkpoint (iter 0)
│   ├── manet_l0_f.pt          # MA model checkpoint (iter 0)
│   ├── danet_l1_f.pt          # Next iteration
│   └── ...
└── bax_log_run_3.pkl          # Pareto front history
```

### Pareto Front Log

The `bax_log_run_X.pkl` file contains:

```python
import pickle
with open('bax_log_run_3.pkl', 'rb') as f:
    pf_history = pickle.load(f)

# Each iteration stores:
# [pf_region_opt, pf_region_idx_opt, SEXT_opt, pf_region_GT_pred,
#  X_bound_DA, Y_bound_DA, X_bound_MA, Y_bound_MA]

# Access iteration 10 Pareto front
pf_iter10 = pf_history[10][0]  # (N, 2) array of (DA, MA) values
configs_iter10 = pf_history[10][2]  # (N, 4) sextupole configurations
```

### Checkpoint Resume

The framework automatically resumes from the latest checkpoint:

```bash
# If models/run_3/ has danet_l5_f.pt and manet_l5_f.pt,
# the script will resume from iteration 6
python run_dama.py --run-id 3
```

---

## Troubleshooting

### "Missing resource files"

Check which files are missing:
```bash
cd examples/dama
ls resources/matlab_data/  # Should have 3 .mat files
ls resources/rings/        # Should have 27 .pkl files
```

### "CUDA out of memory"

Reduce batch size or sampling:
```python
NN_BATCH_SIZE = 500  # Instead of 1000
N_SAMPLING = 25      # Instead of 50
```

### "Training not converging"

- Increase training epochs: `NN_EPOCHS_ITER = 20`
- Adjust learning rate: `NN_LR = 1e-3` (faster) or `1e-5` (more stable)
- Check data normalization in oracle outputs

### "Slow simulations"

- Reduce `N_SAMPLING` to query fewer points per iteration
- Increase SLURM CPU allocation for parallel tracking
- Use GPU for neural network training: `--device cuda`

---

## Advanced Usage

### Custom Acquisition Strategy

Modify `ACQ_METHOD` to try different strategies:

```python
ACQ_METHOD = 0  # Sample around boundary (±10% of threshold)
ACQ_METHOD = 1  # Sample exactly at boundary
ACQ_METHOD = 2  # Sample within survival range (default)
```

### Different Problem Parameters

Try different aperture calculation methods:

```python
DA_METHOD = 0  # Daniel's method
DA_METHOD = 1  # Xiaobiao's method (default)

MA_METHOD = 0  # Daniel's method
MA_METHOD = 1  # Xiaobiao's method
MA_METHOD = 2  # Ground truth match (default)
```

### Hyperparameter Tuning

Experiment with neural network size:

```python
NN_N_NEURONS = 400   # Smaller, faster
NN_N_NEURONS = 1200  # Larger, more capacity
```

Or change architecture:

```python
NN_MODEL_TYPE = 'fc'     # Fully connected
NN_MODEL_TYPE = 'split'  # Split architecture (default)
NN_MODEL_TYPE = 'sine'   # Sine activation
```

---

## Comparison with Other Examples

| Aspect | DAMA | Synthetic |
|--------|------|-----------|
| **Complexity** | Advanced | Minimal |
| **Simulation** | PyAT tracking (~10s) | Analytical (<1ms) |
| **Oracle output** | Survival turns | Direct values |
| **Objective calc** | Complex grid + area | Pass-through |
| **Acquisition** | NSGA2 + boundary | Random sampling |
| **Extra params** | Seeds, grids | None |
| **Lines of code** | ~600 | ~200 |

---

## Key Takeaways

1. **DAMA demonstrates full BAX capabilities**: Seed augmentation, grid generation, complex transformations, GA-based acquisition

2. **Modular design**: Each component (oracles, objectives, algorithm) is independent and reusable

3. **Resource management**: Uses `dama_resources.py` for centralized path management

4. **Production-ready**: Handles pretraining, checkpointing, resuming, and error recovery

5. **Extensible**: Easy to modify parameters, acquisition strategy, or problem formulation

---

## Further Reading

- **BAX Framework**: See `docs/FRAMEWORK_GUIDE.md`
- **Source Code**: Explore `examples/dama/` for implementation details
- **Research Paper**: [Citation to be added]

---

## Citation

If you use the DAMA implementation in your research, please cite:

```
[Citation information to be added]
```
