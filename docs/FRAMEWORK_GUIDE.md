# BAX Framework User Guide

This guide explains how to use the BAX (Bayesian Algorithm Execution) framework for multi-objective optimization with expensive simulations.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Simplified API (NEW!)](#simplified-api-new)
4. [Manual API](#manual-api)
5. [Common Patterns](#common-patterns)
6. [Configuration Options](#configuration-options)
7. [Examples](#examples)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### What is BAX?

BAX is a framework for **multi-objective optimization when simulations are expensive**. It uses neural network surrogate models to efficiently explore the Pareto front without running simulations at every iteration.

**The BAX Loop:**
1. Train surrogate models to predict simulation outputs
2. Use surrogates to find promising candidates (cheap!)
3. Run actual simulations on selected candidates (expensive)
4. Update surrogates with new data
5. Repeat until converged

### When to Use BAX

✅ **Use BAX when:**
- You have competing objectives to optimize (single or multiple)
- Your simulations are expensive (minutes to hours each)
- You want to find the Pareto front efficiently
- You can provide ~100-1000 initial samples for training

❌ **Don't use BAX when:**
- Simulations are cheap (use traditional MOO algorithms)
- You can't provide initial training data

### What You Need to Provide

**Simplified API (Recommended):**
- **3 functions**: oracles, objectives, algorithm
- Automatic initialization, normalization, and configuration

**Manual API (Advanced):**
- **5 functions**: oracles, objectives, algorithm, normalization, initialization
- Full control over all aspects

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install uv
cd DAMA-BAX
uv sync
```

### Minimal Example

See `examples/synthetic/run_synthetic.py` for complete code. Here's the structure:

```python
import sys
sys.path.insert(0, 'path/to/core')
from bax_core import BAXOpt
import da_NN as dann

# Step 1: Define oracle functions (expensive simulations)
def oracle_obj1(X):
    """Run simulation, return intermediate results"""
    return your_expensive_simulation(X)

def oracle_obj2(X):
    """Another expensive simulation"""
    return your_other_simulation(X)

# Step 2: Define objective functions (predictions → objectives)
def objective_obj1(x, fn_model):
    """Convert surrogate predictions to objective value"""
    predictions = fn_model(x)
    return calculate_objective(predictions)

def objective_obj2(x, fn_model):
    """Similar for objective 2"""
    predictions = fn_model(x)
    return calculate_other_objective(predictions)

# Step 3: Define algorithm (acquisition strategy)
def make_algo():
    def algo(fn_model_list):
        """Select next candidates to evaluate"""
        # Use surrogates to find promising regions
        candidates = your_optimization_method(fn_model_list)
        return candidates_obj1, candidates_obj2
    return algo

# Step 4: Generate initial data
X_init = sample_initial_points(n_samples=1000, n_dims=4)
Y1_init = oracle_obj1(X_init)
Y2_init = oracle_obj2(X_init)

# Step 5: Setup normalization
X_mu, X_std = dann.get_norm(X_init)
norm = lambda X: dann.normalize(X.copy(), X_mu, X_std)

# Step 6: Setup initialization functions
init1 = lambda: (X_init, Y1_init)
init2 = lambda: (X_init, Y2_init)

# Step 7: Create and run optimizer
algo = make_algo()
opt = BAXOpt(
    algo=algo,
    fn_oracle=[oracle_obj1, oracle_obj2],
    norm=[norm, norm],
    init=[init1, init2],
    device='cuda'  # or 'cpu'
)

opt.run_acquisition(max_iterations=100)
```

---

## Simplified API (NEW!)

The easiest way to use BAX is with `run_bax_optimization()` - just provide 3 functions and it handles the rest!

### Basic Usage

```python
from bax_core import run_bax_optimization

# Step 1: Define oracle functions
def oracle_obj1(X):
    return your_expensive_simulation_1(X)

def oracle_obj2(X):
    return your_expensive_simulation_2(X)

# Step 2: Define objective functions
def objective_obj1(x, fn_model):
    predictions = fn_model(x)
    return calculate_objective(predictions.T)

def objective_obj2(x, fn_model):
    predictions = fn_model(x)
    return calculate_other_objective(predictions.T)

# Step 3: Define algorithm function
def make_algo():
    def algo(fn_model_list):
        candidates = your_optimization(fn_model_list)
        return candidates_obj1, candidates_obj2
    return algo

# Run! Automatic initialization, normalization, configuration
opt, results = run_bax_optimization(
    oracles=[oracle_obj1, oracle_obj2],
    objectives=[objective_obj1, objective_obj2],
    algorithm=make_algo(),
    n_init=100,           # Automatic LHS sampling
    max_iterations=100
)
```

### What Gets Automated

- **Initial data generation**: Latin Hypercube Sampling in [0,1] (or custom bounds)
- **Normalization**: Automatic mean/std normalization per objective
- **Init functions**: Auto-created from initial data
- **NN configuration**: Sensible defaults (800 neurons, lr=1e-4, etc.)
- **Model naming**: `net0`, `net1`, ... (or custom names)

### Advanced Options

```python
opt, results = run_bax_optimization(
    oracles=[oracle_obj1, oracle_obj2],
    objectives=[objective_obj1, objective_obj2],
    algorithm=algo,

    # Custom bounds
    bounds=[(0, 10), (-5, 5)],

    # Pattern B: Grid expansion
    expansion_funcs=[expand_obj1, expand_obj2],

    # Custom NN config
    nn_config={'n_neur': 1000, 'lr': 1e-3, 'epochs': 200},

    # Custom initialization
    init_sampler=my_custom_sampler,

    # Other
    model_names=['model1', 'model2'],
    device='cuda',
    snapshot=True
)
```

**See:** `examples/synthetic_simple/run_simple_api.py` for complete example

---

## Manual API

For advanced users who need full control over initialization and configuration.

### The 5 Required Functions

### 1-2. Oracle Functions (Expensive Simulations)

```python
def oracle_obj1(X):
    """
    X: (n_samples, n_dims) - input configurations
    Returns: (n_samples,) or (n_samples, n_outputs) - intermediate results
    """
    return your_expensive_simulation(X)

def oracle_obj2(X):
    return your_other_simulation(X)
```

**Key points:**
- Return **intermediate results** that NN will learn to predict
- NOT the final objective - that's what objective functions calculate
- Can augment X with extra parameters (random seeds, etc.)
- Called sparingly during optimization (expensive!)

### 3-4. Objective Functions (Model → Objective)

```python
def objective_obj1(x, fn_model):
    """
    x: (n, n_dims) - candidate configs
    fn_model: surrogate model function
    Returns: (n, 1) - objective values
    """
    predictions = fn_model(x)
    return calculate_your_objective(predictions)

def objective_obj2(x, fn_model):
    predictions = fn_model(x)
    return calculate_other_objective(predictions)
```

**Key points:**
- Use cheap **surrogate model**, not expensive oracle
- Called many times during optimization
- Can apply domain-specific transformations

### 5. Algorithm Function (Acquisition Strategy)

```python
def make_algo():
    def algo(fn_model_list):
        """
        fn_model_list: [fn_model1, fn_model2]
        Returns: (X_obj1, X_obj2) - candidates for each objective
        """
        # Use surrogates to find promising candidates
        candidates = optimize_with_models(fn_model_list)
        return candidates_obj1, candidates_obj2
    return algo
```

**Key points:**
- Implements your acquisition strategy
- Should leverage surrogates to evaluate many candidates
- Can use any method: GA, Bayesian optimization, random sampling, etc.

---

## Understanding Input Expansion (X → X0/X1)

### The Key Pattern

**IMPORTANT**: In most real applications, oracles and objectives work with **different input dimensions**:

- **Base configurations (X)**: What you're optimizing (e.g., n, 4)
- **Expanded inputs (X0, X1)**: What oracles actually evaluate (e.g., n×100, 6)

**Example flow:**
```
Base config X (n, 4)
    ↓ (expand for obj1)
X0 (n×100, 6) → Oracle1 → Y0 (n×100, 1) → NN learns X0→Y0
    ↓ (objective uses NN)
X (n, 4) → expand → X0 (n×100, 6) → NN predicts → Y0_pred → aggregate → obj1 (n, 1)
```

### Two Patterns

#### Pattern A: No Expansion (Simple)

Oracle and objective use the same input:

```python
def oracle_obj1(X):  # X: (n, 2)
    return sphere_function(X)  # (n, 1)

def objective_obj1(x, fn_model):  # x: (n, 2)
    return fn_model(x)  # Direct prediction, no expansion
```

**Use when:** Simulation evaluates configurations directly.

**Example:** `examples/synthetic_simple/run_simple.py`

#### Pattern B: With Expansion (Like DAMA)

Oracle receives expanded input, objective handles expansion:

```python
# Expansion function
def expand_for_obj1(x):  # x: (n, 2) base
    """Expand to evaluation grid."""
    n = x.shape[0]
    grid = np.linspace(0, 1, 10)  # 10 grid points
    x_repeated = np.repeat(x, 10, axis=0)  # (n×10, 2)
    grid_tiled = np.tile(grid, n).reshape(-1, 1)  # (n×10, 1)
    return np.hstack([x_repeated, grid_tiled])  # (n×10, 3)

# Oracle expects EXPANDED input
def oracle_obj1(X0):  # X0: (n×10, 3) - already expanded!
    """Evaluate at each grid point."""
    x1, x2, grid_val = X0[:, 0], X0[:, 1], X0[:, 2]
    return sphere_function(x1, x2, grid_val)  # (n×10, 1)

# Objective expands, predicts, aggregates
def objective_obj1(x, fn_model):  # x: (n, 2) base
    """Complete flow: expand → predict → aggregate."""
    # 1. Expand
    X0 = expand_for_obj1(x)  # (n×10, 3)

    # 2. Predict with surrogate
    Y0_pred = fn_model(X0)  # (n×10, 1)

    # 3. Aggregate to objective
    Y0_reshaped = Y0_pred.T.reshape(n, 10)
    obj = Y0_reshaped.mean(axis=1, keepdims=True)  # (n, 1)
    return obj
```

**Use when:** Simulation evaluates on grids/ensembles (particles, angles, radii, etc.).

**Example:** `examples/synthetic/run_synthetic.py`, `examples/dama/`

### Why Different Expansions for Each Objective?

Each objective may need different evaluation strategies:

```python
# Objective 1: Radial grid (10 points)
X0 = expand_radial(X)  # (n×10, 3): [x, y, radius]
Y0 = oracle_obj1(X0)   # Evaluate at different radii
NN0 learns: X0 → Y0

# Objective 2: Angular grid (8 points)
X1 = expand_angular(X)  # (n×8, 3): [x, y, angle]
Y1 = oracle_obj2(X1)    # Evaluate at different angles
NN1 learns: X1 → Y1
```

This is exactly what DAMA does:
- **DA**: Expands to (x, y) spatial grid
- **MA**: Expands to (s_position, momentum) grid

### Critical Implementation Details

**1. Init data must be expanded:**
```python
X_init_base = np.random.rand(100, 2)  # Base configs

# Expand for each objective
X0_init = expand_for_obj1(X_init_base)  # (1000, 3)
X1_init = expand_for_obj2(X_init_base)  # (800, 3)

# Oracles receive expanded
Y0_init = oracle_obj1(X0_init)
Y1_init = oracle_obj2(X1_init)

# Init functions return expanded data
def init_obj1():
    return X0_init, Y0_init.flatten()
```

**2. Normalization is on expanded space:**
```python
X0_mu, X0_std = dann.get_norm(X0_init)  # Based on expanded dims
norm0 = lambda X: dann.normalize(X.copy(), X0_mu, X0_std)
```

**3. Algorithm returns expanded inputs:**
```python
def algo(fn_model_list):
    # Generate base candidates
    X_candidates_base = np.random.rand(100, 2)

    # Expand before returning
    X0_selected = expand_for_obj1(X_candidates_base)
    X1_selected = expand_for_obj2(X_candidates_base)

    return X0_selected, X1_selected  # Return EXPANDED!
```

**4. Set n_feat to expanded dimensionality:**
```python
opt = BAXOpt(...)
opt.n_feat = 3  # Expanded dims, not base dims!
```

---

## Detailed Function Signatures

### Oracle Function

```python
def oracle(X: np.ndarray) -> np.ndarray:
    """
    Run expensive simulation.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_dims)
        Input configurations (typically normalized to [0, 1])

    Returns
    -------
    Y : np.ndarray, shape (n_samples,) or (n_samples, n_outputs)
        Simulation outputs (intermediate results for NN to learn)

    Notes
    -----
    - Can augment X with extra parameters (seeds, noise levels, etc.)
    - Called sparingly during optimization (expensive!)
    - Returns intermediate results, not final objectives
    - Output will be learned by neural network surrogates
    """
```

### Objective Function

```python
def objective(x: np.ndarray, fn_model: callable) -> np.ndarray:
    """
    Compute objective from surrogate model predictions.

    Parameters
    ----------
    x : np.ndarray, shape (n, n_dims)
        Candidate configurations
    fn_model : callable
        Surrogate model function
        fn_model(X) returns predictions of oracle outputs

    Returns
    -------
    obj : np.ndarray, shape (n, 1)
        Objective values to minimize

    Notes
    -----
    - Called many times (uses cheap surrogate model)
    - Can apply domain-specific transformations
    - This is where you calculate your actual objective from predictions
    """
```

### Algorithm Function

```python
def algo(fn_model_list: list) -> tuple:
    """
    Select next candidates to evaluate.

    Parameters
    ----------
    fn_model_list : list of callable
        Surrogate models for each objective [fn_model1, fn_model2]
        Each fn_model(X) returns predictions like oracle output

    Returns
    -------
    X_obj1 : np.ndarray, shape (n1, n_dims)
        Candidates for objective 1
    X_obj2 : np.ndarray, shape (n2, n_dims)
        Candidates for objective 2

    Notes
    -----
    - Implements acquisition strategy
    - Should use surrogates to evaluate many candidates cheaply
    - Can return different numbers of candidates per objective
    - Common strategies: GA, BO, uncertainty sampling, boundary sampling
    """
```

---

## Common Patterns

### Pattern 1: Oracle with Random Seeds

If your simulation is stochastic, augment inputs with random seeds:

```python
def oracle(X):
    # Augment X with random seeds
    seeds = np.random.choice(10, (X.shape[0], 1))
    X_augmented = np.hstack([X, seeds])

    # Run simulation with augmented input
    Y = expensive_simulation(X_augmented)
    return Y
```

### Pattern 2: Different Evaluation Grids

If objectives need different evaluation points:

```python
def oracle_obj1(X):
    # Generate spatial grid for DA
    X_grid = generate_spatial_grid(X)  # (n*100, dims+2) for x-y coords
    return simulate(X_grid)

def oracle_obj2(X):
    # Generate momentum grid for MA
    X_grid = generate_momentum_grid(X)  # (n*50, dims+1) for momenta
    return simulate(X_grid)

# Objective functions handle different grid structures
def make_obj1(x, fn_model):
    def objective(x, fn_model):
        grid = generate_spatial_grid(x)
        predictions = fn_model(grid)
        # Reshape predictions back to (n, 100) and calculate area
        return calculate_aperture_area(predictions)
    return objective
```

### Pattern 3: Complex Objective Calculation

Use factory pattern to capture problem-specific configuration:

```python
def make_objective(config):
    """Factory that captures problem-specific config."""

    def objective(x, fn_model):
        # Generate evaluation grid
        grid = config.generate_grid(x)

        # Get predictions from surrogate
        predictions = fn_model(grid)

        # Apply thresholds and transformations
        transformed = config.apply_threshold(predictions)
        obj = config.calculate_metric(transformed)

        return obj

    return objective
```

### Pattern 4: GA-based Acquisition

Use genetic algorithm to find Pareto front on surrogates:

```python
def make_algo(ga_config):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize

    def algo(fn_model_list):
        fn_model1, fn_model2 = fn_model_list

        # Define problem using surrogates
        class SurrogateProblem:
            def _evaluate(self, X, out):
                # Evaluate using surrogates
                obj1 = objective_obj1(X, fn_model1)
                obj2 = objective_obj2(X, fn_model2)
                out["F"] = np.column_stack([obj1, obj2])

        # Run GA
        algorithm = NSGA2(pop_size=200)
        res = minimize(SurrogateProblem(), algorithm, n_gen=20)

        # Extract candidates near boundaries/thresholds
        candidates_obj1 = extract_boundary_points(res, objective=1)
        candidates_obj2 = extract_boundary_points(res, objective=2)

        return candidates_obj1, candidates_obj2

    return algo
```

### Pattern 5: Uncertainty Sampling

Sample regions where model is uncertain:

```python
def make_algo():
    def algo(fn_model_list):
        # Generate many random candidates
        candidates = sample_uniform(n=10000, n_dims=4)

        # Evaluate with surrogates
        pred1 = fn_model_list[0](candidates)
        pred2 = fn_model_list[1](candidates)

        # Estimate uncertainty (e.g., via ensemble or dropout)
        uncertainty = estimate_uncertainty(pred1, pred2)

        # Select high-uncertainty points
        idx = np.argsort(uncertainty)[-100:]
        return candidates[idx], candidates[idx]

    return algo
```

---

## Configuration Options

### BAXOpt Parameters

```python
opt = BAXOpt(...)

# Sampling
opt.n_sampling = 50              # Points sampled per iteration

# Neural Network Architecture
opt.n_feat = 4                   # Input dimensionality
opt.n_neur = 800                 # Hidden layer width
opt.dropout = 0.1                # Dropout rate
opt.model_type = 'split'         # 'fc', 'split', or 'sine'

# Training
opt.epochs = 150                 # Initial training epochs
opt.iter_epochs = 10             # Epochs per BAX iteration
opt.lr = 1e-4                    # Learning rate
opt.batch_size = 1000            # Batch size
opt.weight_new_pts = 10          # Weight multiplier for new data
opt.test_ratio = 0.05            # Validation split
opt.early_stop = 10              # Early stopping patience

# Checkpointing
opt.snapshot = True              # Save models each iteration
opt.model_root = './models/'     # Model save directory
opt.data_root = './data/'        # Data save directory
```

---

## Examples

### 1. Synthetic Problem (`examples/synthetic/`)

**Simple analytical functions** - Great for learning the API.

```bash
cd examples/synthetic
python run_synthetic.py
```

**What it demonstrates:**
- Minimal API usage (~200 lines)
- Random sampling acquisition
- Simple transformations

### 2. DAMA Problem (`examples/dama/`)

**Particle accelerator optimization** - Shows advanced features.

```bash
cd examples/dama
python run_dama.py --run-id 3 --max-iter 100
```

**What it demonstrates:**
- Seed augmentation in oracles
- Complex grid generation (spatial + momentum)
- GA + boundary sampling acquisition
- Data transformations (train shape ↔ QAR shape)
- Resource management
- Pretraining and checkpointing

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **"Module not found: bax_core"** | Add `sys.path.insert(0, 'path/to/core')` before imports |
| **Out of memory (GPU/CPU)** | Reduce `opt.batch_size` (e.g., 500) or `opt.n_sampling` (e.g., 25) |
| **Training too slow** | Reduce `opt.n_neur` (e.g., 400) or use GPU with `device='cuda'` |
| **Bad Pareto front** | Increase `opt.n_sampling` or `max_iterations`, check oracle/objective logic |
| **Overfitting to initial data** | Increase `opt.dropout`, `opt.test_ratio`, or generate more initial samples |
| **NaN/Inf in training** | Check data normalization, reduce `opt.lr`, or clip oracle outputs |
| **Model not improving** | Increase `opt.weight_new_pts` to emphasize new data more |

---

## Recommended Project Structure

For custom problems, we recommend:

```
your_problem/
├── __init__.py
├── oracles.py          # Oracle functions
├── objectives.py       # Objective functions
├── algo.py             # Algorithm function
├── utils.py            # Problem-specific utilities
├── resources/          # Data files (if needed)
└── run.py              # Main entry point
```

See `examples/dama/` for a full example following this structure.

---

## Comparison with DAMA Example

| Aspect | DAMA Example | Your Problem |
|--------|--------------|--------------|
| **Base configs** | (n, 4) sextupole settings | Your config space |
| **Expansion DA** | (n×100, 6): [s1...s4, x, y] | Your grid/ensemble |
| **Expansion MA** | (n×50, 6): [s1...s4, s_pos, momentum] | Your grid/ensemble |
| **Oracles** | PyAT particle tracking | Your simulations |
| **Oracle output** | Survival turns (per particle) | Your measurements |
| **Objective calc** | Aperture area from survival | Your aggregation |
| **Acquisition** | NSGA2 + boundary sampling | Your strategy |
| **Extra params** | Random seeds (1-10) | Whatever you need |

**The framework handles:**
- ✅ Surrogate model training (automatic)
- ✅ Data normalization (automatic)
- ✅ Checkpointing/resuming (automatic)
- ✅ Iterative optimization loop (automatic)

**You provide:**
- ✅ Expansion functions (if needed)
- ✅ Your expensive simulations (oracles)
- ✅ How to calculate objectives (objective functions)
- ✅ How to pick next points (algorithm)

---

## Key Concepts

- **Oracle**: Expensive simulation that returns intermediate results
- **Objective**: Cheap transformation that converts predictions → objectives
- **Surrogate**: Neural network that learns oracle behavior
- **Acquisition**: Strategy to select next points to evaluate
- **BAX Loop**: Train surrogates → Acquisition → Query oracles → Repeat

**One-liner:** *BAX trains surrogates to predict expensive simulations, uses them to find promising configurations, queries the real simulation on those points, and repeats until convergence.*

---

## Tips for Success

1. **Start simple**: Use `examples/synthetic/` as a template
2. **Oracle returns intermediate results**: Not final objectives - that's what objective functions do
3. **Leverage surrogates**: Algorithm should evaluate 1000s of candidates using cheap surrogates
4. **Factory pattern**: Use `make_algo()` style to capture configuration
5. **Test incrementally**: Test each function separately before running full BAX
6. **Normalize data**: Ensure oracle outputs are normalized (divide by max value)
7. **Initial data quality**: More diverse initial samples = better surrogates
8. **Monitor training**: Check validation loss to detect overfitting

---

## Further Reading

- **Examples**: `examples/README.md` and `examples/dama/`
- **DAMA Details**: `docs/DAMA_EXAMPLE.md`
- **Contributing**: `docs/CONTRIBUTING.md`
- **Source Code**: `core/bax_core.py` (BAXOpt implementation)

---

## Checklist for New Problems

- [ ] Implement 2 oracle functions
- [ ] Implement 2 objective functions
- [ ] Implement algorithm function
- [ ] Generate initial data (100-1000 samples)
- [ ] Setup normalization
- [ ] Create BAXOpt instance
- [ ] Configure parameters
- [ ] Test each component separately
- [ ] Run and monitor convergence
- [ ] Analyze Pareto front results

---

## Questions?

See `examples/README.md` for more detailed API documentation, or check the DAMA example in `examples/dama/` for a complete working implementation.
