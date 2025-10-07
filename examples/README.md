# BAX Framework Examples

This directory contains examples demonstrating how to use the BAX (Bayesian Algorithm Execution) framework for multi-objective optimization.

## Overview

The BAX framework requires you to provide **5 simple functions** for a 2-objective optimization problem:

1. **Oracle function for objective 1** - Runs expensive simulation
2. **Oracle function for objective 2** - Runs expensive simulation
3. **Objective function for objective 1** - Converts model predictions → objective value
4. **Objective function for objective 2** - Converts model predictions → objective value
5. **Algorithm function** - Selects next candidates to evaluate

## Quick Start

### 1. Define Oracle Functions

Oracle functions run your expensive simulations and return intermediate results that will be predicted by neural network surrogate models.

```python
def oracle_obj1(X):
    """
    Run expensive simulation for objective 1.

    Parameters:
    -----------
    X : np.ndarray, shape (n_samples, n_dims)
        Input configurations

    Returns:
    --------
    Y : np.ndarray, shape (n_samples,) or (n_samples, n_outputs)
        Intermediate results (e.g., survival turns, physical measurements)
    """
    # Run your expensive simulation here
    Y = your_expensive_simulation(X)
    return Y

def oracle_obj2(X):
    """Similar for objective 2"""
    Y = your_other_simulation(X)
    return Y
```

**Key points:**
- Oracle functions can augment X with additional parameters (e.g., random seeds)
- Return intermediate results that surrogates will learn to predict
- These are expensive - called sparingly during optimization

### 2. Define Objective Functions

Objective functions convert surrogate model predictions into actual objective values you want to optimize.

```python
def make_obj1(x, fn_model):
    """
    Convert model predictions to objective 1 values.

    Parameters:
    -----------
    x : np.ndarray, shape (n, n_dims)
        Candidate configurations
    fn_model : function
        Surrogate model that predicts intermediate results

    Returns:
    --------
    obj : np.ndarray, shape (n, 1)
        Objective values
    """
    # Get predictions from surrogate model
    predictions = fn_model(x)

    # Transform predictions to objective
    # (could be simple or complex transformation)
    objectives = calculate_objective_from_predictions(predictions)

    return objectives

def make_obj2(x, fn_model):
    """Similar for objective 2"""
    predictions = fn_model(x)
    objectives = calculate_other_objective(predictions)
    return objectives
```

**Key points:**
- These use cheap surrogate models, not expensive oracles
- Can apply domain-specific transformations
- Called many times during optimization

### 3. Define Algorithm Function

The algorithm function implements your acquisition strategy - it decides which points to query next.

```python
def make_algo():
    """Create acquisition algorithm."""

    def algo(fn_model_list):
        """
        Select next batch of candidates.

        Parameters:
        -----------
        fn_model_list : list of functions
            Surrogate models [fn_model1, fn_model2]

        Returns:
        --------
        X_candidates_obj1 : np.ndarray
            Candidates for objective 1
        X_candidates_obj2 : np.ndarray
            Candidates for objective 2
        """
        fn_model1, fn_model2 = fn_model_list

        # Your acquisition strategy here
        # Could use: GA, random sampling, uncertainty sampling, etc.
        candidates = your_optimization_method(fn_model1, fn_model2)

        # Return candidates for each objective
        X_obj1 = extract_candidates_for_obj1(candidates)
        X_obj2 = extract_candidates_for_obj2(candidates)

        return X_obj1, X_obj2

    return algo
```

**Key points:**
- This is where you implement your optimization strategy
- Can use any algorithm: genetic algorithms, Bayesian optimization, random search, etc.
- Should leverage surrogate models to find promising candidates

### 4. Run BAX Optimization

```python
from bax_core import BAXOpt
import da_NN as dann

# Generate initial data
X_init = generate_initial_samples(n_samples)
Y_init_obj1 = oracle_obj1(X_init)
Y_init_obj2 = oracle_obj2(X_init)

# Setup normalization
X_mu, X_std = dann.get_norm(X_init)
def norm(X):
    return dann.normalize(X.copy(), X_mu, X_std)

# Setup initialization
def init_obj1():
    return X_init, Y_init_obj1

def init_obj2():
    return X_init, Y_init_obj2

# Create optimizer
opt = BAXOpt(
    algo=algo,
    fn_oracle=[oracle_obj1, oracle_obj2],
    norm=[norm, norm],
    init=[init_obj1, init_obj2],
    device=device
)

# Run optimization
opt.run_acquisition(max_iterations=100)
```

## Examples

### 1. Synthetic Example (`synthetic/`)

A minimal example with simple analytical functions. Great for understanding the API.

**Run:**
```bash
cd examples/synthetic
python run_synthetic.py
```

**What it does:**
- Optimizes two simple synthetic functions
- Uses random sampling for acquisition
- Demonstrates the minimal API

### 2. DAMA Example (`dama/`)

Full-featured particle accelerator optimization example. Shows advanced usage with complex simulations.

**Run:**
```bash
cd examples/dama
python run_dama.py --run-id 3 --max-iter 100
```

**What it does:**
- Optimizes Dynamic Aperture (DA) and Momentum Aperture (MA)
- Uses genetic algorithm + boundary sampling
- Handles complex data transformations
- Demonstrates seed augmentation, grid generation, etc.

## API Reference

### Oracle Functions

```python
def oracle(X: np.ndarray) -> np.ndarray:
    """
    Run expensive simulation.

    Args:
        X: Input configurations, shape (n_samples, n_dims)

    Returns:
        Y: Simulation outputs, shape (n_samples,) or (n_samples, n_outputs)
    """
```

### Objective Functions

```python
def objective(x: np.ndarray, fn_model: callable) -> np.ndarray:
    """
    Compute objective from model predictions.

    Args:
        x: Candidate configurations, shape (n, n_dims)
        fn_model: Surrogate model function

    Returns:
        obj: Objective values, shape (n, 1)
    """
```

### Algorithm Function

```python
def algo(fn_model_list: list) -> tuple:
    """
    Select next candidates to evaluate.

    Args:
        fn_model_list: List of surrogate model functions

    Returns:
        tuple: (X_candidates_obj1, X_candidates_obj2)
            Each is np.ndarray of shape (n_candidates, n_dims)
    """
```

## Tips for Custom Problems

### 1. Handling Extra Parameters

If your oracle needs extra parameters (like random seeds), handle it inside the oracle function:

```python
def oracle(X):
    # Augment X with seeds
    seeds = np.random.choice(10, (X.shape[0], 1))
    X_augmented = np.hstack([X, seeds])

    # Run simulation with augmented input
    Y = simulation(X_augmented)
    return Y
```

### 2. Complex Data Transformations

If your objective calculation is complex (like in DAMA), encapsulate the logic:

```python
def make_objective(problem_config):
    """Factory that captures problem-specific config."""

    def objective(x, fn_model):
        # Generate evaluation grid
        grid = problem_config.generate_grid(x)

        # Get predictions
        predictions = fn_model(grid)

        # Transform to objective
        obj = problem_config.calculate_metric(predictions)

        return obj

    return objective
```

### 3. Different Evaluation Grids

If objectives need different evaluation points:

```python
# In oracle functions
def oracle_obj1(X):
    # Generate grid specific to obj1
    X_grid1 = generate_da_grid(X)
    return simulate(X_grid1)

def oracle_obj2(X):
    # Generate different grid for obj2
    X_grid2 = generate_ma_grid(X)
    return simulate(X_grid2)

# In objective functions
def make_obj1(problem):
    def objective(x, fn_model):
        # Use obj1-specific grid
        grid = problem.generate_da_grid(x)
        predictions = fn_model(grid)
        return problem.calc_da_metric(predictions)
    return objective
```

### 4. Custom Acquisition Strategies

You can implement any acquisition strategy:

```python
def make_algo():
    def algo(fn_model_list):
        # Strategy 1: Uncertainty sampling
        candidates = sample_high_uncertainty_regions(fn_model_list)

        # Strategy 2: Boundary sampling
        # candidates = sample_near_boundaries(fn_model_list)

        # Strategy 3: GA optimization
        # candidates = genetic_algorithm(fn_model_list)

        return candidates_obj1, candidates_obj2

    return algo
```

## File Structure for Custom Problems

Recommended structure:

```
examples/
└── your_problem/
    ├── __init__.py
    ├── oracles.py          # Oracle functions
    ├── objectives.py       # Objective functions
    ├── algo.py             # Algorithm function
    ├── problem_utils.py    # Problem-specific utilities
    └── run.py              # Main entry point
```

## Common Patterns

### Pattern 1: Simple Problem

- Oracle returns objective directly
- Objective function just passes through predictions
- Random or grid sampling for acquisition

See: `synthetic/` example

### Pattern 2: Intermediate Predictions

- Oracle returns physical measurements (e.g., survival turns)
- Objective function computes metrics from measurements
- Sophisticated acquisition (GA, Bayesian opt)

See: `dama/` example

### Pattern 3: Stochastic Simulations

- Oracle augments inputs with random seeds
- Multiple evaluations per configuration
- Acquisition accounts for uncertainty

See: `dama/dama_oracles.py`

## Further Reading

- **[Framework Guide](../docs/FRAMEWORK_GUIDE.md)**: Complete BAX user guide with API reference
- **[DAMA Example](../docs/DAMA_EXAMPLE.md)**: Detailed DAMA walkthrough
- **[Contributing](../docs/CONTRIBUTING.md)**: Development guide
- **Main README**: Overview of BAX framework
- **Source code**: `core/bax_core.py` for BAXOpt implementation
- **DAMA example**: Full implementation in `dama/` directory
