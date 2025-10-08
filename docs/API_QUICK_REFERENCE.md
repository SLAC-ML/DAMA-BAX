# BAX API Quick Reference

This document provides a quick reference for both BAX APIs.

## Simplified API (Recommended)

### Basic Usage

```python
from bax_core import run_bax_optimization

opt, results = run_bax_optimization(
    oracles=[oracle_obj1, oracle_obj2, ...],
    objectives=[objective_obj1, objective_obj2, ...],
    algorithm=algo,
    n_init=100,
    max_iterations=100
)
```

### All Parameters

```python
opt, results = run_bax_optimization(
    # Required: Core functions
    oracles,              # List of oracle functions
    objectives,           # List of objective functions
    algorithm,            # Acquisition algorithm function

    # Initialization (automatic by default)
    n_init=100,           # Number of initial samples (int or list)
    input_dims=None,      # Auto-inferred if None
    bounds=None,          # [(lower, upper), ...] per dim, default [0,1]
    init_sampler=None,    # Custom sampler function (optional)
    expansion_funcs=None, # [expand_obj1, ...] for Pattern B

    # Optimization settings
    max_iterations=100,   # BAX iterations
    n_sampling=50,        # Points per iteration

    # Model configuration
    model_root='./models',
    model_names=None,     # Auto: ['net0', 'net1', ...]
    nn_config=None,       # Dict with n_neur, lr, dropout, epochs, ...

    # Other
    device=None,          # Auto-detect CUDA
    snapshot=True,        # Save model each iteration
    verbose=True,
    seed=42
)
```

### NN Config Dictionary

```python
nn_config = {
    'n_neur': 800,        # Number of neurons
    'lr': 1e-4,           # Learning rate
    'dropout': 0.1,       # Dropout rate
    'epochs': 150,        # Initial training epochs
    'iter_epochs': 10,    # Per-iteration epochs
    'batch_size': 1000,   # Batch size
}
```

### Return Values

```python
opt       # BAXOpt instance with trained models
results   # Dict with:
          #   'X_init': initial training data
          #   'Y_init': initial oracle outputs
          #   'final_models': trained models
          #   'optimizer': same as opt
```

---

## Function Requirements

### Oracle Functions

```python
def oracle_obj1(X):
    """
    Run expensive simulation.

    Parameters:
    -----------
    X : np.ndarray, shape (n_samples, n_dims)
        Input configurations

    Returns:
    --------
    Y : np.ndarray, shape (n_samples,) or (n_samples, 1)
        Simulation outputs
    """
    Y = expensive_simulation(X)
    return Y
```

### Objective Functions

```python
def objective_obj1(x, fn_model):
    """
    Convert model predictions to objective values.

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
    predictions = fn_model(x)  # (1, n)
    obj = calculate_objective(predictions.T)  # (n, 1)
    return obj
```

### Algorithm Function

```python
def make_algo():
    """Create acquisition algorithm."""

    def algo(fn_model_list):
        """
        Select next candidates.

        Parameters:
        -----------
        fn_model_list : list of functions
            Surrogate models [fn_model1, fn_model2, ...]

        Returns:
        --------
        X_candidates_obj1, X_candidates_obj2, ... : np.ndarray
            Candidates for each objective
        """
        # Your acquisition strategy
        candidates = your_optimization(fn_model_list)
        return candidates_obj1, candidates_obj2

    return algo
```

### Expansion Functions (Pattern B only)

```python
def expand_for_obj1(x_base):
    """
    Expand base configs to evaluation grid.

    Parameters:
    -----------
    x_base : np.ndarray, shape (n, base_dims)
        Base configurations

    Returns:
    --------
    X_expanded : np.ndarray, shape (n*grid_size, expanded_dims)
        Expanded configurations for oracle
    """
    # Generate grid points for each base config
    X_expanded = generate_grid(x_base)
    return X_expanded
```

---

## Usage Patterns

### Pattern A: Direct Evaluation (No Expansion)

```python
# Oracle and objective work with same input space
opt, results = run_bax_optimization(
    oracles=[oracle_obj1, oracle_obj2],
    objectives=[objective_obj1, objective_obj2],
    algorithm=algo,
    n_init=100
)
```

**Flow:**
```
X (n, dims) → Oracle → Y (n, 1) → NN learns X→Y
X (n, dims) → Objective → NN predicts → obj (n, 1)
```

### Pattern B: Grid/Ensemble Evaluation (With Expansion)

```python
# Different input dimensions for oracle vs optimization
opt, results = run_bax_optimization(
    oracles=[oracle_obj1, oracle_obj2],
    objectives=[objective_obj1, objective_obj2],
    algorithm=algo,
    expansion_funcs=[expand_obj1, expand_obj2],
    n_init=100
)
```

**Flow:**
```
X_base (n, base_dims)
    → expand → X_expanded (n*grid, expanded_dims)
    → Oracle → Y (n*grid, 1)
    → NN learns X_expanded→Y

X_base (n, base_dims)
    → Objective: expand → predict → aggregate → obj (n, 1)
```

---

## Custom Initialization

### Custom Bounds

```python
opt, results = run_bax_optimization(
    oracles=[oracle_obj1, oracle_obj2],
    objectives=[objective_obj1, objective_obj2],
    algorithm=algo,
    bounds=[(0, 10), (-5, 5)],  # Custom bounds per dimension
    n_init=100
)
```

### Custom Sampler

```python
def my_custom_sampler(n_init, input_dims, bounds, seed):
    """
    Custom initialization sampler.

    Returns:
    --------
    X_list : list of np.ndarray
        Initial samples for each objective
    """
    X_list = []
    for dims in input_dims:
        X = my_sampling_strategy(n_init, dims, bounds, seed)
        X_list.append(X)
    return X_list

opt, results = run_bax_optimization(
    ...,
    init_sampler=my_custom_sampler
)
```

---

## Manual API (Advanced)

For full control, use `BAXOpt` directly:

```python
from bax_core import BAXOpt
import da_NN as dann

# Manual setup
X_init = generate_initial_samples(100)
Y0_init = oracle_obj1(X_init)
Y1_init = oracle_obj2(X_init)

X_mu, X_std = dann.get_norm(X_init)
def norm(X):
    return dann.normalize(X.copy(), X_mu, X_std)

def init_obj1():
    return X_init, Y0_init

def init_obj2():
    return X_init, Y1_init

# Create optimizer
opt = BAXOpt(
    algo=algo,
    fn_oracle=[oracle_obj1, oracle_obj2],
    norm=[norm, norm],
    init=[init_obj1, init_obj2],
    device=device,
    model_root='./models',
    model_names=['net0', 'net1']
)

# Configure
opt.n_sampling = 50
opt.n_feat = 2
opt.epochs = 150
opt.n_neur = 800

# Run
opt.run_acquisition(n_iters=100, verbose=True)
```

---

## Examples

| Example | API | Pattern | Lines | Description |
|---------|-----|---------|-------|-------------|
| `synthetic_simple/run_simple_api.py` | Simplified | A | 180 | Simplest example |
| `synthetic_simple/run_simple.py` | Manual | A | 260 | Manual setup |
| `synthetic/run_synthetic_api.py` | Simplified | B | 340 | Grid expansion |
| `synthetic/run_synthetic.py` | Manual | B | 470 | Manual grid setup |
| `dama/run_dama.py` | Manual | B | 600+ | Full DAMA example |

---

## Common Issues

**Q: "Input dimensions don't match"**
- For Pattern A: Ensure oracle and objective use same X dimensions
- For Pattern B: Provide `expansion_funcs` and ensure objective expands correctly

**Q: "How do I use custom model names?"**
```python
run_bax_optimization(..., model_names=['my_model1', 'my_model2'])
```

**Q: "Can I have different initial samples per objective?"**
```python
run_bax_optimization(..., n_init=[100, 50])  # 100 for obj1, 50 for obj2
```

**Q: "How do I resume from checkpoint?"**
- Use manual API with `get_curr_loop_num()` (see `dama/run_dama.py`)
- Simplified API doesn't support resume yet

---

## See Also

- `../examples/README.md` - Detailed examples and patterns
- `FRAMEWORK_GUIDE.md` - Complete framework documentation
- `DAMA_EXAMPLE.md` - Advanced DAMA example walkthrough
