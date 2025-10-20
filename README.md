# BAX Framework

**Bayesian Algorithm Execution for Multi-Objective Optimization with Expensive Simulations**

BAX uses neural network surrogate models to efficiently find Pareto-optimal solutions when simulations are expensive. You provide 3 simple functions (oracles, objectives, algorithm) and BAX handles surrogate training, acquisition, and iterative optimization.

---

## Installation

### Quick Install

```bash
# Install dependencies
pip install uv
cd DAMA-BAX
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Verify installation
python verify.py
```

### Using pip (Alternative)

```bash
# Create virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install
pip install -e .

# Verify
python verify.py
```

---

## Quick Start - Run Your First Example

```bash
# Run the simplest example (takes ~30 seconds)
python run.py --case examples/synthetic_simple --max-iter 5
```

You should see:
- Initial data generation
- Neural network training progress (net_0 and net_1)
- BAX iterations with Pareto front updates
- Final results saved to `./models_simple/`

**Note**: net_1 (Rosenbrock) may show higher loss values (100-900) initially - this is normal as Rosenbrock produces larger values than sphere. Loss should decrease during training.

---

## Minimal Code Example

Here's all you need to use BAX (from `examples/synthetic_simple/run_simple_api.py`):

```python
from bax_core import run_bax_optimization
import numpy as np

# 1. Define oracle functions (your expensive simulations)
def oracle_obj1(X):
    """Sphere function: sum of squares."""
    return np.sum(X**2, axis=1).reshape(-1, 1)

def oracle_obj2(X):
    """Rosenbrock function."""
    return np.sum(100*(X[:, 1:] - X[:, :-1]**2)**2 + (1 - X[:, :-1])**2, axis=1).reshape(-1, 1)

# 2. Define objective functions (convert predictions → objectives)
def objective_obj1(x, fn_model):
    return fn_model(x).T  # Just pass through predictions

def objective_obj2(x, fn_model):
    return fn_model(x).T

# 3. Define algorithm (acquisition strategy)
def make_algo():
    def algo(fn_model_list):
        # Random sampling (simple but effective!)
        candidates = np.random.rand(50, 2)
        return candidates, candidates
    return algo

# That's it! Run optimization:
opt, results = run_bax_optimization(
    oracles=[oracle_obj1, oracle_obj2],
    objectives=[objective_obj1, objective_obj2],
    algorithm=make_algo(),
    n_init=50,           # Initial samples (automatic)
    max_iterations=100
)
```

**Key insight:** BAX trains cheap surrogate models to replace expensive oracles, then uses them to intelligently select the next points to evaluate.

---

## Examples

| Example | Complexity | What it demonstrates | Run command |
|---------|-----------|---------------------|-------------|
| **synthetic_simple** | Starter | Basic 3-function API, random sampling | `python run.py --case examples/synthetic_simple` |
| **synthetic** | Intermediate | Grid expansion, custom initialization | `python run.py --case examples/synthetic --max-iter 5` |
| **dama** | Advanced | Particle accelerator optimization, NSGA2 + boundary sampling | `python run.py --case examples/dama --run-id 3 --max-iter 100` |

**Try them in order!** Each example builds on concepts from the previous one.

### Running Examples

```bash
# Simple: Direct evaluation
python run.py --case examples/synthetic_simple

# Grid expansion pattern
python run.py --case examples/synthetic --max-iter 5

# Full application (requires pretrained models in examples/dama/resources/)
python run.py --case examples/dama --run-id 3 --max-iter 100

# Custom parameters
python run.py --case examples/synthetic \
              --max-iter 10 \
              --n-sampling 20 \
              --nn-neurons 400 \
              --seed 42
```

---

## Creating Your Own Optimization

### Step 1: Create a case directory

```bash
mkdir my_optimization
cd my_optimization
```

### Step 2: Implement `get_bax_config(args)`

Create `run_my_api.py`:

```python
import numpy as np
from bax_core import run_bax_optimization

def oracle_obj1(X):
    # Your expensive simulation here
    return your_simulation_1(X)

def oracle_obj2(X):
    # Another expensive simulation
    return your_simulation_2(X)

def objective_obj1(x, fn_model):
    predictions = fn_model(x)
    return your_metric_calculation(predictions)

def objective_obj2(x, fn_model):
    predictions = fn_model(x)
    return your_other_metric(predictions)

def make_algo():
    def algo(fn_model_list):
        # Your acquisition strategy (GA, Bayesian opt, random, etc.)
        candidates = your_optimization_method(fn_model_list)
        return candidates, candidates
    return algo

def get_bax_config(args):
    """Entry point for unified runner."""
    return {
        'oracles': [oracle_obj1, oracle_obj2],
        'objectives': [objective_obj1, objective_obj2],
        'algorithm': make_algo(),
        'model_root': f'./models_run_{args.run_id}/' if args.run_id else './models/',
    }
```

### Step 3: Run it!

```bash
python /path/to/DAMA-BAX/run.py --case ./my_optimization --max-iter 100
```

**See** `examples/synthetic_simple/run_simple_api.py` for a complete minimal template.

---

## Advanced Configuration

### Available CLI Options

```bash
python run.py --case <directory> [options]

Core options:
  --run-id N              Run identifier for model/data directories
  --max-iter N            Maximum BAX iterations (default: 100)
  --n-sampling N          Points sampled per iteration (default: 50)
  --n-init N              Initial training samples (default: 100)
  --device {auto,cuda,cpu} Compute device (default: auto)
  --seed N                Random seed

Neural network:
  --nn-neurons N          Network width (default: 800)
  --nn-lr FLOAT           Learning rate (default: 1e-4)
  --nn-epochs N           Initial training epochs (default: 150)
  --nn-iter-epochs N      Per-iteration epochs (default: 10)
  --nn-batch-size N       Batch size (default: 1000)

Training:
  --test-ratio FLOAT      Test set ratio (default: 0.05)
  --weight-new FLOAT      Weight for new data points (default: 10)
  --snapshot / --no-snapshot  Save models each iteration
```

### Manual API (Advanced Users)

For full control over initialization, normalization, and configuration, use the `BAXOpt` class directly:

```python
from bax_core import BAXOpt
import da_NN as dann

# Manual initialization
X_init = generate_initial_samples(1000)
Y0_init = oracle_obj1(X_init)
Y1_init = oracle_obj2(X_init)

# Manual normalization
X_mu, X_std = dann.get_norm(X_init)
norm = lambda X: dann.normalize(X.copy(), X_mu, X_std)

# Create optimizer
opt = BAXOpt(
    algo=make_algo(),
    fn_oracle=[oracle_obj1, oracle_obj2],
    norm=[norm, norm],
    init=[lambda: (X_init, Y0_init), lambda: (X_init, Y1_init)],
    device='cuda'
)

# Configure
opt.n_sampling = 50
opt.n_neur = 800
opt.epochs = 150

# Run
opt.run_acquisition(max_iterations=100)
```

**See** `examples/synthetic/run_synthetic.py` for complete manual API example.

---

## Documentation

- **[Framework Guide](docs/FRAMEWORK_GUIDE.md)** - Complete guide with patterns and best practices
- **[API Reference](docs/API_QUICK_REFERENCE.md)** - Quick API reference for both APIs
- **[Examples](examples/README.md)** - Detailed examples documentation
- **[DAMA Example](docs/DAMA_EXAMPLE.md)** - Advanced full-featured example
- **[Contributing](docs/CONTRIBUTING.md)** - Development guidelines

For troubleshooting, SLURM usage, and advanced topics, see the Framework Guide.
