# BAX Framework

**Bayesian Algorithm Execution for Multi-Objective Optimization with Expensive Simulations**

BAX is a framework that uses neural network surrogate models and Bayesian optimization to efficiently explore the Pareto front when simulations are expensive, requiring only a minimal 5-function API from users.

---

## Features

- **Multi-objective optimization** using NSGA2 genetic algorithm
- **Neural network surrogates** (PyTorch) to replace expensive simulations
- **Bayesian acquisition strategy** for efficient sampling
- **Automatic checkpointing** and resume capability
- **Simple API**: Just 5 functions to implement your problem
- **Parallel simulation** support via ProcessPoolExecutor

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for training)

### Using UV (Recommended)

```bash
# Install uv
pip install uv

# Clone and install
cd DAMA-BAX
uv sync
```

### Using pip

```bash
cd DAMA-BAX
pip install -e .
```

### Verify Installation

```bash
python verify.py
```

This checks Python version, dependencies, core modules, and compute devices.

---

## Quick Start

### The 5-Function API

BAX requires you to provide 5 simple functions (no classes!):

```python
import sys
sys.path.insert(0, 'core')

from bax_core import BAXOpt
import da_NN as dann

# 1-2. Oracle functions (expensive simulations)
def oracle_obj1(X):
    return your_expensive_simulation_1(X)

def oracle_obj2(X):
    return your_expensive_simulation_2(X)

# 3-4. Objective functions (convert predictions → objectives)
def objective_obj1(x, fn_model):
    predictions = fn_model(x)
    return calculate_objective(predictions)

def objective_obj2(x, fn_model):
    predictions = fn_model(x)
    return calculate_other_objective(predictions)

# 5. Algorithm function (acquisition strategy)
def make_algo():
    def algo(fn_model_list):
        candidates = optimize_with_surrogates(fn_model_list)
        return candidates_obj1, candidates_obj2
    return algo

# Generate initial data
X_init = sample_initial_points(n=1000, dims=4)
Y1_init = oracle_obj1(X_init)
Y2_init = oracle_obj2(X_init)

# Setup normalization
X_mu, X_std = dann.get_norm(X_init)
norm = lambda X: dann.normalize(X.copy(), X_mu, X_std)

# Create optimizer
opt = BAXOpt(
    algo=make_algo(),
    fn_oracle=[oracle_obj1, oracle_obj2],
    norm=[norm, norm],
    init=[lambda: (X_init, Y1_init), lambda: (X_init, Y2_init)],
    device='auto'
)

# Run optimization
opt.run_acquisition(max_iterations=100)
```

---

## Examples

### 1. Synthetic Problem (Minimal)

Simple analytical functions demonstrating the API:

```bash
cd examples/synthetic
python run_synthetic.py
```

**What it shows:**
- Minimal 5-function implementation (~200 lines)
- Random sampling acquisition
- Direct pass-through objectives

### 2. DAMA Problem (Full-Featured)

Particle accelerator optimization with advanced features:

```bash
cd examples/dama
python run_dama.py --run-id 3 --max-iter 100
```

**What it shows:**
- Seed augmentation in oracles
- Complex grid generation (spatial + momentum)
- NSGA2 genetic algorithm + boundary sampling
- Data transformations and resource management

See `examples/README.md` for detailed API documentation.

---

## Documentation

- **[Framework Guide](docs/FRAMEWORK_GUIDE.md)** - Complete user guide with API reference
- **[DAMA Example](docs/DAMA_EXAMPLE.md)** - Full-featured example walkthrough
- **[Contributing](docs/CONTRIBUTING.md)** - Development guide
- **[Examples README](examples/README.md)** - Detailed examples and patterns

---

## How It Works

### The BAX Loop

1. **Train surrogates**: Neural networks learn to predict expensive simulation outputs
2. **Run acquisition**: Use surrogates to find promising candidates (cheap!)
3. **Query oracles**: Run actual simulations on selected points (expensive)
4. **Update models**: Retrain surrogates with new data
5. **Repeat**: Continue until Pareto front converges

**Key insight**: Most evaluations use fast surrogate models. Only a few carefully selected points require expensive simulations.

---

## When to Use BAX

✅ **Use BAX when:**
- You have 2 competing objectives to optimize
- Your simulations are expensive (minutes to hours each)
- You want to find the Pareto front efficiently
- You can provide ~100-1000 initial samples for training

❌ **Don't use BAX when:**
- Simulations are cheap (use traditional MOO algorithms)
- You have >2 objectives (BAX is designed for 2)
- You can't provide initial training data

---

## Architecture

```
DAMA-BAX/
├── core/                      # Generic BAX framework
│   ├── bax_core.py           # Main optimizer (BAXOpt class)
│   ├── da_NN.py              # Neural network architecture
│   └── config.py             # Configuration utilities
├── examples/                  # Example implementations
│   ├── dama/                 # Particle accelerator optimization
│   └── synthetic/            # Minimal example
├── docs/                      # Documentation
│   ├── FRAMEWORK_GUIDE.md    # User guide
│   ├── DAMA_EXAMPLE.md       # Example walkthrough
│   └── CONTRIBUTING.md       # Development guide
├── pyproject.toml            # Package configuration
├── verify.py                 # Installation checker
└── README.md                 # This file
```

---

## Configuration Options

```python
opt = BAXOpt(...)

# Sampling
opt.n_sampling = 50              # Points per iteration

# Neural Network
opt.n_neur = 800                 # Network width
opt.dropout = 0.1                # Dropout rate
opt.lr = 1e-4                    # Learning rate
opt.batch_size = 1000            # Batch size

# Training
opt.epochs = 150                 # Initial training epochs
opt.iter_epochs = 10             # Per-iteration training
opt.weight_new_pts = 10          # Weight for new data

# Checkpointing
opt.snapshot = True              # Save models each iteration
opt.model_root = './models/'     # Model save directory
```

---

## Dependencies

**Core ML:**
- numpy, scipy, matplotlib, scikit-learn, tqdm
- torch (PyTorch for neural networks)

**Optimization:**
- pymoo (NSGA2 genetic algorithm)
- pyDOE (Latin Hypercube Sampling)

**Optional:**
- umap-learn (dimensionality reduction for visualization)
- at (PyAT - only for DAMA accelerator example)

All dependencies are specified in `pyproject.toml`.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **Out of memory** | Reduce `opt.batch_size` or `opt.n_sampling` |
| **Training too slow** | Use GPU (`device='cuda'`) or reduce `opt.n_neur` |
| **Bad Pareto front** | Increase `opt.n_sampling` or `max_iterations` |
| **Module not found** | Add `sys.path.insert(0, 'path/to/core')` |
| **NaN in training** | Check data normalization, reduce learning rate |

See `docs/FRAMEWORK_GUIDE.md` for more troubleshooting tips.

---

## Automatic Resume

If interrupted, simply restart with the same command. BAX automatically:
- Detects the latest checkpoint
- Reloads models and data
- Continues from where it left off

```bash
# Run initially
python your_optimization.py

# Interrupted? Just re-run
python your_optimization.py  # Resumes automatically
```

---

## Citation

If you use BAX in your research, please cite:

```
[Citation information to be added]
```

---

## License

MIT License - see LICENSE file for details

---

## Support

- **Documentation**: See `docs/` directory
- **Examples**: See `examples/` directory
- **Issues**: Open an issue on GitHub
- **Questions**: Check `docs/FRAMEWORK_GUIDE.md` or examples

---

## Quick Reference

**Create your optimization in 3 steps:**

1. **Implement 5 functions**: 2 oracles + 2 objectives + 1 algorithm
2. **Generate initial data**: ~1000 samples from your simulations
3. **Run BAX**: Create `BAXOpt` instance and call `run_acquisition()`

**See `examples/synthetic/run_synthetic.py` for a complete minimal example.**

---

## What Makes BAX Different?

- **Designed for expensive simulations**: Most MOO algorithms assume cheap evaluations
- **Learns simulation behavior**: Surrogates replace simulations after initial training
- **Acquisition-based**: Strategically selects which points to evaluate
- **Production-ready**: Automatic checkpointing, resuming, and error recovery
- **Simple API**: Just 5 functions, no complex class hierarchies

---

## Comparison

| Method | Evaluations Needed | Best For |
|--------|-------------------|----------|
| **NSGA2 (traditional)** | 10,000+ | Cheap simulations |
| **Bayesian Optimization** | 100-1000 | Single objective |
| **BAX (this framework)** | 100-1000 | Multi-objective + expensive |

BAX combines the efficiency of Bayesian optimization with multi-objective capability.

---

## Getting Started

1. **Install**: `uv sync` or `pip install -e .`
2. **Verify**: `python verify.py`
3. **Learn API**: Read `examples/synthetic/run_synthetic.py`
4. **Understand patterns**: Read `docs/FRAMEWORK_GUIDE.md`
5. **Implement your problem**: Follow the 5-function template
6. **Run and iterate**: Monitor convergence, tune parameters

**Next steps:** See `docs/FRAMEWORK_GUIDE.md` for detailed guide and `examples/` for working implementations.
