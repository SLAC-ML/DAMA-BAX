# Contributing to DAMA-BAX

This guide is for developers who want to contribute to the BAX framework or understand its internal structure.

## Table of Contents

1. [Development Setup](#development-setup)
2. [Project Structure](#project-structure)
3. [Code Style](#code-style)
4. [Testing](#testing)
5. [Adding New Examples](#adding-new-examples)
6. [Core Framework Development](#core-framework-development)

---

## Development Setup

### Installation for Development

```bash
# Clone repository
cd DAMA-BAX

# Install uv (if not already installed)
pip install uv

# Install with development dependencies
uv sync

# Or install in editable mode with pip
pip install -e ".[dev]"
```

### Development Dependencies

Included in `pyproject.toml`:

- **pytest** - Testing framework
- **black** - Code formatting
- **ruff** - Fast linting

### Verify Installation

```bash
python verify.py
```

This checks:
- Python version (≥3.8)
- All dependencies installed
- Core modules importable
- Compute devices available

---

## Project Structure

### Overview

```
DAMA-BAX/
├── core/                  # Generic BAX framework
│   ├── bax_core.py       # Main BAX optimizer
│   ├── da_NN.py          # Neural network architecture
│   └── config.py         # Configuration management
├── examples/              # Example implementations
│   ├── dama/             # Full-featured accelerator example
│   │   ├── run_dama.py
│   │   ├── dama_oracles.py
│   │   ├── dama_objectives.py
│   │   ├── dama_algo.py
│   │   └── resources/    # DAMA-specific data
│   └── synthetic/        # Minimal example
│       └── run_synthetic.py
├── docs/                  # Documentation
│   ├── FRAMEWORK_GUIDE.md
│   ├── DAMA_EXAMPLE.md
│   └── CONTRIBUTING.md (this file)
├── pyproject.toml        # Package configuration
├── verify.py             # Installation verification
└── README.md             # User documentation
```

### Core Modules

**`core/bax_core.py`** (~800 lines)
- `BAXOpt` class - Main optimizer
- Pareto front utilities
- Training functions
- Data generation helpers
- Checkpointing logic

**`core/da_NN.py`** (~300 lines)
- `DA_Net` class - Neural network architecture
- Supports 'fc', 'split', and 'sine' model types
- Training and inference methods
- Normalization utilities

**`core/config.py`** (~200 lines)
- Resource path management (legacy, mainly for DAMA)
- Environment variable support
- File verification utilities

### Package Configuration

**`pyproject.toml`**
- Project metadata (name, version, description)
- Dependencies:
  - **Core ML**: numpy, scipy, torch, scikit-learn, matplotlib, tqdm
  - **Optimization**: pymoo (NSGA2), pyDOE (Latin Hypercube Sampling)
  - **Dimensionality reduction**: umap-learn
  - **Accelerator physics**: at (PyAT) - only for DAMA example
- Build system: hatchling
- Code quality tools: black, ruff

---

## Code Style

### Python Style

We follow PEP 8 with some modifications:

- **Line length**: 120 characters
- **Formatter**: black
- **Linter**: ruff

### Formatting

```bash
# Format code
black . --line-length 120

# Lint code
ruff check .
```

### Configuration

See `pyproject.toml`:

```toml
[tool.black]
line-length = 120
target-version = ["py38", "py39", "py310", "py311"]

[tool.ruff]
line-length = 120
target-version = "py38"
```

### Docstring Style

Use NumPy-style docstrings:

```python
def my_function(param1, param2):
    """
    Brief description.

    Detailed description if needed.

    Parameters
    ----------
    param1 : type
        Description
    param2 : type
        Description

    Returns
    -------
    type
        Description
    """
```

---

## Testing

### Manual Testing

Currently, testing is done manually:

```bash
# Test synthetic example
cd examples/synthetic
python run_synthetic.py

# Test DAMA example
cd examples/dama
python run_dama.py --run-id test --max-iter 5
```

### Future: Automated Testing

We plan to add pytest-based tests:

```
tests/
├── test_bax_core.py     # Test BAX optimizer
├── test_da_nn.py        # Test neural network
├── test_examples.py     # Test examples run
└── fixtures/            # Test data
```

---

## Adding New Examples

### Step 1: Create Directory Structure

```bash
mkdir -p examples/your_problem
cd examples/your_problem
```

### Step 2: Implement Required Functions

Create the following files:

**`oracles.py`**
```python
"""Oracle functions for your problem."""

def oracle_obj1(X):
    """Run simulation for objective 1."""
    # Your expensive simulation here
    return Y

def oracle_obj2(X):
    """Run simulation for objective 2."""
    # Your expensive simulation here
    return Y
```

**`objectives.py`**
```python
"""Objective functions for your problem."""

def make_obj1(x, fn_model):
    """Calculate objective 1 from surrogate predictions."""
    predictions = fn_model(x)
    # Transform to objective
    return objective_values

def make_obj2(x, fn_model):
    """Calculate objective 2 from surrogate predictions."""
    predictions = fn_model(x)
    return objective_values
```

**`algo.py`**
```python
"""Acquisition algorithm for your problem."""

def make_algo(config):
    """Factory for acquisition algorithm."""
    def algo(fn_model_list):
        # Find candidates using surrogates
        candidates_obj1, candidates_obj2 = your_strategy(fn_model_list)
        return candidates_obj1, candidates_obj2
    return algo
```

**`run_your_problem.py`**
```python
"""Main entry point for your problem."""

import sys
sys.path.insert(0, '../../core')

from bax_core import BAXOpt
import da_NN as dann
from oracles import oracle_obj1, oracle_obj2
from objectives import make_obj1, make_obj2
from algo import make_algo

# Generate initial data
X_init = generate_initial_samples(n=1000, dims=4)
Y1_init = oracle_obj1(X_init)
Y2_init = oracle_obj2(X_init)

# Setup normalization
X_mu, X_std = dann.get_norm(X_init)
norm = lambda X: dann.normalize(X.copy(), X_mu, X_std)

# Setup initialization
init1 = lambda: (X_init, Y1_init)
init2 = lambda: (X_init, Y2_init)

# Create algorithm
algo = make_algo()

# Create and run optimizer
opt = BAXOpt(
    algo=algo,
    fn_oracle=[oracle_obj1, oracle_obj2],
    norm=[norm, norm],
    init=[init1, init2],
    device='auto'
)

opt.run_acquisition(max_iterations=100)
```

### Step 3: Test Your Implementation

```bash
cd examples/your_problem
python run_your_problem.py
```

### Step 4: Document Your Example

Add `README.md` in your example directory:

```markdown
# Your Problem

Brief description of the optimization problem.

## Usage

\`\`\`bash
python run_your_problem.py
\`\`\`

## Problem Description

- **Objective 1**: ...
- **Objective 2**: ...
- **Parameters**: ...
- **Constraints**: ...

## Results

Describe expected outputs and how to interpret them.
```

---

## Core Framework Development

### Architecture

**BAXOpt Class** (`bax_core.py`)

Main components:
- `__init__()` - Setup optimizer with oracles, objectives, algorithm
- `run_acquisition()` - Main optimization loop
- `train_models()` - Train surrogate models
- `save_checkpoint()` - Save models and data
- `load_checkpoint()` - Resume from checkpoint

Key methods:
```python
class BAXOpt:
    def __init__(self, algo, fn_oracle, norm, init, device):
        """Initialize optimizer."""

    def run_acquisition(self, max_iterations):
        """Main BAX loop."""
        for i in range(max_iterations):
            # 1. Run acquisition algorithm
            candidates = self.algo(self.surrogate_models)

            # 2. Query oracles
            new_data = self.query_oracles(candidates)

            # 3. Update surrogates
            self.train_models(new_data, weight_new=10)

            # 4. Save checkpoint
            self.save_checkpoint(iteration=i)
```

### Neural Network Architecture

**DA_Net Class** (`da_NN.py`)

Supports three model types:

1. **'fc'**: Fully connected (6 layers, dropout)
2. **'split'**: Reintroduces spatial coords near output
3. **'sine'**: Uses sine activations (for periodic functions)

Architecture:
```python
# Example: 'split' model with 800 neurons
Input (n_feat)
  ↓
FC(n_feat → 800) + ReLU + Dropout
  ↓
FC(800 → 800) + ReLU + Dropout
  ↓
FC(800 → 800) + ReLU + Dropout
  ↓
Concatenate with last 2 input features
  ↓
FC(802 → 800) + ReLU + Dropout
  ↓
FC(800 → 1)
```

### Adding New Model Types

To add a new neural network architecture:

1. Edit `da_NN.py`:

```python
class DA_Net(nn.Module):
    def __init__(self, n_feat, n_neur=800, dropout=0.1, model_type='fc'):
        super().__init__()

        if model_type == 'your_new_type':
            # Define your architecture
            self.fc1 = nn.Linear(n_feat, n_neur)
            # ... more layers
        elif model_type == 'fc':
            # Existing architectures
            pass
```

2. Update forward pass:

```python
def forward(self, x):
    if self.model_type == 'your_new_type':
        # Your forward logic
        return output
    # ... existing logic
```

3. Test:

```python
model = DA_Net(n_feat=6, model_type='your_new_type')
x_test = torch.randn(100, 6)
y = model(x_test)
assert y.shape == (100, 1)
```

### Acquisition Strategies

To implement a new acquisition strategy, create a function in your problem directory:

```python
def make_your_strategy():
    def algo(fn_model_list):
        # Your strategy using surrogates
        # Examples:
        # - Expected improvement
        # - Upper confidence bound
        # - Entropy search
        # - Random sampling
        # - Grid search

        return candidates_obj1, candidates_obj2
    return algo
```

See `examples/dama/dama_algo.py` for a full example using NSGA2 + boundary sampling.

---

## Packaging and Distribution

### Building Package

```bash
# Build distribution
python -m build

# This creates:
# - dist/dama_bax-0.1.0.tar.gz (source)
# - dist/dama_bax-0.1.0-py3-none-any.whl (wheel)
```

### Version Bumping

Update version in `pyproject.toml`:

```toml
[project]
version = "0.2.0"
```

### Creating Releases

1. Tag version:
```bash
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0
```

2. Create release notes describing changes

---

## Code Review Guidelines

When submitting changes:

1. **Follow code style** (run black + ruff)
2. **Test your changes** (manual testing for now)
3. **Update documentation** if adding features
4. **Keep changes focused** (one feature/fix per PR)
5. **Write clear commit messages**

### Good Commit Messages

```
Add support for custom loss functions

- Allow users to specify custom loss in BAXOpt
- Add example in synthetic problem
- Update documentation
```

---

## Common Development Tasks

### Add a New Dependency

1. Update `pyproject.toml`:
```toml
dependencies = [
    ...
    "new-package>=1.0",
]
```

2. Sync environment:
```bash
uv sync
```

### Update Documentation

1. Edit relevant `.md` files in `docs/`
2. Keep examples in sync with code
3. Update `README.md` if changing user-facing features

### Debug Training Issues

Enable verbose output:
```python
opt = BAXOpt(...)
opt.verbose = True  # Print training metrics
opt.run_acquisition(max_iterations=10)
```

Check for:
- NaN/Inf in data (normalize inputs/outputs)
- Learning rate too high/low
- Batch size too small/large
- Dropout too aggressive

---

## Questions or Issues?

- Check existing documentation in `docs/`
- Look at examples in `examples/`
- Open an issue on GitHub

---

## Summary

The BAX framework is designed to be:
- **Simple**: 5-function API for users
- **Modular**: Clear separation of framework vs. application
- **Extensible**: Easy to add new examples and features
- **Well-documented**: Comprehensive guides and examples

When contributing, maintain these principles and test your changes thoroughly.
