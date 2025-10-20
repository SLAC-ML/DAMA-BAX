#!/usr/bin/env python
"""
Unified BAX Runner - Directory-Based Entry Point

This script runs any BAX optimization case from a directory containing a run script.
It works with built-in examples and user-created custom cases.

Directory must contain a run script (run_*_api.py or run_*.py) with either:
  - A run() function that accepts (args, extra_args), OR
  - A main() function (will be called via sys.argv manipulation)

Usage:
  python run.py --case <directory> [options]
  python run.py [options]  # Auto-detect from current directory

Examples:
  # Run built-in examples
  python run.py --case examples/dama --run-id 3 --max-iter 100
  python run.py --case examples/synthetic_simple --n-sampling 20

  # Run custom user case
  python run.py --case ~/my_bax_problem --run-id 0
  python run.py --case ./cases/production_run_v2

  # Auto-detect from current directory
  cd examples/dama && python ../../run.py --run-id 3

  # SLURM job submission
  CASE_DIR=examples/dama RUN_ID=3 sbatch job.sh

Author: Claude (Anthropic)
"""

import sys
import argparse
from pathlib import Path
import importlib.util


def find_case_directory(case_arg=None):
    """
    Find case directory from:
    1. --case argument (if provided)
    2. Current working directory (if it contains run_*.py)
    3. Raise error if ambiguous or not found

    Parameters:
    -----------
    case_arg : str or None
        Value of --case argument

    Returns:
    --------
    Path
        Resolved case directory

    Raises:
    -------
    FileNotFoundError
        If specified directory doesn't exist
    ValueError
        If no case can be determined
    """
    if case_arg:
        case_dir = Path(case_arg).resolve()
        if not case_dir.exists():
            raise FileNotFoundError(f"Case directory not found: {case_arg}")
        if not case_dir.is_dir():
            raise ValueError(f"Not a directory: {case_arg}")
        return case_dir

    # Try to auto-detect from CWD
    cwd = Path.cwd()
    run_scripts = list(cwd.glob("run_*.py"))

    if run_scripts:
        # CWD contains run scripts, use it as case directory
        return cwd

    # Check if CWD is project root (has examples/ but no run_*.py)
    if (cwd / "examples").exists() and (cwd / "core").exists():
        raise ValueError(
            "No --case specified and current directory is project root.\n"
            "Please either:\n"
            "  1. Specify a case: python run.py --case examples/dama\n"
            "  2. Change to case directory: cd examples/dama && python ../../run.py"
        )

    # No run scripts found and not in project root
    raise ValueError(
        "No --case specified and no run_*.py found in current directory.\n"
        "Please specify a case directory: python run.py --case <directory>"
    )


def find_run_script(case_dir):
    """
    Find run script in case directory.

    Priority order:
    1. run_*_api.py files (preferred)
    2. run_*.py files (fallback)

    Parameters:
    -----------
    case_dir : Path
        Case directory to search

    Returns:
    --------
    Path
        Path to run script

    Raises:
    -------
    FileNotFoundError
        If no run script found in directory
    """
    # Prefer API versions (simplified high-level API)
    api_scripts = sorted(case_dir.glob("run_*_api.py"))
    if api_scripts:
        if len(api_scripts) > 1:
            print(f"Info: Multiple API scripts found in {case_dir.name}, using: {api_scripts[0].name}")
        return api_scripts[0]

    # Fall back to any run_*.py
    run_scripts = sorted(case_dir.glob("run_*.py"))
    if run_scripts:
        if len(run_scripts) > 1:
            print(f"Info: Multiple run scripts found in {case_dir.name}, using: {run_scripts[0].name}")
        return run_scripts[0]

    # No run script found
    raise FileNotFoundError(
        f"No run script (run_*.py) found in: {case_dir}\n\n"
        "Case directory must contain a run_*_api.py or run_*.py script.\n\n"
        "To create a new case:\n"
        "  1. Copy an example: cp -r examples/synthetic_simple cases/my_case\n"
        "  2. Edit run script: vim cases/my_case/run_simple_api.py\n"
        "  3. Run it: python run.py --case cases/my_case"
    )


def load_run_module(script_path, case_dir):
    """
    Dynamically load run script as a Python module.

    Parameters:
    -----------
    script_path : Path
        Path to the run script file
    case_dir : Path
        Case directory (added to sys.path for imports)

    Returns:
    --------
    module
        Loaded Python module

    Raises:
    -------
    Exception
        If module cannot be loaded
    """
    # Add case directory to sys.path so local imports work
    # (e.g., "from utils import ..." in dama example)
    case_dir_str = str(case_dir)
    if case_dir_str not in sys.path:
        sys.path.insert(0, case_dir_str)

    # Also add core/ to path for bax_core imports
    project_root = Path(__file__).parent
    core_dir = project_root / "core"
    core_dir_str = str(core_dir)
    if core_dir_str not in sys.path:
        sys.path.insert(0, core_dir_str)

    # Load the module
    module_name = f"case_run_{script_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load run script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def create_parser():
    """
    Create argument parser with common BAX arguments.

    Returns:
    --------
    argparse.ArgumentParser
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='Unified BAX runner - runs any case from a directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run built-in examples
  python run.py --case examples/synthetic_simple
  python run.py --case examples/synthetic --max-iter 5
  python run.py --case examples/dama --run-id 3 --max-iter 100

  # Run user-created case
  python run.py --case ~/my_custom_case
  python run.py --case ./cases/production_run

  # Auto-detect from current directory
  cd examples/dama && python ../../run.py --run-id 3

The case directory must contain a run_*_api.py or run_*.py script.
All additional arguments are passed through to the run script.
        """,
        add_help=True
    )

    # Core argument - optional if running from case directory
    parser.add_argument(
        '--case',
        type=str,
        help='Path to case directory (auto-detects if in case directory)'
    )

    # Common arguments (forwarded to run script)
    parser.add_argument(
        '--run-id',
        type=int,
        help='Run identifier for data/model directories'
    )

    parser.add_argument(
        '--max-iter',
        type=int,
        help='Maximum number of BAX iterations'
    )

    parser.add_argument(
        '--n-sampling',
        type=int,
        help='Number of points sampled per iteration'
    )

    parser.add_argument(
        '--n-init',
        type=int,
        help='Number of initial training samples'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cuda', 'cpu'],
        help='Compute device (auto=detect CUDA, cuda=force CUDA, cpu=force CPU)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='Save model snapshot at each iteration'
    )

    parser.add_argument(
        '--no-snapshot',
        dest='snapshot',
        action='store_false',
        help='Do not save model snapshots'
    )

    parser.set_defaults(snapshot=None)  # Let run script decide default

    # Neural network parameters
    parser.add_argument(
        '--nn-neurons',
        type=int,
        help='Neural network width (number of neurons)'
    )

    parser.add_argument(
        '--nn-lr',
        type=float,
        help='Neural network learning rate'
    )

    parser.add_argument(
        '--nn-dropout',
        type=float,
        help='Neural network dropout rate'
    )

    parser.add_argument(
        '--nn-epochs',
        type=int,
        help='Initial training epochs'
    )

    parser.add_argument(
        '--nn-iter-epochs',
        type=int,
        help='Per-iteration training epochs'
    )

    parser.add_argument(
        '--nn-batch-size',
        type=int,
        help='Training batch size'
    )

    parser.add_argument(
        '--nn-model-type',
        choices=['fc', 'split', 'sine'],
        help='Neural network architecture type'
    )

    # Training parameters
    parser.add_argument(
        '--test-ratio',
        type=float,
        help='Test set ratio (fraction of data for validation)'
    )

    parser.add_argument(
        '--weight-new',
        type=float,
        help='Weight for new data points in loss function'
    )

    return parser


def build_argv_from_args(script_path, args, extra_args):
    """
    Build sys.argv list from parsed arguments for scripts that use argparse.

    Parameters:
    -----------
    script_path : Path
        Path to run script (used as sys.argv[0])
    args : argparse.Namespace
        Parsed common arguments
    extra_args : list
        Additional unparsed arguments

    Returns:
    --------
    list
        sys.argv list
    """
    argv = [str(script_path)]

    # Add common arguments if they were specified
    if args.run_id is not None:
        argv.extend(['--run-id', str(args.run_id)])

    if args.max_iter is not None:
        argv.extend(['--max-iter', str(args.max_iter)])

    if args.n_sampling is not None:
        argv.extend(['--n-sampling', str(args.n_sampling)])

    if args.n_init is not None:
        argv.extend(['--n-init', str(args.n_init)])

    if args.device is not None:
        argv.extend(['--device', args.device])

    if args.seed is not None:
        argv.extend(['--seed', str(args.seed)])

    if args.snapshot is not None:
        if args.snapshot:
            argv.append('--snapshot')
        else:
            argv.append('--no-snapshot')

    # Add any extra arguments
    argv.extend(extra_args)

    return argv


def main():
    """Main entry point for unified runner."""
    # Parse arguments
    parser = create_parser()
    args, unknown_args = parser.parse_known_args()

    # Find case directory
    try:
        case_dir = find_case_directory(args.case)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        print(f"\nRun 'python run.py --help' for usage information.", file=sys.stderr)
        sys.exit(1)

    # Find run script in directory
    try:
        run_script = find_run_script(case_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Print info
    print("=" * 70)
    print("BAX Unified Runner")
    print("=" * 70)
    print(f"Case directory: {case_dir}")
    print(f"Run script:     {run_script.name}")
    print("=" * 70)
    print()

    # Load the run module
    try:
        module = load_run_module(run_script, case_dir)
    except Exception as e:
        print(f"Error loading run script: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Execute optimization
    try:
        # Method 1: Call get_bax_config() if it exists (new approach)
        if hasattr(module, 'get_bax_config'):
            print("Loading case configuration...")
            print()

            # Get configuration from case
            config = module.get_bax_config(args)

            # Import run_bax_optimization
            # Add core to path if needed
            project_root = Path(__file__).parent
            core_dir = project_root / "core"
            core_dir_str = str(core_dir)
            if core_dir_str not in sys.path:
                sys.path.insert(0, core_dir_str)

            from bax_core import run_bax_optimization
            import torch

            # Build nn_config from args (with defaults from config if available)
            nn_config = config.get('nn_config', {})
            if args.nn_neurons is not None:
                nn_config['n_neur'] = args.nn_neurons
            if args.nn_lr is not None:
                nn_config['lr'] = args.nn_lr
            if args.nn_dropout is not None:
                nn_config['dropout'] = args.nn_dropout
            if args.nn_epochs is not None:
                nn_config['epochs'] = args.nn_epochs
            if args.nn_iter_epochs is not None:
                nn_config['iter_epochs'] = args.nn_iter_epochs
            if args.nn_batch_size is not None:
                nn_config['batch_size'] = args.nn_batch_size
            if args.nn_model_type is not None:
                nn_config['model_type'] = args.nn_model_type

            # Handle device
            if args.device == 'auto' or args.device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            elif args.device == 'cuda':
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            # Run BAX optimization with merged configuration
            opt, results = run_bax_optimization(
                oracles=config['oracles'],
                objectives=config['objectives'],
                algorithm=config['algorithm'],
                init_sampler=config.get('init_sampler'),

                # From CLI args (common BAX params)
                n_init=args.n_init if args.n_init is not None else config.get('n_init', 100),
                max_iterations=args.max_iter if args.max_iter is not None else config.get('max_iterations', 100),
                n_sampling=args.n_sampling if args.n_sampling is not None else config.get('n_sampling', 50),
                nn_config=nn_config,
                test_ratio=args.test_ratio if args.test_ratio is not None else config.get('test_ratio', 0.05),
                weight_new=args.weight_new if args.weight_new is not None else config.get('weight_new', 10),
                device=device,
                seed=args.seed if args.seed is not None else config.get('seed', 42),
                snapshot=args.snapshot if args.snapshot is not None else config.get('snapshot', True),

                # From case config (problem-specific)
                model_names=config.get('model_names'),
                model_root=config.get('model_root', './models/'),
                bounds=config.get('bounds'),
                input_dims=config.get('input_dims'),
                verbose=True,
            )

        # Method 2: Call run() function if it exists (old approach for backward compat)
        elif hasattr(module, 'run'):
            print("Calling run() function (legacy mode)...")
            print()
            module.run(args, unknown_args)

        # Method 3: Call main() with sys.argv manipulation (fallback)
        elif hasattr(module, 'main'):
            print("Calling main() function (sys.argv mode)...")
            print()
            # Build sys.argv from our parsed arguments
            sys.argv = build_argv_from_args(run_script, args, unknown_args)
            module.main()

        else:
            raise AttributeError(
                f"Run script must define get_bax_config(args), run(args, extra_args), or main() function.\n"
                f"Script: {run_script.name}\n\n"
                "Recommended approach - define get_bax_config():\n"
                "  def get_bax_config(args):\n"
                "      return {{\n"
                "          'oracles': [oracle1, oracle2],\n"
                "          'objectives': [obj1, obj2],\n"
                "          'algorithm': algo,\n"
                "          'init_sampler': sampler,  # optional\n"
                "      }}\n"
            )

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)

    except Exception as e:
        print(f"\nError during execution: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 70)
    print("BAX run completed successfully")
    print("=" * 70)


if __name__ == '__main__':
    main()
