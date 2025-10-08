#!/usr/bin/env python
"""
Verify DAMA-BAX installation.

This script checks:
1. Python version
2. Required dependencies
3. Core modules can be imported
4. Example resources (DAMA)
5. Compute device availability
"""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ✗ Python 3.8 or higher required")
        return False
    else:
        print("  ✓ Python version OK")
        return True


def check_dependencies():
    """Check if required packages are installed."""
    print("\nChecking dependencies...")

    required = {
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'matplotlib': 'Matplotlib',
        'torch': 'PyTorch',
        'sklearn': 'scikit-learn',
        'tqdm': 'tqdm',
        'pymoo': 'pymoo',
        'pyDOE': 'pyDOE',
    }

    optional = {
        'umap': 'umap-learn (optional)',
        'at': 'Accelerator Toolbox (optional, for DAMA)',
    }

    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            missing.append(name)

    for module, name in optional.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ⚠ {name} - Not installed (needed for DAMA example)")

    if missing:
        print(f"\nMissing required dependencies: {', '.join(missing)}")
        print("Install with: pip install -e .")
        return False
    else:
        print("\n  All required dependencies installed")
        return True


def check_core_modules():
    """Check if core modules can be imported."""
    print("\nChecking core modules...")

    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root / 'core'))

    modules = [
        ('bax_core', 'BAX core (BAXOpt, run_bax_optimization)'),
        ('da_NN', 'Neural network models'),
    ]

    failed = []
    for module, description in modules:
        try:
            mod = __import__(module)
            # Check key components
            if module == 'bax_core':
                if not hasattr(mod, 'BAXOpt'):
                    raise AttributeError("BAXOpt class not found")
                if not hasattr(mod, 'run_bax_optimization'):
                    raise AttributeError("run_bax_optimization function not found")
            elif module == 'da_NN':
                # Check for key neural network components
                if not hasattr(mod, 'DA_Net'):
                    raise AttributeError("DA_Net class not found")
                if not hasattr(mod, 'train_NN'):
                    raise AttributeError("train_NN function not found")
            print(f"  ✓ {description}")
        except Exception as e:
            print(f"  ✗ {description} - IMPORT ERROR")
            print(f"     {str(e)[:80]}")
            failed.append(module)

    if failed:
        print(f"\nFailed to import: {', '.join(failed)}")
        return False
    else:
        print("\n  All core modules imported successfully")
        return True


def check_example_structure():
    """Check example directory structure."""
    print("\nChecking example structure...")

    project_root = Path(__file__).parent
    examples_dir = project_root / 'examples'

    expected_examples = {
        'synthetic_simple': 'Simple synthetic example',
        'synthetic': 'Synthetic with grid expansion',
        'dama': 'DAMA particle accelerator example',
    }

    all_present = True
    for example_name, description in expected_examples.items():
        example_path = examples_dir / example_name
        if example_path.exists():
            print(f"  ✓ {description} ({example_name}/)")
        else:
            print(f"  ✗ {description} ({example_name}/) - MISSING")
            all_present = False

    return all_present


def check_dama_resources():
    """Check DAMA resource files."""
    print("\nChecking DAMA resources...")

    project_root = Path(__file__).parent
    dama_dir = project_root / 'examples' / 'dama'
    resources_dir = dama_dir / 'resources'

    if not resources_dir.exists():
        print(f"  ⚠ DAMA resources directory not found")
        print(f"     Expected: {resources_dir}")
        return True  # Not critical for core functionality

    # Check subdirectories
    subdirs = {
        'matlab_data': 'MATLAB data files',
        'rings': 'Accelerator ring files',
        'setup': 'Setup configuration',
    }

    for subdir, description in subdirs.items():
        subdir_path = resources_dir / subdir
        if subdir_path.exists():
            # Count files
            files = list(subdir_path.glob('*'))
            file_count = len([f for f in files if f.is_file()])
            print(f"  ✓ {description}: {file_count} files")
        else:
            print(f"  ✗ {description} - MISSING")

    # Check specific required files
    required_files = [
        'matlab_data/mopso_run.mat',
        'matlab_data/data_setup_H6BA_10b_6var.mat',
        'matlab_data/init_config.mat',
        'matlab_data/ma_config.mat',
        'matlab_data/dpp0.mat',
        'rings/ring0.pkl',
        'setup/setup.npz',
    ]

    missing = []
    for rel_path in required_files:
        file_path = resources_dir / rel_path
        if not file_path.exists():
            missing.append(rel_path)

    if missing:
        print(f"\n  ⚠ Missing {len(missing)} required DAMA resource files:")
        for path in missing[:5]:
            print(f"     - {path}")
        if len(missing) > 5:
            print(f"     ... and {len(missing) - 5} more")
        print("\n  These are needed to run DAMA example")
        print("  See examples/dama/RESOURCE_AUDIT.md for details")
        return True  # Not critical - user can still use synthetic examples
    else:
        print("\n  ✓ All required DAMA resources present")
        return True


def check_device():
    """Check compute device availability."""
    print("\nChecking compute devices...")

    try:
        import torch

        print(f"  ✓ CPU: Available")

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"  ✓ CUDA: Available ({device_count} device(s))")
            print(f"     Device 0: {device_name}")
        else:
            print(f"  ⚠ CUDA: Not available (will use CPU)")

        return True

    except Exception as e:
        print(f"  ✗ Error checking devices: {e}")
        return False


def check_documentation():
    """Check documentation files."""
    print("\nChecking documentation...")

    project_root = Path(__file__).parent
    docs_dir = project_root / 'docs'

    expected_docs = {
        'README.md': 'Main README (root)',
        'docs/FRAMEWORK_GUIDE.md': 'Framework guide',
        'docs/API_QUICK_REFERENCE.md': 'API quick reference',
        'docs/DAMA_EXAMPLE.md': 'DAMA example walkthrough',
        'docs/CONTRIBUTING.md': 'Contributing guide',
        'examples/README.md': 'Examples guide',
    }

    all_present = True
    for rel_path, description in expected_docs.items():
        doc_path = project_root / rel_path
        if doc_path.exists():
            print(f"  ✓ {description}")
        else:
            print(f"  ✗ {description} - MISSING")
            all_present = False

    return all_present


def main():
    """Run all checks."""
    print("=" * 70)
    print("DAMA-BAX Installation Verification")
    print("=" * 70)

    checks = [
        ("Python version", check_python_version, True),
        ("Dependencies", check_dependencies, True),
        ("Core modules", check_core_modules, True),
        ("Example structure", check_example_structure, True),
        ("Documentation", check_documentation, False),
        ("DAMA resources", check_dama_resources, False),
        ("Compute devices", check_device, False),
    ]

    results = {}
    for name, check_func, critical in checks:
        try:
            results[name] = (check_func(), critical)
        except Exception as e:
            print(f"\n✗ Error during {name} check: {e}")
            import traceback
            traceback.print_exc()
            results[name] = (False, critical)

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    critical_passed = True
    for name, (passed, critical) in results.items():
        if critical:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status:8} {name} {'(critical)' if critical else ''}")
            if critical and not passed:
                critical_passed = False
        else:
            status = "✓ PASS" if passed else "⚠ WARN"
            print(f"{status:8} {name}")

    print("\n" + "=" * 70)
    if critical_passed:
        print("✓ All critical checks passed! DAMA-BAX is ready to use.")
        print("\nNext steps:")
        print("  - Try synthetic example: cd examples/synthetic_simple && python run_simple_api.py")
        print("  - Try DAMA example: cd examples/dama && python run_dama_api.py")
        print("  - Read docs/API_QUICK_REFERENCE.md for API reference")
        print("  - Read docs/FRAMEWORK_GUIDE.md for detailed guide")
    else:
        print("✗ Some critical checks failed. Please resolve the issues above.")
        print("\nFor help:")
        print("  - See README.md for installation instructions")
        print("  - Install dependencies: pip install -e .")
        print("  - Check docs/FRAMEWORK_GUIDE.md for detailed setup")
    print("=" * 70)

    return 0 if critical_passed else 1


if __name__ == "__main__":
    sys.exit(main())
