#!/usr/bin/env python
"""
Verify DAMA-BAX installation.

This script checks:
1. Python version
2. Required dependencies
3. Resource files
4. Import of core modules
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
        'umap': 'umap-learn',
        'at': 'Accelerator Toolbox (at)',
    }

    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            missing.append(name)

    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: uv sync")
        return False
    else:
        print("  All dependencies installed")
        return True


def check_core_modules():
    """Check if core modules can be imported."""
    print("\nChecking core modules...")

    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    modules = [
        'core.config',
        'core.da_NN',
        'core.da_utils',
        'core.da_ssrl',
        'core.da_virtual_opt',
        'core.run_utils',
        'core.dama_utils',
    ]

    failed = []
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except Exception as e:
            print(f"  ✗ {module} - IMPORT ERROR")
            print(f"     {str(e)[:80]}")
            failed.append(module)

    if failed:
        print(f"\nFailed to import: {', '.join(failed)}")
        return False
    else:
        print("  All core modules imported successfully")
        return True


def check_resources():
    """Check resource files."""
    print("\nChecking resource files...")

    try:
        from core.config import check_resources, RESOURCE_DIR

        print(f"  Resource directory: {RESOURCE_DIR}")

        status = check_resources()
        missing = [name for name, exists in status.items() if not exists]
        present = [name for name, exists in status.items() if exists]

        print(f"  Present: {len(present)}/{len(status)} files")

        if missing:
            print(f"  ✗ Missing {len(missing)} files")
            print("\n  Missing files:")
            for name in missing[:10]:  # Show first 10
                print(f"    - {name}")
            if len(missing) > 10:
                print(f"    ... and {len(missing) - 10} more")
            print("\n  Run 'python setup_resources.py' for details")
            return False
        else:
            print("  ✓ All resource files present")
            return True

    except Exception as e:
        print(f"  ✗ Error checking resources: {e}")
        return False


def check_device():
    """Check compute device availability."""
    print("\nChecking compute devices...")

    try:
        import torch

        print(f"  CPU: Available")

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


def main():
    """Run all checks."""
    print("=" * 70)
    print("DAMA-BAX Installation Verification")
    print("=" * 70)

    checks = [
        ("Python version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Core modules", check_core_modules),
        ("Resource files", check_resources),
        ("Compute devices", check_device),
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n✗ Error during {name} check: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {name}")

    all_passed = all(results.values())

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All checks passed! DAMA-BAX is ready to use.")
        print("\nNext steps:")
        print("  - Review configuration in config.example.yaml")
        print("  - Run: python run.py")
        print("  - Or submit job: sbatch job.sh")
    else:
        print("✗ Some checks failed. Please resolve the issues above.")
        print("\nFor help:")
        print("  - See README.md for installation instructions")
        print("  - See RESOURCES.md for resource file details")
        print("  - Run: python setup_resources.py")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
