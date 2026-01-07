"""
Pre-flight checks for AG-SAR experiments.

This module provides installation verification and project root resolution.
All experiment scripts should call check_installation() before importing ag_sar.
"""

import sys
from pathlib import Path


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to project root (directory containing pyproject.toml)
    """
    # Start from this file and walk up to find pyproject.toml
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(
        "Could not find project root (no pyproject.toml found in parent directories)"
    )


def check_installation() -> None:
    """
    Verify that ag_sar is properly installed.

    Raises:
        ImportError: If ag_sar is not installed with clear instructions

    Usage:
        from experiments.utils.preflight import check_installation
        check_installation()  # Call before importing ag_sar

        from ag_sar import AGSAR, AGSARConfig  # Now safe to import
    """
    try:
        import ag_sar
        return  # Successfully imported
    except ImportError:
        pass

    # Provide helpful error message
    project_root = None
    try:
        project_root = get_project_root()
    except RuntimeError:
        pass

    error_msg = """
================================================================================
AG-SAR is not installed. Please install it first:

    pip install -e ".[all]"

If you're in development mode, run from the project root:

    cd {root}
    pip install -e ".[all]"

Then retry your command.
================================================================================
""".format(root=project_root or "<project_root>")

    print(error_msg, file=sys.stderr)
    sys.exit(1)


def check_dependencies(required: list[str]) -> None:
    """
    Check that required dependencies are installed.

    Args:
        required: List of package names to check

    Raises:
        ImportError: If any dependency is missing
    """
    missing = []
    for pkg in required:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}", file=sys.stderr)
        print(f"Install with: pip install {' '.join(missing)}", file=sys.stderr)
        sys.exit(1)
