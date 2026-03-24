"""unpriced package."""

from unpriced.runtime_guard import (
    configure_python_runtime,
    enforce_no_repo_local_artifacts,
    repo_root_from_package,
)

configure_python_runtime()
enforce_no_repo_local_artifacts(repo_root_from_package(__file__))

__all__ = ["__version__"]

__version__ = "0.1.0"
