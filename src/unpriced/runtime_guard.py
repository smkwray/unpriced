from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_CACHE_ROOT = Path.home() / "venvs" / ".cache" / "unpriced"
FORBIDDEN_REPO_DIR_NAMES = {
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".cache",
}
SCAN_PRUNE_DIR_NAMES = {
    ".git",
    ".playwright-cli",
    ".playwright-mcp",
    "data",
    "output",
    "outputs",
    "tmp",
}


def configure_python_runtime() -> None:
    cache_root = default_external_cache_root()
    pycache_prefix = cache_root / "pycache"
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ.setdefault("PYTHONPYCACHEPREFIX", str(pycache_prefix))
    os.environ.setdefault("UV_CACHE_DIR", str(cache_root / "uv"))
    os.environ.setdefault("RUFF_CACHE_DIR", str(cache_root / "ruff"))
    sys.pycache_prefix = str(pycache_prefix)
    sys.dont_write_bytecode = True


def repo_root_from_package(package_file: str) -> Path:
    return Path(package_file).resolve().parents[2]


def default_external_cache_root() -> Path:
    project_venv_root = os.environ.get("PROJECT_VENV_ROOT")
    if not project_venv_root:
        return DEFAULT_CACHE_ROOT
    return Path(project_venv_root).expanduser().resolve().parent / ".cache" / "unpriced"


def find_forbidden_repo_dirs(repo_root: Path) -> list[Path]:
    repo_root = repo_root.resolve()
    offenders: list[Path] = []
    for current_root, dirnames, _filenames in os.walk(repo_root, topdown=True):
        current = Path(current_root)
        kept_dirnames: list[str] = []
        for dirname in dirnames:
            candidate = current / dirname
            if dirname in FORBIDDEN_REPO_DIR_NAMES:
                offenders.append(candidate)
                continue
            if dirname in SCAN_PRUNE_DIR_NAMES:
                continue
            kept_dirnames.append(dirname)
        dirnames[:] = kept_dirnames
    return sorted(offenders)


def enforce_no_repo_local_artifacts(repo_root: Path) -> None:
    offenders = find_forbidden_repo_dirs(repo_root)
    if not offenders:
        return
    rel = [path.relative_to(repo_root).as_posix() for path in offenders[:12]]
    more = "" if len(offenders) <= 12 else f" (+{len(offenders) - 12} more)"
    detail = ", ".join(rel) + more
    raise RuntimeError(
        "Hard blocker: repo-local env/cache artifacts detected. "
        f"Delete these directories before continuing: {detail}. "
        "Run Python entrypoints with `-B` and keep cache dirs redirected outside the repo."
    )
