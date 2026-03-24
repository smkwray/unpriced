from __future__ import annotations

import sys
from pathlib import Path

from unpriced.runtime_guard import (
    configure_python_runtime,
    default_external_cache_root,
    enforce_no_repo_local_artifacts,
    find_forbidden_repo_dirs,
)


def test_configure_python_runtime_sets_external_cache_defaults(monkeypatch) -> None:
    monkeypatch.delenv("PROJECT_VENV_ROOT", raising=False)
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    monkeypatch.delenv("PYTHONPYCACHEPREFIX", raising=False)
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    monkeypatch.delenv("RUFF_CACHE_DIR", raising=False)

    configure_python_runtime()

    assert default_external_cache_root().name == "unpriced"
    assert Path(str(sys.pycache_prefix)).name == "pycache"
    assert sys.dont_write_bytecode is True


def test_find_forbidden_repo_dirs_detects_nested_pycache(tmp_path: Path) -> None:
    (tmp_path / "src" / "unpriced" / "__pycache__").mkdir(parents=True)
    (tmp_path / "tests" / ".pytest_cache").mkdir(parents=True)

    offenders = find_forbidden_repo_dirs(tmp_path)

    rel = {path.relative_to(tmp_path).as_posix() for path in offenders}
    assert rel == {"src/unpriced/__pycache__", "tests/.pytest_cache"}


def test_enforce_no_repo_local_artifacts_raises_with_paths(tmp_path: Path) -> None:
    offender = tmp_path / "src" / "unpriced" / "__pycache__"
    offender.mkdir(parents=True)

    try:
        enforce_no_repo_local_artifacts(tmp_path)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected runtime guard to reject repo-local cache dirs")

    assert "Hard blocker" in message
    assert "src/unpriced/__pycache__" in message
