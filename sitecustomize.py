from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GUARD_PATH = ROOT / "src" / "unpriced" / "runtime_guard.py"

if GUARD_PATH.is_file():
    spec = importlib.util.spec_from_file_location("_unpriced_runtime_guard", GUARD_PATH)
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.configure_python_runtime()
