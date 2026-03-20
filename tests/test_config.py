from __future__ import annotations

from pathlib import Path

from unpriced.config import load_extension_config, resolve_extension_config_path


def test_resolve_extension_config_path_uses_extensions_directory(project_paths) -> None:
    config_dir = project_paths.root / "configs" / "extensions"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "segmented_solver.yaml"
    config_path.write_text("name: segmented_solver\n", encoding="utf-8")

    resolved = resolve_extension_config_path(project_paths.root, "segmented_solver")

    assert resolved == config_path


def test_load_extension_config_accepts_relative_path(project_paths) -> None:
    config_dir = project_paths.root / "configs" / "extensions"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "custom.yaml"
    config_path.write_text("name: custom\nmode: test\n", encoding="utf-8")

    loaded = load_extension_config(project_paths.root, Path("configs/extensions/custom.yaml"))

    assert loaded["name"] == "custom"
    assert loaded["mode"] == "test"
