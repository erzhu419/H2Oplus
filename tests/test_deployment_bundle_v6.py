from pathlib import Path

import pytest

from reproducibility.deployment_bundle_v6 import (
    TOOLCHAIN_INCLUDE,
    _resolve_manifest_runtime_environment,
)


def _deployment_root(path: Path) -> Path:
    include = path / TOOLCHAIN_INCLUDE
    include.mkdir(parents=True)
    (include / "crypt.h").write_text("/* pinned header */\n")
    return path


def test_manifest_runtime_environment_relocates_package_cpath(tmp_path):
    first = _deployment_root(tmp_path / "first")
    second = _deployment_root(tmp_path / "second")
    manifest = {
        "runtime_environment": {
            "CC": "/bin/true",
            "CPATH": TOOLCHAIN_INCLUDE.as_posix(),
        }
    }

    first_environment = _resolve_manifest_runtime_environment(first, manifest)
    second_environment = _resolve_manifest_runtime_environment(second, manifest)

    assert first_environment["CC"] == second_environment["CC"]
    assert first_environment["CPATH"] != second_environment["CPATH"]
    assert first_environment["CPATH"] == str((first / TOOLCHAIN_INCLUDE).resolve())


def test_manifest_runtime_environment_rejects_escaping_cpath(tmp_path):
    root = _deployment_root(tmp_path / "deployment")
    manifest = {
        "runtime_environment": {
            "CC": "/bin/true",
            "CPATH": "../outside",
        }
    }
    with pytest.raises(ValueError, match="package-relative"):
        _resolve_manifest_runtime_environment(root, manifest)
