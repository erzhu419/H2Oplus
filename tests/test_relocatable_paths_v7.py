from pathlib import Path

import pytest

from reproducibility.relocatable_paths_v7 import (
    ArtifactPathRemap,
    resolve_evidence_path,
)


def test_v7_artifact_path_remap_preserves_relative_suffix(tmp_path):
    target = tmp_path / "bundle"
    target.mkdir()
    remap = ArtifactPathRemap.build(Path("/original/H2Oplus"), target)

    assert remap is not None
    assert remap.resolve(
        "/original/H2Oplus/bus_h2o/datasets_v7/shard.npz"
    ) == (target / "bus_h2o/datasets_v7/shard.npz").resolve()
    assert remap.resolve("/outside/evidence.json") == Path(
        "/outside/evidence.json"
    )


def test_v7_artifact_path_remap_is_explicit_and_absolute(tmp_path):
    target = tmp_path / "bundle"
    target.mkdir()
    assert ArtifactPathRemap.build(None, None) is None
    with pytest.raises(ValueError, match="both source and target"):
        ArtifactPathRemap.build(Path("/source"), None)
    with pytest.raises(ValueError, match="must be absolute"):
        ArtifactPathRemap.build(Path("relative"), target)
    with pytest.raises(ValueError, match="not absolute"):
        resolve_evidence_path("relative/file.json", None)
