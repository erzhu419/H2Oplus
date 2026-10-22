import json

import pytest

from reproducibility.build_evaluation_environment_design_v4 import (
    _validate_design_inputs,
    _write_design,
)


def test_environment_design_rejects_seed_leakage_and_pseudoreplication():
    with pytest.raises(ValueError, match="forbidden training seeds"):
        _validate_design_inputs((3101,), (2101,), (1.0,), {3101}, set())
    with pytest.raises(ValueError, match="SUMO seeds must be unique"):
        _validate_design_inputs(
            (3101, 3102), (2101, 2101), (0.75, 1.0), set(), set()
        )
    with pytest.raises(ValueError, match="demand scales"):
        _validate_design_inputs(
            (3101,), (2101,), (1.0, 1.0), set(), set()
        )


def test_environment_design_is_idempotent_and_refuses_silent_replacement(tmp_path):
    path = tmp_path / "design.json"
    payload = {"schema_version": "test", "blocks": [{"name": "one"}]}
    _write_design(path, payload, force=False)
    first = path.read_bytes()
    _write_design(path, payload, force=False)
    assert path.read_bytes() == first

    with pytest.raises(FileExistsError, match="different environment design"):
        _write_design(
            path,
            {"schema_version": "test", "blocks": [{"name": "two"}]},
            force=False,
        )
    _write_design(
        path,
        {"schema_version": "test", "blocks": [{"name": "two"}]},
        force=True,
    )
    assert json.loads(path.read_text())["blocks"] == [{"name": "two"}]
