from __future__ import annotations

import json
from pathlib import Path

import pytest

from reproducibility import build_training_environment_design_v6 as builder
from reproducibility import run_collection_matrix_v6 as matrix


def test_training_and_evaluation_seed_sets_must_be_disjoint(tmp_path):
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "schema_version": "transit-evaluation-environment-design-v1",
                "blocks": [
                    {"demand_realization_seed": 3101, "sumo_seed": 2101}
                ],
            }
        )
    )
    evidence = builder.validate_evaluation_disjointness(
        evaluation, (1201,), (1101,)
    )
    assert evidence["verified_disjoint"] is True
    with pytest.raises(ValueError, match="seed overlap"):
        builder.validate_evaluation_disjointness(
            evaluation, (3101,), (1101,)
        )


def test_collection_design_preserves_realization_blocks_across_scales(
    tmp_path, monkeypatch
):
    manifests = []
    for seed in (1201, 1202):
        for scale in (0.75, 1.0, 1.25):
            path = tmp_path / f"seed{seed}-scale{scale}.json"
            path.write_text("{}")
            manifests.append(path)
    design = tmp_path / "training.json"
    design.write_text(
        json.dumps(
            {
                "schema_version": builder.TRAINING_DESIGN_SCHEMA,
                "independent_realization_blocks": 2,
                "environment_cells": 6,
                "demand_scales": [0.75, 1.0, 1.25],
                "evaluation_disjointness": {"verified_disjoint": True},
                "blocks": [
                    {
                        "name": "r1",
                        "demand_realization_seed": 1201,
                        "sumo_seed": 1101,
                        "scenario_manifests": [str(path) for path in manifests[:3]],
                    },
                    {
                        "name": "r2",
                        "demand_realization_seed": 1202,
                        "sumo_seed": 1102,
                        "scenario_manifests": [str(path) for path in manifests[3:]],
                    },
                ],
            }
        )
    )

    monkeypatch.setattr(matrix, "sha256", lambda path: "hash")
    payload = json.loads(design.read_text())
    payload.update(
        {
            "generator": str(
                matrix.ROOT
                / "reproducibility"
                / "build_training_environment_design_v6.py"
            ),
            "generator_sha256": "hash",
            "scenario_generator_sha256": "hash",
            "source_sha256": {},
            "depart_jitter_seconds": 120.0,
            "evaluation_disjointness": {
                "verified_disjoint": True,
                "path": str(tmp_path / "evaluation.json"),
                "sha256": "hash",
            },
        }
    )
    (tmp_path / "evaluation.json").write_text(json.dumps({"blocks": []}))
    design.write_text(json.dumps(payload))

    def fake_scenario(path: Path) -> dict:
        stem = path.stem
        seed = int(stem.split("-")[0][len("seed") :])
        scale = float(stem.split("scale")[1])
        return {
            "path": path.resolve(),
            "sha256": stem,
            "tag": stem,
            "demand_scale": scale,
            "selection_seed": seed,
            "depart_jitter_seconds": 120.0,
            "runtime_input_sha256": {stem: "hash"},
        }

    monkeypatch.setattr(matrix, "_scenario", fake_scenario)
    cells, evidence = matrix.load_environment_design(design)
    assert len(cells) == 6
    assert evidence["independent_realization_blocks"] == 2
    assert evidence["environment_cells"] == 6
    assert [cell["realization_block_id"] for cell in cells] == [0, 0, 0, 1, 1, 1]
    assert [cell["sumo_seed"] for cell in cells] == [1101] * 3 + [1102] * 3
