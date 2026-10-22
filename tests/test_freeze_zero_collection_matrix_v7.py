import copy

import pytest

from bus_h2o.mdp_contract_v7 import DEFAULT_CONFIG_V7
from bus_h2o.full_service_trace_v7 import full_service_trace_path
from reproducibility.freeze_zero_collection_matrix_v7 import (
    EXPECTED_DEMAND_SCALES,
    _validate_exact_zero_artifact_set,
    validate_formal_zero_manifest,
)
from reproducibility.run_collection_matrix_v7 import MATRIX_SCHEMA


def _manifest():
    jobs = []
    for realization in range(10):
        for scale_index, scale in enumerate(EXPECTED_DEMAND_SCALES):
            block = realization * len(EXPECTED_DEMAND_SCALES) + scale_index
            jobs.append(
                {
                    "block_id": block,
                    "realization_block_id": realization,
                    "policy": {"policy": "zero", "policy_name": "zero"},
                    "policy_seed": 1000 + block,
                    "sumo_seed": 1101 + realization,
                    "split": "train",
                    "scenario": {
                        "demand_scale": scale,
                        "selection_seed": 1201 + realization,
                        "sha256": f"scenario-{block}",
                    },
                    "episode_seconds": int(DEFAULT_CONFIG_V7.horizon_seconds),
                    "max_events": 0,
                    "diagnostic": False,
                    "output": f"/tmp/zero-{block}.npz",
                    "source_sha256": {"source": "hash"},
                    "sumo_version": "SUMO test",
                    "signature": f"signature-{block}",
                }
            )
    return {
        "schema_version": MATRIX_SCHEMA,
        "study_stage": "formal",
        "independent_realization_blocks": 10,
        "environment_cells": 30,
        "environment_design": {
            "demand_scales": list(EXPECTED_DEMAND_SCALES),
            "evaluation_disjointness": {"verified_disjoint": True},
        },
        "behavior_mixture": {"selected_policy_names": ["zero"]},
        "source_sha256": {"source": "hash"},
        "sumo_version": "SUMO test",
        "jobs": jobs,
    }


def test_formal_zero_manifest_accepts_registered_10_by_3_grid():
    topology = validate_formal_zero_manifest(_manifest())
    assert topology["environment_cells"] == 30
    assert topology["independent_realization_blocks"] == 10
    assert topology["calibration_realization_blocks"] == list(range(5))
    assert topology["reserved_fidelity_realization_blocks"] == list(range(5, 10))


def test_formal_zero_manifest_rejects_nonzero_policy():
    manifest = _manifest()
    manifest["jobs"][0]["policy"] = {
        "policy": "headway-balance",
        "policy_name": "headway-gain-0.1",
    }
    with pytest.raises(ValueError, match="nonzero policy"):
        validate_formal_zero_manifest(manifest)


def test_formal_zero_manifest_rejects_missing_environment_cell():
    manifest = _manifest()
    manifest["jobs"] = manifest["jobs"][:-1]
    with pytest.raises(ValueError, match="exactly 30 jobs"):
        validate_formal_zero_manifest(manifest)


def test_formal_zero_manifest_rejects_seed_reuse_across_realizations():
    manifest = copy.deepcopy(_manifest())
    for job in manifest["jobs"]:
        if job["realization_block_id"] == 1:
            job["sumo_seed"] = 1101
    with pytest.raises(ValueError, match="reuse demand or SUMO seeds"):
        validate_formal_zero_manifest(manifest)


def test_exact_zero_artifact_set_rejects_unregistered_file(tmp_path):
    manifest = _manifest()
    zero_root = tmp_path / "zero"
    for job in manifest["jobs"]:
        output = zero_root / f"scenario-{job['block_id']}" / "shard.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        job["output"] = str(output)
        output.write_bytes(b"shard")
        output.with_suffix(".npz.json").write_text("{}")
        output.with_suffix(".npz.log").write_text("")
        trace = full_service_trace_path(output)
        trace.write_bytes(b"trace")
        trace.with_suffix(".npz.json").write_text("{}")
    report = _validate_exact_zero_artifact_set(manifest)
    assert report["exact_match"] is True
    assert report["observed_files"] == 150

    (zero_root / "stale.invalid-collection").write_text("stale")
    with pytest.raises(ValueError, match="artifact set is not closed"):
        _validate_exact_zero_artifact_set(manifest)
