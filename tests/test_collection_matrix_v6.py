import json

import pytest

from reproducibility import run_collection_matrix_v6 as matrix


def _scenario(name: str, digest: str) -> dict:
    return {
        "tag": name,
        "sha256": digest,
        "runtime_input_sha256": {f"{name}.xml": digest},
    }


def test_v6_blocks_require_unique_paired_environment_realizations():
    scenarios = [_scenario("a", "a"), _scenario("b", "b")]
    blocks = matrix._blocks(scenarios, [11, 12], "paired")
    assert [(row["scenario"]["tag"], row["sumo_seed"]) for row in blocks] == [
        ("a", 11),
        ("b", 12),
    ]
    with pytest.raises(ValueError, match="equal scenario/seed counts"):
        matrix._blocks(scenarios, [11], "paired")
    with pytest.raises(ValueError, match="must be unique"):
        matrix._blocks([scenarios[0], scenarios[0]], [11, 12], "paired")


def test_v6_complete_shard_requires_exact_coverage_and_provenance(tmp_path):
    output = tmp_path / "shard.npz"
    output.write_bytes(b"immutable v6 shard")
    source_hashes = {"collector.py": "source"}
    runtime_hashes = {"scenario.xml": "runtime"}
    job = {
        "output": output,
        "signature": "exact",
        "source_sha256": source_hashes,
        "sumo_version": "SUMO 1.27.1",
        "scenario": {"runtime_input_sha256": runtime_hashes},
        "split": "train",
        "diagnostic": False,
    }
    summary_path = output.with_suffix(".npz.json")
    summary = {
        "schema_version": matrix.SHARD_SCHEMA,
        "job_signature": "exact",
        "shard_sha256": matrix.sha256(output),
        "source_sha256": source_hashes,
        "runtime": {"sumo_version": "SUMO 1.27.1"},
        "scenario": {"runtime_input_sha256": runtime_hashes},
        "split": "train",
        "diagnostic": False,
        "training_eligible": True,
        "training_reward": {
            "chronological_coverage_errors": {
                "wait_passenger_seconds": 0.0,
                "in_vehicle_passenger_seconds": 0.0,
                "reward": 1e-12,
            }
        },
    }
    summary_path.write_text(json.dumps(summary))
    assert matrix._is_complete(job)

    summary["training_reward"]["chronological_coverage_errors"]["reward"] = 1e-5
    summary_path.write_text(json.dumps(summary))
    assert not matrix._is_complete(job)
