import json

from reproducibility import run_collection_matrix_v4 as matrix


def test_policy_seed_does_not_depend_on_selected_policy_subset():
    assert matrix.policy_seed(100, "headway-gain-0.1", 7) == 30_107
    assert matrix.policy_seed(100, "headway-gain-0.1", 7) == matrix.policy_seed(
        100, matrix.POLICIES[3]["policy_name"], 7
    )


def test_complete_shard_requires_signature_source_runtime_and_hash(tmp_path):
    output = tmp_path / "shard.npz"
    output.write_bytes(b"immutable shard")
    source_hashes = {"collector.py": "source"}
    runtime_hashes = {"scenario.xml": "runtime"}
    job = {
        "output": output,
        "signature": "exact",
        "source_sha256": source_hashes,
        "sumo_version": "SUMO 1.25.0",
        "scenario": {"runtime_input_sha256": runtime_hashes},
    }
    summary_path = output.with_suffix(".npz.json")
    summary = {
        "schema_version": "transit-smdp-v4-shard-v1",
        "job_signature": "exact",
        "shard_sha256": matrix.sha256(output),
        "source_sha256": source_hashes,
        "runtime": {"sumo_version": "SUMO 1.25.0"},
        "scenario": {"runtime_input_sha256": runtime_hashes},
    }
    summary_path.write_text(json.dumps(summary))
    assert matrix._is_complete(job)

    summary["source_sha256"] = {"collector.py": "changed"}
    summary_path.write_text(json.dumps(summary))
    assert not matrix._is_complete(job)
