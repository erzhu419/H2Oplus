from __future__ import annotations

import json
from pathlib import Path

import pytest

from reproducibility import merge_evaluation_matrix_v7 as merger
from reproducibility import run_evaluation_matrix_v4 as matrix


def _metadata(methods_hash: str, methods_path: str) -> dict:
    return {
        "schema_version": matrix.MATRIX_SCHEMAS["v7"],
        "mdp_version": "v7",
        "methods_manifest": methods_path,
        "methods_manifest_sha256": methods_hash,
        "environment_design": {"sha256": "e" * 64},
        "dataset": "/data/formal.h5",
        "dataset_sha256": "d" * 64,
        "dataset_size_bytes": 123,
        "dataset_mtime_ns": 456,
        "evaluator": "/code/evaluate_transit_v7.py",
        "evaluator_sha256": "a" * 64,
        "runner_sha256": "b" * 64,
        "source_sha256": {"/code/source.py": "c" * 64},
        "sumo_version": "1.25.0",
        "sumo_backend": "libsumo",
        "sumo_runtime_signature": "5" * 64,
        "sumo_runtime_lock": {"sha256": "6" * 64},
        "workers": 4,
    }


def _job(tmp_path: Path, method: str, algorithm: str, cell: int) -> dict:
    return {
        "method": {
            "name": method,
            "algorithm": algorithm,
            "training_seed": 5101,
            "type": "checkpoint",
            "checkpoint": f"/checkpoints/{method}.pt",
            "checkpoint_sha256": ("1" if algorithm == "h2oplus" else "2") * 64,
        },
        "mdp_version": "v7",
        "evaluator": "/code/evaluate_transit_v7.py",
        "result_schema": "transit-evaluation-v7",
        "scenario": {
            "path": f"/scenario/{cell}/manifest.json",
            "tag": f"demand-cell-{cell}",
            "manifest_sha256": f"{cell:064x}",
            "runtime_input_sha256": {f"/scenario/{cell}/input.xml": f"{cell + 8:064x}"},
        },
        "environment_block": f"environment-{cell}",
        "sumo_seed": 2100 + cell,
        "dataset": "/data/formal.h5",
        "dataset_sha256": "d" * 64,
        "dataset_size_bytes": 123,
        "dataset_mtime_ns": 456,
        "device": "cpu",
        "max_events": 0,
        "max_sim_time_seconds": 0,
        "sumo_dir": "/sumo",
        "edge_xml": "/code/edge.xml",
        "output": str(tmp_path / "outputs" / method / f"cell-{cell}.json"),
        "event_log": None,
        "source_sha256": {"/code/source.py": "c" * 64},
        "sumo_version": "1.25.0",
        "sumo_backend": "libsumo",
        "sumo_runtime_version": "1.25.0",
        "sumo_runtime_signature": "5" * 64,
        "sumo_runtime_lock": "/runtime/runtime_lock.json",
        "sumo_runtime_lock_sha256": "6" * 64,
        "signature": f"{algorithm}-{cell}",
    }


def _write_group(
    tmp_path: Path,
    *,
    group: str,
    method: str,
    algorithm: str,
    cells: tuple[int, ...] = (1, 2),
) -> list[Path]:
    methods_hash = ("3" if algorithm == "h2oplus" else "4") * 64
    metadata = _metadata(methods_hash, f"/methods/{group}.json")
    jobs = [_job(tmp_path, method, algorithm, cell) for cell in cells]
    count = min(2, len(jobs))
    paths = []
    for index in range(count):
        selected, partition = matrix.partition_jobs(
            jobs, metadata, shard_count=count, shard_index=index
        )
        payload = {
            **metadata,
            "jobs": selected,
            "matrix_partition": partition,
        }
        path = tmp_path / f"{group}-shard-{index}.json"
        path.write_text(json.dumps(payload))
        paths.append(path)
    return paths


def test_merge_complete_partition_groups_and_execution_classes(tmp_path):
    manifests = [
        *_write_group(
            tmp_path,
            group="server-h2oplus",
            method="h2oplus-seed5101",
            algorithm="h2oplus",
        ),
        *_write_group(
            tmp_path,
            group="server",
            method="rlpd-seed5101",
            algorithm="rlpd",
        ),
    ]

    destination = merger.merge_manifests(
        manifests, tmp_path / "merged", verify_outputs=False
    )
    payload = json.loads(destination.read_text())
    assert len(payload["jobs"]) == 4
    assert payload["matrix_merge"]["method_count"] == 2
    assert payload["matrix_merge"]["environment_cells_per_method"] == 2
    assert payload["matrix_merge"]["all_outputs_verified"] is False
    assert len(payload["methods_manifests"]) == 2


def test_merge_allows_distinct_checkpoints_from_same_algorithm_and_seed(tmp_path):
    manifests = [
        *_write_group(
            tmp_path,
            group="fewshot-10000",
            method="h2oplus-seed5101-events000010000",
            algorithm="h2oplus",
        ),
        *_write_group(
            tmp_path,
            group="fewshot-25000",
            method="h2oplus-seed5101-events000025000",
            algorithm="h2oplus",
        ),
    ]

    destination = merger.merge_manifests(
        manifests, tmp_path / "merged", verify_outputs=False
    )
    payload = json.loads(destination.read_text())
    assert len(payload["jobs"]) == 4
    assert payload["matrix_merge"]["method_count"] == 2
    assert payload["matrix_merge"]["environment_cells_per_method"] == 2


def test_merge_rejects_missing_partition(tmp_path):
    manifests = _write_group(
        tmp_path,
        group="server",
        method="rlpd-seed5101",
        algorithm="rlpd",
    )
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        merger.merge_manifests(
            manifests[:1], tmp_path / "merged", verify_outputs=False
        )


def test_merge_rejects_inconsistent_method_coverage(tmp_path):
    manifests = [
        *_write_group(
            tmp_path,
            group="server-h2oplus",
            method="h2oplus-seed5101",
            algorithm="h2oplus",
        ),
        *_write_group(
            tmp_path,
            group="server",
            method="rlpd-seed5101",
            algorithm="rlpd",
            cells=(1,),
        ),
    ]
    with pytest.raises(ValueError, match="complete environment design"):
        merger.merge_manifests(
            manifests, tmp_path / "merged", verify_outputs=False
        )
