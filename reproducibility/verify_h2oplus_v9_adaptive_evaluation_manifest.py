#!/usr/bin/env python3
"""Independently verify the H2O+ v9 adaptive evaluation file manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reproducibility/h2oplus_v9_adaptive_results"
PARTITIONS = RESULTS / "evaluation_partitions_v3"
CANONICAL = RESULTS / "evaluation_canonical_v3"
DEFAULT_MANIFEST = RESULTS / "evaluation_file_manifest_v3.json"
DEFAULT_OUTPUT = RESULTS / "evaluation_file_manifest_verification_v3.json"

EXPECTED_CONTROLS = {
    "reproducibility/h2oplus_v9_adaptive_registration_v2.json": "f4f6922324ef518acde9b5489328bf13c7154012dfd6f4cc2b61797adb90785e",
    "reproducibility/h2oplus_v9_adaptive_protocol_v2.json": "e0652db511c5ca46e0e14801895bdfdc57761223842a4e30673075b9163d3c7d",
    "reproducibility/h2oplus_v9_adaptive_evaluation_count_correction_v1.json": "4fc9324b140f6af387a4b8011b40c1d6994c57636d909ca065b17d4d0e27c6ba",
    "reproducibility/h2oplus_v9_adaptive_evaluation_portability_correction_v1.json": "b5063671012b48c55f537ab5f017356ff2a72229f59aa74b5eabf67214842d23",
    "reproducibility/h2oplus_v9_adaptive_evaluation_source_correction_v2.json": "2e28bd6dfe71e3d1adf82866137da0a8a678048c7a6a91aaa00df8b39e782f74",
    "reproducibility/h2oplus_v9_adaptive_evaluation_source_freeze_v2.json": "f717bebbe0ed838c104ebb716161fcc7d9be79b64ed9ac3b207ba638139a37cc",
    "reproducibility/h2oplus_v9_adaptive_evaluation_methods_v2.json": "e1c83b52c78dc60d4a8da8888f3b2a0d82f2c6ea77ca4b9d372d56a33eb58f0c",
    "reproducibility/h2oplus_v9_adaptive_evaluation_submission_v3.json": "21088cc14d0b4714ba1a311e14638f4107e6f7c46eea537111afc35d602403df",
    "reproducibility/deployments/h2oplus_v9_adaptive_evaluation_v3/workspace/deployment_manifest.json": "64180676bf222d8cf8877c9a3c3d80b8f7aa86d60ee2a60e042bc28b242ff0e3",
    "reproducibility/h2oplus_v9_adaptive_results/training_closeout.json": "fc263b3d51ac5a532bd13dc3fdaecf43153e1336d9d57a4ce6b87e6eac18bafd",
    "reproducibility/h2oplus_v9_adaptive_results/training_audit.json": "f126301d52ccdc665c8c1fdc20611c5f87d7879fcce14296d4984b47de997959",
    "reproducibility/h2oplus_v9_adaptive_results/training_file_manifest.json": "010ddf60b394230cdeb28edd32e4df52d189d754c1a2a2301e9c40f0ed46f4ce",
    "reproducibility/h2oplus_v9_adaptive_results/training_file_manifest_verification.json": "9926dea64d692619808c2a2e03382e7620fd71358223119ac066d217ceff080e",
}
EXPECTED_DATASET = "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899"
EXPECTED_RUNTIME = "264d3c412b8e6c42320995208d2e009eb809b85626047206c4de752c2dae06e7"
BLOCKS = ("v9-adaptive-development-01", "v9-adaptive-development-02")
SCALES = {0.85, 1.0, 1.15}
RESULT_SOURCE_SUFFIXES = {
    "SimpleSAC/evaluate_transit_v9_adaptive.py",
    "reproducibility/run_evaluation_matrix_v9_adaptive.py",
}
METHOD_BINDING_FIELDS = {
    "actor_tensor_sha256",
    "actual_online_events",
    "algorithm",
    "artifact_lineage",
    "learner_update_steps",
    "name",
    "online_events",
    "ratio_optimizer_steps",
    "training_admission_sha256",
    "training_checkpoint_sha256",
    "training_run_signature",
    "training_seed",
    "type",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite(item) for item in value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    return False


def aggregate(records: list[dict]) -> str:
    digest = hashlib.sha256()
    for item in sorted(records, key=lambda row: row["path"]):
        digest.update(item["path"].encode())
        digest.update(b"\0")
        digest.update(str(item["size_bytes"]).encode())
        digest.update(b"\0")
        digest.update(item["sha256"].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _suffix_hash(source: dict, suffix: str) -> str | None:
    values = [value for path, value in source.items() if path.endswith(suffix)]
    return values[0] if len(values) == 1 else None


def _validate_result(
    job: dict,
    path: Path,
    method: dict,
    source_reference: dict | None,
    frozen_sources: dict[str, str],
) -> dict:
    payload = json.loads(path.read_text())
    job_method = job.get("method", {})
    source = payload.get("source_sha256", {})
    training = payload.get("policy", {}).get("training_metadata", {})
    coverage = payload.get("passenger_welfare", {}).get(
        "chronological_coverage_errors", {}
    )
    if (
        payload.get("schema_version") != "transit-evaluation-v7"
        or payload.get("evaluation_job", {}).get("signature") != job.get("signature")
        or payload.get("method") != method.get("name")
        or payload.get("sumo_seed") != job.get("sumo_seed")
        or not finite(payload)
        or source != job.get("source_sha256")
        or len(source) != 41
        or any(
            _suffix_hash(source, suffix) != frozen_sources[suffix]
            for suffix in RESULT_SOURCE_SUFFIXES
        )
        or any(job_method.get(key) != method.get(key) for key in METHOD_BINDING_FIELDS)
        or job.get("dataset_sha256") != EXPECTED_DATASET
        or payload.get("dataset", {}).get("sha256") != EXPECTED_DATASET
        or job.get("sumo_runtime_signature") != EXPECTED_RUNTIME
        or payload.get("runtime", {}).get("sumo_runtime_signature") != EXPECTED_RUNTIME
        or payload.get("policy", {}).get("sha256")
        != job_method.get("checkpoint_sha256")
        or payload.get("policy", {}).get("actor_tensor_sha256")
        != method.get("actor_tensor_sha256")
        or training.get("study_stage") != "development"
        or training.get("formal_held_out_outcomes_read") is not False
        or training.get("run_signature") != method.get("training_run_signature")
        or training.get("training_checkpoint_sha256")
        != method.get("training_checkpoint_sha256")
        or training.get("training_admission", {}).get("sha256")
        != method.get("training_admission_sha256")
        or not coverage
        or any(abs(float(value)) > 1e-9 for value in coverage.values())
    ):
        raise ValueError(f"independent v9 result provenance check failed: {path}")
    if source_reference is not None and source != source_reference:
        raise ValueError(f"source map differs across sealed v9 results: {path}")
    return source


def verify(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version")
        != "transit-h2oplus-v9-adaptive-evaluation-file-manifest-v3"
    ):
        raise ValueError("unexpected v9 evaluation manifest schema")
    records = manifest.get("files", [])
    record_paths = {row["path"] for row in records}
    if len(records) != manifest.get("file_count") or len(record_paths) != len(records):
        raise ValueError("file records are incomplete or duplicated")
    total = 0
    for record in records:
        path = (REPO / record["path"]).resolve()
        if not path.is_file() or REPO not in path.parents:
            raise ValueError(f"missing or out-of-root file: {record['path']}")
        if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"file identity mismatch: {record['path']}")
        total += path.stat().st_size
    if (
        total != manifest.get("total_size_bytes")
        or aggregate(records) != manifest.get("file_records_aggregate_sha256")
    ):
        raise ValueError("file record aggregate mismatch")
    for relative, digest in EXPECTED_CONTROLS.items():
        if sha256(REPO / relative) != digest or relative not in record_paths:
            raise ValueError(f"sealed control drifted or is absent: {relative}")

    artifact = manifest.get("artifact_integrity", {})
    if (
        artifact.get("passed") is not True
        or artifact.get("result_count") != 252
        or artifact.get("unique_job_signature_count") != 252
        or artifact.get("canonical_matrix_count") != 2
        or artifact.get("source_manifest_count") != 4
        or artifact.get("corrected_source_file_count") != 41
        or artifact.get("physical_method_count") != 42
        or manifest.get("study_stage") != "development"
        or manifest.get("formal_held_out_outcomes_read") is not False
        or manifest.get("supports_current_formal_held_out_model_effect_claim") is not False
        or manifest.get("scientific_analysis", {}).get("completed") is not False
    ):
        raise ValueError("artifact/scientific scope summary failed")

    audit = json.loads((RESULTS / "evaluation_audit_v3.json").read_text())
    if (
        audit.get("passed") is not True
        or audit.get("artifact_integrity", {}).get("passed") is not True
        or audit.get("scientific_results", {}).get("analysis_completed") is not False
        or audit.get("training_stability", {}).get(
            "adaptation_seed6202_late_stability_passed"
        )
        is not False
        or audit.get("training_stability", {}).get("intervention_attribution_available")
        is not False
    ):
        raise ValueError("evaluation audit did not preserve integrity/stability separation")

    scheduler = json.loads(
        (RESULTS / "evidence/evaluation_v3/scheduler_evidence_v3.json").read_text()
    )
    if (
        scheduler.get("passed") is not True
        or scheduler.get("token_bound_backend_exit0") is not True
        or set(scheduler.get("tasks", {})) != {"t82575", "t82576"}
    ):
        raise ValueError("scheduler evidence failed")
    for task_id, item in scheduler["tasks"].items():
        status = RESULTS / f"evidence/evaluation_v3/backend_exit_{task_id}.status"
        fields = status.read_text().strip().split("\t")
        if (
            len(fields) != 3
            or fields[0] != "0"
            or fields[2] != item.get("exit_status_token")
            or item.get("queue_status") != "done"
            or item.get("exit_code") != 0
        ):
            raise ValueError(f"independent token-bound exit check failed: {task_id}")

    methods_payload = json.loads(
        (REPO / "reproducibility/h2oplus_v9_adaptive_evaluation_methods_v2.json").read_text()
    )
    methods = {item["name"]: item for item in methods_payload["methods"]}
    source_freeze = json.loads(
        (REPO / "reproducibility/h2oplus_v9_adaptive_evaluation_source_freeze_v2.json").read_text()
    )
    frozen_sources = {item["path"]: item["sha256"] for item in source_freeze["sources"]}
    if len(methods) != 42 or len(frozen_sources) != 4:
        raise ValueError("independent methods/source-freeze check failed")

    raw_files = {
        str(path.resolve().relative_to(REPO))
        for path in PARTITIONS.rglob("*")
        if path.is_file()
    }
    if not raw_files.issubset(record_paths):
        raise ValueError("not every synchronized v9 partition file is sealed")

    partition_signatures: set[str] = set()
    source_reference = None
    source_manifest_hashes: set[str] = set()
    for shard_index, shard in enumerate(("shard0_jtl110gpu", "shard1_jtl110gpu")):
        root = PARTITIONS / shard
        host = json.loads((root / "host_execution.json").read_text())
        results = list(root.rglob("sumo_seed*.json"))
        matrices = sorted(root.rglob("matrix_manifest.shard-*.json"))
        if (
            host.get("status") != "completed"
            or host.get("returncode") != 0
            or host.get("shard_index") != shard_index
            or host.get("shard_count") != 2
            or len(results) != 126
            or len(matrices) != 2
        ):
            raise ValueError(f"independent source partition check failed: {shard}")
        for matrix_path in matrices:
            matrix = json.loads(matrix_path.read_text())
            jobs = matrix.get("jobs", [])
            partition = matrix.get("matrix_partition", {})
            if (
                len(jobs) != 63
                or partition.get("shard_index") != shard_index
                or partition.get("shard_count") != 2
                or partition.get("full_job_count") != 126
                or len(matrix.get("source_sha256", {})) != 41
            ):
                raise ValueError(f"independent partition matrix check failed: {matrix_path}")
            source_manifest_hashes.add(sha256(matrix_path))
            for job in jobs:
                name = str(job.get("method", {}).get("name"))
                method = methods.get(name)
                if method is None:
                    raise ValueError(f"unregistered physical method: {name}")
                output = Path(job["output"]).expanduser().resolve()
                if not output.is_file() or PARTITIONS.resolve() not in output.parents:
                    raise ValueError(f"missing or out-of-root partition output: {output}")
                source_reference = _validate_result(
                    job, output, method, source_reference, frozen_sources
                )
                signature = str(job["signature"])
                if signature in partition_signatures:
                    raise ValueError("duplicate result signature in source partitions")
                partition_signatures.add(signature)
    if (
        len(partition_signatures) != 252
        or len(source_manifest_hashes) != 4
        or source_reference is None
    ):
        raise ValueError("independent partition union check failed")

    canonical_signatures: set[str] = set()
    expected_cells = {(name, scale) for name in methods for scale in SCALES}
    for block in BLOCKS:
        path = CANONICAL / block / "matrix_manifest.json"
        matrix = json.loads(path.read_text())
        jobs = matrix.get("jobs", [])
        merge = matrix.get("matrix_merge", {})
        cells = set()
        if (
            len(jobs) != 126
            or merge.get("all_outputs_verified") is not True
            or merge.get("method_count") != 42
            or merge.get("environment_cells_per_method") != 3
            or len(merge.get("source_manifests", [])) != 2
        ):
            raise ValueError(f"independent canonical matrix check failed: {block}")
        observed_sources = {
            descriptor.get("sha256") for descriptor in merge["source_manifests"]
        }
        if not observed_sources.issubset(source_manifest_hashes):
            raise ValueError(f"canonical source-manifest binding failed: {block}")
        for job in jobs:
            name = str(job.get("method", {}).get("name"))
            scale = float(job["scenario"]["demand_scale"])
            cells.add((name, scale))
            signature = str(job["signature"])
            if name not in methods or signature in canonical_signatures:
                raise ValueError("invalid or duplicate canonical v9 signature")
            canonical_signatures.add(signature)
        if cells != expected_cells:
            raise ValueError(f"canonical Cartesian coverage failed: {block}")
    if canonical_signatures != partition_signatures:
        raise ValueError("canonical and synchronized source unions differ")

    return {
        "schema_version": (
            "transit-h2oplus-v9-adaptive-evaluation-file-manifest-verification-v3"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": {
            "file_records_aggregate_sha256": manifest[
                "file_records_aggregate_sha256"
            ],
            "path": str(manifest_path.relative_to(REPO)),
            "sha256": sha256(manifest_path),
        },
        "independent_artifact_audit": {
            "all_numeric_values_finite": True,
            "canonical_matrix_count": 2,
            "corrected_source_file_count": 41,
            "file_count": len(records),
            "passed": True,
            "physical_method_count": 42,
            "result_count": 252,
            "source_manifest_count": 4,
            "source_partition_count": 2,
            "total_size_bytes": total,
            "unique_job_signature_count": 252,
        },
        "training_stability": {
            "adaptation_seed6202_late_stability_passed": False,
            "intervention_attribution_available": False,
        },
        "scientific_analysis_completed": False,
        "formal_confirmation_authorized_or_submitted": False,
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "supports_current_formal_held_out_model_effect_claim": False,
        "passed": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest.expanduser().resolve(strict=True)
    payload = verify(manifest)
    output = args.output.expanduser().resolve()
    atomic_json(output, payload)
    print(output)
    print(payload["manifest"]["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
