#!/usr/bin/env python3
"""Independently verify the v15 hold-calibration evaluation file manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reproducibility/h2oplus_v15_hold_calibration_results"
PARTITION = RESULTS / "evaluation_partition_v1/single_jtl110gpu"
MANIFEST = RESULTS / "evaluation_file_manifest_v1.json"
OUTPUT = RESULTS / "evaluation_file_manifest_verification_v1.json"
METHODS = REPO / "reproducibility/h2oplus_v15_hold_calibration_evaluation_methods_v1.json"
SOURCE_SYNC = RESULTS / "evidence/evaluation_v1/source_sync_v1.json"
SCHEDULER = RESULTS / "evidence/evaluation_v1/scheduler_evidence_v1.json"
EXPECTED_METHODS_SHA256 = (
    "7712a35cbb1adbc55acf50bd8843bb3f4f7dc17090d0e65deeeb10ffe5177e5a"
)
EXPECTED_SOURCE_SYNC_SHA256 = (
    "b15b9dac18f429bc566f3f5b9f8cec5628e3d725d2b0b695efcb8843693c5a72"
)
EXPECTED_SCHEDULER_SHA256 = (
    "1cdc2569924549b061ce3e245ea08ae0ba0ede33276c8d63902807cf079467d7"
)
EXPECTED_DATASET_SHA256 = (
    "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899"
)
EXPECTED_RUNTIME_SIGNATURE = (
    "264d3c412b8e6c42320995208d2e009eb809b85626047206c4de752c2dae06e7"
)
EXPECTED_BACKEND_TOKEN = "1787553772281703586:1528677:126605756510208"


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
    for record in sorted(records, key=lambda item: item["path"]):
        digest.update(record["path"].encode())
        digest.update(b"\0")
        digest.update(str(record["size_bytes"]).encode())
        digest.update(b"\0")
        digest.update(record["sha256"].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def resolve_record(record: dict) -> Path:
    path = (REPO / record.get("path", "")).resolve()
    if (
        not path.is_file()
        or REPO not in path.parents
        or path.stat().st_size != record.get("size_bytes")
        or sha256(path) != record.get("sha256")
    ):
        raise ValueError(f"v15 manifest file record failed: {record}")
    return path


def verify_results() -> dict:
    results = sorted(PARTITION.rglob("sumo_seed*.json"))
    if len(results) != 135:
        raise ValueError("independent v15 result count failed")
    signatures: set[str] = set()
    methods: set[str] = set()
    source_reference: dict | None = None
    maximum_coverage_error = 0.0
    maximum_reward_reconstruction_error = 0.0
    for path in results:
        payload = json.loads(path.read_text())
        signature = payload.get("evaluation_job", {}).get("signature")
        source = payload.get("source_sha256", {})
        service = payload.get("service", {})
        semantics = payload.get("bus_service_semantics_audit", {})
        violations = semantics.get("violations", {})
        ledger = payload.get("ledger_audit", {})
        count_audit = ledger.get("count_source_audit", {})
        passengers = payload.get("passengers", {})
        welfare = payload.get("passenger_welfare", {})
        coverage = welfare.get("chronological_coverage_errors", {})
        if coverage:
            maximum_coverage_error = max(
                maximum_coverage_error,
                *(abs(float(value)) for value in coverage.values()),
            )
        maximum_reward_reconstruction_error = max(
            maximum_reward_reconstruction_error,
            abs(float(welfare.get("maximum_reward_reconstruction_error", math.inf))),
        )
        if (
            payload.get("schema_version") != "transit-evaluation-v7"
            or not finite(payload)
            or not isinstance(signature, str)
            or len(signature) != 64
            or signature in signatures
            or len(source) != 49
            or payload.get("dataset", {}).get("sha256") != EXPECTED_DATASET_SHA256
            or payload.get("runtime", {}).get("sumo_runtime_signature")
            != EXPECTED_RUNTIME_SIGNATURE
            or semantics.get("passed") is not True
            or semantics.get("require_all_departed_trips_completed") is not True
            or any(value not in ([], {}) for value in violations.values())
            or service.get("active_trips_at_horizon") != 0
            or service.get("completed_trips") != service.get("departed_trips")
            or service.get("completion_rate") != 1.0
            or ledger.get("legacy_sampled_trip_histories_used_as_ground_truth")
            is not False
            or count_audit.get("max_in_vehicle_count_error") != 0
            or count_audit.get("max_waiting_count_error") != 0
            or passengers.get("legacy_sampled_trip_histories_used_for_welfare")
            is not False
            or passengers.get("active_state_count_balance_error") != 0
            or welfare.get("primary") is not True
            or not coverage
            or maximum_coverage_error > 1e-9
            or maximum_reward_reconstruction_error > 1e-9
            or payload.get("policy", {}).get("hold_calibration", {}).get(
                "formal_held_out_outcomes_read"
            )
            is not False
            or payload.get("policy", {}).get("hold_calibration", {}).get(
                "checkpoint_selection_by_behavior"
            )
            is not False
            or payload.get("policy", {}).get("training_metadata", {}).get(
                "formal_held_out_outcomes_read"
            )
            is not False
        ):
            raise ValueError(f"independent v15 result audit failed: {path}")
        if source_reference is not None and source != source_reference:
            raise ValueError(f"independent v15 source-map equality failed: {path}")
        source_reference = source
        signatures.add(signature)
        methods.add(str(payload.get("method")))
    if len(signatures) != 135 or len(methods) != 15 or source_reference is None:
        raise ValueError("independent v15 behavior/signature cardinality failed")
    return {
        "all_numeric_values_finite": True,
        "all_service_semantics_passed": True,
        "all_welfare_ledgers_passed": True,
        "maximum_welfare_chronological_coverage_error": maximum_coverage_error,
        "maximum_reward_reconstruction_error": maximum_reward_reconstruction_error,
        "physical_behavior_count": len(methods),
        "result_count": len(results),
        "source_file_count": len(source_reference),
        "unique_job_signature_count": len(signatures),
    }


def verify_checkpoints() -> dict:
    if sha256(METHODS) != EXPECTED_METHODS_SHA256:
        raise ValueError("independent v15 methods identity failed")
    payload = json.loads(METHODS.read_text())
    checkpoints: dict[Path, str] = {}
    behaviors: set[tuple[str, int, str, float]] = set()
    for method in payload.get("methods", []):
        checkpoint = Path(method["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = METHODS.parent / checkpoint
        checkpoint = checkpoint.resolve(strict=True)
        expected = method.get("checkpoint_sha256")
        previous = checkpoints.setdefault(checkpoint, expected)
        behavior = (
            method.get("actor_tensor_sha256"),
            int(method.get("training_seed", -1)),
            str(method.get("algorithm")),
            float(method.get("hold_scale", math.nan)),
        )
        if previous != expected or sha256(checkpoint) != expected or behavior in behaviors:
            raise ValueError(f"independent v15 checkpoint/behavior audit failed: {checkpoint}")
        behaviors.add(behavior)
    if len(behaviors) != 15 or len(checkpoints) != 5:
        raise ValueError("independent v15 checkpoint cardinality failed")
    return {
        "physical_behaviors_verified": len(behaviors),
        "unique_parent_checkpoints_rehashed": len(checkpoints),
    }


def verify_sync_and_scheduler() -> None:
    if sha256(SOURCE_SYNC) != EXPECTED_SOURCE_SYNC_SHA256:
        raise ValueError("independent v15 source-sync identity failed")
    sync = json.loads(SOURCE_SYNC.read_text())
    rows = sync.get("partition", {}).get("files", [])
    local = {
        str(path.relative_to(PARTITION)): path
        for path in PARTITION.rglob("*")
        if path.is_file()
    }
    if (
        sync.get("passed") is not True
        or sync.get("local_sumo_executed") is not False
        or sync.get("formal_held_out_outcomes_read") is not False
        or len(rows) != 290
        or {row.get("path") for row in rows} != set(local)
        or any(
            row.get("bytes") != local[row["path"]].stat().st_size
            or row.get("sha256") != sha256(local[row["path"]])
            for row in rows
        )
    ):
        raise ValueError("independent v15 source-sync audit failed")
    if sha256(SCHEDULER) != EXPECTED_SCHEDULER_SHA256:
        raise ValueError("independent v15 scheduler identity failed")
    scheduler = json.loads(SCHEDULER.read_text())
    task = scheduler.get("task", {})
    if (
        scheduler.get("passed") is not True
        or scheduler.get("token_bound_backend_exit0") is not True
        or task.get("task_id") != "t83647"
        or task.get("backend_returncode") != 0
        or task.get("backend_token") != EXPECTED_BACKEND_TOKEN
        or task.get("token_matches") is not True
        or task.get("residual_evaluation_process_count") != 0
        or scheduler.get("scheduler_frontend_false_negative", {}).get(
            "scientific_result_status_inferred_from_scheduler_frontend"
        )
        is not False
    ):
        raise ValueError("independent v15 token-bound backend audit failed")


def verify(expected_manifest: str, expected_audit: str) -> dict:
    if sha256(MANIFEST) != expected_manifest:
        raise ValueError("v15 evaluation manifest identity drifted")
    manifest = json.loads(MANIFEST.read_text())
    records = manifest.get("files", [])
    if (
        manifest.get("schema_version")
        != "transit-h2oplus-v15-hold-calibration-evaluation-file-manifest-v1"
        or manifest.get("study_stage") != "development"
        or manifest.get("formal_held_out_outcomes_read") is not False
        or manifest.get("supports_formal_held_out_model_effect_claim") is not False
        or manifest.get("artifact_integrity", {}).get("passed") is not True
        or manifest.get("artifact_integrity", {}).get("result_count") != 135
        or manifest.get("synchronized_partition_file_count") != 290
        or manifest.get("file_count") != len(records)
        or len({record.get("path") for record in records}) != len(records)
        or manifest.get("file_records_aggregate_sha256") != aggregate(records)
    ):
        raise ValueError("v15 evaluation manifest scope failed")
    paths = [resolve_record(record) for record in records]
    audit_record = manifest.get("audit", {})
    audit_path = resolve_record(audit_record)
    if sha256(audit_path) != expected_audit:
        raise ValueError("v15 evaluation audit identity drifted")
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("schema_version")
        != "transit-h2oplus-v15-hold-calibration-evaluation-audit-v1"
        or audit.get("passed") is not True
        or audit.get("artifact_integrity", {}).get("passed") is not True
        or audit.get("artifact_integrity", {}).get("result_count") != 135
        or audit.get("parent_training_scientific_stability", {}).get(
            "scientific_stability_passed"
        )
        is not True
        or audit.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("v15 evaluation audit scope failed")
    result_stats = verify_results()
    checkpoint_stats = verify_checkpoints()
    verify_sync_and_scheduler()
    result_records = [
        path
        for path in paths
        if path.name.startswith("sumo_seed") and path.suffix == ".json"
    ]
    if len(result_records) != 135:
        raise ValueError("v15 manifest does not bind all physical result files")
    return {
        "schema_version": (
            "transit-h2oplus-v15-hold-calibration-evaluation-file-manifest-"
            "verification-v1"
        ),
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": {
            "path": str(MANIFEST.relative_to(REPO)),
            "sha256": sha256(MANIFEST),
            "size_bytes": MANIFEST.stat().st_size,
        },
        "audit": {
            "path": str(audit_path.relative_to(REPO)),
            "sha256": sha256(audit_path),
            "size_bytes": audit_path.stat().st_size,
        },
        "file_count_rehashed": len(records),
        "file_records_aggregate_sha256": aggregate(records),
        "independent_artifact_audit": {**result_stats, **checkpoint_stats},
        "scheduler_frontend_false_negative_preserved": True,
        "token_bound_backend_exit0_verified": True,
        "parent_training_scientific_stability_verified_separately": True,
        "prior_v14_development_outcomes_used_for_v15_design": True,
        "formal_held_out_outcomes_read": False,
        "supports_formal_held_out_model_effect_claim": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "passed": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-audit-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    values = (args.expected_manifest_sha256, args.expected_audit_sha256)
    if any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in values
    ):
        raise ValueError("invalid expected v15 manifest/audit SHA256")
    payload = verify(*values)
    atomic_json(OUTPUT, payload)
    print(OUTPUT)
    print(sha256(OUTPUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
