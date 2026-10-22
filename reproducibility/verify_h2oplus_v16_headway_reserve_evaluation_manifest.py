#!/usr/bin/env python3
"""Independently verify the v16 headway-reserve evaluation file manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reproducibility/h2oplus_v16_headway_reserve_results"
PARTITION = RESULTS / "evaluation_partition_v1/single_jtl110gpu"
MANIFEST = RESULTS / "evaluation_file_manifest_v1.json"
OUTPUT = RESULTS / "evaluation_file_manifest_verification_v1.json"
METHODS = REPO / "reproducibility/h2oplus_v16_headway_reserve_evaluation_methods_v1.json"
SOURCE_FREEZE = REPO / "reproducibility/h2oplus_v16_headway_reserve_evaluation_source_freeze_v1.json"
SOURCE_SYNC = RESULTS / "evidence/evaluation_v1/source_sync_v1.json"
SCHEDULER = RESULTS / "evidence/evaluation_v1/scheduler_evidence_v1.json"
EXPECTED_METHODS_SHA256 = (
    "35047fd2ce616acbccd4592ae333a7ba84a1e4231c1a2505c7768d96b91de06a"
)
EXPECTED_SOURCE_FREEZE_SHA256 = (
    "606a15bf7db8b9aa1c4dbefa7b9bcc12bc94ca60e605de5a3e185eefec679577"
)
EXPECTED_PROTOCOL_SHA256 = (
    "978faf1b8ddf51128b5fac570552e8388c3d882e39244c4b6f0ec7615e79a43b"
)
EXPECTED_SOURCE_SYNC_SHA256 = (
    "be6009cd2fade6a1f39475463065c9f3b3d90c89e33e10fffa7edf48a86036eb"
)
EXPECTED_SCHEDULER_SHA256 = (
    "39786f5b16f6a8f3b125565ad08510f20fa1ca73a117be699c57a925d63ede65"
)
EXPECTED_DATASET_SHA256 = (
    "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899"
)
EXPECTED_RUNTIME_SIGNATURE = (
    "264d3c412b8e6c42320995208d2e009eb809b85626047206c4de752c2dae06e7"
)
EXPECTED_BACKEND_TOKEN = "1787571481648149575:1528677:126605756510208"
EXPECTED_MATRIX_MANIFESTS = {
    "v12-staged-release-development-01": "8fa37fce8888a561584162ebbc1cb2e2e712b9b177bd16e5c701cddaee74946b",
    "v12-staged-release-development-02": "6885e883723a4bb6baf46bbfb01b07387c69384c4c527b5a339c0f61a039c2b9",
    "v12-staged-release-development-03": "03c9810ade93b29bb843e481023c6ebb11fbfcac3bc48f2059e7906e60970d5c",
}
METHOD_BINDING_FIELDS = {
    "actor_tensor_sha256",
    "actual_online_events",
    "algorithm",
    "artifact_lineage",
    "base_hold_scale",
    "checkpoint_sha256",
    "headway_balance_gain",
    "learner_update_steps",
    "name",
    "online_events",
    "parent_artifact_lineage",
    "parent_method",
    "ratio_optimizer_steps",
    "reserve_horizon_fraction",
    "temperature_tensor_sha256",
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
        raise ValueError(f"v16 manifest file record failed: {record}")
    return path


def suffix_hash(source: dict, suffix: str) -> str | None:
    values = [digest for path, digest in source.items() if str(path).endswith(suffix)]
    return values[0] if len(values) == 1 else None


def verify_results() -> dict:
    if sha256(METHODS) != EXPECTED_METHODS_SHA256:
        raise ValueError("independent v16 methods identity failed")
    if sha256(SOURCE_FREEZE) != EXPECTED_SOURCE_FREEZE_SHA256:
        raise ValueError("independent v16 source-freeze identity failed")
    methods_payload = json.loads(METHODS.read_text())
    method_rows = {row["name"]: row for row in methods_payload.get("methods", [])}
    source_freeze = json.loads(SOURCE_FREEZE.read_text())
    frozen_sources = {
        row["path"]: row["sha256"] for row in source_freeze.get("matrix_sources", [])
    }
    if (
        methods_payload.get("schema_version")
        != "transit-evaluation-methods-v16-headway-reserve-v1"
        or methods_payload.get("predesignated_candidate")
        != "h2oplus-v16-headway-reserve-three-quarter"
        or len(method_rows) != 15
        or source_freeze.get("schema_version")
        != "transit-h2oplus-v16-headway-reserve-evaluation-source-freeze-v1"
        or len(frozen_sources) != 12
        or source_freeze.get("expected_matrix_source_file_count") != 51
        or any(
            not (REPO / relative).is_file()
            or sha256(REPO / relative) != digest
            for relative, digest in frozen_sources.items()
        )
    ):
        raise ValueError("independent v16 method/source contract failed")

    jobs: dict[str, dict] = {}
    for block, expected_hash in EXPECTED_MATRIX_MANIFESTS.items():
        manifest_path = PARTITION / block / "matrix_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        source = manifest.get("source_sha256", {})
        rows = manifest.get("jobs", [])
        partition = manifest.get("matrix_partition", {})
        if (
            sha256(manifest_path) != expected_hash
            or manifest.get("schema_version") != "transit-evaluation-matrix-v7"
            or manifest.get("methods_manifest_sha256") != EXPECTED_METHODS_SHA256
            or manifest.get("dataset_sha256") != EXPECTED_DATASET_SHA256
            or manifest.get("sumo_runtime_signature") != EXPECTED_RUNTIME_SIGNATURE
            or len(source) != 51
            or any(
                suffix_hash(source, suffix) != digest
                for suffix, digest in frozen_sources.items()
            )
            or len(rows) != 45
            or partition.get("full_job_count") != 45
            or partition.get("selected_job_count") != 45
            or partition.get("shard_count") != 1
            or partition.get("shard_index") != 0
        ):
            raise ValueError(f"independent v16 matrix audit failed: {block}")
        for job in rows:
            signature = job.get("signature")
            if not isinstance(signature, str) or signature in jobs:
                raise ValueError(f"independent v16 matrix signature failed: {block}")
            jobs[signature] = job

    results = sorted(PARTITION.rglob("sumo_seed*.json"))
    if len(results) != 135 or len(jobs) != 135:
        raise ValueError("independent v16 result count failed")
    signatures: set[str] = set()
    method_names: set[str] = set()
    source_reference: dict | None = None
    maximum_coverage_error = 0.0
    maximum_reward_reconstruction_error = 0.0
    for path in results:
        payload = json.loads(path.read_text())
        signature = payload.get("evaluation_job", {}).get("signature")
        job = jobs.get(signature)
        method = method_rows.get(payload.get("method"))
        source = payload.get("source_sha256", {})
        policy = payload.get("policy", {})
        reserve = policy.get("headway_reserve", {})
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
            or job is None
            or method is None
            or len(source) != 51
            or source != job.get("source_sha256")
            or any(
                suffix_hash(source, suffix) != digest
                for suffix, digest in frozen_sources.items()
            )
            or Path(job.get("output", "")).resolve() != path.resolve()
            or job.get("method", {}).get("name") != method.get("name")
            or any(
                job.get("method", {}).get(key) != method.get(key)
                for key in METHOD_BINDING_FIELDS
            )
            or job.get("dataset_sha256") != EXPECTED_DATASET_SHA256
            or job.get("sumo_runtime_signature") != EXPECTED_RUNTIME_SIGNATURE
            or payload.get("dataset", {}).get("sha256") != EXPECTED_DATASET_SHA256
            or payload.get("runtime", {}).get("sumo_runtime_signature")
            != EXPECTED_RUNTIME_SIGNATURE
            or payload.get("scenario", {}).get("manifest_sha256")
            != job.get("scenario", {}).get("manifest_sha256")
            or policy.get("artifact_lineage") != "v16-headway-reserve"
            or policy.get("parent_artifact_lineage") != "v15-hold-calibration"
            or policy.get("parent_method") != method.get("parent_method")
            or policy.get("sha256") != method.get("checkpoint_sha256")
            or policy.get("actor_tensor_sha256") != method.get("actor_tensor_sha256")
            or policy.get("temperature_tensor_sha256")
            != method.get("temperature_tensor_sha256")
            or reserve.get("schema_version")
            != "transit-h2oplus-v16-headway-reserve-v1"
            or reserve.get("base_hold_scale") != 0.125
            or reserve.get("headway_balance_gain") != 1.0
            or reserve.get("reserve_horizon_fraction")
            != method.get("reserve_horizon_fraction")
            or reserve.get("v16_protocol_sha256") != EXPECTED_PROTOCOL_SHA256
            or reserve.get("predesignated_candidate")
            is not (
                method.get("algorithm")
                == "h2oplus-v16-headway-reserve-three-quarter"
            )
            or reserve.get("zero_hold_preserved_exactly") is not True
            or reserve.get("transformed_hold_never_exceeds_parent") is not True
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
            or reserve.get("formal_held_out_outcomes_read") is not False
            or reserve.get("checkpoint_selection_by_behavior") is not False
            or policy.get("training_metadata", {}).get(
                "formal_held_out_outcomes_read"
            )
            is not False
        ):
            raise ValueError(f"independent v16 result audit failed: {path}")
        if source_reference is not None and source != source_reference:
            raise ValueError(f"independent v16 source-map equality failed: {path}")
        source_reference = source
        signatures.add(signature)
        method_names.add(str(payload.get("method")))
    if (
        len(signatures) != 135
        or signatures != set(jobs)
        or len(method_names) != 15
        or source_reference is None
    ):
        raise ValueError("independent v16 behavior/signature cardinality failed")
    return {
        "all_numeric_values_finite": True,
        "all_service_semantics_passed": True,
        "all_welfare_ledgers_passed": True,
        "maximum_welfare_chronological_coverage_error": maximum_coverage_error,
        "maximum_reward_reconstruction_error": maximum_reward_reconstruction_error,
        "physical_behavior_count": len(method_names),
        "result_count": len(results),
        "source_file_count": len(source_reference),
        "unique_job_signature_count": len(signatures),
    }


def verify_checkpoints() -> dict:
    if sha256(METHODS) != EXPECTED_METHODS_SHA256:
        raise ValueError("independent v16 methods identity failed")
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
            float(method.get("reserve_horizon_fraction", math.nan)),
        )
        if previous != expected or sha256(checkpoint) != expected or behavior in behaviors:
            raise ValueError(f"independent v16 checkpoint/behavior audit failed: {checkpoint}")
        behaviors.add(behavior)
    if len(behaviors) != 15 or len(checkpoints) != 5:
        raise ValueError("independent v16 checkpoint cardinality failed")
    return {
        "physical_behaviors_verified": len(behaviors),
        "unique_parent_checkpoints_rehashed": len(checkpoints),
    }


def verify_sync_and_scheduler() -> None:
    if sha256(SOURCE_SYNC) != EXPECTED_SOURCE_SYNC_SHA256:
        raise ValueError("independent v16 source-sync identity failed")
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
        or len(rows) != 281
        or {row.get("path") for row in rows} != set(local)
        or any(
            row.get("bytes") != local[row["path"]].stat().st_size
            or row.get("sha256") != sha256(local[row["path"]])
            for row in rows
        )
    ):
        raise ValueError("independent v16 source-sync audit failed")
    if sha256(SCHEDULER) != EXPECTED_SCHEDULER_SHA256:
        raise ValueError("independent v16 scheduler identity failed")
    scheduler = json.loads(SCHEDULER.read_text())
    task = scheduler.get("task", {})
    if (
        scheduler.get("passed") is not True
        or scheduler.get("token_bound_backend_exit0") is not True
        or task.get("task_id") != "t83704"
        or task.get("backend_returncode") != 0
        or task.get("backend_token") != EXPECTED_BACKEND_TOKEN
        or task.get("token_matches") is not True
        or task.get("residual_evaluation_process_count") != 0
        or scheduler.get("scheduler_frontend_false_negative", {}).get(
            "scientific_result_status_inferred_from_scheduler_frontend"
        )
        is not False
    ):
        raise ValueError("independent v16 token-bound backend audit failed")


def verify(expected_manifest: str, expected_audit: str) -> dict:
    if sha256(MANIFEST) != expected_manifest:
        raise ValueError("v16 evaluation manifest identity drifted")
    manifest = json.loads(MANIFEST.read_text())
    records = manifest.get("files", [])
    if (
        manifest.get("schema_version")
        != "transit-h2oplus-v16-headway-reserve-evaluation-file-manifest-v1"
        or manifest.get("study_stage") != "development"
        or manifest.get("formal_held_out_outcomes_read") is not False
        or manifest.get("supports_formal_held_out_model_effect_claim") is not False
        or manifest.get("artifact_integrity", {}).get("passed") is not True
        or manifest.get("artifact_integrity", {}).get("result_count") != 135
        or manifest.get("synchronized_partition_file_count") != 281
        or manifest.get("file_count") != len(records)
        or len({record.get("path") for record in records}) != len(records)
        or manifest.get("file_records_aggregate_sha256") != aggregate(records)
    ):
        raise ValueError("v16 evaluation manifest scope failed")
    paths = [resolve_record(record) for record in records]
    audit_record = manifest.get("audit", {})
    audit_path = resolve_record(audit_record)
    if sha256(audit_path) != expected_audit:
        raise ValueError("v16 evaluation audit identity drifted")
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("schema_version")
        != "transit-h2oplus-v16-headway-reserve-evaluation-audit-v1"
        or audit.get("passed") is not True
        or audit.get("artifact_integrity", {}).get("passed") is not True
        or audit.get("artifact_integrity", {}).get("result_count") != 135
        or audit.get("parent_training_scientific_stability", {}).get(
            "scientific_stability_passed"
        )
        is not True
        or audit.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("v16 evaluation audit scope failed")
    result_stats = verify_results()
    checkpoint_stats = verify_checkpoints()
    verify_sync_and_scheduler()
    result_records = [
        path
        for path in paths
        if path.name.startswith("sumo_seed") and path.suffix == ".json"
    ]
    if len(result_records) != 135:
        raise ValueError("v16 manifest does not bind all physical result files")
    return {
        "schema_version": (
            "transit-h2oplus-v16-headway-reserve-evaluation-file-manifest-"
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
        "prior_v14_and_v15_development_outcomes_used_for_v16_design": True,
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
        raise ValueError("invalid expected v16 manifest/audit SHA256")
    payload = verify(*values)
    atomic_json(OUTPUT, payload)
    print(OUTPUT)
    print(sha256(OUTPUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
