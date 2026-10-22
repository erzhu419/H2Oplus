#!/usr/bin/env python3
"""Independently audit the synchronized v18-v3 development evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
RESULTS = REPRO / "h2oplus_v18_empty_tail_reserve_results"
PARTITION = RESULTS / "evaluation_partition_v3/single_jtl110gpu"
EVIDENCE = RESULTS / "evidence/evaluation_v3"
SUBMISSION_SHA256 = "cb92eba9cdb583f5921817d355477c62a53abf75908c987d86574b99c787ca0b"
SCHEDULER_SHA256 = "b514e9e354d15b919e1d2ed2f8562b1a3e94160adfcecb5968e95adaaec8a8ad"
SOURCE_SYNC_SHA256 = "da38e79455ee5c8d78ac7386c17d0cd38d04396efd8d8dbf2f3ba2a050e2af15"
RUNTIME_SIGNATURE = "264d3c412b8e6c42320995208d2e009eb809b85626047206c4de752c2dae06e7"
DATASET_SHA256 = "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899"
RUNTIME_LOCK_SHA256 = "ac60ae58e0e6db6bb1386ffdfe0d164b96f81e49e4983b8b062ceddb65da51a4"
BLOCK_MAP = {
    "v18-empty-tail-development-01": ("sumo-seed-7101", 7101),
    "v18-empty-tail-development-02": ("sumo-seed-7102", 7102),
    "v18-empty-tail-development-03": ("sumo-seed-7103", 7103),
}
ALGORITHMS = (
    "h2oplus-v18-empty-tail-reserve-baseline",
    "h2oplus-v18-empty-tail-reserve-empty-forty",
    "h2oplus-v18-empty-tail-reserve-one-seat-forty",
)
SEEDS = tuple(range(6201, 6206))
SCALES = (0.85, 1.0, 1.15)
METHOD_RE = re.compile(
    r"^(?P<algorithm>h2oplus-v18-empty-tail-reserve-"
    r"(?P<variant>baseline|empty-forty|one-seat-forty))"
    r"-seed(?P<seed>62\d{2})-events000025000$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved.relative_to(ROOT)),
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", type=Path, default=PARTITION)
    parser.add_argument(
        "--output", type=Path, default=RESULTS / "evaluation_audit_v3.json"
    )
    return parser.parse_args()


def finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite {label}")
    return result


def ledger_and_service_pass(payload: dict[str, Any]) -> bool:
    semantics = payload.get("bus_service_semantics_audit", {})
    violations = semantics.get("violations")
    service = payload.get("service", {})
    ledger = payload.get("ledger_audit", {})
    counts = ledger.get("count_source_audit", {})
    passengers = payload.get("passengers", {})
    welfare = payload.get("passenger_welfare", {})
    coverage = welfare.get("chronological_coverage_errors", {})
    return (
        semantics.get("passed") is True
        and semantics.get("require_all_departed_trips_completed") is True
        and isinstance(violations, dict)
        and all(value in ([], {}) for value in violations.values())
        and service.get("active_trips_at_horizon") == 0
        and service.get("completed_trips") == service.get("departed_trips")
        and service.get("completion_rate") == 1.0
        and ledger.get("legacy_sampled_trip_histories_used_as_ground_truth") is False
        and counts.get("max_in_vehicle_count_error") == 0
        and counts.get("max_waiting_count_error") == 0
        and counts.get("subscription_fallback_queries") == 0
        and passengers.get("legacy_sampled_trip_histories_used_for_welfare") is False
        and passengers.get("active_state_count_balance_error") == 0
        and welfare.get("primary") is True
        and isinstance(coverage, dict)
        and bool(coverage)
        and all(abs(float(value)) <= 1e-9 for value in coverage.values())
        and abs(float(welfare.get("maximum_reward_reconstruction_error", math.inf)))
        <= 1e-9
    )


def sources_bound(observed: dict[str, str], freeze: dict[str, Any]) -> bool:
    for source in freeze.get("matrix_sources", []):
        suffix = str(source.get("path", ""))
        expected = source.get("sha256")
        if not any(str(path).endswith(suffix) and digest == expected for path, digest in observed.items()):
            return False
    return True


def load_controls() -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    submission_path = REPRO / "h2oplus_v18_empty_tail_reserve_evaluation_submission_v3.json"
    if sha256(submission_path) != SUBMISSION_SHA256:
        raise ValueError("v18 submission identity drifted")
    submission = json.loads(submission_path.read_text())
    controls = {
        "authorization": REPRO / "h2oplus_v18_empty_tail_reserve_successor_authorization_v3.json",
        "correction": REPRO / "h2oplus_v18_empty_tail_reserve_namespace_binding_correction_v3.json",
        "registration": REPRO / "h2oplus_v18_empty_tail_reserve_successor_registration_v3.json",
        "protocol": REPRO / "h2oplus_v18_empty_tail_reserve_protocol_v1.json",
        "methods": REPRO / "h2oplus_v18_empty_tail_reserve_evaluation_methods_v1.json",
        "environment_design": REPRO / "h2oplus_v18_empty_tail_reserve_environment_v1.json",
        "source_freeze": REPRO / "h2oplus_v18_empty_tail_reserve_source_freeze_v3.json",
        "deployment_manifest": REPRO / "deployments/h2oplus_v18_empty_tail_reserve_evaluation_v3/workspace/deployment_manifest.json",
        "deployment_sync": REPRO / "h2oplus_v18_empty_tail_reserve_evaluation_deployment_sync_v3.json",
        "local_verification": REPRO / "h2oplus_v18_empty_tail_reserve_evaluation_deployment_audit_v3.json",
        "remote_verification": REPRO / "h2oplus_v18_empty_tail_reserve_evaluation_deployment_v3_jtl110gpu_verification_v5.json",
        "parent_v17_analysis": RESULTS.parent / "h2oplus_v17_tail_service_reserve_results/tail_service_reserve_analysis_v2.json",
        "parent_v17_audit": RESULTS.parent / "h2oplus_v17_tail_service_reserve_results/evaluation_audit_v2.json",
        "parent_v17_closeout": RESULTS.parent / "h2oplus_v17_tail_service_reserve_results/evaluation_closeout_v2.json",
        "parent_v17_verification": RESULTS.parent / "h2oplus_v17_tail_service_reserve_results/evaluation_closeout_verification_v2.json",
        "data_environment_audit": REPRO / "h2oplus_v17_data_environment_convergence_audit_v1.json",
        "data_environment_verification": REPRO / "h2oplus_v17_data_environment_convergence_verification_v1.json",
        "v1_incident_closeout": RESULTS / "evaluation_incidents/v1_parser_contract/incident_closeout_v1.json",
        "v1_incident_verification": RESULTS / "evaluation_incidents/v1_parser_contract/incident_verification_v1.json",
        "v2_incident_closeout": RESULTS / "evaluation_incidents/v2_namespace_binding/incident_closeout_v2.json",
        "v2_incident_verification": RESULTS / "evaluation_incidents/v2_namespace_binding/incident_verification_v2.json",
    }
    expected = submission.get("immutable_hashes", {})
    for name, path in controls.items():
        if sha256(path.resolve(strict=True)) != expected.get(name):
            raise ValueError(f"v18 immutable control drifted: {name}")
    methods_payload = json.loads(controls["methods"].read_text())
    methods = {row["name"]: row for row in methods_payload.get("methods", [])}
    expected_names = {
        f"{algorithm}-seed{seed}-events000025000"
        for algorithm in ALGORITHMS
        for seed in SEEDS
    }
    if (
        methods_payload.get("schema_version")
        != "transit-evaluation-methods-v18-empty-tail-reserve-v1"
        or methods_payload.get("study_stage") != "development"
        or methods_payload.get("screen_type") != "paired-mechanism-screen"
        or methods_payload.get("formal_held_out_outcomes_read") is not False
        or methods_payload.get("v18_development_evaluation_outcomes_read") is not False
        or methods_payload.get("physical_method_count") != 15
        or methods_payload.get("logical_method_count") != 15
        or set(methods) != expected_names
    ):
        raise ValueError("v18 methods grid drifted")
    for name, method in methods.items():
        match = METHOD_RE.fullmatch(name)
        variant = match.group("variant") if match else None
        expected_enabled = variant != "baseline"
        expected_fraction = 0.02 if variant == "one-seat-forty" else 0.0
        if (
            match is None
            or method.get("algorithm") != match.group("algorithm")
            or method.get("training_seed") != int(match.group("seed"))
            or method.get("online_events") != 25_000
            or method.get("empty_reserve_enabled") is not expected_enabled
            or method.get("empty_reserve_floor_seconds") != 40.0
            or method.get("activation_start_fraction") != 0.25
            or method.get("activation_end_fraction") != 0.75
            or method.get("max_onboard_capacity_fraction") != expected_fraction
            or method.get("predesignated_candidate") is not (variant == "empty-forty")
        ):
            raise ValueError(f"malformed v18 method: {name}")
    return submission, methods, {name: descriptor(path) for name, path in controls.items()}


def validate_policy(payload: dict[str, Any], method: dict[str, Any]) -> None:
    policy = payload.get("policy", {})
    reserve = policy.get("empty_tail_reserve", {})
    variant = method["variant"]
    if (
        policy.get("artifact_lineage") != "v18-empty-tail-reserve"
        or policy.get("parent_artifact_lineage") != "v17-tail-service-reserve"
        or policy.get("parent_method") != method["parent_method"]
        or policy.get("sha256") != method["checkpoint_sha256"]
        or policy.get("actor_tensor_sha256") != method["actor_tensor_sha256"]
        or policy.get("temperature_tensor_sha256") != method["temperature_tensor_sha256"]
        or reserve.get("schema_version") != "transit-h2oplus-v18-empty-tail-reserve-v1"
        or reserve.get("variant") != variant
        or reserve.get("empty_reserve_enabled") is not method["empty_reserve_enabled"]
        or reserve.get("empty_reserve_floor_seconds") != 40.0
        or reserve.get("activation_start_fraction") != 0.25
        or reserve.get("activation_end_fraction") != 0.75
        or reserve.get("max_onboard_capacity_fraction") != method["max_onboard_capacity_fraction"]
        or reserve.get("requires_forward_bus_present") is not True
        or reserve.get("requires_backward_bus_absent") is not True
        or reserve.get("requires_positive_eligible_arrival_rate") is not True
        or reserve.get("parent_action_preserved_outside_guard") is not True
        or reserve.get("physical_max_hold_preserved") is not True
        or reserve.get("predesignated_candidate") is not (variant == "empty-forty")
        or reserve.get("checkpoint_selection_by_behavior") is not False
        or reserve.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError(f"v18 policy binding drifted: {method['name']}")


def main() -> int:
    args = parse_args()
    partition = args.partition.expanduser().resolve(strict=True)
    submission, methods, controls = load_controls()
    scheduler_path = EVIDENCE / "scheduler_evidence_v3.json"
    sync_path = EVIDENCE / "source_sync_v3.json"
    if sha256(scheduler_path) != SCHEDULER_SHA256 or sha256(sync_path) != SOURCE_SYNC_SHA256:
        raise ValueError("v18 observed scheduler or source-sync evidence drifted")
    scheduler = json.loads(scheduler_path.read_text())
    sync = json.loads(sync_path.read_text())
    if (
        scheduler.get("passed") is not True
        or scheduler.get("token_bound_backend_exit0") is not True
        or scheduler.get("task", {}).get("task_id") != "t83816"
        or scheduler.get("task", {}).get("backend_token")
        != "1787639205147330111:1528677:126605756510208"
        or sync.get("passed") is not True
        or sync.get("formal_held_out_outcomes_read") is not False
        or sync.get("v18_v1_or_v2_results_reused") is not False
        or sync.get("local_sumo_executed") is not False
    ):
        raise ValueError("v18 completion evidence contract failed")
    listed = sync.get("partition", {}).get("files", [])
    actual = []
    for path in sorted(p for p in partition.rglob("*") if p.is_file()):
        actual.append({"path": str(path.relative_to(partition)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    if listed != actual:
        raise ValueError("v18 synchronized file inventory drifted after transfer")

    host_path = partition / "host_execution.json"
    summary_path = partition / "host_summary.json"
    host = json.loads(host_path.read_text())
    summary = json.loads(summary_path.read_text())
    immutable = host.get("immutable_evidence", {})
    if (
        host.get("schema_version")
        != "transit-h2oplus-v18-empty-tail-reserve-evaluation-host-execution-v3"
        or host.get("status") != "completed"
        or host.get("completed_results") != 135
        or host.get("expected_results") != 135
        or host.get("workers") != 12
        or host.get("training") is not False
        or host.get("formal_held_out_outcomes_read") is not False
        or immutable.get("v18_v1_or_v2_results_reused") is not False
        or immutable.get("v18_v1_result_count") != 0
        or immutable.get("v18_v2_result_count") != 0
        or summary.get("schema_version")
        != "transit-h2oplus-v18-empty-tail-reserve-evaluation-host-summary-v3"
        or summary.get("status") != "completed"
        or summary.get("physical_evaluation_results") != 135
        or summary.get("training") is not False
        or summary.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("v18 host execution boundary failed")

    freeze = json.loads((REPRO / "h2oplus_v18_empty_tail_reserve_source_freeze_v3.json").read_text())
    remote_root = Path(sync["partition"]["source_root"])
    manifests = []
    result_files = []
    signatures = set()
    grid = set()
    counts = Counter()
    expected_grid = {
        (algorithm, seed, registered, scale)
        for algorithm in ALGORITHMS
        for seed in SEEDS
        for registered in BLOCK_MAP
        for scale in SCALES
    }
    for registered, (matrix_block, sumo_seed) in BLOCK_MAP.items():
        block_root = partition / registered
        manifest_path = block_root / "matrix_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        jobs = manifest.get("jobs", [])
        source_hashes = manifest.get("source_sha256", {})
        matrix_partition = manifest.get("matrix_partition", {})
        if (
            manifest.get("schema_version") != "transit-evaluation-matrix-v7"
            or manifest.get("methods_manifest_sha256")
            != submission["immutable_hashes"]["methods"]
            or manifest.get("dataset_sha256") != DATASET_SHA256
            or manifest.get("sumo_runtime_signature") != RUNTIME_SIGNATURE
            or len(source_hashes) != 59
            or not sources_bound(source_hashes, freeze)
            or len(jobs) != 45
            or {job.get("environment_block") for job in jobs} != {matrix_block}
            or {job.get("sumo_seed") for job in jobs} != {sumo_seed}
            or matrix_partition.get("full_job_count") != 45
            or matrix_partition.get("selected_job_count") != 45
            or matrix_partition.get("shard_count") != 1
            or matrix_partition.get("shard_index") != 0
        ):
            raise ValueError(f"v18 matrix binding failed: {registered}")
        manifests.append({"registered_block": registered, "matrix_block": matrix_block, **descriptor(manifest_path), "jobs": 45, "source_file_count": 59})
        for job in jobs:
            method_name = job.get("method", {}).get("name")
            if method_name not in methods:
                raise ValueError("v18 matrix contains an unregistered method")
            method = methods[method_name]
            signature = job.get("signature")
            output_remote = Path(job["output"])
            output = partition / output_remote.relative_to(remote_root)
            payload = json.loads(output.read_text())
            scale = finite(job.get("scenario", {}).get("demand_scale"), "demand scale")
            runtime = payload.get("runtime", {})
            backend_api = runtime.get("sumo_runtime_fingerprint", {}).get("backend_api", {})
            if (
                payload.get("schema_version") != "transit-evaluation-v7"
                or payload.get("evaluation_job", {}).get("signature") != signature
                or payload.get("method") != method_name
                or payload.get("source_sha256") != source_hashes
                or payload.get("sumo_seed") != sumo_seed
                or payload.get("scenario", {}).get("demand_scale") != scale
                or payload.get("scenario", {}).get("manifest_sha256")
                != job.get("scenario", {}).get("manifest_sha256")
                or job.get("dataset_sha256") != DATASET_SHA256
                or job.get("sumo_runtime_lock_sha256") != RUNTIME_LOCK_SHA256
                or job.get("sumo_runtime_signature") != RUNTIME_SIGNATURE
                or runtime.get("sumo_runtime_signature") != RUNTIME_SIGNATURE
                or runtime.get("device") != "cpu"
                or runtime.get("python_version") != "3.8.20"
                or runtime.get("sumo_backend") != "libsumo"
                or backend_api.get("version_line") != "SUMO 1.25.0"
                or signature in signatures
                or not ledger_and_service_pass(payload)
            ):
                raise ValueError(f"v18 result binding failed: {output}")
            validate_policy(payload, method)
            finite(payload["passenger_welfare"]["accrued_generalized_seconds_per_departed_passenger"], "primary endpoint")
            finite(payload["passengers"]["completion_rate"], "completion rate")
            int(payload["passengers"]["unfinished_passengers"])
            signatures.add(signature)
            key = (method["algorithm"], method["training_seed"], registered, scale)
            grid.add(key)
            counts[("algorithm", method["algorithm"])] += 1
            counts[("block", registered)] += 1
            counts[("scale", str(scale))] += 1
            result_files.append(descriptor(output))
    if len(result_files) != 135 or len(signatures) != 135 or grid != expected_grid:
        raise ValueError("v18 result grid is not exactly 135 unique episodes")
    for algorithm in ALGORITHMS:
        if counts[("algorithm", algorithm)] != 45:
            raise ValueError("v18 algorithm coverage is not balanced")
    for block in BLOCK_MAP:
        if counts[("block", block)] != 45:
            raise ValueError("v18 block coverage is not balanced")

    output = args.output.expanduser().resolve()
    payload = {
        "schema_version": "transit-h2oplus-v18-empty-tail-reserve-evaluation-audit-v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": "development",
        "screen_type": "paired-mechanism-screen",
        "passed": True,
        "artifact_integrity": {
            "passed": True,
            "physical_result_count": 135,
            "unique_job_signatures": 135,
            "physical_behavior_count": 15,
            "fresh_environment_cell_count": 9,
            "matrix_manifest_count": 3,
            "matrix_manifests": manifests,
            "all_results_source_protocol_runtime_and_job_bound": True,
            "all_ledgers_and_service_semantics_passed": True,
        },
        "parent_training_stability": {
            "passed": True,
            "parent_v17_closeout": controls["parent_v17_closeout"],
            "parent_v17_verification": controls["parent_v17_verification"],
        },
        "data_environment_validity_risks": {
            "material_factors_present": True,
            "fresh_development_environment_used": True,
            "remaining_runtime_shift_from_training": True,
            "audit": controls["data_environment_audit"],
            "verification": controls["data_environment_verification"],
        },
        "coverage": {
            "algorithms": {algorithm: counts[("algorithm", algorithm)] for algorithm in ALGORITHMS},
            "registered_blocks": {block: counts[("block", block)] for block in BLOCK_MAP},
            "demand_scales": {str(scale): counts[("scale", str(scale))] for scale in SCALES},
        },
        "bindings": {
            "submission": descriptor(REPRO / "h2oplus_v18_empty_tail_reserve_evaluation_submission_v3.json"),
            "scheduler_evidence": descriptor(scheduler_path),
            "source_sync": descriptor(sync_path),
            "host_execution": descriptor(host_path),
            "host_summary": descriptor(summary_path),
            "immutable_controls": controls,
        },
        "result_files": sorted(result_files, key=lambda item: item["path"]),
        "prior_v14_v15_v16_v17_development_outcomes_used_for_design": True,
        "v18_v1_or_v2_results_reused": False,
        "local_sumo_executed": False,
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_eligible": False,
        "formal_confirmation_authorized": False,
        "claim_boundary": (
            "This audit establishes artifact integrity only. Scientific gate results "
            "must come from the unchanged frozen analyzer; a passing screen can only "
            "permit a later full development control matrix."
        ),
    }
    atomic_json(output, payload)
    print(json.dumps({"output": str(output), "sha256": sha256(output), "results": 135, "passed": True}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
