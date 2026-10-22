#!/usr/bin/env python3
"""Independently audit the synchronized v20 development control matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
RESULTS = REPRO / "h2oplus_v20_full_control_results"
PARTITION = RESULTS / "evaluation_partition_v1/single_jtl110gpu"
EVIDENCE = RESULTS / "evidence/evaluation_v1"
SUBMISSION = REPRO / "h2oplus_v20_full_control_evaluation_submission_v1.json"
SUBMISSION_SHA256 = "801e601c00976590aca84fecd3957ac28d27ec1623e9f93bb33bce7a7d0c4372"
SCHEDULER_SHA256 = "63b5e39a9fa3edc2483f06ecb7685f41e3f24cd442fbeac899f63f2949d841dd"
SOURCE_SYNC_SHA256 = "02a63f240717b20d4e03d9fd6bd838ff13778f3d95340d8b8fb2ea3b83caa736"
RUNTIME_SIGNATURE = "264d3c412b8e6c42320995208d2e009eb809b85626047206c4de752c2dae06e7"
DATASET_SHA256 = "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899"
RUNTIME_LOCK_SHA256 = "ac60ae58e0e6db6bb1386ffdfe0d164b96f81e49e4983b8b062ceddb65da51a4"
METHODS_SHA256 = "77162e22cc3b748fc2ad65a229ebca5c9212fd2af172b37ba81cdaded2d2e2e9"
SOURCE_FREEZE_SHA256 = "b30de4416829a0bfa683184712a842dcff64732d27c3c51a5f8a3b107da22159"
BLOCK_MAP = {
    "v20-full-control-development-01": ("sumo-seed-7501", 7501),
    "v20-full-control-development-02": ("sumo-seed-7502", 7502),
    "v20-full-control-development-03": ("sumo-seed-7503", 7503),
}
ALGORITHMS = (
    "h2oplus-v19-completion-safety-quarter-load-sixty",
    "h2oplus-v19-completion-safety-baseline",
    "h2oplus-v14-terminal-quarter",
    "h2oplus",
    "sac-online",
    "h2oplus-v10-delayed-actor",
    "h2oplus-v11-actor-hold",
    "h2oplus-v12-actor-ramp",
    "h2oplus-v13-proximal-strong",
)
SEEDS = tuple(range(6201, 6206))
SCALES = (0.85, 1.0, 1.15)
CANDIDATE = "h2oplus-v19-completion-safety-quarter-load-sixty"
DIRECT_PARENT = "h2oplus-v19-completion-safety-baseline"


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
        "--output", type=Path, default=RESULTS / "evaluation_audit_v1.json"
    )
    return parser.parse_args()


def finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite {label}")
    return result


def inventory(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    blob = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode()
    return {
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "aggregate_sha256": hashlib.sha256(blob).hexdigest(),
    }


def ledger_and_service_pass(payload: dict[str, Any]) -> bool:
    semantics = payload.get("bus_service_semantics_audit", {})
    violations = semantics.get("violations", {})
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
    return all(
        any(
            str(path).endswith(str(source.get("path", "")))
            and digest == source.get("sha256")
            for path, digest in observed.items()
        )
        for source in freeze.get("matrix_sources", [])
    )


def checked_bindings(submission: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = {}
    for name, item in submission.get("bindings", {}).items():
        path = Path(item.get("path", "")).expanduser().resolve(strict=True)
        if ROOT not in path.parents or sha256(path) != item.get("sha256"):
            raise ValueError(f"v20 immutable binding drifted: {name}")
        records[name] = descriptor(path)
    expected = {
        "authorization",
        "data_environment_audit",
        "data_environment_verification",
        "dataset",
        "deployment_manifest",
        "deployment_sync",
        "design_audit",
        "design_verification",
        "environment",
        "local_verification",
        "methods",
        "parent_v19_audit",
        "parent_v19_closeout",
        "parent_v19_verification",
        "protocol",
        "rationale",
        "registration",
        "remote_verification",
        "runtime_lock",
        "source_freeze",
    }
    if set(records) != expected:
        raise ValueError("v20 immutable binding set drifted")
    return records


def load_controls() -> tuple[
    dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]
]:
    if sha256(SUBMISSION) != SUBMISSION_SHA256:
        raise ValueError("v20 submission identity drifted")
    submission = json.loads(SUBMISSION.read_text())
    if (
        submission.get("schema_version")
        != "transit-h2oplus-v20-full-control-evaluation-submission-v1"
        or submission.get("status") != "submitted"
        or submission.get("study_stage") != "development"
        or submission.get("matrix_type") != "full-development-control-matrix"
        or submission.get("task_count") != 1
        or submission.get("total_remote_sumo_episodes") != 405
        or submission.get("physical_behavior_count") != 45
        or submission.get("formal_held_out_outcomes_read") is not False
        or submission.get("formal_confirmation_stage_authorized_or_submitted")
        is not False
        or submission.get("task", {}).get("task_id") != "t83955"
    ):
        raise ValueError("v20 submission contract drifted")
    controls = checked_bindings(submission)
    if (
        controls["methods"]["sha256"] != METHODS_SHA256
        or controls["source_freeze"]["sha256"] != SOURCE_FREEZE_SHA256
        or controls["dataset"]["sha256"] != DATASET_SHA256
        or controls["runtime_lock"]["sha256"] != RUNTIME_LOCK_SHA256
    ):
        raise ValueError("v20 core immutable identity drifted")

    methods_path = ROOT / controls["methods"]["path"]
    methods_payload = json.loads(methods_path.read_text())
    methods = {row["name"]: row for row in methods_payload.get("methods", [])}
    expected_names = {
        f"{algorithm}-seed{seed}-events000025000"
        for algorithm in ALGORITHMS
        for seed in SEEDS
    }
    if (
        methods_payload.get("schema_version")
        != "transit-evaluation-methods-v20-full-control-v1"
        or methods_payload.get("study_stage") != "development"
        or methods_payload.get("matrix_type") != "full-development-control-matrix"
        or methods_payload.get("physical_method_count") != 45
        or methods_payload.get("logical_method_count") != 45
        or methods_payload.get("formal_held_out_outcomes_read") is not False
        or methods_payload.get("v20_development_evaluation_outcomes_read") is not False
        or set(methods) != expected_names
        or set(methods_payload.get("algorithms", [])) != set(ALGORITHMS)
    ):
        raise ValueError("v20 methods grid drifted")
    for name, method in methods.items():
        algorithm = method.get("algorithm")
        seed = method.get("training_seed")
        if (
            algorithm not in ALGORITHMS
            or seed not in SEEDS
            or name != f"{algorithm}-seed{seed}-events000025000"
            or method.get("online_events") != 25_000
            or method.get("predesignated_candidate")
            is not (algorithm == CANDIDATE)
            or not isinstance(method.get("checkpoint_sha256"), str)
            or not isinstance(method.get("actor_tensor_sha256"), str)
        ):
            raise ValueError(f"malformed v20 method: {name}")

    raw_path = REPRO / "h2oplus_v19_completion_safety_evaluation_methods_v1.json"
    raw_payload = json.loads(raw_path.read_text())
    raw_v19 = {row["name"]: row for row in raw_payload.get("methods", [])}
    freeze = json.loads((ROOT / controls["source_freeze"]["path"]).read_text())
    if (
        freeze.get("schema_version")
        != "transit-h2oplus-v20-full-control-source-freeze-v1"
        or freeze.get("formal_held_out_outcomes_read") is not False
        or freeze.get("v20_development_evaluation_outcomes_read") is not False
        or freeze.get("matrix_source_count") != 18
        or freeze.get("expected_matrix_source_file_count") != 57
        or freeze.get("physical_behavior_count") != 45
        or freeze.get("analyzer", {}).get("sha256")
        != "19815b30eaab6278e737ae15b1bea86a91c2621581041fef6c48f9f0ccdae5ce"
    ):
        raise ValueError("v20 source freeze contract drifted")

    parent_audit = json.loads((ROOT / controls["parent_v19_audit"]["path"]).read_text())
    parent_closeout = json.loads(
        (ROOT / controls["parent_v19_closeout"]["path"]).read_text()
    )
    parent_verification = json.loads(
        (ROOT / controls["parent_v19_verification"]["path"]).read_text()
    )
    data_audit = json.loads((ROOT / controls["data_environment_audit"]["path"]).read_text())
    data_verification = json.loads(
        (ROOT / controls["data_environment_verification"]["path"]).read_text()
    )
    if (
        parent_audit.get("passed") is not True
        or parent_audit.get("artifact_integrity", {}).get("passed") is not True
        or parent_audit.get("parent_training_stability", {}).get("passed") is not True
        or parent_closeout.get("passed") is not True
        or parent_closeout.get("parent_training_stability", {}).get("passed") is not True
        or parent_closeout.get("eligible_for_full_development_control_matrix") is not True
        or parent_verification.get("passed") is not True
        or parent_verification.get("parent_training_stability_verified") is not True
        or data_audit.get("conclusions", {}).get(
            "material_data_or_environment_factors_present"
        )
        is not True
        or data_verification.get("passed") is not True
    ):
        raise ValueError("v20 parent or data/environment evidence failed")
    return submission, methods, raw_v19, {"freeze": freeze, "bindings": controls}


def validate_policy(
    payload: dict[str, Any], method: dict[str, Any], raw_v19: dict[str, dict[str, Any]]
) -> None:
    policy = payload.get("policy", {})
    training = policy.get("training_metadata", {})
    if (
        policy.get("sha256") != method["checkpoint_sha256"]
        or policy.get("actor_tensor_sha256") != method["actor_tensor_sha256"]
        or training.get("training_seed") != method["training_seed"]
        or training.get("dataset_sha256") != DATASET_SHA256
        or training.get("phase") not in {"complete", "online_training"}
        or training.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError(f"v20 policy provenance drifted: {method['name']}")
    if method["algorithm"] not in {CANDIDATE, DIRECT_PARENT}:
        return
    raw = raw_v19.get(method["name"])
    reserve = policy.get("completion_safety_reserve", {})
    variant = method["algorithm"][len("h2oplus-v19-completion-safety-") :]
    if raw is None or (
        policy.get("artifact_lineage") != "v19-completion-safety-reserve"
        or policy.get("parent_method") != raw.get("parent_method")
        or training.get("run_signature") != raw.get("training_run_signature")
        or training.get("training_checkpoint_sha256")
        != raw.get("training_checkpoint_sha256")
        or training.get("training_admission", {}).get("sha256")
        != raw.get("training_admission_sha256")
        or reserve.get("schema_version")
        != "transit-h2oplus-v19-completion-safety-v1"
        or reserve.get("variant") != variant
        or reserve.get("reserve_enabled")
        is not raw.get("completion_safety_reserve_enabled")
        or reserve.get("reserve_floor_seconds")
        != raw.get("completion_safety_reserve_floor_seconds")
        or reserve.get("activation_start_fraction") != 0.15
        or reserve.get("activation_end_fraction") != 0.75
        or reserve.get("max_onboard_capacity_fraction") != 0.25
        or reserve.get("requires_forward_bus_present") is not True
        or reserve.get("requires_backward_bus_absent") is not True
        or reserve.get("requires_positive_eligible_arrival_rate") is not True
        or reserve.get("checkpoint_selection_by_behavior") is not False
        or reserve.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError(f"v20 v19 transform drifted: {method['name']}")


def resolve_output(root: Path, raw: str) -> Path:
    value = Path(raw)
    if root.name in value.parts:
        index = value.parts.index(root.name)
        local = root.joinpath(*value.parts[index + 1 :])
        if local.is_file():
            return local.resolve()
    suffix = value.parts[-3:]
    local = root.joinpath(*suffix)
    if local.is_file():
        return local.resolve()
    matches = [path for path in root.rglob(value.name) if path.parts[-3:] == suffix]
    if len(matches) != 1:
        raise FileNotFoundError(f"cannot uniquely resolve {raw!r} under {root}")
    return matches[0].resolve()


def main() -> int:
    args = parse_args()
    partition = args.partition.expanduser().resolve(strict=True)
    submission, methods, raw_v19, controls = load_controls()
    scheduler_path = EVIDENCE / "scheduler_evidence_v1.json"
    sync_path = EVIDENCE / "source_sync_v1.json"
    if (
        sha256(scheduler_path) != SCHEDULER_SHA256
        or sha256(sync_path) != SOURCE_SYNC_SHA256
    ):
        raise ValueError("v20 observed scheduler or source-sync evidence drifted")
    scheduler = json.loads(scheduler_path.read_text())
    sync = json.loads(sync_path.read_text())
    if (
        scheduler.get("passed") is not True
        or scheduler.get("token_bound_backend_exit0") is not True
        or scheduler.get("authorized_scientific_task") != "t83955"
        or scheduler.get("backend_token")
        != "1787705802108374398:365:140143939756032"
        or scheduler.get("formal_held_out_outcomes_read") is not False
        or scheduler.get("formal_confirmation_authorized_or_submitted") is not False
        or sync.get("passed") is not True
        or sync.get("exact_remote_local_match") is not True
        or sync.get("physical_result_count") != 405
        or sync.get("formal_held_out_outcomes_read") is not False
        or inventory(partition) != sync.get("local_inventory")
    ):
        raise ValueError("v20 completion evidence contract failed")

    host_path = partition / "host_execution.json"
    summary_path = partition / "host_summary.json"
    host = json.loads(host_path.read_text())
    summary = json.loads(summary_path.read_text())
    immutable = host.get("immutable_evidence", {})
    if (
        host.get("schema_version")
        != "transit-h2oplus-v20-full-control-host-execution-v1"
        or host.get("status") != "completed"
        or host.get("completed_results") != 405
        or host.get("expected_results") != 405
        or host.get("workers") != 12
        or host.get("training") is not False
        or host.get("formal_held_out_outcomes_read") is not False
        or immutable.get("parent_v19_artifact_integrity_passed") is not True
        or immutable.get("parent_training_stability_passed") is not True
        or immutable.get("data_environment_material_factors_present") is not True
        or immutable.get("required_environment", {}).get("CUDA_VISIBLE_DEVICES") != ""
        or summary.get("schema_version")
        != "transit-h2oplus-v20-full-control-host-summary-v1"
        or summary.get("status") != "completed"
        or summary.get("physical_evaluation_results") != 405
        or summary.get("training") is not False
        or summary.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("v20 host execution boundary failed")

    freeze = controls["freeze"]
    manifests = []
    result_files = []
    signatures = set()
    grid = set()
    counts: Counter[tuple[str, str]] = Counter()
    expected_grid = {
        (algorithm, seed, block, scale)
        for algorithm in ALGORITHMS
        for seed in SEEDS
        for block in BLOCK_MAP
        for scale in SCALES
    }
    for registered_block, (matrix_block, sumo_seed) in BLOCK_MAP.items():
        root = partition / registered_block
        manifest_path = root / "matrix_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        jobs = manifest.get("jobs", [])
        source_hashes = manifest.get("source_sha256", {})
        matrix_partition = manifest.get("matrix_partition", {})
        if (
            manifest.get("schema_version") != "transit-evaluation-matrix-v7"
            or manifest.get("methods_manifest_sha256") != METHODS_SHA256
            or manifest.get("dataset_sha256") != DATASET_SHA256
            or manifest.get("sumo_runtime_signature") != RUNTIME_SIGNATURE
            or len(source_hashes) != 57
            or not sources_bound(source_hashes, freeze)
            or len(jobs) != 135
            or {job.get("environment_block") for job in jobs} != {matrix_block}
            or {job.get("sumo_seed") for job in jobs} != {sumo_seed}
            or matrix_partition.get("full_job_count") != 135
            or matrix_partition.get("selected_job_count") != 135
            or matrix_partition.get("shard_count") != 1
            or matrix_partition.get("shard_index") != 0
        ):
            raise ValueError(f"v20 matrix binding failed: {registered_block}")
        manifests.append(
            {
                "registered_block": registered_block,
                "matrix_block": matrix_block,
                **descriptor(manifest_path),
                "jobs": 135,
                "source_file_count": 57,
            }
        )
        for job in jobs:
            method_name = job.get("method", {}).get("name")
            if method_name not in methods:
                raise ValueError("v20 matrix contains an unregistered method")
            method = methods[method_name]
            signature = job.get("signature")
            output = resolve_output(root, job["output"])
            payload = json.loads(output.read_text())
            scale = finite(job.get("scenario", {}).get("demand_scale"), "demand scale")
            runtime = payload.get("runtime", {})
            backend_api = runtime.get("sumo_runtime_fingerprint", {}).get(
                "backend_api", {}
            )
            completion = finite(
                payload.get("passengers", {}).get("completion_rate"),
                "completion rate",
            )
            unfinished = int(payload.get("passengers", {}).get("unfinished_passengers"))
            finite(
                payload.get("passenger_welfare", {}).get(
                    "accrued_generalized_seconds_per_departed_passenger"
                ),
                "primary endpoint",
            )
            if (
                payload.get("schema_version") != "transit-evaluation-v7"
                or payload.get("formal_held_out_outcomes_read") not in (None, False)
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
                or scale not in SCALES
                or not (0.0 <= completion <= 1.0)
                or unfinished < 0
                or not ledger_and_service_pass(payload)
            ):
                raise ValueError(f"v20 result binding failed: {output}")
            validate_policy(payload, method, raw_v19)
            signatures.add(signature)
            key = (
                method["algorithm"],
                method["training_seed"],
                registered_block,
                scale,
            )
            grid.add(key)
            counts[("algorithm", method["algorithm"])] += 1
            counts[("block", registered_block)] += 1
            counts[("scale", str(scale))] += 1
            result_files.append(descriptor(output))
    if len(result_files) != 405 or len(signatures) != 405 or grid != expected_grid:
        raise ValueError("v20 result grid is not exactly 405 unique episodes")
    if any(counts[("algorithm", algorithm)] != 45 for algorithm in ALGORITHMS):
        raise ValueError("v20 algorithm coverage is not balanced")
    if any(counts[("block", block)] != 135 for block in BLOCK_MAP):
        raise ValueError("v20 block coverage is not balanced")
    if any(counts[("scale", str(scale))] != 135 for scale in SCALES):
        raise ValueError("v20 demand-scale coverage is not balanced")

    output = args.output.expanduser().resolve()
    payload = {
        "schema_version": "transit-h2oplus-v20-full-control-evaluation-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": "development",
        "matrix_type": "full-development-control-matrix",
        "passed": True,
        "artifact_integrity": {
            "passed": True,
            "physical_result_count": 405,
            "unique_job_signatures": 405,
            "physical_behavior_count": 45,
            "fresh_environment_cell_count": 9,
            "matrix_manifest_count": 3,
            "matrix_manifests": manifests,
            "partition_inventory": inventory(partition),
            "all_results_source_protocol_runtime_policy_and_job_bound": True,
            "all_ledgers_and_service_semantics_passed": True,
        },
        "parent_training_stability": {
            "passed": True,
            "parent_v19_audit": controls["bindings"]["parent_v19_audit"],
            "parent_v19_closeout": controls["bindings"]["parent_v19_closeout"],
            "parent_v19_verification": controls["bindings"][
                "parent_v19_verification"
            ],
        },
        "data_environment_validity_risks": {
            "material_factors_present": True,
            "fresh_development_environment_used": True,
            "fresh_development_environment_cell_count": 9,
            "remaining_runtime_shift_from_training": True,
            "adaptive_prior_development_used_for_design": True,
            "audit": controls["bindings"]["data_environment_audit"],
            "verification": controls["bindings"]["data_environment_verification"],
        },
        "coverage": {
            "algorithms": {
                algorithm: counts[("algorithm", algorithm)]
                for algorithm in ALGORITHMS
            },
            "blocks": {block: counts[("block", block)] for block in BLOCK_MAP},
            "demand_scales": {
                str(scale): counts[("scale", str(scale))] for scale in SCALES
            },
        },
        "bindings": {
            "submission": descriptor(SUBMISSION),
            "scheduler_evidence": descriptor(scheduler_path),
            "source_sync": descriptor(sync_path),
            "host_execution": descriptor(host_path),
            "host_summary": descriptor(summary_path),
            "immutable_controls": controls["bindings"],
        },
        "result_files": sorted(result_files, key=lambda item: item["path"]),
        "prior_v14_through_v19_development_outcomes_used_for_design": True,
        "local_sumo_executed": False,
        "formal_held_out_outcomes_read": False,
        "supports_formal_held_out_effect_claim": False,
        "formal_confirmation_authorized_or_submitted": False,
        "claim_boundary": (
            "This independent audit establishes v20 artifact integrity and provenance "
            "only. Scientific gate results must come from the unchanged frozen analyzer."
        ),
    }
    atomic_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256(output),
                "results": 405,
                "passed": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
