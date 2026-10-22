#!/usr/bin/env python3
"""Independently audit the synchronized v20 formal-confirmation matrix."""

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
RESULTS = REPRO / "h2oplus_v20_formal_confirmation_results"
PARTITION = RESULTS / "evaluation_v4/single_jtl110gpu"
EVIDENCE = RESULTS / "evidence/evaluation_v4"
SUBMISSION = REPRO / "h2oplus_v20_formal_confirmation_submission_v4.json"
SYNC = REPRO / "h2oplus_v20_formal_confirmation_minimal_sync_v4.json"
SCHEDULER = EVIDENCE / "scheduler_evidence_v4.json"

SUBMISSION_SHA256 = "6dea4c147d9ee987ef3f9bb970c0e9f5b79be6a9e85a8f8a92fc8e0365da35dc"
SYNC_SHA256 = "c593de405c43cf2c28085072fff593a5cd913526f89b60e8ee61d9b8aa1eed88"
SCHEDULER_SHA256 = "19efdd5b7942eda3be9ed122950eb30130364cc5fde56ca8252b7dac252df190"
METHODS_SHA256 = "b4bcad09e13cb50c94d5541f2a52340ce99f15b281b5ddef57be38e792d052cd"
SOURCE_FREEZE_SHA256 = "70f387eeb232d5021f33b12ce5e653d0dea500b905827d7c8d525fb738e23c25"
DATASET_SHA256 = "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899"
RUNTIME_LOCK_SHA256 = "ac60ae58e0e6db6bb1386ffdfe0d164b96f81e49e4983b8b062ceddb65da51a4"
RUNTIME_SIGNATURE = "264d3c412b8e6c42320995208d2e009eb809b85626047206c4de752c2dae06e7"
ANALYZER_SHA256 = "ea69c2fe6b31de622952e2328f749f337ae94cb651a39633bdaec1031f9fdc4b"
V19_PROTOCOL_SHA256 = "fa5f629588d35b1435e28bd39931f49d9851811ea3e308d62672f9eacf1f7184"

SEEDS = tuple(range(6201, 6206))
SCALES = (0.75, 1.0, 1.25)
CANDIDATE = "h2oplus-v19-completion-safety-quarter-load-sixty"
DIRECT_PARENT = "h2oplus-v19-completion-safety-baseline"
RAW_PARENT = "h2oplus-v14-terminal-quarter"
LINEAGE_CONTROLS = (
    "h2oplus",
    "sac-online",
    "h2oplus-v10-delayed-actor",
    "h2oplus-v11-actor-hold",
    "h2oplus-v12-actor-ramp",
    "h2oplus-v13-proximal-strong",
)
ALGORITHMS = (CANDIDATE, DIRECT_PARENT, RAW_PARENT, *LINEAGE_CONTROLS)
EXPECTED_ROLES = {
    CANDIDATE: "predesignated-candidate",
    DIRECT_PARENT: "direct-behavior-parent",
    RAW_PARENT: "raw-actor-parent",
    **{algorithm: "independent-lineage-control" for algorithm in LINEAGE_CONTROLS},
}
BLOCK_MAP = {
    f"v20-formal-confirmation-{index:02d}": (f"sumo-seed-{8100 + index}", 8100 + index)
    for index in range(1, 11)
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    resolved = path.resolve(strict=True)
    return str(resolved.relative_to(ROOT))


def descriptor(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": relative(resolved),
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def load_json_record(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.resolve(strict=True)
    raw = resolved.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {resolved}")
    return payload, {
        "path": relative(resolved),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
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


def resolve_owned(owner: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve(strict=True)


def resolve_output(root: Path, raw: str) -> Path:
    value = Path(raw).expanduser()
    suffix = value.parts[-3:]
    local = root.joinpath(*suffix)
    if local.is_file():
        return local.resolve()
    matches = [path for path in root.rglob(value.name) if path.parts[-3:] == suffix]
    if len(matches) == 1:
        return matches[0].resolve()
    raise FileNotFoundError(f"cannot resolve formal output {raw!r} under {root}")


def checked_submission_bindings(submission: dict[str, Any]) -> dict[str, dict[str, Any]]:
    expected = {
        "data_environment_audit",
        "data_environment_verification",
        "dataset",
        "deployment_correction",
        "deployment_manager",
        "deployment_manifest",
        "deployment_sync",
        "design_audit",
        "design_verification",
        "dispatch_wrapper",
        "environment",
        "execution_authorization",
        "local_deployment_audit",
        "methods",
        "parent_v20_analysis",
        "parent_v20_audit",
        "parent_v20_closeout",
        "parent_v20_verification",
        "preparation_authorization",
        "protocol",
        "rationale",
        "registration",
        "remote_verification",
        "runtime_lock",
        "source_freeze",
        "v1_incident_closeout",
        "v1_incident_verification",
        "v2_incident_closeout",
        "v2_incident_verification",
        "v3_incident_closeout",
        "v3_incident_verification",
    }
    bindings = submission.get("bindings", {})
    if set(bindings) != expected:
        raise ValueError("formal submission binding set drifted")
    records = {}
    for name, item in bindings.items():
        path = Path(item.get("path", "")).expanduser().resolve(strict=True)
        if ROOT not in path.parents or sha256(path) != item.get("sha256"):
            raise ValueError(f"formal immutable binding drifted: {name}")
        records[name] = descriptor(path)
    return records


def load_methods(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text())
    selections = payload.get("methods", [])
    parents = payload.get("parent_method_manifests", {})
    if (
        payload.get("schema_version")
        != "transit-evaluation-methods-v20-formal-confirmation-v1"
        or payload.get("status")
        != "frozen-before-any-v20-formal-confirmation-outcome"
        or payload.get("study_stage") != "formal-confirmation"
        or payload.get("matrix_type") != "endpoint-only-formal-confirmation"
        or payload.get("formal_held_out_outcomes_read") is not False
        or payload.get("formal_confirmation_outcomes_read") is not False
        or payload.get("physical_method_count") != 45
        or payload.get("logical_method_count") != 45
        or payload.get("expected_new_remote_sumo_episodes") != 1350
        or payload.get("predesignated_candidate") != CANDIDATE
        or payload.get("algorithms") != list(ALGORITHMS)
        or payload.get("training_seeds") != list(SEEDS)
        or len(selections) != 45
        or set(parents) != {"v14", "v19"}
    ):
        raise ValueError("formal methods contract drifted")

    source_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for source, item in parents.items():
        parent_path = resolve_owned(path, item["path"])
        if sha256(parent_path) != item.get("sha256"):
            raise ValueError(f"formal parent methods drifted: {source}")
        parent = json.loads(parent_path.read_text())
        source_rows[source] = {row["name"]: row for row in parent.get("methods", [])}

    methods: dict[str, dict[str, Any]] = {}
    pairs = set()
    for selected in selections:
        source = selected.get("source_manifest")
        name = selected.get("name")
        parent = source_rows.get(source, {}).get(name)
        if parent is None:
            raise ValueError(f"formal method absent from parent manifest: {name}")
        algorithm = parent.get("algorithm")
        seed = parent.get("training_seed")
        pair = (algorithm, seed)
        if (
            name in methods
            or pair in pairs
            or algorithm not in ALGORITHMS
            or seed not in SEEDS
            or parent.get("online_events") != 25_000
            or selected.get("algorithm") != algorithm
            or selected.get("training_seed") != seed
            or selected.get("role") != EXPECTED_ROLES[algorithm]
            or selected.get("checkpoint_sha256") != parent.get("checkpoint_sha256")
            or selected.get("actor_tensor_sha256")
            != parent.get("actor_tensor_sha256")
            or selected.get("predesignated_candidate")
            is not (algorithm == CANDIDATE)
        ):
            raise ValueError(f"malformed formal method: {name}")
        methods[name] = dict(parent)
        pairs.add(pair)
    expected = {(algorithm, seed) for algorithm in ALGORITHMS for seed in SEEDS}
    if pairs != expected or len(methods) != 45:
        raise ValueError("formal methods are not the exact 9-by-5 grid")
    return methods, source_rows["v19"]


def validate_policy(
    payload: dict[str, Any], method: dict[str, Any], raw_v19: dict[str, dict[str, Any]]
) -> None:
    policy = payload.get("policy", {})
    training = policy.get("training_metadata", {})
    mismatches = {
        "sha256": (policy.get("sha256"), method.get("checkpoint_sha256")),
        "actor_tensor_sha256": (
            policy.get("actor_tensor_sha256"),
            method.get("actor_tensor_sha256"),
        ),
        "training_seed": (training.get("training_seed"), method.get("training_seed")),
        "dataset_sha256": (training.get("dataset_sha256"), DATASET_SHA256),
        "run_signature": (
            training.get("run_signature"),
            method.get("training_run_signature"),
        ),
        "training_checkpoint_sha256": (
            training.get("training_checkpoint_sha256"),
            method.get("training_checkpoint_sha256"),
        ),
        "training_admission_sha256": (
            training.get("training_admission", {}).get("sha256"),
            method.get("training_admission_sha256"),
        ),
    }
    mismatches = {key: value for key, value in mismatches.items() if value[0] != value[1]}
    if (
        mismatches
        or training.get("phase") not in {"complete", "online_training"}
        or training.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError(f"formal policy provenance drifted: {method.get('name')}: {mismatches}")
    if method.get("algorithm") not in {CANDIDATE, DIRECT_PARENT}:
        return
    raw = raw_v19[method["name"]]
    reserve = policy.get("completion_safety_reserve", {})
    variant = method["algorithm"][len("h2oplus-v19-completion-safety-") :]
    if (
        policy.get("artifact_lineage") != "v19-completion-safety-reserve"
        or policy.get("parent_method") != raw.get("parent_method")
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
        or reserve.get("v19_protocol_sha256") != V19_PROTOCOL_SHA256
        or reserve.get("checkpoint_selection_by_behavior") is not False
        or reserve.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError(f"formal v19 policy transform drifted: {method['name']}")


def load_controls() -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    if sha256(SUBMISSION) != SUBMISSION_SHA256:
        raise ValueError("formal submission identity drifted")
    submission = json.loads(SUBMISSION.read_text())
    task = submission.get("task", {})
    if (
        submission.get("schema_version")
        != "transit-h2oplus-v20-formal-confirmation-submission-v4"
        or submission.get("status") != "submitted"
        or submission.get("study_stage") != "formal-confirmation"
        or submission.get("matrix_type") != "endpoint-only-formal-confirmation"
        or submission.get("task_count") != 1
        or submission.get("physical_behavior_count") != 45
        or submission.get("formal_block_count") != 10
        or submission.get("jobs_per_block") != 135
        or submission.get("total_remote_sumo_episodes") != 1350
        or submission.get("predesignated_candidate") != CANDIDATE
        or submission.get("formal_held_out_outcome_access_authorized") is not True
        or submission.get("formal_held_out_outcomes_read") is not False
        or submission.get("training") is not False
        or submission.get("prior_incident_results_reused") is not False
        or submission.get("scheduler_result_sync_opted_in") is not False
        or submission.get("checkpoint_optimizer_replay_or_workspace_sync_authorized")
        is not False
        or task.get("task_id") != "t84212"
        or task.get("host") != "jtl110gpu"
        or task.get("cpu_cores") != 12
        or task.get("ram_mb") != 32768
        or task.get("est_vram_mb") != 0
        or task.get("expected_jobs") != 1350
    ):
        raise ValueError("formal submission contract drifted")
    controls = checked_submission_bindings(submission)
    if (
        controls["methods"]["sha256"] != METHODS_SHA256
        or controls["source_freeze"]["sha256"] != SOURCE_FREEZE_SHA256
        or controls["dataset"]["sha256"] != DATASET_SHA256
        or controls["runtime_lock"]["sha256"] != RUNTIME_LOCK_SHA256
    ):
        raise ValueError("formal core immutable identity drifted")

    methods_path = ROOT / controls["methods"]["path"]
    methods, raw_v19 = load_methods(methods_path)
    freeze = json.loads((ROOT / controls["source_freeze"]["path"]).read_text())
    analyzer_path = ROOT / freeze.get("analyzer", {}).get("path", "")
    if (
        freeze.get("schema_version")
        != "transit-h2oplus-v20-formal-confirmation-source-freeze-v1"
        or freeze.get("status")
        != "frozen-before-any-v20-formal-confirmation-outcome"
        or freeze.get("formal_held_out_outcomes_read") is not False
        or freeze.get("formal_confirmation_outcomes_read") is not False
        or freeze.get("matrix_source_count") != 57
        or len(freeze.get("matrix_sources", [])) != 57
        or freeze.get("physical_behavior_count") != 45
        or freeze.get("authorized_remote_sumo_episodes") != 0
        or freeze.get("planned_remote_sumo_episodes") != 1350
        or freeze.get("analyzer", {}).get("sha256") != ANALYZER_SHA256
        or sha256(analyzer_path.resolve(strict=True)) != ANALYZER_SHA256
    ):
        raise ValueError("formal source freeze contract drifted")

    parent_analysis = json.loads(
        (ROOT / controls["parent_v20_analysis"]["path"]).read_text()
    )
    parent_audit = json.loads((ROOT / controls["parent_v20_audit"]["path"]).read_text())
    parent_closeout = json.loads(
        (ROOT / controls["parent_v20_closeout"]["path"]).read_text()
    )
    parent_verification = json.loads(
        (ROOT / controls["parent_v20_verification"]["path"]).read_text()
    )
    data_audit = json.loads(
        (ROOT / controls["data_environment_audit"]["path"]).read_text()
    )
    data_verification = json.loads(
        (ROOT / controls["data_environment_verification"]["path"]).read_text()
    )
    if (
        parent_analysis.get("formal_confirmation_eligible") is not True
        or parent_analysis.get("artifact_integrity", {}).get("passed") is not True
        or parent_analysis.get("parent_training_stability", {}).get("passed") is not True
        or parent_audit.get("passed") is not True
        or parent_audit.get("artifact_integrity", {}).get("passed") is not True
        or parent_audit.get("parent_training_stability", {}).get("passed") is not True
        or parent_closeout.get("passed") is not True
        or parent_closeout.get("formal_confirmation_eligible") is not True
        or parent_verification.get("passed") is not True
        or parent_verification.get("formal_confirmation_eligible") is not True
        or parent_verification.get("parent_training_stability_verified") is not True
        or data_audit.get("conclusions", {}).get(
            "material_data_or_environment_factors_present"
        )
        is not True
        or data_verification.get("passed") is not True
    ):
        raise ValueError("formal parent or data/environment evidence failed")
    return submission, methods, raw_v19, freeze, controls


def main() -> int:
    args = parse_args()
    partition = args.partition.expanduser().resolve(strict=True)
    submission, methods, raw_v19, freeze, controls = load_controls()

    if sha256(SYNC) != SYNC_SHA256 or sha256(SCHEDULER) != SCHEDULER_SHA256:
        raise ValueError("formal sync or scheduler evidence identity drifted")
    sync = json.loads(SYNC.read_text())
    scheduler = json.loads(SCHEDULER.read_text())
    if (
        sync.get("passed") is not True
        or sync.get("task_id") != "t84212"
        or sync.get("allowlist_only") is not True
        or sync.get("scientific_result_json_count") != 1350
        or sync.get("matrix_manifest_count") != 10
        or sync.get("checkpoint_optimizer_replay_or_workspace_files_synchronized")
        is not False
        or sync.get("scientific_numeric_outcomes_parsed_by_sync") is not False
        or scheduler.get("passed") is not True
        or scheduler.get("authorized_scientific_task") != "t84212"
        or scheduler.get("token_bound_backend_exit0") is not True
        or scheduler.get("backend_returncode") != 0
        or scheduler.get("physical_result_count") != 1350
        or scheduler.get("matrix_manifest_count") != 10
        or scheduler.get("minimal_allowlist_sync_passed") is not True
        or scheduler.get("formal_held_out_outcomes_read_before_analysis") is not False
        or scheduler.get("task", {}).get("status") != "done"
    ):
        raise ValueError("formal completion evidence contract failed")

    host_path = partition / "host_execution.json"
    summary_path = partition / "host_summary.json"
    host = json.loads(host_path.read_text())
    summary = json.loads(summary_path.read_text())
    immutable = host.get("immutable_evidence", {})
    required_environment = {
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcurl.so.4",
        "CC": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc",
        "CPATH": "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/v7_formal_v2_training_v7r1/workspace/reproducibility/toolchain/libxcrypt-4.4.36-hd590300_1/include",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "H2OPLUS_V19_COMPLETION_SAFETY_PROTOCOL_SHA256": V19_PROTOCOL_SHA256,
        "H2OPLUS_V20_FORMAL_CONFIRMATION_PROTOCOL_SHA256": controls["protocol"][
            "sha256"
        ],
    }
    host_binding_names = {
        "data_environment_audit",
        "data_environment_verification",
        "dataset",
        "deployment_manifest",
        "design_audit",
        "design_verification",
        "environment",
        "execution_authorization",
        "methods",
        "parent_v20_analysis",
        "parent_v20_audit",
        "parent_v20_closeout",
        "parent_v20_verification",
        "preparation_authorization",
        "protocol",
        "rationale",
        "registration",
        "runtime_lock",
        "source_freeze",
    }
    for name in host_binding_names:
        if immutable.get(name, {}).get("sha256") != controls[name]["sha256"]:
            raise ValueError(f"formal host immutable binding drifted: {name}")
    if (
        host.get("schema_version")
        != "transit-h2oplus-v20-formal-confirmation-host-execution-v1"
        or host.get("status") != "completed"
        or host.get("study_stage") != "formal-confirmation"
        or host.get("matrix_type") != "endpoint-only-formal-confirmation"
        or host.get("completed_results") != 1350
        or host.get("expected_results") != 1350
        or len(host.get("completed_blocks", [])) != 10
        or host.get("workers") != 12
        or host.get("training") is not False
        or host.get("formal_held_out_outcome_access_authorized") is not True
        or host.get("formal_held_out_outcomes_read") is not False
        or immutable.get("parent_artifact_integrity_passed") is not True
        or immutable.get("parent_training_stability_passed") is not True
        or immutable.get("formal_held_out_outcome_access_authorized") is not True
        or immutable.get("required_environment") != required_environment
        or immutable.get("runtime_lock", {}).get("runtime_signature")
        != RUNTIME_SIGNATURE
        or immutable.get("runtime_fingerprint", {}).get("portable_signature")
        != RUNTIME_SIGNATURE
        or summary.get("schema_version")
        != "transit-h2oplus-v20-formal-confirmation-host-summary-v1"
        or summary.get("status") != "completed"
        or summary.get("physical_evaluation_results") != 1350
        or summary.get("training") is not False
        or summary.get("formal_held_out_outcome_access_authorized") is not True
        or summary.get("formal_held_out_outcomes_read") is not False
        or len(summary.get("matrix_manifests", [])) != 10
    ):
        raise ValueError("formal host execution boundary failed")

    environment = json.loads((ROOT / controls["environment"]["path"]).read_text())
    expected_blocks = {
        row["name"]: row for row in environment.get("blocks", [])
    }
    if (
        set(expected_blocks) != set(BLOCK_MAP)
        or environment.get("demand_scales") != list(SCALES)
        or environment.get("formal_held_out_outcomes_read") is not False
        or environment.get("prior_used_formal_v7_environment_reused") is not False
    ):
        raise ValueError("formal environment design drifted")

    manifests = []
    result_files = []
    signatures = set()
    outputs = set()
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
        block_design = expected_blocks[registered_block]
        if block_design.get("sumo_seed") != sumo_seed:
            raise ValueError(f"formal environment seed drifted: {registered_block}")
        root = partition / registered_block
        manifest_path = root / "matrix_manifest.json"
        manifest, manifest_record = load_json_record(manifest_path)
        jobs = manifest.get("jobs", [])
        source_hashes = manifest.get("source_sha256", {})
        matrix_partition = manifest.get("matrix_partition", {})
        if (
            manifest.get("schema_version") != "transit-evaluation-matrix-v7"
            or manifest.get("methods_manifest_sha256") != METHODS_SHA256
            or manifest.get("dataset_sha256") != DATASET_SHA256
            or manifest.get("sumo_runtime_signature") != RUNTIME_SIGNATURE
            or manifest.get("sumo_backend") != "libsumo"
            or manifest.get("sumo_version") != "1.25.0"
            or manifest.get("workers") != 12
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
            raise ValueError(f"formal matrix binding failed: {registered_block}")
        manifests.append(
            {
                "registered_block": registered_block,
                "matrix_block": matrix_block,
                **manifest_record,
                "jobs": 135,
                "source_file_count": 57,
            }
        )
        for job in jobs:
            method_name = job.get("method", {}).get("name")
            if method_name not in methods:
                raise ValueError("formal matrix contains an unregistered method")
            method = methods[method_name]
            signature = job.get("signature")
            output = resolve_output(root, job["output"])
            payload, result_record = load_json_record(output)
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
                or not isinstance(signature, str)
                or signature in signatures
                or output in outputs
                or scale not in SCALES
                or not (0.0 <= completion <= 1.0)
                or unfinished < 0
                or not ledger_and_service_pass(payload)
            ):
                raise ValueError(f"formal result binding failed: {output}")
            validate_policy(payload, method, raw_v19)
            signatures.add(signature)
            outputs.add(output)
            key = (
                method["algorithm"],
                method["training_seed"],
                registered_block,
                scale,
            )
            grid.add(key)
            counts[("algorithm", method["algorithm"])] += 1
            counts[("seed", str(method["training_seed"]))] += 1
            counts[("block", registered_block)] += 1
            counts[("scale", str(scale))] += 1
            result_files.append(result_record)

    actual_results = set(partition.rglob("sumo_seed*.json"))
    actual_manifests = set(partition.rglob("matrix_manifest.json"))
    if (
        len(result_files) != 1350
        or len(signatures) != 1350
        or set(outputs) != actual_results
        or len(actual_manifests) != 10
        or grid != expected_grid
    ):
        raise ValueError("formal result grid is not exactly 1350 unique episodes")
    if any(counts[("algorithm", algorithm)] != 150 for algorithm in ALGORITHMS):
        raise ValueError("formal algorithm coverage is not balanced")
    if any(counts[("seed", str(seed))] != 270 for seed in SEEDS):
        raise ValueError("formal seed coverage is not balanced")
    if any(counts[("block", block)] != 135 for block in BLOCK_MAP):
        raise ValueError("formal block coverage is not balanced")
    if any(counts[("scale", str(scale))] != 450 for scale in SCALES):
        raise ValueError("formal demand-scale coverage is not balanced")

    output = args.output.expanduser().resolve()
    payload = {
        "schema_version": "transit-h2oplus-v20-formal-confirmation-evaluation-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": "formal-confirmation",
        "matrix_type": "endpoint-only-formal-confirmation",
        "passed": True,
        "artifact_integrity": {
            "passed": True,
            "physical_result_count": 1350,
            "unique_job_signatures": 1350,
            "physical_behavior_count": 45,
            "fresh_formal_environment_cell_count": 30,
            "matrix_manifest_count": 10,
            "matrix_manifests": manifests,
            "all_results_source_protocol_runtime_policy_and_job_bound": True,
            "all_ledgers_and_service_semantics_passed": True,
        },
        "parent_training_stability": {
            "passed": True,
            "parent_v20_analysis": controls["parent_v20_analysis"],
            "parent_v20_audit": controls["parent_v20_audit"],
            "parent_v20_closeout": controls["parent_v20_closeout"],
            "parent_v20_verification": controls["parent_v20_verification"],
        },
        "data_environment_validity_risks": {
            "material_factors_present": True,
            "fresh_formal_environment_used": True,
            "fresh_formal_environment_cell_count": 30,
            "remaining_runtime_shift_from_training": True,
            "adaptive_prior_development_used_for_design": True,
            "audit": controls["data_environment_audit"],
            "verification": controls["data_environment_verification"],
        },
        "coverage": {
            "algorithms": {
                algorithm: counts[("algorithm", algorithm)] for algorithm in ALGORITHMS
            },
            "training_seeds": {
                str(seed): counts[("seed", str(seed))] for seed in SEEDS
            },
            "blocks": {block: counts[("block", block)] for block in BLOCK_MAP},
            "demand_scales": {
                str(scale): counts[("scale", str(scale))] for scale in SCALES
            },
        },
        "bindings": {
            "submission": descriptor(SUBMISSION),
            "scheduler_evidence": descriptor(SCHEDULER),
            "minimal_sync": descriptor(SYNC),
            "host_execution": descriptor(host_path),
            "host_summary": descriptor(summary_path),
            "immutable_controls": controls,
        },
        "result_files": sorted(result_files, key=lambda item: item["path"]),
        "formal_held_out_outcome_access_authorized": True,
        "formal_held_out_result_files_read_after_terminal_authorization": True,
        "scientific_effects_computed": False,
        "formal_held_out_outcomes_read": True,
        "local_sumo_executed": False,
        "training_executed": False,
        "supports_formal_held_out_effect_claim": False,
        "claim_boundary": (
            "This audit establishes terminal artifact integrity, provenance, exact "
            "coverage and service-ledger validity only. Formal scientific effects "
            "must come from the unchanged frozen analyzer."
        ),
    }
    atomic_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256(output),
                "results": 1350,
                "passed": True,
                "scientific_effects_computed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
