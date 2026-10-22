#!/usr/bin/env python3
"""Independently verify the H2O+ v14 terminal-taper evaluation file manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reproducibility/h2oplus_v14_terminal_taper_results"
PARTITIONS = RESULTS / "evaluation_partitions_v1"
CANONICAL = RESULTS / "evaluation_canonical_v1"
DEFAULT_MANIFEST = RESULTS / "evaluation_file_manifest_v1.json"
DEFAULT_OUTPUT = RESULTS / "evaluation_file_manifest_verification_v1.json"

EXPECTED_CONTROLS = {
    "reproducibility/h2oplus_v14_terminal_taper_registration_v1.json": "6be53642facea16013dc90f84f60a1d861c112e08a9b748a61011eb985dd2a41",
    "reproducibility/h2oplus_v14_terminal_taper_authorization_v1.json": "fc3abf9d22a99a8b6fe48e2a74e778f4515d77f9b49943c0f690809d17126f8b",
    "reproducibility/h2oplus_v12_staged_release_environment_v2.json": "bfe97d430921f38fbbec04490ac980e330d2000f6c1ad4d62627c95d3deba5e3",
    "reproducibility/h2oplus_v14_terminal_taper_training_plan_v1.json": "71e43c0a6ec063f2b96459388835e0b29277695d1714f375d5115921cbd8c831",
    "reproducibility/h2oplus_v14_terminal_taper_protocol_v1.json": "ba275fc4c134feadb564b49548699571ca5bf6bfd1aee529759d6ce791df12c5",
    "reproducibility/h2oplus_v14_terminal_taper_freeze_v1.json": "3a1eb6a31f6cc4708714541917ef101ce68a3eddae4b64228d27d8ce529b5b5d",
    "reproducibility/h2oplus_v14_terminal_taper_evaluation_methods_v1.json": "560a91eaca3e279cb986daedf6978c8275ce4c937859d60cd2eed37e27e52c3b",
    "reproducibility/h2oplus_v14_terminal_taper_evaluation_source_freeze_v1.json": "82e1404e8471fe01ad4fa4311ce442b8348dcc0444855424cf2b89db281b49bb",
    "reproducibility/h2oplus_v14_terminal_taper_evaluation_deployment_local_verification_v1.json": "f1e7b0347b30f1b9ac62f353b18836a5a3fe3f11c6f0e805a761be0e1a763812",
    "reproducibility/h2oplus_v14_terminal_taper_evaluation_deployment_sync_v1.json": "75eb8aa753387d9122c2d5a407069f54e75952c53c1bb55618c5ac0739e7161e",
    "reproducibility/h2oplus_v14_terminal_taper_evaluation_deployment_v1_jtl110gpu_verification.json": "bbf7fbb67185b80230cd898c2d74c9338a4b7c9fec57d39562e601a08add6184",
    "reproducibility/h2oplus_v14_terminal_taper_evaluation_submission_v1.json": "cf1745d10adf13548cfea82563428a46646218a0ecb31b591d15d07727783b35",
    "reproducibility/h2oplus_v14_terminal_taper_branch_closeout_v1.json": "5d609f76f6d139dc8982a0ce4f8c9e95a4f0e5f4c4d9655b3545fa15270d2ffc",
    "reproducibility/h2oplus_v14_terminal_taper_branch_closeout_verification_v1.json": "56cf282615364dc1b93bbf118cd1f3101b20fa2398ca87e08b202322c833c15b",
    "reproducibility/h2oplus_v14_terminal_taper_results/branch_audit_v1.json": "51296e4eb91cf785dd484ae726f172d6828b16b1aeaa8af8469a6356b45ea136",
    "reproducibility/h2oplus_v14_terminal_taper_results/branch_file_manifest_v1.json": "bb1e4bb04e437091c137595399abfae1d4f80294b37a471c39ee53fe3eaae558",
    "reproducibility/deployments/h2oplus_v14_terminal_taper_evaluation_v1/workspace/deployment_manifest.json": "fe227722da303dd38be684222845bbbd84cb80f14e0cf2fb175b333c2495ee04",
    "reproducibility/h2oplus_v8_stage_a_sumo_runtime_lock_py38.json": "ac60ae58e0e6db6bb1386ffdfe0d164b96f81e49e4983b8b062ceddb65da51a4",
    "reproducibility/h2oplus_v14_terminal_taper_results/evidence/evaluation_v1/scheduler_evidence_v1.json": None,
    "reproducibility/h2oplus_v14_terminal_taper_results/evidence/evaluation_v1/source_sync_v1.json": None,
}
EXPECTED_DATASET = "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899"
EXPECTED_RUNTIME = "264d3c412b8e6c42320995208d2e009eb809b85626047206c4de752c2dae06e7"
METHODS_PATH = REPO / "reproducibility/h2oplus_v14_terminal_taper_evaluation_methods_v1.json"
SOURCE_FREEZE_PATH = REPO / "reproducibility/h2oplus_v14_terminal_taper_evaluation_source_freeze_v1.json"
BLOCKS = (
    "v12-staged-release-development-01",
    "v12-staged-release-development-02",
    "v12-staged-release-development-03",
)
SCALES = {0.85, 1.0, 1.15}
RESULT_SOURCE_SUFFIXES = {
    "SimpleSAC/evaluate_transit_v11_actor_hold.py",
    "SimpleSAC/evaluate_transit_v12_staged_release.py",
    "SimpleSAC/evaluate_transit_v13_proximal_release.py",
    "SimpleSAC/evaluate_transit_v14_terminal_taper.py",
    "reproducibility/run_evaluation_matrix_v11_actor_hold.py",
    "reproducibility/run_evaluation_matrix_v12_staged_release.py",
    "reproducibility/run_evaluation_matrix_v13_proximal_release.py",
    "reproducibility/run_evaluation_matrix_v14_terminal_taper.py",
}
V13_PROXIMAL_RATES = {
    "proximal-weak": 1e-5,
    "proximal-medium": 2.5e-5,
    "proximal-strong": 5e-5,
}
V14_TERMINAL_MULTIPLIERS = {
    "terminal-half": 0.5,
    "terminal-quarter": 0.25,
    "terminal-zero": 0.0,
}
V14_FULL_RATE_EQUIVALENT_UPDATES = {
    "terminal-half": 10249.449999999913,
    "terminal-quarter": 8999.449999999915,
    "terminal-zero": 7749.449999999926,
}
METHOD_FIELDS = {
    "actor_tensor_sha256",
    "actual_online_events",
    "algorithm",
    "artifact_lineage",
    "checkpoint_sha256",
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


def _validate_training(payload: dict, method: dict, path: Path) -> None:
    training = payload.get("policy", {}).get("training_metadata", {})
    admission = training.get("training_admission", {})
    common = {
        "training_seed": method["training_seed"],
        "run_signature": method["training_run_signature"],
        "training_checkpoint_sha256": method["training_checkpoint_sha256"],
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
    }
    if any(training.get(key) != value for key, value in common.items()):
        raise ValueError(f"independent common training provenance failed: {path}")
    if method["artifact_lineage"] == "v8-fewshot":
        fields = {
            "method": method["algorithm"],
            "fewshot_online_events_nominal": method["online_events"],
            "fewshot_online_events_actual": method["actual_online_events"],
            "fewshot_learner_update_steps": method["learner_update_steps"],
            "fewshot_ratio_optimizer_steps": method["ratio_optimizer_steps"],
        }
        protocol_hash = training.get("fewshot_protocol", {}).get("sha256")
        admission_schema = "transit-training-admission-v8-fewshot-development-v1"
    elif method["artifact_lineage"] == "v10-delayed":
        variant = method["algorithm"][len("h2oplus-v10-") :]
        fields = {
            "method": "h2oplus",
            "variant": variant,
            "v10_online_events_nominal": method["online_events"],
            "v10_online_events_actual": method["actual_online_events"],
            "v10_learner_update_steps": method["learner_update_steps"],
            "v10_ratio_optimizer_steps": method["ratio_optimizer_steps"],
        }
        intervention = training.get("v10_intervention", {})
        if intervention.get("variant") != variant or intervention.get("unit_ratio_weights") is not True:
            raise ValueError(f"independent v10 intervention provenance failed: {path}")
        protocol_hash = training.get("v10_protocol", {}).get("sha256")
        admission_schema = "transit-training-admission-v10-delayed-development-v4"
    elif method["artifact_lineage"] == "v11-actor-hold":
        fields = {
            "method": "h2oplus",
            "variant": "actor-hold-online-only",
            "stage": "candidate",
            "v11_online_events_nominal": method["online_events"],
            "v11_online_events_actual": method["actual_online_events"],
            "v11_learner_update_steps": method["learner_update_steps"],
            "v11_ratio_optimizer_steps": method["ratio_optimizer_steps"],
            "v11_actor_tensor_sha256": method["actor_tensor_sha256"],
            "v11_temperature_tensor_sha256": method["temperature_tensor_sha256"],
        }
        intervention = training.get("v11_intervention", {})
        if (
            intervention.get("variant") != "actor-hold-online-only"
            or intervention.get("unit_ratio_weights") is not True
            or intervention.get("checkpoint_selection_by_behavior") is not False
        ):
            raise ValueError(f"independent v11 intervention provenance failed: {path}")
        protocol_hash = training.get("v11_protocol", {}).get("sha256")
        admission_schema = "transit-training-admission-v11-actor-hold-development-v2"
    elif method["artifact_lineage"] == "v12-staged-release":
        variant = method["algorithm"][len("h2oplus-v12-") :]
        fields = {
            "method": "h2oplus",
            "variant": variant,
            "stage": "branch",
            "phase": "complete",
            "v12_online_events_nominal": method["online_events"],
            "v12_online_events_actual": method["actual_online_events"],
            "v12_learner_update_steps": method["learner_update_steps"],
            "v12_ratio_optimizer_steps": method["ratio_optimizer_steps"],
            "v12_actor_tensor_sha256": method["actor_tensor_sha256"],
            "v12_temperature_tensor_sha256": method["temperature_tensor_sha256"],
        }
        flags = {
            "abrupt-control": (False, False, False),
            "actor-ramp": (True, False, False),
            "temperature-ramp": (False, True, False),
            "actor-anchor": (False, False, True),
            "staged-combined": (True, True, True),
        }[variant]
        intervention = training.get("v12_intervention", {})
        if (
            intervention.get("variant") != variant
            or intervention.get("actor_learning_rate_ramp") is not flags[0]
            or intervention.get("temperature_learning_rate_ramp") is not flags[1]
            or intervention.get("actor_state_anchor") is not flags[2]
            or intervention.get("unit_ratio_weights") is not True
            or intervention.get("checkpoint_selection_by_behavior") is not False
        ):
            raise ValueError(f"independent v12 intervention provenance failed: {path}")
        protocol_hash = training.get("v12_protocol", {}).get("sha256")
        admission_schema = "transit-training-admission-v12-staged-release-development-v2"
    elif method["artifact_lineage"] == "v13-proximal-release":
        variant = method["algorithm"][len("h2oplus-v13-") :]
        rate = V13_PROXIMAL_RATES[variant]
        fields = {
            "method": "h2oplus",
            "variant": variant,
            "stage": "branch",
            "phase": "complete",
            "v13_online_events_nominal": method["online_events"],
            "v13_online_events_actual": method["actual_online_events"],
            "v13_learner_update_steps": method["learner_update_steps"],
            "v13_ratio_optimizer_steps": method["ratio_optimizer_steps"],
            "v13_actor_tensor_sha256": method["actor_tensor_sha256"],
            "v13_temperature_tensor_sha256": method["temperature_tensor_sha256"],
        }
        intervention = training.get("v13_intervention", {})
        retention = float(
            intervention.get("no_gradient_anchor_retention_at_25k", math.nan)
        )
        if (
            intervention.get("variant") != variant
            or intervention.get("proximal_rate_per_actor_update") != rate
            or intervention.get("registered_post_10k_actor_updates") != 15_000
            or not math.isclose(
                retention,
                (1.0 - rate) ** 15_000,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or intervention.get("predesignated_candidate")
            is not (method["algorithm"] == "h2oplus-v13-proximal-medium")
            or intervention.get("unit_ratio_weights") is not True
            or intervention.get("checkpoint_selection_by_behavior") is not False
        ):
            raise ValueError(
                f"independent v13 intervention provenance failed: {path}"
            )
        protocol_hash = training.get("v13_protocol", {}).get("sha256")
        admission_schema = (
            "transit-training-admission-v13-proximal-release-development-v2"
        )
    elif method["artifact_lineage"] == "v14-terminal-taper":
        variant = method["algorithm"][len("h2oplus-v14-") :]
        terminal_multiplier = V14_TERMINAL_MULTIPLIERS[variant]
        equivalent_updates = V14_FULL_RATE_EQUIVALENT_UPDATES[variant]
        rate = 5e-5
        fields = {
            "method": "h2oplus",
            "variant": variant,
            "stage": "branch",
            "phase": "complete",
            "v14_online_events_nominal": method["online_events"],
            "v14_online_events_actual": method["actual_online_events"],
            "v14_learner_update_steps": method["learner_update_steps"],
            "v14_ratio_optimizer_steps": method["ratio_optimizer_steps"],
            "v14_actor_tensor_sha256": method["actor_tensor_sha256"],
            "v14_temperature_tensor_sha256": method["temperature_tensor_sha256"],
        }
        intervention = training.get("v14_intervention", {})
        retention = float(
            intervention.get("no_gradient_anchor_retention_at_25k", math.nan)
        )
        if (
            intervention.get("variant") != variant
            or intervention.get("proximal_rate_per_actor_update") != rate
            or intervention.get("registered_post_10k_actor_updates") != 15_000
            or not math.isclose(
                retention,
                (1.0 - rate) ** 15_000,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or intervention.get("terminal_actor_lr_multiplier")
            != terminal_multiplier
            or not math.isclose(
                float(
                    intervention.get(
                        "full_rate_equivalent_actor_updates", math.nan
                    )
                ),
                equivalent_updates,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or intervention.get("terminal_actor_lr_taper_events")
            != [15_000, 25_000]
            or intervention.get("terminal_actor_lr_taper_critic_updates")
            != [10_000, 19_999]
            or intervention.get("predesignated_candidate")
            is not (method["algorithm"] == "h2oplus-v14-terminal-quarter")
            or intervention.get("unit_ratio_weights") is not True
            or intervention.get("checkpoint_selection_by_behavior") is not False
        ):
            raise ValueError(
                f"independent v14 intervention provenance failed: {path}"
            )
        protocol_hash = training.get("v14_protocol", {}).get("sha256")
        admission_schema = (
            "transit-training-admission-v14-terminal-taper-development-v1"
        )
    else:
        raise ValueError(f"unknown physical actor lineage: {path}")
    if (
        any(training.get(key) != value for key, value in fields.items())
        or admission.get("schema_version") != admission_schema
        or admission.get("sha256") != method["training_admission_sha256"]
        or protocol_hash != method["source_protocol_sha256"]
    ):
        raise ValueError(f"independent lineage-specific training provenance failed: {path}")


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
    coverage = payload.get("passenger_welfare", {}).get("chronological_coverage_errors", {})
    if (
        payload.get("schema_version") != "transit-evaluation-v7"
        or payload.get("evaluation_job", {}).get("signature") != job.get("signature")
        or payload.get("method") != method.get("name")
        or payload.get("sumo_seed") != job.get("sumo_seed")
        or not finite(payload)
        or source != job.get("source_sha256")
        or len(source) != 47
        or any(_suffix_hash(source, suffix) != frozen_sources[suffix] for suffix in RESULT_SOURCE_SUFFIXES)
        or any(job_method.get(key) != method.get(key) for key in METHOD_FIELDS)
        or job_method.get("training_plan_sha256") != method.get("source_protocol_sha256")
        or job.get("dataset_sha256") != EXPECTED_DATASET
        or payload.get("dataset", {}).get("sha256") != EXPECTED_DATASET
        or job.get("sumo_runtime_signature") != EXPECTED_RUNTIME
        or payload.get("runtime", {}).get("sumo_runtime_signature") != EXPECTED_RUNTIME
        or payload.get("policy", {}).get("sha256") != method.get("checkpoint_sha256")
        or payload.get("policy", {}).get("actor_tensor_sha256") != method.get("actor_tensor_sha256")
        or method.get("artifact_lineage")
        in {"v12-staged-release", "v13-proximal-release", "v14-terminal-taper"}
        and payload.get("policy", {}).get("temperature_tensor_sha256")
        != method.get("temperature_tensor_sha256")
        or payload.get("scenario", {}).get("manifest_sha256")
        != job.get("scenario", {}).get("manifest_sha256")
        or not coverage
        or any(abs(float(value)) > 1e-9 for value in coverage.values())
        or payload.get("bus_service_semantics_audit", {}).get("passed") is not True
        or payload.get("bus_service_semantics_audit", {}).get("require_all_departed_trips_completed") is not True
        or payload.get("ledger_audit", {}).get("legacy_sampled_trip_histories_used_as_ground_truth") is not False
        or payload.get("passengers", {}).get("legacy_sampled_trip_histories_used_for_welfare") is not False
        or payload.get("passengers", {}).get("active_state_count_balance_error") != 0
        or payload.get("passenger_welfare", {}).get("primary") is not True
        or abs(
            float(
                payload.get("passenger_welfare", {}).get(
                    "maximum_reward_reconstruction_error", math.inf
                )
            )
        )
        > 1e-9
    ):
        raise ValueError(f"independent v14 result provenance check failed: {path}")
    _validate_training(payload, method, path)
    if source_reference is not None and source != source_reference:
        raise ValueError(f"source map differs across independently checked v14 results: {path}")
    return source


def verify(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "transit-h2oplus-v14-terminal-taper-evaluation-file-manifest-v1":
        raise ValueError("unexpected v14 evaluation manifest schema")
    records = manifest.get("files", [])
    record_paths = {row["path"] for row in records}
    if len(records) != manifest.get("file_count") or len(record_paths) != len(records):
        raise ValueError("v14 file records are incomplete or duplicated")
    total = 0
    for record in records:
        path = (REPO / record["path"]).resolve()
        if not path.is_file() or REPO not in path.parents:
            raise ValueError(f"missing or out-of-root v14 file: {record['path']}")
        if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"v14 file identity mismatch: {record['path']}")
        total += path.stat().st_size
    if total != manifest.get("total_size_bytes") or aggregate(records) != manifest.get("file_records_aggregate_sha256"):
        raise ValueError("v14 file record aggregate mismatch")
    for relative, digest in EXPECTED_CONTROLS.items():
        if sha256(REPO / relative) != digest or relative not in record_paths:
            raise ValueError(f"sealed v14 control drifted or is absent: {relative}")

    artifact = manifest.get("artifact_integrity", {})
    if (
        artifact.get("passed") is not True
        or artifact.get("result_count") != 630
        or artifact.get("unique_job_signature_count") != 630
        or artifact.get("canonical_matrix_count") != 3
        or artifact.get("source_manifest_count") != 6
        or artifact.get("source_file_count") != 47
        or artifact.get("physical_method_count") != 70
        or artifact.get("logical_method_budget_alias_count") != 180
        or manifest.get("study_stage") != "development"
        or manifest.get("formal_held_out_outcomes_read") is not False
        or manifest.get("supports_current_formal_held_out_model_effect_claim") is not False
        or manifest.get("scientific_analysis", {}).get("completed") is not False
    ):
        raise ValueError("v14 artifact/scientific scope summary failed")

    audit = json.loads((RESULTS / "evaluation_audit_v1.json").read_text())
    if (
        audit.get("passed") is not True
        or audit.get("artifact_integrity", {}).get("passed") is not True
        or audit.get("scientific_results", {}).get("analysis_completed") is not False
        or audit.get("training_scientific_stability", {}).get("all_15_training_branches_stability_gate_passed") is not True
        or audit.get("training_scientific_stability", {}).get("failure_runs") != []
    ):
        raise ValueError("v14 evaluation audit did not preserve integrity/stability separation")

    scheduler = json.loads((RESULTS / "evidence/evaluation_v1/scheduler_evidence_v1.json").read_text())
    if (
        scheduler.get("passed") is not True
        or scheduler.get("token_bound_backend_exit0") is not True
        or set(scheduler.get("tasks", {})) != {"t83584", "t83585"}
    ):
        raise ValueError("v14 scheduler evidence failed")
    for task_id, item in scheduler["tasks"].items():
        status = RESULTS / f"evidence/evaluation_v1/backend_exit_{task_id}.status"
        fields = status.read_text().strip().split("\t")
        if (
            len(fields) != 3
            or fields[0] != "0"
            or fields[2] != item.get("backend_token")
            or item.get("status") != "done"
            or item.get("backend_returncode") != 0
            or item.get("token_matches") is not True
            or item.get("residual_evaluation_process_count") != 0
        ):
            raise ValueError(f"independent token-bound v14 exit check failed: {task_id}")

    methods_payload = json.loads(METHODS_PATH.read_text())
    methods = {item["name"]: item for item in methods_payload["methods"]}
    aliases = methods_payload.get("aliases", [])
    source_freeze = json.loads(SOURCE_FREEZE_PATH.read_text())
    frozen_sources = {item["path"]: item["sha256"] for item in source_freeze["sources"]}
    identities = methods_payload.get("identity_evidence", [])
    if (
        len(methods) != 70
        or len(aliases) != 180
        or len({item["name"] for item in aliases}) != 180
        or len(frozen_sources) != 10
        or len(identities) != 5
    ):
        raise ValueError("independent v14 methods/source-freeze check failed")
    for alias in aliases:
        physical = methods.get(alias.get("physical_method"))
        if (
            physical is None
            or alias.get("training_seed") != physical.get("training_seed")
            or alias.get("actor_tensor_sha256") != physical.get("actor_tensor_sha256")
        ):
            raise ValueError("independent v14 logical alias identity check failed")
    for identity in identities:
        if (
            identity.get("v14_logical_aliases") != 12
            or identity.get("prefix_zero_5k_10k_equal_fixed_h2oplus_zero") is not True
            or len(identity.get("prefix_source_policies", [])) != 3
            or len(identity.get("branch_source_policies", [])) != 3
            or identity.get("exact_seed_matched_v12_10k_handoff") is not True
        ):
            raise ValueError("independent v14 terminal-taper identity evidence check failed")

    raw_files = {
        str(path.resolve().relative_to(REPO))
        for path in PARTITIONS.rglob("*")
        if path.is_file()
    }
    if not raw_files.issubset(record_paths):
        raise ValueError("not every synchronized v14 partition file is sealed")

    partition_signatures = set()
    source_manifest_hashes = set()
    deployed_checkpoint_hashes = {}
    source_reference = None
    expected_jobs = (315, 315)
    selected_jobs = (105, 105)
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
            or host.get("formal_held_out_outcomes_read") is not False
            or len(results) != expected_jobs[shard_index]
            or len(matrices) != 3
        ):
            raise ValueError(f"independent v14 source partition check failed: {shard}")
        for matrix_path in matrices:
            matrix = json.loads(matrix_path.read_text())
            jobs = matrix.get("jobs", [])
            partition = matrix.get("matrix_partition", {})
            if (
                len(jobs) != selected_jobs[shard_index]
                or partition.get("shard_index") != shard_index
                or partition.get("shard_count") != 2
                or partition.get("selected_job_count")
                != selected_jobs[shard_index]
                or partition.get("full_job_count") != 210
                or len(matrix.get("source_sha256", {})) != 47
                or matrix.get("methods_manifest_sha256") != sha256(METHODS_PATH)
                or matrix.get("sumo_runtime_signature") != EXPECTED_RUNTIME
            ):
                raise ValueError(f"independent v14 partition matrix check failed: {matrix_path}")
            source_manifest_hashes.add(sha256(matrix_path))
            for job in jobs:
                name = str(job.get("method", {}).get("name"))
                method = methods.get(name)
                output = Path(job["output"]).expanduser().resolve()
                if method is None or not output.is_file() or PARTITIONS.resolve() not in output.parents:
                    raise ValueError(f"missing or out-of-root independent v14 output: {output}")
                deployed_checkpoint = Path(job["method"]["checkpoint"]).expanduser().resolve()
                if deployed_checkpoint not in deployed_checkpoint_hashes:
                    if not deployed_checkpoint.is_file():
                        raise ValueError(
                            f"missing deployed policy checkpoint: {deployed_checkpoint}"
                        )
                    deployed_checkpoint_hashes[deployed_checkpoint] = sha256(deployed_checkpoint)
                if deployed_checkpoint_hashes[deployed_checkpoint] != method["checkpoint_sha256"]:
                    raise ValueError(
                        f"independent deployed policy checkpoint check failed: {deployed_checkpoint}"
                    )
                source_reference = _validate_result(job, output, method, source_reference, frozen_sources)
                signature = str(job["signature"])
                if signature in partition_signatures:
                    raise ValueError("duplicate independent v14 result signature")
                partition_signatures.add(signature)
    if len(partition_signatures) != 630 or len(source_manifest_hashes) != 6 or source_reference is None:
        raise ValueError("independent v14 partition union check failed")
    if len(deployed_checkpoint_hashes) != 70:
        raise ValueError("independent deployed policy checkpoint cardinality failed")

    canonical_signatures = set()
    expected_cells = {(name, scale) for name in methods for scale in SCALES}
    for block in BLOCKS:
        path = CANONICAL / block / "matrix_manifest.json"
        matrix = json.loads(path.read_text())
        jobs = matrix.get("jobs", [])
        merge = matrix.get("matrix_merge", {})
        cells = set()
        if (
            len(jobs) != 210
            or merge.get("all_outputs_verified") is not True
            or merge.get("method_count") != 70
            or merge.get("environment_cells_per_method") != 3
            or len(merge.get("source_manifests", [])) != 2
            or not {item.get("sha256") for item in merge["source_manifests"]}.issubset(source_manifest_hashes)
        ):
            raise ValueError(f"independent v14 canonical matrix check failed: {block}")
        for job in jobs:
            name = str(job.get("method", {}).get("name"))
            scale = float(job["scenario"]["demand_scale"])
            signature = str(job["signature"])
            cells.add((name, scale))
            if name not in methods or signature in canonical_signatures:
                raise ValueError("invalid or duplicate canonical v14 signature")
            canonical_signatures.add(signature)
        if cells != expected_cells:
            raise ValueError(f"canonical v14 Cartesian coverage failed: {block}")
    if canonical_signatures != partition_signatures:
        raise ValueError("canonical and synchronized v14 source unions differ")

    return {
        "schema_version": "transit-h2oplus-v14-terminal-taper-evaluation-file-manifest-verification-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": {
            "file_records_aggregate_sha256": manifest["file_records_aggregate_sha256"],
            "path": str(manifest_path.relative_to(REPO)),
            "sha256": sha256(manifest_path),
        },
        "independent_artifact_audit": {
            "all_numeric_values_finite": True,
            "canonical_matrix_count": 3,
            "source_file_count": 47,
            "file_count": len(records),
            "logical_method_budget_alias_count": 180,
            "passed": True,
            "physical_method_count": 70,
            "physical_policy_checkpoints_rehashed": 70,
            "result_count": 630,
            "source_manifest_count": 6,
            "source_partition_count": 2,
            "total_size_bytes": total,
            "unique_job_signature_count": 630,
        },
        "training_scientific_stability": {
            "all_15_training_branches_stability_gate_passed": True,
            "failure_runs": [],
            "separate_from_artifact_integrity": True,
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
    parser.add_argument("--expected-scheduler-sync-sha256", required=True)
    parser.add_argument("--expected-source-sync-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dynamic_controls = {
        "reproducibility/h2oplus_v14_terminal_taper_results/evidence/evaluation_v1/scheduler_evidence_v1.json": args.expected_scheduler_sync_sha256,
        "reproducibility/h2oplus_v14_terminal_taper_results/evidence/evaluation_v1/source_sync_v1.json": args.expected_source_sync_sha256,
    }
    for path, value in dynamic_controls.items():
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"invalid expected SHA256 for {path}")
        EXPECTED_CONTROLS[path] = value
    manifest = args.manifest.expanduser().resolve(strict=True)
    payload = verify(manifest)
    output = args.output.expanduser().resolve()
    atomic_json(output, payload)
    print(output)
    print(payload["manifest"]["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
