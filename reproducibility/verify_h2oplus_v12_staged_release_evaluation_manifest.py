#!/usr/bin/env python3
"""Independently verify the H2O+ v12 staged-release evaluation file manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reproducibility/h2oplus_v12_staged_release_results"
PARTITIONS = RESULTS / "evaluation_partitions_v1"
CANONICAL = RESULTS / "evaluation_canonical_v1"
DEFAULT_MANIFEST = RESULTS / "evaluation_file_manifest_v1.json"
DEFAULT_OUTPUT = RESULTS / "evaluation_file_manifest_verification_v1.json"

EXPECTED_CONTROLS = {
    "reproducibility/h2oplus_v12_staged_release_registration_v2.json": "59cb1e14ad0bbc8944472f5bbf3e849a62b65c4517766e3fd3845b76a5ec3ba3",
    "reproducibility/h2oplus_v12_staged_release_authorization_v2.json": "bb1eeda1c451916aba7576f0592418471f42afaefe499386482be540def7fa23",
    "reproducibility/h2oplus_v12_staged_release_deployment_correction_v2.json": "40dabd0dddab6ce7fd8f08ac025f8cc96a5545d1c36ff24730ce885b5eb67b25",
    "reproducibility/h2oplus_v12_staged_release_environment_v2.json": "bfe97d430921f38fbbec04490ac980e330d2000f6c1ad4d62627c95d3deba5e3",
    "reproducibility/h2oplus_v12_staged_release_training_plan_v2.json": "1bb58d5ec2dca0b49a04e10be6d2f0705511b49aaf24e6ce69b4e4b70a688ffa",
    "reproducibility/h2oplus_v12_staged_release_protocol_v2.json": "ed6cd5e32f4719a0ef8925272688fdb3393e5c5056027ae9d6430b1ba20d5f1c",
    "reproducibility/h2oplus_v12_staged_release_freeze_v2.json": "0d8ab1e7852c679ae6e444d1c767799635c8790e4bc787587210a63b0e92dfe1",
    "reproducibility/h2oplus_v12_staged_release_evaluation_methods_v1.json": "d61acae2314d0588b0dc6aa84c0a0bf2f168d4bdf1f18c5d19b2d0dbdec5c374",
    "reproducibility/h2oplus_v12_staged_release_evaluation_source_freeze_v1.json": "2ab7b6c4d8eb94977df097ab37ff8671becdd11fbf4c2ffe562eecb6bce6cd28",
    "reproducibility/h2oplus_v12_staged_release_evaluation_deployment_local_verification_v1.json": "8dd611ab2d434fd03e9ab445e705fc9023e6ebd47dd2d30f3dc42fd5b14d8c06",
    "reproducibility/h2oplus_v12_staged_release_evaluation_deployment_sync_v1.json": "f882c926cb6badd8699d49a95d79ba4b5d7a6ffdad31d69d82ff240684cd0b56",
    "reproducibility/h2oplus_v12_staged_release_evaluation_deployment_v1_jtl110gpu_verification.json": "ea3e403f44482ff92c3fd4758198dea62a40c914472a71df6875a722d887f038",
    "reproducibility/h2oplus_v12_staged_release_evaluation_submission_v1.json": "331a62b1659f9f309718443e75c9e952ff64116c48272da14f1d9296fa603dd7",
    "reproducibility/h2oplus_v12_staged_release_branch_closeout_v2.json": "998f47488b50b2cd40f9a6ddd5a399712bb9208cb1ef883cad1181f800385b66",
    "reproducibility/h2oplus_v12_staged_release_branch_closeout_verification_v2.json": "6835e3c545be22c25bd9e3c7eae7ef987bb1864e027c4fef9849788f744868c8",
    "reproducibility/h2oplus_v12_staged_release_results/branch_audit_v2.json": "8e34c4cd69f86d7333dfd5d2a57696ce427388349c07c377bb279b957fd1bfc7",
    "reproducibility/h2oplus_v12_staged_release_results/branch_file_manifest_v2.json": "9ed7ccbddf0832e0c8b4532421e000745f9c60f03b9dfc01a40d970503c2b08b",
    "reproducibility/deployments/h2oplus_v12_staged_release_evaluation_v1/workspace/deployment_manifest.json": "68155a903e305ce04cd1654487502ae5356219ba43b1fc953f474efc5eb8e973",
    "reproducibility/h2oplus_v8_stage_a_sumo_runtime_lock_py38.json": "ac60ae58e0e6db6bb1386ffdfe0d164b96f81e49e4983b8b062ceddb65da51a4",
    "reproducibility/h2oplus_v12_staged_release_results/evidence/evaluation_v1/scheduler_evidence_v1.json": "e9ea76b428831f6669eb625506ca76fb401aec07f098899a4a6528f2991bc379",
    "reproducibility/h2oplus_v12_staged_release_results/evidence/evaluation_v1/source_sync_v1.json": "4d4076e6c5c9d6a7c882302eca8252220cf8c8554e75c871ac6db88f077dd193",
}
EXPECTED_DATASET = "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899"
EXPECTED_RUNTIME = "264d3c412b8e6c42320995208d2e009eb809b85626047206c4de752c2dae06e7"
METHODS_PATH = REPO / "reproducibility/h2oplus_v12_staged_release_evaluation_methods_v1.json"
SOURCE_FREEZE_PATH = REPO / "reproducibility/h2oplus_v12_staged_release_evaluation_source_freeze_v1.json"
BLOCKS = (
    "v12-staged-release-development-01",
    "v12-staged-release-development-02",
    "v12-staged-release-development-03",
)
SCALES = {0.85, 1.0, 1.15}
RESULT_SOURCE_SUFFIXES = {
    "SimpleSAC/evaluate_transit_v11_actor_hold.py",
    "SimpleSAC/evaluate_transit_v12_staged_release.py",
    "reproducibility/run_evaluation_matrix_v11_actor_hold.py",
    "reproducibility/run_evaluation_matrix_v12_staged_release.py",
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
        or len(source) != 43
        or any(_suffix_hash(source, suffix) != frozen_sources[suffix] for suffix in RESULT_SOURCE_SUFFIXES)
        or any(job_method.get(key) != method.get(key) for key in METHOD_FIELDS)
        or job_method.get("training_plan_sha256") != method.get("source_protocol_sha256")
        or job.get("dataset_sha256") != EXPECTED_DATASET
        or payload.get("dataset", {}).get("sha256") != EXPECTED_DATASET
        or job.get("sumo_runtime_signature") != EXPECTED_RUNTIME
        or payload.get("runtime", {}).get("sumo_runtime_signature") != EXPECTED_RUNTIME
        or payload.get("policy", {}).get("sha256") != method.get("checkpoint_sha256")
        or payload.get("policy", {}).get("actor_tensor_sha256") != method.get("actor_tensor_sha256")
        or method.get("artifact_lineage") == "v12-staged-release"
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
        raise ValueError(f"independent v12 result provenance check failed: {path}")
    _validate_training(payload, method, path)
    if source_reference is not None and source != source_reference:
        raise ValueError(f"source map differs across independently checked v12 results: {path}")
    return source


def verify(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "transit-h2oplus-v12-staged-release-evaluation-file-manifest-v1":
        raise ValueError("unexpected v12 evaluation manifest schema")
    records = manifest.get("files", [])
    record_paths = {row["path"] for row in records}
    if len(records) != manifest.get("file_count") or len(record_paths) != len(records):
        raise ValueError("v12 file records are incomplete or duplicated")
    total = 0
    for record in records:
        path = (REPO / record["path"]).resolve()
        if not path.is_file() or REPO not in path.parents:
            raise ValueError(f"missing or out-of-root v12 file: {record['path']}")
        if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"v12 file identity mismatch: {record['path']}")
        total += path.stat().st_size
    if total != manifest.get("total_size_bytes") or aggregate(records) != manifest.get("file_records_aggregate_sha256"):
        raise ValueError("v12 file record aggregate mismatch")
    for relative, digest in EXPECTED_CONTROLS.items():
        if sha256(REPO / relative) != digest or relative not in record_paths:
            raise ValueError(f"sealed v12 control drifted or is absent: {relative}")

    artifact = manifest.get("artifact_integrity", {})
    if (
        artifact.get("passed") is not True
        or artifact.get("result_count") != 630
        or artifact.get("unique_job_signature_count") != 630
        or artifact.get("canonical_matrix_count") != 3
        or artifact.get("source_manifest_count") != 6
        or artifact.get("corrected_source_file_count") != 43
        or artifact.get("physical_method_count") != 70
        or artifact.get("logical_method_budget_alias_count") != 180
        or manifest.get("study_stage") != "development"
        or manifest.get("formal_held_out_outcomes_read") is not False
        or manifest.get("supports_current_formal_held_out_model_effect_claim") is not False
        or manifest.get("scientific_analysis", {}).get("completed") is not False
    ):
        raise ValueError("v12 artifact/scientific scope summary failed")

    audit = json.loads((RESULTS / "evaluation_audit_v1.json").read_text())
    if (
        audit.get("passed") is not True
        or audit.get("artifact_integrity", {}).get("passed") is not True
        or audit.get("scientific_results", {}).get("analysis_completed") is not False
        or audit.get("training_scientific_stability", {}).get("all_30_training_runs_stability_gate_passed") is not True
        or audit.get("training_scientific_stability", {}).get("failure_runs") != []
    ):
        raise ValueError("v12 evaluation audit did not preserve integrity/stability separation")

    scheduler = json.loads((RESULTS / "evidence/evaluation_v1/scheduler_evidence_v1.json").read_text())
    if (
        scheduler.get("passed") is not True
        or scheduler.get("token_bound_backend_exit0") is not True
        or set(scheduler.get("tasks", {})) != {"t83115", "t83116"}
    ):
        raise ValueError("v12 scheduler evidence failed")
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
            raise ValueError(f"independent token-bound v12 exit check failed: {task_id}")

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
        or len(frozen_sources) != 6
        or len(identities) != 5
    ):
        raise ValueError("independent v12 methods/source-freeze check failed")
    for alias in aliases:
        physical = methods.get(alias.get("physical_method"))
        if (
            physical is None
            or alias.get("training_seed") != physical.get("training_seed")
            or alias.get("actor_tensor_sha256") != physical.get("actor_tensor_sha256")
        ):
            raise ValueError("independent v12 logical alias identity check failed")
    for identity in identities:
        if (
            identity.get("v12_logical_aliases") != 20
            or identity.get("prefix_zero_5k_10k_equal_fixed_h2oplus_zero") is not True
            or identity.get("prefix_zero_5k_10k_temperature_equal") is not True
            or len(identity.get("prefix_source_policies", [])) != 3
            or len(identity.get("branch_source_policies", [])) != 5
        ):
            raise ValueError("independent v12 staged-release identity evidence check failed")

    raw_files = {
        str(path.resolve().relative_to(REPO))
        for path in PARTITIONS.rglob("*")
        if path.is_file()
    }
    if not raw_files.issubset(record_paths):
        raise ValueError("not every synchronized v12 partition file is sealed")

    partition_signatures = set()
    source_manifest_hashes = set()
    deployed_checkpoint_hashes = {}
    source_reference = None
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
            or len(results) != 315
            or len(matrices) != 3
        ):
            raise ValueError(f"independent v12 source partition check failed: {shard}")
        for matrix_path in matrices:
            matrix = json.loads(matrix_path.read_text())
            jobs = matrix.get("jobs", [])
            partition = matrix.get("matrix_partition", {})
            if (
                len(jobs) != 105
                or partition.get("shard_index") != shard_index
                or partition.get("shard_count") != 2
                or partition.get("selected_job_count") != 105
                or partition.get("full_job_count") != 210
                or len(matrix.get("source_sha256", {})) != 43
                or matrix.get("methods_manifest_sha256") != sha256(METHODS_PATH)
                or matrix.get("sumo_runtime_signature") != EXPECTED_RUNTIME
            ):
                raise ValueError(f"independent v12 partition matrix check failed: {matrix_path}")
            source_manifest_hashes.add(sha256(matrix_path))
            for job in jobs:
                name = str(job.get("method", {}).get("name"))
                method = methods.get(name)
                output = Path(job["output"]).expanduser().resolve()
                if method is None or not output.is_file() or PARTITIONS.resolve() not in output.parents:
                    raise ValueError(f"missing or out-of-root independent v12 output: {output}")
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
                    raise ValueError("duplicate independent v12 result signature")
                partition_signatures.add(signature)
    if len(partition_signatures) != 630 or len(source_manifest_hashes) != 6 or source_reference is None:
        raise ValueError("independent v12 partition union check failed")
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
            raise ValueError(f"independent v12 canonical matrix check failed: {block}")
        for job in jobs:
            name = str(job.get("method", {}).get("name"))
            scale = float(job["scenario"]["demand_scale"])
            signature = str(job["signature"])
            cells.add((name, scale))
            if name not in methods or signature in canonical_signatures:
                raise ValueError("invalid or duplicate canonical v12 signature")
            canonical_signatures.add(signature)
        if cells != expected_cells:
            raise ValueError(f"canonical v12 Cartesian coverage failed: {block}")
    if canonical_signatures != partition_signatures:
        raise ValueError("canonical and synchronized v12 source unions differ")

    return {
        "schema_version": "transit-h2oplus-v12-staged-release-evaluation-file-manifest-verification-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": {
            "file_records_aggregate_sha256": manifest["file_records_aggregate_sha256"],
            "path": str(manifest_path.relative_to(REPO)),
            "sha256": sha256(manifest_path),
        },
        "independent_artifact_audit": {
            "all_numeric_values_finite": True,
            "canonical_matrix_count": 3,
            "corrected_source_file_count": 43,
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
            "all_30_training_runs_stability_gate_passed": True,
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
