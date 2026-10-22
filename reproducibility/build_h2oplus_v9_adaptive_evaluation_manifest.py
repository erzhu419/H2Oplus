#!/usr/bin/env python3
"""Audit and seal the corrected H2O+ v9 adaptive development evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reproducibility/h2oplus_v9_adaptive_results"
PARTITIONS = RESULTS / "evaluation_partitions_v3"
CANONICAL = RESULTS / "evaluation_canonical_v3"
EVIDENCE = RESULTS / "evidence/evaluation_v3"
QUEUE = Path("/home/erzhu419/.claude/scheduler/queue.json")

AUDIT_PATH = RESULTS / "evaluation_audit_v3.json"
SCHEDULER_EVIDENCE_PATH = EVIDENCE / "scheduler_evidence_v3.json"
MANIFEST_PATH = RESULTS / "evaluation_file_manifest_v3.json"

EXPECTED_HASHES = {
    "registration": "f4f6922324ef518acde9b5489328bf13c7154012dfd6f4cc2b61797adb90785e",
    "authorization": "c99b2cb637018a67c04d3be8e088195c8ec9c4d05f3fcf02ca7e672b967f0fa6",
    "freeze": "b3265d5e55e4df53afacd50c9f54cc3530a5d8f28a7d3eceec217655dc9bd822",
    "environment": "cf6faac3b684b4bddfd1391925d67d7c3d5202978bd2c6d02e6326a6bb5dd8f5",
    "protocol": "e0652db511c5ca46e0e14801895bdfdc57761223842a4e30673075b9163d3c7d",
    "count_correction": "4fc9324b140f6af387a4b8011b40c1d6994c57636d909ca065b17d4d0e27c6ba",
    "portability_correction": "b5063671012b48c55f537ab5f017356ff2a72229f59aa74b5eabf67214842d23",
    "source_correction": "2e28bd6dfe71e3d1adf82866137da0a8a678048c7a6a91aaa00df8b39e782f74",
    "source_freeze": "f717bebbe0ed838c104ebb716161fcc7d9be79b64ed9ac3b207ba638139a37cc",
    "methods": "e1c83b52c78dc60d4a8da8888f3b2a0d82f2c6ea77ca4b9d372d56a33eb58f0c",
    "submission": "21088cc14d0b4714ba1a311e14638f4107e6f7c46eea537111afc35d602403df",
    "deployment": "64180676bf222d8cf8877c9a3c3d80b8f7aa86d60ee2a60e042bc28b242ff0e3",
    "training_closeout": "fc263b3d51ac5a532bd13dc3fdaecf43153e1336d9d57a4ce6b87e6eac18bafd",
    "training_audit": "f126301d52ccdc665c8c1fdc20611c5f87d7879fcce14296d4984b47de997959",
    "training_manifest": "010ddf60b394230cdeb28edd32e4df52d189d754c1a2a2301e9c40f0ed46f4ce",
    "training_verification": "9926dea64d692619808c2a2e03382e7620fd71358223119ac066d217ceff080e",
}

CONTROL_PATHS = {
    "registration": REPO / "reproducibility/h2oplus_v9_adaptive_registration_v2.json",
    "authorization": REPO / "reproducibility/h2oplus_v9_adaptive_authorization_v2.json",
    "freeze": REPO / "reproducibility/h2oplus_v9_adaptive_freeze_v2.json",
    "environment": REPO / "reproducibility/h2oplus_v9_adaptive_environment_v2.json",
    "protocol": REPO / "reproducibility/h2oplus_v9_adaptive_protocol_v2.json",
    "count_correction": REPO
    / "reproducibility/h2oplus_v9_adaptive_evaluation_count_correction_v1.json",
    "portability_correction": REPO
    / "reproducibility/h2oplus_v9_adaptive_evaluation_portability_correction_v1.json",
    "source_correction": REPO
    / "reproducibility/h2oplus_v9_adaptive_evaluation_source_correction_v2.json",
    "source_freeze": REPO
    / "reproducibility/h2oplus_v9_adaptive_evaluation_source_freeze_v2.json",
    "methods": REPO / "reproducibility/h2oplus_v9_adaptive_evaluation_methods_v2.json",
    "submission": REPO
    / "reproducibility/h2oplus_v9_adaptive_evaluation_submission_v3.json",
    "deployment": REPO
    / "reproducibility/deployments/h2oplus_v9_adaptive_evaluation_v3/workspace/deployment_manifest.json",
    "training_closeout": RESULTS / "training_closeout.json",
    "training_audit": RESULTS / "training_audit.json",
    "training_manifest": RESULTS / "training_file_manifest.json",
    "training_verification": RESULTS / "training_file_manifest_verification.json",
}

EXPECTED_DEPLOYMENT_AGGREGATE = (
    "a0e180239fdeb5b6046e4ca59a83afbe08e4258fd648164991d3f6a5c3c085d8"
)
EXPECTED_DATASET_SHA256 = (
    "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899"
)
EXPECTED_RUNTIME_SIGNATURE = (
    "264d3c412b8e6c42320995208d2e009eb809b85626047206c4de752c2dae06e7"
)
EXPECTED_ENV = {
    "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcurl.so.4",
    "CC": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc",
    "CPATH": (
        "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/"
        "deployments/v7_formal_v2_training_v7r1/workspace/reproducibility/"
        "toolchain/libxcrypt-4.4.36-hd590300_1/include"
    ),
    "PYTHONDONTWRITEBYTECODE": "1",
}
TASKS = {
    "t82575": {
        "shard": "shard0_jtl110gpu",
        "status_file": EVIDENCE / "backend_exit_t82575.status",
        "log_file": EVIDENCE / "scheduler_t82575.log",
    },
    "t82576": {
        "shard": "shard1_jtl110gpu",
        "status_file": EVIDENCE / "backend_exit_t82576.status",
        "log_file": EVIDENCE / "scheduler_t82576.log",
    },
}
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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


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


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def file_record(path: Path) -> dict:
    return {
        "path": relative(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def records_aggregate(records: Iterable[dict]) -> str:
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


def validate_controls() -> tuple[dict, set[Path], dict[str, dict], dict[str, str]]:
    records = {}
    for name, path in CONTROL_PATHS.items():
        actual = sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"control identity drifted for {name}: {actual}")
        records[name] = file_record(path)

    deployment = load_json(CONTROL_PATHS["deployment"])
    submission = load_json(CONTROL_PATHS["submission"])
    methods_payload = load_json(CONTROL_PATHS["methods"])
    source_freeze = load_json(CONTROL_PATHS["source_freeze"])
    if (
        deployment.get("aggregate_sha256") != EXPECTED_DEPLOYMENT_AGGREGATE
        or len(deployment.get("entries", [])) != 434
        or submission.get("formal_held_out_outcomes_read") is not False
        or submission.get("formal_confirmation_stage_authorized_or_submitted")
        is not False
        or submission.get("physical_actor_count") != 42
        or submission.get("total_remote_sumo_episodes") != 252
        or methods_payload.get("schema_version")
        != "transit-evaluation-methods-v9-adaptive-v1"
        or methods_payload.get("physical_method_count") != 42
        or methods_payload.get("method_budget_alias_count") != 60
        or len(methods_payload.get("methods", [])) != 42
        or len(methods_payload.get("aliases", [])) != 60
        or source_freeze.get("formal_held_out_outcomes_read") is not False
        or source_freeze.get("training_or_policy_artifacts_changed") is not False
    ):
        raise ValueError("v9 corrected evaluation control contract failed")
    methods = {item["name"]: item for item in methods_payload["methods"]}
    if len(methods) != 42:
        raise ValueError("duplicate physical methods in v9 manifest")
    frozen_sources = {item["path"]: item["sha256"] for item in source_freeze["sources"]}
    if len(frozen_sources) != 4:
        raise ValueError("v9 source-freeze cardinality drifted")
    return records, set(CONTROL_PATHS.values()), methods, frozen_sources


def _queue_tasks() -> dict[str, dict]:
    payload = load_json(QUEUE)
    rows = payload.get("tasks", payload if isinstance(payload, list) else [])
    return {row["id"]: row for row in rows if row.get("id") in TASKS}


def validate_scheduler() -> tuple[dict, set[Path]]:
    queue_tasks = _queue_tasks()
    submission = load_json(CONTROL_PATHS["submission"])
    submitted = {item["task_id"]: item for item in submission["tasks"]}
    if set(queue_tasks) != set(TASKS) or set(submitted) != set(TASKS):
        raise ValueError("v9 scheduler task coverage drifted")
    evidence = {}
    files: set[Path] = set()
    for task_id, expected in TASKS.items():
        task = queue_tasks[task_id]
        task_submission = submitted[task_id]
        status_path = expected["status_file"]
        log_path = expected["log_file"]
        fields = status_path.read_text().strip().split("\t")
        if len(fields) != 3:
            raise ValueError(f"malformed token-bound backend status for {task_id}")
        exit_code, backend_finished_at, token = fields
        host_path = PARTITIONS / expected["shard"] / "host_execution.json"
        host = load_json(host_path)
        env = task.get("extra_env", {})
        shard_index = int(task_submission["shard_index"])
        if (
            exit_code != "0"
            or token != task.get("exit_status_token")
            or int(backend_finished_at) != int(task.get("backend_finished_at"))
            or task.get("status") != "done"
            or task.get("exit_code") != 0
            or task.get("node") != "jtl110gpu"
            or task.get("cpu_declared_cores") != 12
            or task.get("est_vram_mb") != 0
            or task.get("signature") != task_submission.get("signature")
            or task_submission.get("cpu_cores") != 12
            or task_submission.get("ram_mb") != 32768
            or any(env.get(name) != value for name, value in EXPECTED_ENV.items())
            or host.get("status") != "completed"
            or host.get("returncode") != 0
            or host.get("shard_index") != shard_index
            or host.get("shard_count") != 2
            or host.get("formal_held_out_outcomes_read") is not False
            or log_path.read_text().strip()
            != f"DONE H2O+ v9 adaptive shard {shard_index}: 126/126 jobs"
        ):
            raise ValueError(f"scheduler/backend/host contract failed for {task_id}")
        files.update({status_path, log_path, host_path})
        evidence[task_id] = {
            "backend_finished_at": int(backend_finished_at),
            "cpu_declared_cores": 12,
            "exit_code": 0,
            "exit_status_token": token,
            "host_execution": file_record(host_path),
            "node": "jtl110gpu",
            "queue_finished_at": task.get("finished_at"),
            "queue_status": "done",
            "scheduler_log": file_record(log_path),
            "status_file": file_record(status_path),
        }
    payload = {
        "schema_version": "transit-h2oplus-v9-adaptive-evaluation-scheduler-evidence-v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": evidence,
        "task_count": 2,
        "token_bound_backend_exit0": True,
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "passed": True,
    }
    atomic_json(SCHEDULER_EVIDENCE_PATH, payload)
    files.add(SCHEDULER_EVIDENCE_PATH)
    return payload, files


def _suffix_hash(source: dict, suffix: str) -> str | None:
    matches = [value for path, value in source.items() if path.endswith(suffix)]
    return matches[0] if len(matches) == 1 else None


def validate_result(
    job: dict,
    output: Path,
    method: dict,
    source_reference: dict | None,
    frozen_sources: dict[str, str],
) -> dict:
    payload = load_json(output)
    job_method = job.get("method", {})
    training = payload.get("policy", {}).get("training_metadata", {})
    source = payload.get("source_sha256", {})
    coverage = payload.get("passenger_welfare", {}).get(
        "chronological_coverage_errors", {}
    )
    if (
        payload.get("schema_version") != "transit-evaluation-v7"
        or payload.get("evaluation_job", {}).get("signature") != job.get("signature")
        or payload.get("method") != method.get("name")
        or int(payload.get("sumo_seed")) != int(job.get("sumo_seed"))
        or not finite(payload)
        or source != job.get("source_sha256")
        or len(source) != 41
        or any(
            _suffix_hash(source, suffix) != frozen_sources[suffix]
            for suffix in RESULT_SOURCE_SUFFIXES
        )
        or any(job_method.get(key) != method.get(key) for key in METHOD_BINDING_FIELDS)
        or job.get("dataset_sha256") != EXPECTED_DATASET_SHA256
        or payload.get("dataset", {}).get("sha256") != EXPECTED_DATASET_SHA256
        or job.get("sumo_runtime_signature") != EXPECTED_RUNTIME_SIGNATURE
        or payload.get("runtime", {}).get("sumo_runtime_signature")
        != EXPECTED_RUNTIME_SIGNATURE
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
        or payload.get("scenario", {}).get("manifest_sha256")
        != job.get("scenario", {}).get("manifest_sha256")
        or not coverage
        or any(abs(float(value)) > 1e-9 for value in coverage.values())
    ):
        raise ValueError(f"v9 result provenance failed: {output}")
    if source_reference is not None and source != source_reference:
        raise ValueError(f"source map differs across v9 results: {output}")
    return source


def validate_partitions(
    methods: dict[str, dict], frozen_sources: dict[str, str]
) -> tuple[dict, set[Path], dict, set[str]]:
    stats = {}
    files: set[Path] = set()
    source_manifests = {}
    signatures: set[str] = set()
    source_reference = None
    for shard_index, shard in enumerate(("shard0_jtl110gpu", "shard1_jtl110gpu")):
        root = PARTITIONS / shard
        shard_files = {path for path in root.rglob("*") if path.is_file()}
        results = sorted(root.rglob("sumo_seed*.json"))
        manifests = sorted(root.rglob("matrix_manifest.shard-*.json"))
        host_path = root / "host_execution.json"
        host = load_json(host_path)
        if (
            len(results) != 126
            or len(manifests) != 2
            or host.get("status") != "completed"
            or host.get("returncode") != 0
            or host.get("shard_index") != shard_index
            or host.get("shard_count") != 2
            or host.get("training") is not False
            or host.get("formal_held_out_outcomes_read") is not False
            or len(host.get("completed_blocks", [])) != 2
            or sum(item.get("selected_job_count", 0) for item in host["completed_blocks"])
            != 126
        ):
            raise ValueError(f"partition host/result contract failed: {shard}")
        completed = {Path(item["path"]).parent.name: item for item in host["completed_blocks"]}
        shard_signatures: set[str] = set()
        for manifest_path in manifests:
            block = manifest_path.parent.name
            manifest = load_json(manifest_path)
            jobs = manifest.get("jobs", [])
            partition = manifest.get("matrix_partition", {})
            item = completed.get(block)
            if (
                item is None
                or item.get("sha256") != sha256(manifest_path)
                or item.get("selected_job_count") != 63
                or item.get("source_file_count") != 41
                or manifest.get("schema_version") != "transit-evaluation-matrix-v7"
                or manifest.get("dataset_sha256") != EXPECTED_DATASET_SHA256
                or manifest.get("methods_manifest_sha256") != EXPECTED_HASHES["methods"]
                or manifest.get("sumo_runtime_signature") != EXPECTED_RUNTIME_SIGNATURE
                or len(manifest.get("source_sha256", {})) != 41
                or partition.get("shard_index") != shard_index
                or partition.get("shard_count") != 2
                or partition.get("selected_job_count") != 63
                or partition.get("full_job_count") != 126
                or len(jobs) != 63
            ):
                raise ValueError(f"partition matrix contract failed: {manifest_path}")
            observed: set[str] = set()
            for job in jobs:
                name = str(job.get("method", {}).get("name"))
                method = methods.get(name)
                if method is None:
                    raise ValueError(f"unregistered physical method: {name}")
                output = Path(job["output"]).expanduser().resolve()
                if not output.is_file() or PARTITIONS.resolve() not in output.parents:
                    raise ValueError(f"partition output missing or outside sealed root: {output}")
                source_reference = validate_result(
                    job, output, method, source_reference, frozen_sources
                )
                signature = str(job["signature"])
                if signature in observed:
                    raise ValueError(f"duplicate signature in partition block: {signature}")
                observed.add(signature)
            if len(observed) != 63 or shard_signatures & observed:
                raise ValueError(f"partition block signature coverage failed: {manifest_path}")
            shard_signatures.update(observed)
            source_manifests[(block, shard_index)] = {
                "path": manifest_path,
                "sha256": sha256(manifest_path),
            }
        if len(shard_signatures) != 126 or signatures & shard_signatures:
            raise ValueError(f"partition signature union failed: {shard}")
        signatures.update(shard_signatures)
        files.update(shard_files)
        stats[shard] = {
            "block_manifest_count": 2,
            "file_count": len(shard_files),
            "host_execution_sha256": sha256(host_path),
            "result_count": 126,
            "size_bytes": sum(path.stat().st_size for path in shard_files),
        }
    if len(signatures) != 252 or len(source_manifests) != 4 or source_reference is None:
        raise ValueError("partition union is not exactly 252 source-bound results")
    return stats, files, source_manifests, signatures


def validate_canonical(
    methods: dict[str, dict], source_manifests: dict, partition_signatures: set[str]
) -> tuple[dict, set[Path]]:
    registration = load_json(CONTROL_PATHS["registration"])
    blocks = {
        block["name"]: {
            "sumo_seed": int(block["sumo_seed"]),
            "selection_seed": int(block["demand_realization_seed"]),
        }
        for block in registration["development_evaluation"]["blocks"]
    }
    if set(blocks) != set(BLOCKS):
        raise ValueError("registered development block design drifted")
    signatures: set[str] = set()
    outputs: set[Path] = set()
    canonical_files: set[Path] = set()
    block_stats = {}
    expected_cells = {(name, scale) for name in methods for scale in SCALES}
    for block in BLOCKS:
        path = CANONICAL / block / "matrix_manifest.json"
        manifest = load_json(path)
        merge = manifest.get("matrix_merge", {})
        jobs = manifest.get("jobs", [])
        if (
            manifest.get("schema_version") != "transit-evaluation-matrix-v7"
            or len(jobs) != 126
            or merge.get("all_outputs_verified") is not True
            or merge.get("job_count") != 126
            or merge.get("method_count") != 42
            or merge.get("environment_cells_per_method") != 3
            or len(merge.get("source_manifests", [])) != 2
        ):
            raise ValueError(f"canonical matrix contract failed: {block}")
        expected_sources = {
            source_manifests[(block, shard)]["sha256"] for shard in (0, 1)
        }
        observed_sources = {
            descriptor.get("sha256") for descriptor in merge["source_manifests"]
        }
        if observed_sources != expected_sources:
            raise ValueError(f"canonical source-manifest binding failed: {block}")
        observed_cells = set()
        for job in jobs:
            name = str(job.get("method", {}).get("name"))
            scale = float(job["scenario"]["demand_scale"])
            if (
                name not in methods
                or scale not in SCALES
                or int(job["sumo_seed"]) != blocks[block]["sumo_seed"]
                or int(job["scenario"]["selection_seed"])
                != blocks[block]["selection_seed"]
                or job.get("environment_block")
                not in {block, f"sumo-seed-{blocks[block]['sumo_seed']}"}
            ):
                raise ValueError(f"canonical Cartesian cell drifted: {block}")
            observed_cells.add((name, scale))
            signature = str(job["signature"])
            output = Path(job["output"]).expanduser().resolve()
            if (
                signature in signatures
                or output in outputs
                or signature not in partition_signatures
                or not output.is_file()
            ):
                raise ValueError("canonical signature/output binding failed")
            payload = load_json(output)
            if payload.get("evaluation_job", {}).get("signature") != signature or not finite(payload):
                raise ValueError(f"canonical output validation failed: {output}")
            signatures.add(signature)
            outputs.add(output)
        if observed_cells != expected_cells:
            raise ValueError(f"incomplete canonical Cartesian matrix: {block}")
        canonical_files.add(path)
        block_stats[block] = {
            "job_count": 126,
            "matrix_manifest_sha256": sha256(path),
            "source_manifest_count": 2,
        }
    if signatures != partition_signatures or len(outputs) != 252:
        raise ValueError("canonical union differs from the two source partitions")
    return {
        "blocks": block_stats,
        "demand_scales": sorted(SCALES),
        "physical_method_count": 42,
        "result_count": 252,
        "unique_job_signature_count": 252,
    }, canonical_files


def build() -> tuple[dict, dict]:
    controls, control_files, methods, frozen_sources = validate_controls()
    scheduler, scheduler_files = validate_scheduler()
    partition_stats, partition_files, source_manifests, signatures = validate_partitions(
        methods, frozen_sources
    )
    canonical_stats, canonical_files = validate_canonical(
        methods, source_manifests, signatures
    )
    audit = {
        "schema_version": "transit-h2oplus-v9-adaptive-evaluation-audit-v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_integrity": {
            "all_numeric_values_finite": True,
            "canonical_matrix_count": 2,
            "corrected_source_file_count": 41,
            "physical_method_count": 42,
            "passed": True,
            "result_count": 252,
            "source_manifest_count": 4,
            "source_partition_count": 2,
            "unique_job_signature_count": 252,
        },
        "canonical": canonical_stats,
        "controls": controls,
        "deployment_aggregate_sha256": EXPECTED_DEPLOYMENT_AGGREGATE,
        "partitions": partition_stats,
        "scheduler": scheduler,
        "training_stability": {
            "artifact_integrity_passed": True,
            "adaptation_seed6202_late_stability_passed": False,
            "ratio_intervention_ever_applied": False,
            "intervention_attribution_available": False,
        },
        "scientific_results": {
            "analysis_completed": False,
            "artifact_integrity_is_not_scientific_performance": True,
            "formal_confirmation_authorized_or_submitted": False,
            "preregistered_gates_applied": False,
        },
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "supports_current_formal_held_out_model_effect_claim": False,
        "passed": True,
    }
    atomic_json(AUDIT_PATH, audit)

    tool_files = {
        Path(__file__).resolve(),
        REPO / "reproducibility/merge_evaluation_matrix_v7.py",
        REPO / "reproducibility/analyze_h2oplus_v9_adaptive.py",
    }
    files = (
        control_files
        | scheduler_files
        | partition_files
        | canonical_files
        | tool_files
        | {AUDIT_PATH}
    )
    records = [file_record(path) for path in sorted(files)]
    manifest = {
        "schema_version": "transit-h2oplus-v9-adaptive-evaluation-file-manifest-v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_integrity": audit["artifact_integrity"],
        "audit": file_record(AUDIT_PATH),
        "file_count": len(records),
        "file_records_aggregate_sha256": records_aggregate(records),
        "files": records,
        "formal_held_out_outcomes_read": False,
        "scientific_analysis": {
            "completed": False,
            "formal_confirmation_authorized_or_submitted": False,
        },
        "study_stage": "development",
        "supports_current_formal_held_out_model_effect_claim": False,
        "total_size_bytes": sum(record["size_bytes"] for record in records),
    }
    atomic_json(MANIFEST_PATH, manifest)
    return audit, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--scheduler-evidence", type=Path, default=SCHEDULER_EVIDENCE_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global AUDIT_PATH, MANIFEST_PATH, SCHEDULER_EVIDENCE_PATH
    AUDIT_PATH = args.audit.expanduser().resolve()
    MANIFEST_PATH = args.manifest.expanduser().resolve()
    SCHEDULER_EVIDENCE_PATH = args.scheduler_evidence.expanduser().resolve()
    _, manifest = build()
    print(MANIFEST_PATH)
    print(manifest["file_records_aggregate_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
