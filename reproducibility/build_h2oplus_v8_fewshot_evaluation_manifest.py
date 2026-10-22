#!/usr/bin/env python3
"""Audit and seal the corrected H2O+ v8 few-shot development evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reproducibility/h2oplus_v8_fewshot_results"
PARTITIONS = RESULTS / "evaluation_partitions_source_bound_v3"
CANONICAL = RESULTS / "evaluation_canonical_source_bound_v3"
EVIDENCE = RESULTS / "evidence/evaluation_v3"
QUEUE = Path("/home/erzhu419/.claude/scheduler/queue.json")

AUDIT_PATH = RESULTS / "evaluation_audit_v3.json"
SCHEDULER_EVIDENCE_PATH = EVIDENCE / "scheduler_evidence_v3.json"
MANIFEST_PATH = RESULTS / "evaluation_file_manifest_v3.json"

EXPECTED_HASHES = {
    "authorization": "0bc8b342cf7618c3a0d0b6f0beff15ee021ca26dbaf78cca8005eaeffdcc5e3c",
    "environment": "4ca7ac74ea9a4f327b03f88a1f7d8a32591852905fdf01142454737fa2dc71cd",
    "freeze": "2fb2b581591bb382cabe045eaebb5c1236825e818a26443339d33b1b9342b204",
    "registration": "fb632f92998d82cfb87b2cd1a534058925d4e410be17ecea9070b7a50db7c115",
    "protocol": "26d4f5c4ca9f337c11886855658127ef93a35954ca58eb43710aa70350610294",
    "correction": "16e55e7ec22105cb8065db47a4047fabfb16ebd302be6e7f74e8fbb2f5ec0398",
    "submission": "5080462cf0710e16c6f707259fdc11c5d132449af88b3c621ca1580751d5afe9",
    "deployment": "6f668ffb4cfb83643bc1f840bdb0be4692f716ca23a3b4de0f11ceae861a9e2f",
    "methods": "8dbca88cd754d3fe711f2408b3328cf3947efcad478c221f5d8be8bba017e7f7",
    "training_closeout": "8a12385844f7949d3f7c2cbc16d8e021787c9de04fb14b7e64b46779d2373822",
    "training_audit": "58d5f871a2563469a434918aa9d4f9433956b10b6b36a61ddea52ca5cfe4457e",
    "training_manifest": "6742d8c0d9c965209f0c2ac56a2ac3351290f18564e0573f92b9fff644acf239",
    "training_verification": "c6f124d80952577ddb6d6c6597c0fbf67704c4af92deb35fc0b5f5a0e99809aa",
}

CONTROL_PATHS = {
    "authorization": REPO / "reproducibility/h2oplus_v8_fewshot_authorization_v1.json",
    "environment": REPO / "reproducibility/h2oplus_v8_fewshot_environment_v1.json",
    "freeze": REPO / "reproducibility/h2oplus_v8_fewshot_freeze_v1.json",
    "registration": REPO / "reproducibility/h2oplus_v8_fewshot_registration_v1.json",
    "protocol": REPO / "reproducibility/h2oplus_v8_fewshot_protocol_v1.json",
    "correction": REPO / "reproducibility/h2oplus_v8_fewshot_evaluation_correction_v2.json",
    "submission": REPO / "reproducibility/h2oplus_v8_fewshot_evaluation_submission_v3.json",
    "deployment": REPO
    / "reproducibility/deployments/h2oplus_v8_fewshot_evaluation_v3/workspace/deployment_manifest.json",
    "methods": RESULTS / "evaluation_methods_v8_fewshot.json",
    "training_closeout": RESULTS / "training_closeout.json",
    "training_audit": RESULTS / "training_audit.json",
    "training_manifest": RESULTS / "training_file_manifest.json",
    "training_verification": RESULTS / "training_file_manifest_verification.json",
}

EXPECTED_DEPLOYMENT_AGGREGATE = (
    "723ba0c05f00f675d14ae225d401d446aed4a8e973317e8f06024f243a1237ec"
)
EXPECTED_DATASET_SHA256 = (
    "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899"
)
EXPECTED_RUNTIME_SIGNATURE = (
    "264d3c412b8e6c42320995208d2e009eb809b85626047206c4de752c2dae06e7"
)
EXPECTED_WRAPPERS = {
    "SimpleSAC/evaluate_transit_v8_fewshot.py": (
        "e78d5bf337baf25c7d9d2bd366dd2305dbca8b540caff1638ee3463c8d9e73ea"
    ),
    "reproducibility/run_evaluation_matrix_v8_fewshot.py": (
        "ac895cfec83cb1812d11e62e4348e1c6b9487d3eca85b3a512baff0a1aa5500c"
    ),
}
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
    "t82450": {
        "shard": "shard0_jtl110gpu",
        "status_file": EVIDENCE / "backend_exit_t82450.status",
        "log_file": EVIDENCE / "scheduler_t82450.log",
    },
    "t82451": {
        "shard": "shard1_jtl110gpu",
        "status_file": EVIDENCE / "backend_exit_t82451.status",
        "log_file": EVIDENCE / "scheduler_t82451.log",
    },
}
METHOD = re.compile(
    r"^(?P<algorithm>h2oplus|sac-online)-seed(?P<seed>62\d{2})-events(?P<events>\d{9})$"
)
SEEDS = set(range(6201, 6206))
BUDGETS = {
    "h2oplus": {0, 10_000, 25_000, 50_000, 100_000},
    "sac-online": {0, 10_000, 25_000, 50_000, 55_000, 60_000, 75_000, 100_000, 150_000},
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


def validate_controls() -> tuple[dict, set[Path]]:
    records = {}
    for name, path in CONTROL_PATHS.items():
        actual = sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"control identity drifted for {name}: {actual}")
        records[name] = file_record(path)
    deployment = load_json(CONTROL_PATHS["deployment"])
    if (
        deployment.get("schema_version")
        != "transit-h2oplus-v8-fewshot-evaluation-deployment-v3"
        or deployment.get("aggregate_sha256") != EXPECTED_DEPLOYMENT_AGGREGATE
        or len(deployment.get("entries", [])) != 446
    ):
        raise ValueError("corrected v3 deployment contract drifted")
    correction = load_json(CONTROL_PATHS["correction"])
    submission = load_json(CONTROL_PATHS["submission"])
    if (
        correction.get("formal_held_out_outcomes_read") is not False
        or correction.get("training_or_policy_artifacts_changed") is not False
        or submission.get("source_contract", {}).get("local_equal") is not True
        or submission.get("source_contract", {}).get("remote_equal") is not True
        or submission.get("source_contract", {}).get("evaluator_source_file_count") != 41
        or submission.get("source_contract", {}).get("matrix_source_file_count") != 41
    ):
        raise ValueError("v3 correction or source-equality contract failed")
    return records, set(CONTROL_PATHS.values())


def _queue_tasks() -> dict[str, dict]:
    payload = load_json(QUEUE)
    rows = payload.get("tasks", payload if isinstance(payload, list) else [])
    return {row["id"]: row for row in rows if row.get("id") in TASKS}


def validate_scheduler() -> tuple[dict, set[Path]]:
    queue_tasks = _queue_tasks()
    submission = load_json(CONTROL_PATHS["submission"])
    submitted = {item["task_id"]: item for item in submission["tasks"]}
    if set(queue_tasks) != set(TASKS) or set(submitted) != set(TASKS):
        raise ValueError("v3 scheduler task coverage drifted")
    evidence = {}
    files = set()
    for task_id, expected in TASKS.items():
        task = queue_tasks[task_id]
        task_submission = submitted[task_id]
        status_path = expected["status_file"]
        log_path = expected["log_file"]
        files.update({status_path, log_path})
        fields = status_path.read_text().strip().split("\t")
        if len(fields) != 3:
            raise ValueError(f"malformed token-bound backend status for {task_id}")
        exit_code, backend_finished_at, token = fields
        env = task.get("extra_env", {})
        host_path = PARTITIONS / expected["shard"] / "host_execution.json"
        host = load_json(host_path)
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
            or log_path.read_text().strip()
            != f"DONE H2O+ v8 few-shot shard {host.get('shard_index')}: 315/315 jobs"
        ):
            raise ValueError(f"scheduler/backend/host contract failed for {task_id}")
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
        "schema_version": "transit-h2oplus-v8-fewshot-evaluation-scheduler-evidence-v3",
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


def _method_identity(method: dict) -> tuple[str, int, int]:
    name = str(method.get("name"))
    match = METHOD.fullmatch(name)
    if match is None:
        raise ValueError(f"unregistered few-shot method: {name}")
    algorithm = match.group("algorithm")
    seed = int(match.group("seed"))
    events = int(match.group("events"))
    if (
        seed not in SEEDS
        or events not in BUDGETS[algorithm]
        or method.get("algorithm") != algorithm
        or int(method.get("training_seed")) != seed
        or int(method.get("online_events")) != events
        or method.get("training_plan_sha256") != EXPECTED_HASHES["protocol"]
    ):
        raise ValueError(f"few-shot method contract drifted: {name}")
    return algorithm, seed, events


def _source_wrapper_hash(source: dict, suffix: str) -> str | None:
    matches = [value for path, value in source.items() if path.endswith(suffix)]
    return matches[0] if len(matches) == 1 else None


def validate_result(job: dict, output: Path, canonical_source: dict | None) -> dict:
    payload = load_json(output)
    method = job["method"]
    algorithm, seed, events = _method_identity(method)
    training = payload.get("policy", {}).get("training_metadata", {})
    source = payload.get("source_sha256", {})
    job_source = job.get("source_sha256", {})
    if (
        payload.get("schema_version") != "transit-evaluation-v7"
        or payload.get("evaluation_job", {}).get("signature") != job.get("signature")
        or payload.get("method") != method.get("name")
        or int(payload.get("sumo_seed")) != int(job.get("sumo_seed"))
        or not finite(payload)
        or source != job_source
        or len(source) != 41
        or any(
            _source_wrapper_hash(source, suffix) != digest
            for suffix, digest in EXPECTED_WRAPPERS.items()
        )
        or job.get("dataset_sha256") != EXPECTED_DATASET_SHA256
        or payload.get("dataset", {}).get("sha256") != EXPECTED_DATASET_SHA256
        or job.get("sumo_runtime_signature") != EXPECTED_RUNTIME_SIGNATURE
        or payload.get("runtime", {}).get("sumo_runtime_signature")
        != EXPECTED_RUNTIME_SIGNATURE
        or payload.get("policy", {}).get("sha256") != method.get("checkpoint_sha256")
        or training.get("study_stage") != "development"
        or training.get("formal_held_out_outcomes_read") is not False
        or training.get("fewshot_online_events_nominal") != events
        or training.get("fewshot_protocol", {}).get("sha256")
        != EXPECTED_HASHES["protocol"]
        or training.get("run_signature") != method.get("training_run_signature")
        or training.get("fewshot_model_checkpoint", {}).get("sha256")
        != method.get("training_checkpoint_sha256")
        or training.get("training_admission", {}).get("sha256")
        != method.get("training_admission_sha256")
        or payload.get("scenario", {}).get("manifest_sha256")
        != job.get("scenario", {}).get("manifest_sha256")
        or any(
            abs(float(value)) > 1e-9
            for value in payload.get("passenger_welfare", {})
            .get("chronological_coverage_errors", {})
            .values()
        )
    ):
        raise ValueError(f"few-shot result provenance failed: {output}")
    if canonical_source is not None and source != canonical_source:
        raise ValueError(f"source map differs across corrected v3 results: {output}")
    return {
        "algorithm": algorithm,
        "events": events,
        "seed": seed,
        "source": source,
    }


def validate_partitions() -> tuple[dict, set[Path], dict, set[str]]:
    stats = {}
    files: set[Path] = set()
    source_manifests = {}
    signatures: set[str] = set()
    canonical_source = None
    for shard_index, shard in enumerate(("shard0_jtl110gpu", "shard1_jtl110gpu")):
        root = PARTITIONS / shard
        shard_files = {path for path in root.rglob("*") if path.is_file()}
        results = sorted(root.rglob("sumo_seed*.json"))
        manifests = sorted(root.rglob("matrix_manifest.shard-*.json"))
        host_path = root / "host_execution.json"
        host = load_json(host_path)
        if (
            len(results) != 315
            or len(manifests) != 3
            or host.get("status") != "completed"
            or host.get("returncode") != 0
            or host.get("shard_index") != shard_index
            or host.get("shard_count") != 2
            or host.get("training") is not False
            or host.get("formal_held_out_outcomes_read") is not False
            or len(host.get("completed_blocks", [])) != 3
            or sum(item.get("selected_job_count", 0) for item in host["completed_blocks"])
            != 315
        ):
            raise ValueError(f"partition host/result contract failed: {shard}")
        completed = {
            Path(item["path"]).parent.name: item for item in host["completed_blocks"]
        }
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
                or item.get("selected_job_count") != 105
                or manifest.get("schema_version") != "transit-evaluation-matrix-v7"
                or manifest.get("dataset_sha256") != EXPECTED_DATASET_SHA256
                or manifest.get("methods_manifest_sha256") != EXPECTED_HASHES["methods"]
                or manifest.get("sumo_runtime_signature") != EXPECTED_RUNTIME_SIGNATURE
                or partition.get("shard_index") != shard_index
                or partition.get("shard_count") != 2
                or partition.get("selected_job_count") != 105
                or len(jobs) != 105
            ):
                raise ValueError(f"partition matrix contract failed: {manifest_path}")
            observed = set()
            for job in jobs:
                output = Path(job["output"]).resolve()
                if not output.is_file() or PARTITIONS.resolve() not in output.parents:
                    raise ValueError(f"partition output missing or outside sealed root: {output}")
                details = validate_result(job, output, canonical_source)
                if canonical_source is None:
                    canonical_source = details["source"]
                signature = str(job["signature"])
                if signature in observed:
                    raise ValueError(f"duplicate signature in partition block: {signature}")
                observed.add(signature)
            if len(observed) != 105 or shard_signatures & observed:
                raise ValueError(f"partition block signature coverage failed: {manifest_path}")
            shard_signatures.update(observed)
            source_manifests[(block, shard_index)] = {
                "path": manifest_path,
                "sha256": sha256(manifest_path),
            }
        if len(shard_signatures) != 315 or signatures & shard_signatures:
            raise ValueError(f"partition signature union failed: {shard}")
        signatures.update(shard_signatures)
        files.update(shard_files)
        stats[shard] = {
            "block_manifest_count": 3,
            "file_count": len(shard_files),
            "host_execution_sha256": sha256(host_path),
            "result_count": 315,
            "size_bytes": sum(path.stat().st_size for path in shard_files),
        }
    if len(signatures) != 630 or len(source_manifests) != 6 or canonical_source is None:
        raise ValueError("partition union is not exactly 630 source-bound results")
    return stats, files, source_manifests, signatures


def validate_canonical(source_manifests: dict, partition_signatures: set[str]) -> tuple[dict, set[Path]]:
    registration = load_json(CONTROL_PATHS["registration"])
    blocks = {
        block["name"]: {
            "sumo_seed": int(block["sumo_seed"]),
            "selection_seed": int(block["demand_realization_seed"]),
        }
        for block in registration["development_environment"]["blocks"]
    }
    expected_blocks = {f"v8-fewshot-development-{index:02d}" for index in range(1, 4)}
    if set(blocks) != expected_blocks:
        raise ValueError("registered development block design drifted")
    expected_methods = {
        f"{algorithm}-seed{seed}-events{events:09d}"
        for algorithm, budgets in BUDGETS.items()
        for seed in SEEDS
        for events in budgets
    }
    if len(expected_methods) != 70:
        raise AssertionError("internal method-grid error")
    signatures: set[str] = set()
    outputs: set[Path] = set()
    canonical_files = set()
    block_stats = {}
    for block in sorted(blocks):
        path = CANONICAL / block / "matrix_manifest.json"
        manifest = load_json(path)
        merge = manifest.get("matrix_merge", {})
        jobs = manifest.get("jobs", [])
        if (
            manifest.get("schema_version") != "transit-evaluation-matrix-v7"
            or len(jobs) != 210
            or merge.get("all_outputs_verified") is not True
            or merge.get("job_count") != 210
            or merge.get("method_count") != 70
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
            algorithm, seed, events = _method_identity(job["method"])
            name = str(job["method"]["name"])
            scale = float(job["scenario"]["demand_scale"])
            if (
                name not in expected_methods
                or scale not in {0.85, 1.0, 1.15}
                or int(job["sumo_seed"]) != blocks[block]["sumo_seed"]
                or int(job["scenario"]["selection_seed"])
                != blocks[block]["selection_seed"]
                or job.get("environment_block") not in {
                    block,
                    f"sumo-seed-{blocks[block]['sumo_seed']}",
                }
            ):
                raise ValueError(f"canonical Cartesian cell drifted: {block}")
            observed_cells.add((name, scale))
            signature = str(job["signature"])
            output = Path(job["output"]).resolve()
            if (
                signature in signatures
                or output in outputs
                or signature not in partition_signatures
                or not output.is_file()
            ):
                raise ValueError("canonical signature/output binding failed")
            payload = load_json(output)
            if (
                payload.get("evaluation_job", {}).get("signature") != signature
                or not finite(payload)
            ):
                raise ValueError(f"canonical output validation failed: {output}")
            signatures.add(signature)
            outputs.add(output)
        expected_cells = {
            (method, scale)
            for method in expected_methods
            for scale in (0.85, 1.0, 1.15)
        }
        if observed_cells != expected_cells:
            raise ValueError(f"incomplete canonical Cartesian matrix: {block}")
        canonical_files.add(path)
        block_stats[block] = {
            "job_count": 210,
            "matrix_manifest_sha256": sha256(path),
            "source_manifest_count": 2,
        }
    if signatures != partition_signatures or len(outputs) != 630:
        raise ValueError("canonical union differs from the two source partitions")
    return {
        "blocks": block_stats,
        "demand_scales": [0.85, 1.0, 1.15],
        "method_count": 70,
        "result_count": 630,
        "training_seeds": sorted(SEEDS),
        "unique_job_signature_count": 630,
    }, canonical_files


def build() -> tuple[dict, dict]:
    controls, control_files = validate_controls()
    scheduler, scheduler_files = validate_scheduler()
    partition_stats, partition_files, source_manifests, signatures = validate_partitions()
    canonical_stats, canonical_files = validate_canonical(source_manifests, signatures)
    audit = {
        "schema_version": "transit-h2oplus-v8-fewshot-evaluation-audit-v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_integrity": {
            "all_numeric_values_finite": True,
            "canonical_matrix_count": 3,
            "corrected_source_file_count": 41,
            "method_count": 70,
            "passed": True,
            "result_count": 630,
            "source_manifest_count": 6,
            "source_partition_count": 2,
            "unique_job_signature_count": 630,
        },
        "canonical": canonical_stats,
        "controls": controls,
        "deployment_aggregate_sha256": EXPECTED_DEPLOYMENT_AGGREGATE,
        "partitions": partition_stats,
        "scheduler": scheduler,
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
        REPO / "reproducibility/analyze_h2oplus_v8_fewshot.py",
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
        "schema_version": "transit-h2oplus-v8-fewshot-evaluation-file-manifest-v3",
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
