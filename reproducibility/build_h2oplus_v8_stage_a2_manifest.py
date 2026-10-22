#!/usr/bin/env python3
"""Audit and seal the H2O+ v8 Stage A2 SAC development artifacts."""

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
RESULTS = (
    REPO
    / "reproducibility/deployments/h2oplus_v8_stage_a2_sac_baseline_v1/results"
)
CANONICAL = REPO / "reproducibility/h2oplus_v8_stage_a2_results/canonical"
EVIDENCE = REPO / "reproducibility/h2oplus_v8_stage_a2_results/evidence"
POLICIES = REPO / "reproducibility/h2oplus_v8_stage_a2_policy_export_v1"

EXPECTED_IDENTITIES = {
    "deployment_manifest": "5f879bd6e16d13d2f969e6c7a3f619dc10426c633d095ad5df8d479120ba782d",
    "registration": "ac84a4ce4612a7c8a5e03c64e0edc767f1c8a5a8c1c33de1f838781f3bde9308",
    "authorization": "147344656c073f518af79c3d682a75c01d255a7e0fdf13c03130bf1021d5cabb",
    "freeze": "2186a2dfe0968446075f654ba1657aff6c059e78cdc5fb8310e5b9f4ab3fbd64",
    "methods": "7904e95a1d010234f60214813876b0cf51afbbd35d69a3365403b850a44d7b35",
    "policy_export": "9307ece8974bb22aa12341aafcf4d0d334495444ce0e70b13cc1481b2e393e59",
    "submission": "9e9c083f7c80542e9cb63a10497bdccaa4dbb8862e2d1f46167fe35d2f9b8980",
    "baseline_audit": "52c88d9d33a4aae518a8bf3889f7cedd4e3e971e84e18d6b23fcd3b951b53a12",
    "baseline_manifest": "9d35a4397c758b06b03f1c4e399e727d116c6a9198656ba2b13a49f4a5e2826f",
    "environment_design": "d4a40a900cf9f72043c1d7c2067d02d6c79f2081f856b38cb257ec2ad0d18f3c",
    "reference_stage_a_analysis": "20c8532b264d3f0048bf2b654ea6a1968aa4c272fcc87e48c989e578bf4e95c8",
    "reference_stage_a_manifest": "a424b65d330d8338a2aaaf1f7af3c381ef787a6c85267553ff9e7dec8a6ceb8d",
    "reference_stage_a_verification": "bfc52b45946b66440e27e4248d5a790e9b9cab9e2053cd6b60c0c37bf1dacf3c",
}

CONTROL_PATHS = {
    "deployment_manifest": REPO
    / "reproducibility/deployments/h2oplus_v8_stage_a2_sac_baseline_v1/workspace/deployment_manifest.json",
    "registration": REPO / "reproducibility/h2oplus_v8_stage_a2_registration_v1.json",
    "authorization": REPO
    / "reproducibility/h2oplus_v8_stage_a2_execution_authorization_v1.json",
    "freeze": REPO / "reproducibility/h2oplus_v8_stage_a2_freeze_v1.json",
    "methods": POLICIES / "evaluation_methods_v8_stage_a2.json",
    "policy_export": POLICIES / "policy_export_manifest_v8_stage_a2.json",
    "submission": REPO / "reproducibility/h2oplus_v8_stage_a2_submission_v1.json",
    "baseline_audit": REPO
    / "reproducibility/formal_results_v7_v7r1/baseline_matrix_audit.json",
    "baseline_manifest": REPO
    / "reproducibility/formal_results_v7_v7r1/baseline_manifest.json",
    "environment_design": REPO
    / "reproducibility/h2oplus_v8_development_environment_stage_a_v1.json",
    "reference_stage_a_analysis": REPO
    / "reproducibility/h2oplus_v8_stage_a_results/stage_a_gate_analysis.json",
    "reference_stage_a_manifest": REPO
    / "reproducibility/h2oplus_v8_stage_a_results/stage_a_file_manifest.json",
    "reference_stage_a_verification": REPO
    / "reproducibility/h2oplus_v8_stage_a_results/stage_a_file_manifest_verification.json",
}

EXPECTED_ALGORITHMS = {"sac-mixed", "sac-online"}
EXPECTED_SEEDS = set(range(5101, 5106))
EXPECTED_SCALES = {0.85, 1.0, 1.15}
EXPECTED_DATASET_SHA256 = "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899"
EXPECTED_DEPLOYMENT_AGGREGATE = (
    "81fbd12f3c18c4d0702306cf36e7e6312ea9384d87c56f9729433197ce067b90"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


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


def validate_controls() -> dict[str, dict]:
    descriptors = {}
    for name, path in CONTROL_PATHS.items():
        actual = sha256(path)
        if actual != EXPECTED_IDENTITIES[name]:
            raise ValueError(f"{name} identity drifted: {actual}")
        descriptors[name] = file_record(path)
    deployment = load_json(CONTROL_PATHS["deployment_manifest"])
    if deployment.get("aggregate_sha256") != EXPECTED_DEPLOYMENT_AGGREGATE:
        raise ValueError("deployment aggregate identity drifted")
    reference_verification = load_json(CONTROL_PATHS["reference_stage_a_verification"])
    if reference_verification.get("passed") is not True:
        raise ValueError("sealed Stage A reference verification did not pass")
    return descriptors


def _method_contracts() -> dict[str, dict]:
    payload = load_json(CONTROL_PATHS["methods"])
    methods = payload.get("methods", [])
    contracts = {str(item["name"]): item for item in methods}
    if len(methods) != 10 or len(contracts) != 10:
        raise ValueError("Stage A2 methods manifest does not contain ten unique methods")
    return contracts


def _validate_method(job_method: dict, contracts: dict[str, dict]) -> None:
    name = str(job_method.get("name"))
    expected = contracts.get(name)
    if expected is None:
        raise ValueError(f"unregistered method: {name}")
    for key in (
        "algorithm",
        "training_seed",
        "training_admission_sha256",
        "training_checkpoint_sha256",
        "training_plan_sha256",
        "training_run_signature",
        "type",
    ):
        if job_method.get(key) != expected.get(key):
            raise ValueError(f"method contract drifted for {name}: {key}")
    actor = POLICIES / str(expected["checkpoint"])
    if not actor.is_file() or sha256(actor) != job_method.get("checkpoint_sha256"):
        raise ValueError(f"compact actor identity drifted for {name}")


def validate_partitions() -> tuple[dict, set[Path], dict[tuple[str, int], dict]]:
    contracts = _method_contracts()
    stats = {}
    files: set[Path] = set()
    source_manifests: dict[tuple[str, int], dict] = {}
    global_signatures: set[str] = set()
    for shard_index, shard_name in enumerate(("shard0_jtl110gpu", "shard1_jtl110gpu")):
        root = RESULTS / shard_name
        shard_files = {path for path in root.rglob("*") if path.is_file()}
        result_paths = sorted(root.rglob("sumo_seed*.json"))
        manifest_paths = sorted(root.rglob("matrix_manifest*.json"))
        if len(result_paths) != 75 or len(manifest_paths) != 5:
            raise ValueError(f"{shard_name} is not 75 results plus five manifests")
        host_path = root / "host_execution.json"
        host = load_json(host_path)
        if (
            host.get("status") != "completed"
            or host.get("returncode") != 0
            or host.get("shard_index") != shard_index
            or host.get("shard_count") != 2
            or host.get("workers") != 12
            or host.get("training") is not False
            or host.get("formal_held_out_outcomes_read") is not False
            or len(host.get("completed_blocks", [])) != 5
            or sum(item.get("selected_job_count", 0) for item in host["completed_blocks"]) != 75
        ):
            raise ValueError(f"{shard_name} host execution contract failed")
        evidence = host.get("evidence", {})
        for name in ("registration", "authorization", "freeze", "methods", "policy_export"):
            if evidence.get(name, {}).get("sha256") != EXPECTED_IDENTITIES[name]:
                raise ValueError(f"{shard_name} host evidence drifted for {name}")

        completed = {
            Path(item["path"]).parent.name: item for item in host["completed_blocks"]
        }
        shard_signatures: set[str] = set()
        for manifest_path in manifest_paths:
            block = manifest_path.parent.name
            manifest = load_json(manifest_path)
            partition = manifest.get("matrix_partition", {})
            jobs = manifest.get("jobs", [])
            item = completed.get(block)
            if (
                item is None
                or item.get("sha256") != sha256(manifest_path)
                or item.get("selected_job_count") != 15
                or manifest.get("schema_version") != "transit-evaluation-matrix-v7"
                or manifest.get("mdp_version") != "v7"
                or manifest.get("dataset_sha256") != EXPECTED_DATASET_SHA256
                or manifest.get("methods_manifest_sha256") != EXPECTED_IDENTITIES["methods"]
                or partition.get("shard_index") != shard_index
                or partition.get("shard_count") != 2
                or partition.get("selected_job_count") != 15
                or len(jobs) != 15
            ):
                raise ValueError(f"partition manifest contract failed: {manifest_path}")
            expected_signatures = set()
            for job in jobs:
                _validate_method(job["method"], contracts)
                expected_signatures.add(str(job["signature"]))
            observed_signatures = set()
            for path in sorted(manifest_path.parent.rglob("sumo_seed*.json")):
                payload = load_json(path)
                signature = str(payload.get("evaluation_job", {}).get("signature"))
                if (
                    payload.get("schema_version") != "transit-evaluation-v7"
                    or signature in observed_signatures
                    or not finite(payload)
                ):
                    raise ValueError(f"invalid evaluation output: {path}")
                observed_signatures.add(signature)
            if expected_signatures != observed_signatures:
                raise ValueError(f"partition result binding failed: {manifest_path}")
            if shard_signatures & observed_signatures:
                raise ValueError(f"duplicate result signatures in {shard_name}")
            shard_signatures.update(observed_signatures)
            source_manifests[(block, shard_index)] = {
                "path": manifest_path,
                "sha256": sha256(manifest_path),
            }
        if len(shard_signatures) != 75 or global_signatures & shard_signatures:
            raise ValueError(f"{shard_name} signature coverage failed")
        global_signatures.update(shard_signatures)
        files.update(shard_files)
        stats[shard_name] = {
            "result_count": 75,
            "block_manifest_count": 5,
            "file_count": len(shard_files),
            "size_bytes": sum(path.stat().st_size for path in shard_files),
            "host_execution_sha256": sha256(host_path),
        }
    if len(global_signatures) != 150 or len(source_manifests) != 10:
        raise ValueError("Stage A2 partition union is not 150 unique bound results")
    return stats, files, source_manifests


def validate_canonical(source_manifests: dict[tuple[str, int], dict]) -> tuple[dict, set[Path]]:
    registration = load_json(CONTROL_PATHS["registration"])
    blocks = {
        str(block["name"]): {
            "sumo_seed": int(block["sumo_seed"]),
            "realization_seed": int(block["demand_realization_seed"]),
        }
        for block in registration["development_environment"]["blocks"]
    }
    expected_blocks = {f"v8-development-{index:02d}" for index in range(1, 6)}
    if set(blocks) != expected_blocks:
        raise ValueError("registered Stage A2 block design drifted")
    contracts = _method_contracts()
    signatures: set[str] = set()
    outputs: set[Path] = set()
    canonical_files: set[Path] = set()
    block_stats = {}
    for block_name in sorted(blocks):
        path = CANONICAL / block_name / "matrix_manifest.json"
        manifest = load_json(path)
        merge = manifest.get("matrix_merge", {})
        jobs = manifest.get("jobs", [])
        if (
            manifest.get("schema_version") != "transit-evaluation-matrix-v7"
            or len(jobs) != 30
            or merge.get("all_outputs_verified") is not True
            or merge.get("job_count") != 30
            or merge.get("method_count") != 10
            or merge.get("environment_cells_per_method") != 3
            or len(merge.get("source_manifests", [])) != 2
        ):
            raise ValueError(f"canonical matrix contract failed: {block_name}")
        expected_sources = {
            source_manifests[(block_name, shard)]["sha256"] for shard in (0, 1)
        }
        observed_sources = {
            str(item.get("sha256")) for item in merge["source_manifests"]
        }
        if observed_sources != expected_sources:
            raise ValueError(f"canonical source binding failed: {block_name}")

        observed_cells = set()
        for job in jobs:
            method = job["method"]
            _validate_method(method, contracts)
            algorithm = str(method["algorithm"])
            seed = int(method["training_seed"])
            scale = float(job["scenario"]["demand_scale"])
            sumo_seed = int(job["sumo_seed"])
            realization_seed = int(job["scenario"]["selection_seed"])
            if (
                algorithm not in EXPECTED_ALGORITHMS
                or seed not in EXPECTED_SEEDS
                or scale not in EXPECTED_SCALES
                or sumo_seed != blocks[block_name]["sumo_seed"]
                or realization_seed != blocks[block_name]["realization_seed"]
                or job.get("environment_block") not in {
                    block_name,
                    f"sumo-seed-{sumo_seed}",
                }
            ):
                raise ValueError(f"canonical job coverage drifted: {block_name}")
            observed_cells.add((algorithm, seed, scale))
            signature = str(job["signature"])
            output = Path(job["output"]).resolve()
            if signature in signatures or output in outputs or not output.is_file():
                raise ValueError("duplicate signature/output or missing canonical output")
            payload = load_json(output)
            if (
                payload.get("schema_version") != "transit-evaluation-v7"
                or payload.get("evaluation_job", {}).get("signature") != signature
                or not finite(payload)
            ):
                raise ValueError(f"canonical result binding failed: {output}")
            signatures.add(signature)
            outputs.add(output)
        expected_cells = {
            (algorithm, seed, scale)
            for algorithm in EXPECTED_ALGORITHMS
            for seed in EXPECTED_SEEDS
            for scale in EXPECTED_SCALES
        }
        if observed_cells != expected_cells:
            raise ValueError(f"incomplete Stage A2 Cartesian matrix: {block_name}")
        canonical_files.add(path)
        block_stats[block_name] = {
            "job_count": 30,
            "matrix_manifest_sha256": sha256(path),
            "source_manifest_count": 2,
        }
    if len(signatures) != 150 or len(outputs) != 150:
        raise ValueError("canonical Stage A2 matrix is not 150 unique bound results")
    return {
        "blocks": block_stats,
        "result_count": 150,
        "unique_job_signature_count": 150,
        "algorithms": sorted(EXPECTED_ALGORITHMS),
        "training_seeds": sorted(EXPECTED_SEEDS),
        "sumo_seeds": list(range(4101, 4106)),
        "demand_realization_seeds": list(range(4201, 4206)),
        "demand_scales": sorted(EXPECTED_SCALES),
    }, canonical_files


def validate_closeout_evidence() -> tuple[dict, set[Path]]:
    paths = {
        EVIDENCE / "backend_exit_t82300.status",
        EVIDENCE / "backend_exit_t82301.status",
        EVIDENCE / "stage_a2_scheduler_evidence.json",
        EVIDENCE / "stage_a2_partition_sync_evidence.json",
        EVIDENCE / "stage_a2_analysis_identity_reconciliation.json",
    }
    for path in paths:
        if not path.is_file():
            raise ValueError(f"missing Stage A2 closeout evidence: {path}")
    scheduler = load_json(EVIDENCE / "stage_a2_scheduler_evidence.json")
    sync = load_json(EVIDENCE / "stage_a2_partition_sync_evidence.json")
    identity = load_json(EVIDENCE / "stage_a2_analysis_identity_reconciliation.json")
    if scheduler.get("passed") is not True or sync.get("passed") is not True:
        raise ValueError("Stage A2 scheduler or synchronization evidence failed")
    tasks = scheduler.get("tasks", {})
    if set(tasks) != {"t82300", "t82301"} or any(
        item.get("exit_code") != 0
        or item.get("token_bound") is not True
        or item.get("host_execution_completed") is not True
        or item.get("result_count") != 75
        or item.get("manifest_count") != 5
        for item in tasks.values()
    ):
        raise ValueError("Stage A2 scheduler closeout contract failed")
    if sync.get("partition_count") != 2 or sync.get("exact_match") is not True:
        raise ValueError("Stage A2 exact partition synchronization failed")
    if (
        identity.get("passed") is not True
        or identity.get("failure_stage") != "pre-statistical matrix identity validation"
        or identity.get("gate_statistics_computed_before_reconciliation") is not False
        or identity.get("formal_held_out_outcomes_read") is not False
        or identity.get("matrix_count") != 10
        or identity.get("portable_identity_count") != 1
        or identity.get("source_file_count") != 39
        or identity.get("scenario_block_count") != 5
        or identity.get("scenario_content_identity_matches_between_stages") is not True
        or identity.get("training_or_evaluation_rerun") is not False
        or identity.get("threshold_or_metric_change") is not False
    ):
        raise ValueError("Stage A2 analysis identity reconciliation failed")
    return {
        "scheduler_evidence_sha256": sha256(EVIDENCE / "stage_a2_scheduler_evidence.json"),
        "partition_sync_evidence_sha256": sha256(
            EVIDENCE / "stage_a2_partition_sync_evidence.json"
        ),
        "analysis_identity_reconciliation_sha256": sha256(
            EVIDENCE / "stage_a2_analysis_identity_reconciliation.json"
        ),
    }, paths


def build_manifest() -> dict:
    controls = validate_controls()
    partition_stats, partition_files, source_manifests = validate_partitions()
    matrix_audit, canonical_files = validate_canonical(source_manifests)
    closeout, evidence_files = validate_closeout_evidence()
    policy_files = {path for path in POLICIES.rglob("*") if path.is_file()}
    implementation_files = {
        Path(__file__).resolve(),
        REPO / "reproducibility/verify_h2oplus_v8_stage_a2_manifest.py",
        REPO / "reproducibility/analyze_h2oplus_v8_stage_a2.py",
        REPO / "reproducibility/merge_evaluation_matrix_v7.py",
        REPO / "tests/test_h2oplus_v8_stage_a2.py",
        REPO / "tests/test_h2oplus_v8_stage_a2_manifest.py",
    }
    all_files = (
        partition_files
        | canonical_files
        | evidence_files
        | policy_files
        | set(CONTROL_PATHS.values())
        | implementation_files
    )
    missing = sorted(path for path in all_files if not path.is_file())
    if missing:
        raise ValueError(f"Stage A2 manifest inputs missing: {missing}")
    records = [file_record(path) for path in sorted(all_files)]
    return {
        "schema_version": "transit-h2oplus-v8-stage-a2-file-manifest-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "A2-H2Oplus-vs-SAC-development-baseline-confirmation",
        "study_stage": "development",
        "training": False,
        "formal_held_out_outcomes_read": False,
        "supports_current_formal_held_out_model_effect_claim": False,
        "artifact_integrity": {
            "passed": True,
            "independent_verification_required": True,
            "partitions": partition_stats,
            "source_manifest_count": len(source_manifests),
            "canonical_matrix_count": 5,
            **matrix_audit,
        },
        "scientific_analysis": {
            "run_after_independent_artifact_verification": True,
            "completed": False,
            "stage_b_authorized_or_submitted": False,
        },
        "closeout_evidence": closeout,
        "control_identities": controls,
        "file_count": len(records),
        "total_size_bytes": sum(item["size_bytes"] for item in records),
        "file_records_aggregate_sha256": records_aggregate(records),
        "files": sorted(records, key=lambda item: item["path"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO
        / "reproducibility/h2oplus_v8_stage_a2_results/stage_a2_file_manifest.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_manifest()
    output = args.output.expanduser().resolve()
    atomic_json(output, payload)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
