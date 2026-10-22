#!/usr/bin/env python3
"""Audit and seal the complete H2O+ v8 Stage A development artifact set."""

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
RESULTS = REPO / "reproducibility/deployments/h2oplus_v8_development_stage_a_v2/results"
CANONICAL = REPO / "reproducibility/h2oplus_v8_stage_a_results/canonical"
EVIDENCE = REPO / "reproducibility/h2oplus_v8_stage_a_results/evidence"
ANALYSIS = REPO / "reproducibility/h2oplus_v8_stage_a_results/stage_a_gate_analysis.json"
POLICIES = REPO / "reproducibility/h2oplus_v8_stage_a_policy_export_v1"

EXPECTED_IDENTITIES = {
    "deployment_manifest": "60f9e4212673eb6cf3491e1a1d21321a7f68d9bf4071cffbb2702e2ad0288826",
    "registration": "d76dbde2d54b53c18b12770523f02a85221d0622839b3619c882362a2cc6a820",
    "authorization": "cebc1083ad6d94b6643f6343e862e022794e69c272d42d1e8be2cf0f911c5cf5",
    "design": "d4a40a900cf9f72043c1d7c2067d02d6c79f2081f856b38cb257ec2ad0d18f3c",
    "freeze": "0291c8f78ecb76a4ea36f25e1a70a1ea3b8237d02811adc28181c1fbd12615c0",
    "methods": "7fba8cc26f6854ac07963df69cf39451d92a6eed6cb0dfa9c52e13708f913a86",
    "policy_export": "c6e5eb408629bfdc0c046bd025f64947ceaf66b337f47cc2abb4e07d749e7183",
}

CONTROL_PATHS = {
    "deployment_manifest": REPO
    / "reproducibility/deployments/h2oplus_v8_development_stage_a_v2/workspace/deployment_manifest.json",
    "registration": REPO / "reproducibility/h2oplus_v8_development_pilot_registration_v1.json",
    "authorization": REPO
    / "reproducibility/h2oplus_v8_development_stage_a_execution_authorization_v1.json",
    "design": REPO / "reproducibility/h2oplus_v8_development_environment_stage_a_v1.json",
    "freeze": REPO / "reproducibility/h2oplus_v8_stage_a_freeze_v1.json",
    "methods": POLICIES / "evaluation_methods_v8_stage_a.json",
    "policy_export": POLICIES / "policy_export_manifest_v8_stage_a.json",
}


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


def all_numbers_finite(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(all_numbers_finite(item) for item in value)
    if isinstance(value, dict):
        return all(all_numbers_finite(item) for item in value.values())
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


def validate_control_identities() -> dict:
    descriptors = {}
    for name, path in CONTROL_PATHS.items():
        actual = sha256(path)
        if actual != EXPECTED_IDENTITIES[name]:
            raise ValueError(f"{name} identity drifted: {actual}")
        descriptors[name] = file_record(path)
    deployment = load_json(CONTROL_PATHS["deployment_manifest"])
    if deployment.get("aggregate_sha256") != (
        "ab56483dbe329043a3fd659f83f2d6b32562e49a262c8db7af63d0d8fb89fe8a"
    ):
        raise ValueError("deployment aggregate identity drifted")
    return descriptors


def validate_partitions() -> tuple[dict, set[Path], dict[str, dict[str, str]]]:
    stats = {}
    files = set()
    source_manifests: dict[str, dict[str, str]] = {}
    for shard_index, shard_name in enumerate(("shard0_jtl110gpu", "shard1_jtl110gpu")):
        root = RESULTS / shard_name
        shard_files = {path for path in root.rglob("*") if path.is_file()}
        result_paths = sorted(root.rglob("sumo_seed*.json"))
        manifest_paths = sorted(root.rglob("matrix_manifest*.json"))
        if len(result_paths) != 75 or len(manifest_paths) != 5:
            raise ValueError(f"{shard_name} coverage is not 75 results plus five manifests")
        host_path = root / "host_execution.json"
        host = load_json(host_path)
        if (
            host.get("status") != "completed"
            or host.get("returncode") != 0
            or host.get("shard_index") != shard_index
            or host.get("shard_count") != 2
            or host.get("training") is not False
            or host.get("formal_held_out_outcomes_read") is not False
            or len(host.get("completed_blocks", [])) != 5
            or sum(item.get("selected_job_count", 0) for item in host["completed_blocks"]) != 75
        ):
            raise ValueError(f"{shard_name} host execution contract failed")
        for item in host["completed_blocks"]:
            path = Path(item["path"])
            if not path.is_file() or sha256(path) != item["sha256"]:
                raise ValueError(f"source manifest binding failed: {path}")
            source_manifests[relative(path)] = {"sha256": item["sha256"], "shard": shard_name}
        for path in result_paths:
            payload = load_json(path)
            if payload.get("schema_version") != "transit-evaluation-v7" or not all_numbers_finite(payload):
                raise ValueError(f"invalid or nonfinite evaluation output: {path}")
        files.update(shard_files)
        stats[shard_name] = {
            "result_count": len(result_paths),
            "block_manifest_count": len(manifest_paths),
            "file_count": len(shard_files),
            "size_bytes": sum(path.stat().st_size for path in shard_files),
            "host_execution_sha256": sha256(host_path),
        }
    if len(source_manifests) != 10:
        raise ValueError("expected ten distinct source manifests")
    return stats, files, source_manifests


def validate_canonical(source_manifests: dict[str, dict[str, str]]) -> tuple[dict, set[Path]]:
    design = load_json(CONTROL_PATHS["design"])
    blocks = {
        block["name"]: {
            "sumo_seed": int(block["sumo_seed"]),
            "realization_seed": int(block["demand_realization_seed"]),
        }
        for block in design["blocks"]
    }
    expected_names = {f"v8-development-{index:02d}" for index in range(1, 6)}
    if set(blocks) != expected_names or design.get("demand_scales") != [0.85, 1.0, 1.15]:
        raise ValueError("registered development-block design drifted")

    canonical_files = set()
    output_paths: set[Path] = set()
    signatures: set[str] = set()
    block_stats = {}
    for block_name in sorted(blocks):
        manifest_path = CANONICAL / block_name / "matrix_manifest.json"
        manifest = load_json(manifest_path)
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
            raise ValueError(f"merged matrix contract failed: {block_name}")
        for descriptor in merge["source_manifests"]:
            path = Path(descriptor["path"])
            key = relative(path)
            if key not in source_manifests or sha256(path) != descriptor["sha256"]:
                raise ValueError(f"merged source binding failed: {path}")

        observed_cells = set()
        for job in jobs:
            method = job["method"]
            algorithm = str(method["algorithm"])
            training_seed = int(method["training_seed"])
            sumo_seed = int(job["sumo_seed"])
            realization_seed = int(job["scenario"]["selection_seed"])
            demand_scale = float(job["scenario"]["demand_scale"])
            if (
                algorithm not in {"h2oplus", "h2oplus-lambda0p5"}
                or training_seed not in range(5101, 5106)
                or sumo_seed != blocks[block_name]["sumo_seed"]
                or realization_seed != blocks[block_name]["realization_seed"]
                or demand_scale not in {0.85, 1.0, 1.15}
                or job.get("environment_block") not in {block_name, f"sumo-seed-{sumo_seed}"}
            ):
                raise ValueError(f"job coverage drifted in {block_name}")
            observed_cells.add((algorithm, training_seed, demand_scale))
            signature = str(job["signature"])
            output = Path(job["output"])
            if signature in signatures or output in output_paths or not output.is_file():
                raise ValueError("duplicate signature/output or missing output")
            result = load_json(output)
            if (
                result.get("schema_version") != "transit-evaluation-v7"
                or result.get("evaluation_job", {}).get("signature") != signature
                or int(result.get("sumo_seed")) != sumo_seed
                or not all_numbers_finite(result)
            ):
                raise ValueError(f"result binding failed: {output}")
            signatures.add(signature)
            output_paths.add(output)
        expected_cells = {
            (algorithm, seed, scale)
            for algorithm in ("h2oplus", "h2oplus-lambda0p5")
            for seed in range(5101, 5106)
            for scale in (0.85, 1.0, 1.15)
        }
        if observed_cells != expected_cells:
            raise ValueError(f"incomplete Cartesian matrix: {block_name}")
        canonical_files.add(manifest_path)
        block_stats[block_name] = {
            "job_count": 30,
            "matrix_manifest_sha256": sha256(manifest_path),
            "source_manifest_count": 2,
        }
    if len(output_paths) != 150 or len(signatures) != 150:
        raise ValueError("Stage A does not contain exactly 150 unique bound results")
    return {
        "blocks": block_stats,
        "result_count": len(output_paths),
        "unique_job_signature_count": len(signatures),
        "algorithms": ["h2oplus", "h2oplus-lambda0p5"],
        "training_seeds": list(range(5101, 5106)),
        "sumo_seeds": list(range(4101, 4106)),
        "demand_realization_seeds": list(range(4201, 4206)),
        "demand_scales": [0.85, 1.0, 1.15],
    }, canonical_files


def validate_evidence_and_analysis() -> tuple[dict, set[Path]]:
    evidence_paths = {
        EVIDENCE / "backend_exit_t82278.status",
        EVIDENCE / "backend_exit_t82281.status",
        EVIDENCE / "stage_a_scheduler_evidence.json",
        EVIDENCE / "stage_a_partition_sync_evidence.json",
        EVIDENCE / "stage_a_analysis_correction_v1.json",
    }
    for path in evidence_paths:
        if not path.is_file():
            raise ValueError(f"missing closeout evidence: {path}")
    for name in (
        "stage_a_scheduler_evidence.json",
        "stage_a_partition_sync_evidence.json",
        "stage_a_analysis_correction_v1.json",
    ):
        if load_json(EVIDENCE / name).get("passed") is not True:
            raise ValueError(f"closeout evidence failed: {name}")

    report = load_json(ANALYSIS)
    if (
        report.get("schema_version") != "transit-h2oplus-v8-stage-a-development-gate-v1"
        or report.get("study_stage") != "development"
        or report.get("training") is not False
        or report.get("formal_held_out_outcomes_read") is not False
        or report.get("supports_current_held_out_model_effect_claim") is not False
        or report.get("stage_a_gate_passed") is not False
        or report.get("stage_b_gate_eligible") is not False
        or report.get("stage_b_authorized_or_submitted") is not False
        or len(report.get("paired_units", [])) != 25
    ):
        raise ValueError("Stage A development analysis scope or outcome drifted")
    checks = report.get("checks", {})
    if set(checks) != {
        "mean_relative_improvement",
        "training_seeds_with_lower_paired_mean",
        "one_sided_paired_permutation",
        "completion_rate_noninferiority",
        "no_demand_scale_relative_degradation",
    } or all(item.get("passed") for item in checks.values()):
        raise ValueError("preregistered Stage A checks are incomplete or misreported")
    evidence_paths.add(ANALYSIS)
    return {
        "analysis_sha256": sha256(ANALYSIS),
        "artifact_integrity_outcome": "pass",
        "development_scientific_gate_outcome": "fail",
        "stage_a_gate_passed": False,
        "stage_b_gate_eligible": False,
        "stage_b_authorized_or_submitted": False,
        "formal_held_out_model_effect_claim_supported": False,
    }, evidence_paths


def build_manifest() -> dict:
    controls = validate_control_identities()
    partition_stats, partition_files, source_manifests = validate_partitions()
    matrix_audit, canonical_files = validate_canonical(source_manifests)
    outcome, evidence_files = validate_evidence_and_analysis()

    policy_files = {path for path in POLICIES.rglob("*") if path.is_file()}
    implementation_files = {
        Path(__file__).resolve(),
        REPO / "reproducibility/verify_h2oplus_v8_stage_a_manifest.py",
        REPO / "reproducibility/analyze_h2oplus_v8_stage_a.py",
        REPO / "tests/test_analyze_h2oplus_v8_stage_a.py",
        REPO / "tests/test_h2oplus_v8_stage_a_dispatch.py",
        REPO / "tests/test_h2oplus_v8_stage_a_manifest.py",
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
        raise ValueError(f"manifest inputs missing: {missing}")
    records = [file_record(path) for path in sorted(all_files)]
    return {
        "schema_version": "transit-h2oplus-v8-stage-a-file-manifest-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "A-existing-policy-development-screen",
        "study_stage": "development",
        "training": False,
        "formal_held_out_outcomes_read": False,
        "supports_current_held_out_model_effect_claim": False,
        "artifact_integrity": {
            "passed": True,
            "independent_verification_required": True,
            "partitions": partition_stats,
            "source_manifest_count": len(source_manifests),
            "canonical_matrix_count": 5,
            **matrix_audit,
        },
        "scientific_outcome": outcome,
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
        default=REPO / "reproducibility/h2oplus_v8_stage_a_results/stage_a_file_manifest.json",
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
