#!/usr/bin/env python3
"""Independently verify the H2O+ v10 delayed evaluation file manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reproducibility/h2oplus_v10_delayed_results"
PARTITIONS = RESULTS / "evaluation_partitions_v1"
CANONICAL = RESULTS / "evaluation_canonical_v1"
DEFAULT_MANIFEST = RESULTS / "evaluation_file_manifest_v1.json"
DEFAULT_OUTPUT = RESULTS / "evaluation_file_manifest_verification_v1.json"

EXPECTED_CONTROLS = {
    "reproducibility/h2oplus_v10_delayed_registration_v4.json": "59fbc0c7f1148fe74c4c8a0c8af0879c7900753520b5b112e0526eb3c05acf56",
    "reproducibility/h2oplus_v10_delayed_authorization_v4.json": "7aa6e68db01503dc9e1c3bbfbce1639dd08890e5fb46fec2cda347ade6942741",
    "reproducibility/h2oplus_v10_delayed_correction_v4.json": "2bf2a6d707283ae694e18ce796a209810415a2e079ca6b023c88a7b2e6857163",
    "reproducibility/h2oplus_v10_delayed_environment_v4.json": "ee3de3b3990e35c579f5fcfdbafe16168a83f5011f9e5b5ce44b59cea5365bb4",
    "reproducibility/h2oplus_v10_delayed_training_plan_v4.json": "20d87242072b4aab8332d25e2d2d27af986818339b6c63672d781607e7a0b829",
    "reproducibility/h2oplus_v10_delayed_protocol_v4.json": "83a45e7d06ec30bdaba3631d7d8c6854e7609b8dcff9c46aa735d9a7944daaf3",
    "reproducibility/h2oplus_v10_delayed_freeze_v4.json": "3776bf088ba7c7cf3c2e65fb8407afb2adb9c8fbcc2266b7781bfd8d1b04cb64",
    "reproducibility/h2oplus_v10_delayed_evaluation_methods_v1.json": "dca00ae59b34e0810ec407c0df3a08583d35eac2b404a0b879866922b2d0d9f7",
    "reproducibility/h2oplus_v10_delayed_evaluation_source_freeze_v1.json": "449d5432720b1b59ff9865b4bb83bb15ea1a6f5c0096ea01cad33766a9c73d9f",
    "reproducibility/h2oplus_v10_delayed_evaluation_deployment_sync_v1.json": "ae879be0635b8b0816ef1c8d2c534237aba4d1d30454d0070d4a0018284e1ab5",
    "reproducibility/h2oplus_v10_delayed_evaluation_deployment_v1_jtl110gpu_verification.json": "9dbffa503a52d8cbe60918d2b9b35251b681e5499239d4137039ca9847ab778e",
    "reproducibility/h2oplus_v10_delayed_evaluation_submission_v1.json": "d15c770cb40edc0659d618d5eaecd561ae7e1ec350045c832c47101fc29158ee",
    "reproducibility/h2oplus_v10_delayed_results/prefix_closeout_v4.json": "6a05ddc9ef1a3a07605851360d3733d6442f6aa27e4fb2b65d69f4b2a25951ad",
    "reproducibility/h2oplus_v10_delayed_results/prefix_closeout_verification_v4.json": "1d1d8b592485d24038cf363b233a4228c96d4a156c00804c6eadfbe3b0906c65",
    "reproducibility/h2oplus_v10_delayed_branch_submission_v4.json": "9a55e9c7ad208cf1d4f69f569c61734ff355bd7b9cfadeb81d864d977e371648",
    "reproducibility/h2oplus_v10_delayed_results/branch_closeout_v4.json": "9204b4196f40a7fa4ce0bab1277d8c7bbbe930aafa644760b7251d463fbc6498",
    "reproducibility/h2oplus_v10_delayed_results/branch_audit_v4.json": "e851b3413d19062b2e981df89bfc92bdd9908f256bc80031cf3ae4c13328fabe",
    "reproducibility/h2oplus_v10_delayed_results/branch_file_manifest_v4.json": "d794161e0faaec6d5ebbcd2039aab25c0051166802d577743188c8e55e29c926",
    "reproducibility/h2oplus_v10_delayed_results/branch_closeout_verification_v4.json": "101cadbfc03bde2719f0a786a961006ed2d1d86a27080376afd59842ff4bcb17",
    "reproducibility/deployments/h2oplus_v10_delayed_evaluation_v1/workspace/deployment_manifest.json": "ebb36040134679458f7ee1e818e8a3631c6929916b95e4b627542defc9fa2d4c",
    "reproducibility/h2oplus_v8_stage_a_sumo_runtime_lock_py38.json": "ac60ae58e0e6db6bb1386ffdfe0d164b96f81e49e4983b8b062ceddb65da51a4",
}
EXPECTED_DATASET = "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899"
EXPECTED_RUNTIME = "264d3c412b8e6c42320995208d2e009eb809b85626047206c4de752c2dae06e7"
METHODS_PATH = REPO / "reproducibility/h2oplus_v10_delayed_evaluation_methods_v1.json"
SOURCE_FREEZE_PATH = REPO / "reproducibility/h2oplus_v10_delayed_evaluation_source_freeze_v1.json"
BLOCKS = (
    "v10-delayed-development-01",
    "v10-delayed-development-02",
    "v10-delayed-development-03",
)
SCALES = {0.85, 1.0, 1.15}
RESULT_SOURCE_SUFFIXES = {
    "SimpleSAC/evaluate_transit_v10_delayed.py",
    "reproducibility/run_evaluation_matrix_v10_delayed.py",
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
    training = payload.get("policy", {}).get("training_metadata", {})
    admission = training.get("training_admission", {})
    coverage = payload.get("passenger_welfare", {}).get("chronological_coverage_errors", {})
    if (
        payload.get("schema_version") != "transit-evaluation-v7"
        or payload.get("evaluation_job", {}).get("signature") != job.get("signature")
        or payload.get("method") != method.get("name")
        or payload.get("sumo_seed") != job.get("sumo_seed")
        or not finite(payload)
        or source != job.get("source_sha256")
        or len(source) != 41
        or any(_suffix_hash(source, suffix) != frozen_sources[suffix] for suffix in RESULT_SOURCE_SUFFIXES)
        or any(job_method.get(key) != method.get(key) for key in METHOD_FIELDS)
        or job_method.get("training_plan_sha256") != method.get("source_protocol_sha256")
        or job.get("dataset_sha256") != EXPECTED_DATASET
        or payload.get("dataset", {}).get("sha256") != EXPECTED_DATASET
        or job.get("sumo_runtime_signature") != EXPECTED_RUNTIME
        or payload.get("runtime", {}).get("sumo_runtime_signature") != EXPECTED_RUNTIME
        or payload.get("policy", {}).get("sha256") != method.get("checkpoint_sha256")
        or payload.get("policy", {}).get("actor_tensor_sha256") != method.get("actor_tensor_sha256")
        or training.get("study_stage") != "development"
        or training.get("formal_held_out_outcomes_read") is not False
        or training.get("training_seed") != method.get("training_seed")
        or training.get("run_signature") != method.get("training_run_signature")
        or training.get("training_checkpoint_sha256") != method.get("training_checkpoint_sha256")
        or admission.get("sha256") != method.get("training_admission_sha256")
        or not coverage
        or any(abs(float(value)) > 1e-9 for value in coverage.values())
        or payload.get("bus_service_semantics_audit", {}).get("passed") is not True
        or payload.get("ledger_audit", {}).get("legacy_sampled_trip_histories_used_as_ground_truth") is not False
        or payload.get("passengers", {}).get("legacy_sampled_trip_histories_used_for_welfare") is not False
        or payload.get("passengers", {}).get("active_state_count_balance_error") != 0
        or payload.get("passenger_welfare", {}).get("primary") is not True
    ):
        raise ValueError(f"independent v10 result provenance check failed: {path}")
    if method["artifact_lineage"] == "v8-fewshot":
        if (
            admission.get("schema_version") != "transit-training-admission-v8-fewshot-development-v1"
            or training.get("fewshot_protocol", {}).get("sha256") != method.get("source_protocol_sha256")
        ):
            raise ValueError(f"independent v8 actor lineage check failed: {path}")
    elif method["artifact_lineage"] == "v10-delayed":
        variant = method["algorithm"][len("h2oplus-v10-") :]
        intervention = training.get("v10_intervention", {})
        if (
            admission.get("schema_version") != "transit-training-admission-v10-delayed-development-v4"
            or training.get("v10_protocol", {}).get("sha256") != method.get("source_protocol_sha256")
            or training.get("variant") != variant
            or intervention.get("variant") != variant
            or intervention.get("unit_ratio_weights") is not True
        ):
            raise ValueError(f"independent v10 actor lineage check failed: {path}")
    else:
        raise ValueError(f"unknown v10 actor lineage: {path}")
    if source_reference is not None and source != source_reference:
        raise ValueError(f"source map differs across independently checked v10 results: {path}")
    return source


def verify(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "transit-h2oplus-v10-delayed-evaluation-file-manifest-v1":
        raise ValueError("unexpected v10 evaluation manifest schema")
    records = manifest.get("files", [])
    record_paths = {row["path"] for row in records}
    if len(records) != manifest.get("file_count") or len(record_paths) != len(records):
        raise ValueError("v10 file records are incomplete or duplicated")
    total = 0
    for record in records:
        path = (REPO / record["path"]).resolve()
        if not path.is_file() or REPO not in path.parents:
            raise ValueError(f"missing or out-of-root v10 file: {record['path']}")
        if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"v10 file identity mismatch: {record['path']}")
        total += path.stat().st_size
    if total != manifest.get("total_size_bytes") or aggregate(records) != manifest.get("file_records_aggregate_sha256"):
        raise ValueError("v10 file record aggregate mismatch")
    for relative, digest in EXPECTED_CONTROLS.items():
        if sha256(REPO / relative) != digest or relative not in record_paths:
            raise ValueError(f"sealed v10 control drifted or is absent: {relative}")

    artifact = manifest.get("artifact_integrity", {})
    if (
        artifact.get("passed") is not True
        or artifact.get("result_count") != 540
        or artifact.get("unique_job_signature_count") != 540
        or artifact.get("canonical_matrix_count") != 3
        or artifact.get("source_manifest_count") != 6
        or artifact.get("corrected_source_file_count") != 41
        or artifact.get("physical_method_count") != 60
        or artifact.get("logical_method_budget_alias_count") != 140
        or manifest.get("study_stage") != "development"
        or manifest.get("formal_held_out_outcomes_read") is not False
        or manifest.get("supports_current_formal_held_out_model_effect_claim") is not False
        or manifest.get("scientific_analysis", {}).get("completed") is not False
    ):
        raise ValueError("v10 artifact/scientific scope summary failed")

    audit = json.loads((RESULTS / "evaluation_audit_v1.json").read_text())
    if (
        audit.get("passed") is not True
        or audit.get("artifact_integrity", {}).get("passed") is not True
        or audit.get("scientific_results", {}).get("analysis_completed") is not False
        or audit.get("training_scientific_stability", {}).get("all_30_training_runs_stability_gate_passed") is not False
        or audit.get("training_scientific_stability", {}).get("failure_runs")
        != ["h2oplus_v10c4_delayed_order_seed6205"]
    ):
        raise ValueError("v10 evaluation audit did not preserve integrity/stability separation")

    scheduler = json.loads((RESULTS / "evidence/evaluation_v1/scheduler_evidence_v1.json").read_text())
    if (
        scheduler.get("passed") is not True
        or scheduler.get("token_bound_backend_exit0") is not True
        or set(scheduler.get("tasks", {})) != {"t82706", "t82707"}
    ):
        raise ValueError("v10 scheduler evidence failed")
    for task_id, item in scheduler["tasks"].items():
        status = RESULTS / f"evidence/evaluation_v1/backend_exit_{task_id}.status"
        fields = status.read_text().strip().split("\t")
        if (
            len(fields) != 3
            or fields[0] != "0"
            or fields[2] != item.get("exit_status_token")
            or item.get("queue_status") != "done"
            or item.get("exit_code") != 0
        ):
            raise ValueError(f"independent token-bound v10 exit check failed: {task_id}")

    methods_payload = json.loads(METHODS_PATH.read_text())
    methods = {item["name"]: item for item in methods_payload["methods"]}
    aliases = methods_payload.get("aliases", [])
    source_freeze = json.loads(SOURCE_FREEZE_PATH.read_text())
    frozen_sources = {item["path"]: item["sha256"] for item in source_freeze["sources"]}
    if len(methods) != 60 or len(aliases) != 140 or len({item["name"] for item in aliases}) != 140 or len(frozen_sources) != 4:
        raise ValueError("independent v10 methods/source-freeze check failed")

    raw_files = {
        str(path.resolve().relative_to(REPO))
        for path in PARTITIONS.rglob("*")
        if path.is_file()
    }
    if not raw_files.issubset(record_paths):
        raise ValueError("not every synchronized v10 partition file is sealed")

    partition_signatures = set()
    source_manifest_hashes = set()
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
            or len(results) != 270
            or len(matrices) != 3
        ):
            raise ValueError(f"independent v10 source partition check failed: {shard}")
        for matrix_path in matrices:
            matrix = json.loads(matrix_path.read_text())
            jobs = matrix.get("jobs", [])
            partition = matrix.get("matrix_partition", {})
            if (
                len(jobs) != 90
                or partition.get("shard_index") != shard_index
                or partition.get("shard_count") != 2
                or partition.get("full_job_count") != 180
                or len(matrix.get("source_sha256", {})) != 41
                or matrix.get("methods_manifest_sha256") != sha256(METHODS_PATH)
                or matrix.get("sumo_runtime_signature") != EXPECTED_RUNTIME
            ):
                raise ValueError(f"independent v10 partition matrix check failed: {matrix_path}")
            source_manifest_hashes.add(sha256(matrix_path))
            for job in jobs:
                name = str(job.get("method", {}).get("name"))
                method = methods.get(name)
                output = Path(job["output"]).expanduser().resolve()
                if method is None or not output.is_file() or PARTITIONS.resolve() not in output.parents:
                    raise ValueError(f"missing or out-of-root independent v10 output: {output}")
                source_reference = _validate_result(job, output, method, source_reference, frozen_sources)
                signature = str(job["signature"])
                if signature in partition_signatures:
                    raise ValueError("duplicate independent v10 result signature")
                partition_signatures.add(signature)
    if len(partition_signatures) != 540 or len(source_manifest_hashes) != 6 or source_reference is None:
        raise ValueError("independent v10 partition union check failed")

    canonical_signatures = set()
    expected_cells = {(name, scale) for name in methods for scale in SCALES}
    for block in BLOCKS:
        path = CANONICAL / block / "matrix_manifest.json"
        matrix = json.loads(path.read_text())
        jobs = matrix.get("jobs", [])
        merge = matrix.get("matrix_merge", {})
        cells = set()
        if (
            len(jobs) != 180
            or merge.get("all_outputs_verified") is not True
            or merge.get("method_count") != 60
            or merge.get("environment_cells_per_method") != 3
            or len(merge.get("source_manifests", [])) != 2
            or not {item.get("sha256") for item in merge["source_manifests"]}.issubset(source_manifest_hashes)
        ):
            raise ValueError(f"independent v10 canonical matrix check failed: {block}")
        for job in jobs:
            name = str(job.get("method", {}).get("name"))
            scale = float(job["scenario"]["demand_scale"])
            signature = str(job["signature"])
            cells.add((name, scale))
            if name not in methods or signature in canonical_signatures:
                raise ValueError("invalid or duplicate canonical v10 signature")
            canonical_signatures.add(signature)
        if cells != expected_cells:
            raise ValueError(f"canonical v10 Cartesian coverage failed: {block}")
    if canonical_signatures != partition_signatures:
        raise ValueError("canonical and synchronized v10 source unions differ")

    return {
        "schema_version": "transit-h2oplus-v10-delayed-evaluation-file-manifest-verification-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": {
            "file_records_aggregate_sha256": manifest["file_records_aggregate_sha256"],
            "path": str(manifest_path.relative_to(REPO)),
            "sha256": sha256(manifest_path),
        },
        "independent_artifact_audit": {
            "all_numeric_values_finite": True,
            "canonical_matrix_count": 3,
            "corrected_source_file_count": 41,
            "file_count": len(records),
            "logical_method_budget_alias_count": 140,
            "passed": True,
            "physical_method_count": 60,
            "result_count": 540,
            "source_manifest_count": 6,
            "source_partition_count": 2,
            "total_size_bytes": total,
            "unique_job_signature_count": 540,
        },
        "training_scientific_stability": {
            "all_30_training_runs_stability_gate_passed": False,
            "failure_runs": ["h2oplus_v10c4_delayed_order_seed6205"],
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
