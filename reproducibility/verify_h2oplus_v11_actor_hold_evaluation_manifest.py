#!/usr/bin/env python3
"""Independently verify the H2O+ v11 actor-hold evaluation file manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reproducibility/h2oplus_v11_actor_hold_results"
PARTITIONS = RESULTS / "evaluation_partitions_v1"
CANONICAL = RESULTS / "evaluation_canonical_v1"
DEFAULT_MANIFEST = RESULTS / "evaluation_file_manifest_v1.json"
DEFAULT_OUTPUT = RESULTS / "evaluation_file_manifest_verification_v1.json"

EXPECTED_CONTROLS = {
    "reproducibility/h2oplus_v11_actor_hold_registration_v2.json": "caab9c2c3aa7ee2ce977c13403b415be9b0debc2d7c88fc6b7a12c51b3e9efb5",
    "reproducibility/h2oplus_v11_actor_hold_authorization_v2.json": "8dc0ca71f8c45abc3e5e86af789281e4e43a882533ba4202ca567f00d87281aa",
    "reproducibility/h2oplus_v11_actor_hold_correction_v2.json": "8f1defa7d756f6336d26af0b1fe8ebf87db39f81554c89891fc2fbaaf17d1258",
    "reproducibility/h2oplus_v11_actor_hold_environment_v2.json": "0f005b94d5ecd361d7e10bac363be8dac5e6c9cdad7cbaf3f063744d280871de",
    "reproducibility/h2oplus_v11_actor_hold_training_plan_v2.json": "61aa351f7e36c9abd3f4822c20be17c7e5b5343cd822e031d0ede001889e7234",
    "reproducibility/h2oplus_v11_actor_hold_protocol_v2.json": "c5b162541c6ce2476bb0f57f6dc936de13b85b5e1e6ede77b533a5cf9a4dfced",
    "reproducibility/h2oplus_v11_actor_hold_freeze_v2.json": "095754d08f14b6e2eb84baadfa6ce1a8d4100ebac5b92f34891b2d1e01afed04",
    "reproducibility/h2oplus_v11_actor_hold_evaluation_methods_v1.json": "6716c56c73e9d491431b1ab019727f9bfd18a2242cb7500826cf7ccc0628b260",
    "reproducibility/h2oplus_v11_actor_hold_evaluation_source_freeze_v1.json": "fe61024af63450c6502767c2fcee7915c2b2e8216cf23812f63523e241b237c7",
    "reproducibility/h2oplus_v11_actor_hold_evaluation_deployment_local_verification_v1.json": "70b8b4473b7c4c24a146a1c25a276f7bb0b7bde642d774461bbca62dadede40a",
    "reproducibility/h2oplus_v11_actor_hold_evaluation_deployment_sync_v1.json": "75dcacf761206bcd64f7b74fc2b7668c843f3b12f3dfa5302599a5db68d885a4",
    "reproducibility/h2oplus_v11_actor_hold_evaluation_deployment_v1_jtl110gpu_verification.json": "e393ebcb2f025cd485e5dd455baa83467c0bfed0c86b2e0ef66d84501afd1446",
    "reproducibility/h2oplus_v11_actor_hold_evaluation_submission_v1.json": "f510ea8766d9d808447c04da252a2511b442b2b82fbc0989d8ea79fabbbaed9a",
    "reproducibility/h2oplus_v11_actor_hold_results/training_closeout_v2.json": "489c7efe5cf91f73bb31a9b6df03441c2fb3bcd4d280c65493c5ae8ec3af01a6",
    "reproducibility/h2oplus_v11_actor_hold_results/training_closeout_verification_v2.json": "ff0ef7c69bac850bf9f743275ffd9bbfc87c0b1c9713a83769a052bd936d256e",
    "reproducibility/h2oplus_v11_actor_hold_results/training_audit_v2.json": "623ae58932c5ec3ffb80fbe82b2d159c6f0fe96ee0a1f12eeb4ca1c49385d516",
    "reproducibility/h2oplus_v11_actor_hold_results/training_file_manifest_v2.json": "0a70f10026e1a80d14176648eea866e3e52c00a56ba30acfc48b414e323025fa",
    "reproducibility/deployments/h2oplus_v11_actor_hold_evaluation_v1/workspace/deployment_manifest.json": "b579fade4a0c4ea6dac164620dd50818818fc49666d365ddff63a26ab989246d",
    "reproducibility/h2oplus_v8_stage_a_sumo_runtime_lock_py38.json": "ac60ae58e0e6db6bb1386ffdfe0d164b96f81e49e4983b8b062ceddb65da51a4",
    "reproducibility/h2oplus_v11_actor_hold_results/evidence/evaluation_v1/scheduler_evidence_v1.json": "315c54b46d32dfb01d073682ebc06ea5526ded30fa0dd794bde933f359309647",
    "reproducibility/h2oplus_v11_actor_hold_results/evidence/evaluation_v1/source_sync_v1.json": "d3a07694b945c7b8a4a4f9ed8787a9d3f9217c218b8ad83c157977d803ecccef",
}
EXPECTED_DATASET = "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899"
EXPECTED_RUNTIME = "264d3c412b8e6c42320995208d2e009eb809b85626047206c4de752c2dae06e7"
METHODS_PATH = REPO / "reproducibility/h2oplus_v11_actor_hold_evaluation_methods_v1.json"
SOURCE_FREEZE_PATH = REPO / "reproducibility/h2oplus_v11_actor_hold_evaluation_source_freeze_v1.json"
BLOCKS = (
    "v11-actor-hold-development-01",
    "v11-actor-hold-development-02",
    "v11-actor-hold-development-03",
)
SCALES = {0.85, 1.0, 1.15}
RESULT_SOURCE_SUFFIXES = {
    "SimpleSAC/evaluate_transit_v11_actor_hold.py",
    "reproducibility/run_evaluation_matrix_v11_actor_hold.py",
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
        raise ValueError(f"independent v11 result provenance check failed: {path}")
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
    elif method["artifact_lineage"] == "v11-actor-hold":
        intervention = training.get("v11_intervention", {})
        if (
            admission.get("schema_version")
            != "transit-training-admission-v11-actor-hold-development-v2"
            or training.get("v11_protocol", {}).get("sha256")
            != method.get("source_protocol_sha256")
            or training.get("variant") != "actor-hold-online-only"
            or training.get("stage") != "candidate"
            or training.get("v11_actor_tensor_sha256") != method.get("actor_tensor_sha256")
            or training.get("v11_temperature_tensor_sha256")
            != method.get("temperature_tensor_sha256")
            or intervention.get("variant") != "actor-hold-online-only"
            or intervention.get("unit_ratio_weights") is not True
            or intervention.get("checkpoint_selection_by_behavior") is not False
        ):
            raise ValueError(f"independent v11 actor lineage check failed: {path}")
    else:
        raise ValueError(f"unknown v11 actor lineage: {path}")
    if source_reference is not None and source != source_reference:
        raise ValueError(f"source map differs across independently checked v11 results: {path}")
    return source


def verify(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "transit-h2oplus-v11-actor-hold-evaluation-file-manifest-v1":
        raise ValueError("unexpected v11 evaluation manifest schema")
    records = manifest.get("files", [])
    record_paths = {row["path"] for row in records}
    if len(records) != manifest.get("file_count") or len(record_paths) != len(records):
        raise ValueError("v11 file records are incomplete or duplicated")
    total = 0
    for record in records:
        path = (REPO / record["path"]).resolve()
        if not path.is_file() or REPO not in path.parents:
            raise ValueError(f"missing or out-of-root v11 file: {record['path']}")
        if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"v11 file identity mismatch: {record['path']}")
        total += path.stat().st_size
    if total != manifest.get("total_size_bytes") or aggregate(records) != manifest.get("file_records_aggregate_sha256"):
        raise ValueError("v11 file record aggregate mismatch")
    for relative, digest in EXPECTED_CONTROLS.items():
        if sha256(REPO / relative) != digest or relative not in record_paths:
            raise ValueError(f"sealed v11 control drifted or is absent: {relative}")

    artifact = manifest.get("artifact_integrity", {})
    if (
        artifact.get("passed") is not True
        or artifact.get("result_count") != 405
        or artifact.get("unique_job_signature_count") != 405
        or artifact.get("canonical_matrix_count") != 3
        or artifact.get("source_manifest_count") != 6
        or artifact.get("corrected_source_file_count") != 41
        or artifact.get("physical_method_count") != 45
        or artifact.get("logical_method_budget_alias_count") != 80
        or manifest.get("study_stage") != "development"
        or manifest.get("formal_held_out_outcomes_read") is not False
        or manifest.get("supports_current_formal_held_out_model_effect_claim") is not False
        or manifest.get("scientific_analysis", {}).get("completed") is not False
    ):
        raise ValueError("v11 artifact/scientific scope summary failed")

    audit = json.loads((RESULTS / "evaluation_audit_v1.json").read_text())
    if (
        audit.get("passed") is not True
        or audit.get("artifact_integrity", {}).get("passed") is not True
        or audit.get("scientific_results", {}).get("analysis_completed") is not False
        or audit.get("training_scientific_stability", {}).get("all_5_training_runs_stability_gate_passed") is not True
        or audit.get("training_scientific_stability", {}).get("failure_runs") != []
    ):
        raise ValueError("v11 evaluation audit did not preserve integrity/stability separation")

    scheduler = json.loads((RESULTS / "evidence/evaluation_v1/scheduler_evidence_v1.json").read_text())
    if (
        scheduler.get("passed") is not True
        or scheduler.get("token_bound_backend_exit0") is not True
        or set(scheduler.get("tasks", {})) != {"t82929", "t82930"}
    ):
        raise ValueError("v11 scheduler evidence failed")
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
            raise ValueError(f"independent token-bound v11 exit check failed: {task_id}")

    methods_payload = json.loads(METHODS_PATH.read_text())
    methods = {item["name"]: item for item in methods_payload["methods"]}
    aliases = methods_payload.get("aliases", [])
    source_freeze = json.loads(SOURCE_FREEZE_PATH.read_text())
    frozen_sources = {item["path"]: item["sha256"] for item in source_freeze["sources"]}
    identities = methods_payload.get("identity_evidence", [])
    if (
        len(methods) != 45
        or len(aliases) != 80
        or len({item["name"] for item in aliases}) != 80
        or len(frozen_sources) != 4
        or len(identities) != 5
    ):
        raise ValueError("independent v11 methods/source-freeze check failed")
    for alias in aliases:
        physical = methods.get(alias.get("physical_method"))
        if (
            physical is None
            or alias.get("training_seed") != physical.get("training_seed")
            or alias.get("actor_tensor_sha256") != physical.get("actor_tensor_sha256")
        ):
            raise ValueError("independent v11 logical alias identity check failed")
    for identity in identities:
        if (
            identity.get("v11_zero_5k_10k_equal_v8_h2oplus_zero") is not True
            or identity.get("v11_zero_5k_10k_temperature_equal") is not True
            or len(identity.get("v11_source_policies", [])) != 4
        ):
            raise ValueError("independent v11 hold identity evidence check failed")

    raw_files = {
        str(path.resolve().relative_to(REPO))
        for path in PARTITIONS.rglob("*")
        if path.is_file()
    }
    if not raw_files.issubset(record_paths):
        raise ValueError("not every synchronized v11 partition file is sealed")

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
            or len(results) != (204 if shard_index == 0 else 201)
            or len(matrices) != 3
        ):
            raise ValueError(f"independent v11 source partition check failed: {shard}")
        for matrix_path in matrices:
            matrix = json.loads(matrix_path.read_text())
            jobs = matrix.get("jobs", [])
            partition = matrix.get("matrix_partition", {})
            if (
                len(jobs) != (68 if shard_index == 0 else 67)
                or partition.get("shard_index") != shard_index
                or partition.get("shard_count") != 2
                or partition.get("selected_job_count") != (68 if shard_index == 0 else 67)
                or partition.get("full_job_count") != 135
                or len(matrix.get("source_sha256", {})) != 41
                or matrix.get("methods_manifest_sha256") != sha256(METHODS_PATH)
                or matrix.get("sumo_runtime_signature") != EXPECTED_RUNTIME
            ):
                raise ValueError(f"independent v11 partition matrix check failed: {matrix_path}")
            source_manifest_hashes.add(sha256(matrix_path))
            for job in jobs:
                name = str(job.get("method", {}).get("name"))
                method = methods.get(name)
                output = Path(job["output"]).expanduser().resolve()
                if method is None or not output.is_file() or PARTITIONS.resolve() not in output.parents:
                    raise ValueError(f"missing or out-of-root independent v11 output: {output}")
                source_reference = _validate_result(job, output, method, source_reference, frozen_sources)
                signature = str(job["signature"])
                if signature in partition_signatures:
                    raise ValueError("duplicate independent v11 result signature")
                partition_signatures.add(signature)
    if len(partition_signatures) != 405 or len(source_manifest_hashes) != 6 or source_reference is None:
        raise ValueError("independent v11 partition union check failed")

    canonical_signatures = set()
    expected_cells = {(name, scale) for name in methods for scale in SCALES}
    for block in BLOCKS:
        path = CANONICAL / block / "matrix_manifest.json"
        matrix = json.loads(path.read_text())
        jobs = matrix.get("jobs", [])
        merge = matrix.get("matrix_merge", {})
        cells = set()
        if (
            len(jobs) != 135
            or merge.get("all_outputs_verified") is not True
            or merge.get("method_count") != 45
            or merge.get("environment_cells_per_method") != 3
            or len(merge.get("source_manifests", [])) != 2
            or not {item.get("sha256") for item in merge["source_manifests"]}.issubset(source_manifest_hashes)
        ):
            raise ValueError(f"independent v11 canonical matrix check failed: {block}")
        for job in jobs:
            name = str(job.get("method", {}).get("name"))
            scale = float(job["scenario"]["demand_scale"])
            signature = str(job["signature"])
            cells.add((name, scale))
            if name not in methods or signature in canonical_signatures:
                raise ValueError("invalid or duplicate canonical v11 signature")
            canonical_signatures.add(signature)
        if cells != expected_cells:
            raise ValueError(f"canonical v11 Cartesian coverage failed: {block}")
    if canonical_signatures != partition_signatures:
        raise ValueError("canonical and synchronized v11 source unions differ")

    return {
        "schema_version": "transit-h2oplus-v11-actor-hold-evaluation-file-manifest-verification-v1",
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
            "logical_method_budget_alias_count": 80,
            "passed": True,
            "physical_method_count": 45,
            "result_count": 405,
            "source_manifest_count": 6,
            "source_partition_count": 2,
            "total_size_bytes": total,
            "unique_job_signature_count": 405,
        },
        "training_scientific_stability": {
            "all_5_training_runs_stability_gate_passed": True,
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
