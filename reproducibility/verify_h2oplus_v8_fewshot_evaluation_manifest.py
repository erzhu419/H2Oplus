#!/usr/bin/env python3
"""Independently verify the H2O+ v8 few-shot evaluation file manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reproducibility/h2oplus_v8_fewshot_results"
PARTITIONS = RESULTS / "evaluation_partitions_source_bound_v3"
CANONICAL = RESULTS / "evaluation_canonical_source_bound_v3"
DEFAULT_MANIFEST = RESULTS / "evaluation_file_manifest_v3.json"
DEFAULT_OUTPUT = RESULTS / "evaluation_file_manifest_verification_v3.json"

EXPECTED_CONTROLS = {
    "reproducibility/h2oplus_v8_fewshot_registration_v1.json": (
        "fb632f92998d82cfb87b2cd1a534058925d4e410be17ecea9070b7a50db7c115"
    ),
    "reproducibility/h2oplus_v8_fewshot_protocol_v1.json": (
        "26d4f5c4ca9f337c11886855658127ef93a35954ca58eb43710aa70350610294"
    ),
    "reproducibility/h2oplus_v8_fewshot_evaluation_correction_v2.json": (
        "16e55e7ec22105cb8065db47a4047fabfb16ebd302be6e7f74e8fbb2f5ec0398"
    ),
    "reproducibility/h2oplus_v8_fewshot_evaluation_submission_v3.json": (
        "5080462cf0710e16c6f707259fdc11c5d132449af88b3c621ca1580751d5afe9"
    ),
    (
        "reproducibility/deployments/h2oplus_v8_fewshot_evaluation_v3/"
        "workspace/deployment_manifest.json"
    ): "6f668ffb4cfb83643bc1f840bdb0be4692f716ca23a3b4de0f11ceae861a9e2f",
    "reproducibility/h2oplus_v8_fewshot_results/evaluation_methods_v8_fewshot.json": (
        "8dbca88cd754d3fe711f2408b3328cf3947efcad478c221f5d8be8bba017e7f7"
    ),
    "reproducibility/h2oplus_v8_fewshot_results/training_closeout.json": (
        "8a12385844f7949d3f7c2cbc16d8e021787c9de04fb14b7e64b46779d2373822"
    ),
    "reproducibility/h2oplus_v8_fewshot_results/training_audit.json": (
        "58d5f871a2563469a434918aa9d4f9433956b10b6b36a61ddea52ca5cfe4457e"
    ),
    "reproducibility/h2oplus_v8_fewshot_results/training_file_manifest.json": (
        "6742d8c0d9c965209f0c2ac56a2ac3351290f18564e0573f92b9fff644acf239"
    ),
    (
        "reproducibility/h2oplus_v8_fewshot_results/"
        "training_file_manifest_verification.json"
    ): "c6f124d80952577ddb6d6c6597c0fbf67704c4af92deb35fc0b5f5a0e99809aa",
}
EXPECTED_PROTOCOL = "26d4f5c4ca9f337c11886855658127ef93a35954ca58eb43710aa70350610294"
EXPECTED_DATASET = "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899"
EXPECTED_RUNTIME = "264d3c412b8e6c42320995208d2e009eb809b85626047206c4de752c2dae06e7"
EXPECTED_WRAPPERS = {
    "SimpleSAC/evaluate_transit_v8_fewshot.py": (
        "e78d5bf337baf25c7d9d2bd366dd2305dbca8b540caff1638ee3463c8d9e73ea"
    ),
    "reproducibility/run_evaluation_matrix_v8_fewshot.py": (
        "ac895cfec83cb1812d11e62e4348e1c6b9487d3eca85b3a512baff0a1aa5500c"
    ),
}
METHOD = re.compile(
    r"^(?P<algorithm>h2oplus|sac-online)-seed(?P<seed>62\d{2})-events(?P<events>\d{9})$"
)
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


def _wrapper_hash(source: dict, suffix: str) -> str | None:
    values = [value for path, value in source.items() if path.endswith(suffix)]
    return values[0] if len(values) == 1 else None


def _validate_method(method: dict) -> tuple[str, int, int]:
    match = METHOD.fullmatch(str(method.get("name")))
    if match is None:
        raise ValueError("unregistered method in sealed matrix")
    algorithm = match.group("algorithm")
    seed = int(match.group("seed"))
    events = int(match.group("events"))
    if (
        seed not in range(6201, 6206)
        or events not in BUDGETS[algorithm]
        or method.get("algorithm") != algorithm
        or int(method.get("training_seed")) != seed
        or int(method.get("online_events")) != events
    ):
        raise ValueError("method identity does not match the frozen grid")
    return algorithm, seed, events


def _validate_result(job: dict, path: Path, source_reference: dict | None) -> dict:
    payload = json.loads(path.read_text())
    method = job["method"]
    _, _, events = _validate_method(method)
    source = payload.get("source_sha256", {})
    training = payload.get("policy", {}).get("training_metadata", {})
    if (
        payload.get("schema_version") != "transit-evaluation-v7"
        or payload.get("evaluation_job", {}).get("signature") != job.get("signature")
        or payload.get("method") != method.get("name")
        or payload.get("sumo_seed") != job.get("sumo_seed")
        or not finite(payload)
        or source != job.get("source_sha256")
        or len(source) != 41
        or any(
            _wrapper_hash(source, suffix) != digest
            for suffix, digest in EXPECTED_WRAPPERS.items()
        )
        or job.get("dataset_sha256") != EXPECTED_DATASET
        or payload.get("dataset", {}).get("sha256") != EXPECTED_DATASET
        or job.get("sumo_runtime_signature") != EXPECTED_RUNTIME
        or payload.get("runtime", {}).get("sumo_runtime_signature") != EXPECTED_RUNTIME
        or training.get("study_stage") != "development"
        or training.get("formal_held_out_outcomes_read") is not False
        or training.get("fewshot_online_events_nominal") != events
        or training.get("fewshot_protocol", {}).get("sha256") != EXPECTED_PROTOCOL
        or training.get("run_signature") != method.get("training_run_signature")
        or training.get("fewshot_model_checkpoint", {}).get("sha256")
        != method.get("training_checkpoint_sha256")
        or training.get("training_admission", {}).get("sha256")
        != method.get("training_admission_sha256")
        or any(
            abs(float(value)) > 1e-9
            for value in payload.get("passenger_welfare", {})
            .get("chronological_coverage_errors", {})
            .values()
        )
    ):
        raise ValueError(f"independent result provenance check failed: {path}")
    if source_reference is not None and source != source_reference:
        raise ValueError(f"source map differs across sealed v3 results: {path}")
    return source


def verify(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version")
        != "transit-h2oplus-v8-fewshot-evaluation-file-manifest-v3"
    ):
        raise ValueError("unexpected few-shot evaluation manifest schema")
    records = manifest.get("files", [])
    record_paths = {row["path"] for row in records}
    if len(records) != manifest.get("file_count") or len(record_paths) != len(records):
        raise ValueError("file records are incomplete or duplicated")
    total = 0
    for record in records:
        path = (REPO / record["path"]).resolve()
        if not path.is_file() or REPO not in path.parents:
            raise ValueError(f"missing or out-of-root file: {record['path']}")
        if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"file identity mismatch: {record['path']}")
        total += path.stat().st_size
    if (
        total != manifest.get("total_size_bytes")
        or aggregate(records) != manifest.get("file_records_aggregate_sha256")
    ):
        raise ValueError("file record aggregate mismatch")
    for relative, digest in EXPECTED_CONTROLS.items():
        if sha256(REPO / relative) != digest or relative not in record_paths:
            raise ValueError(f"sealed control drifted or is absent: {relative}")

    artifact = manifest.get("artifact_integrity", {})
    if (
        artifact.get("passed") is not True
        or artifact.get("result_count") != 630
        or artifact.get("unique_job_signature_count") != 630
        or artifact.get("canonical_matrix_count") != 3
        or artifact.get("source_manifest_count") != 6
        or artifact.get("corrected_source_file_count") != 41
        or manifest.get("study_stage") != "development"
        or manifest.get("formal_held_out_outcomes_read") is not False
        or manifest.get("supports_current_formal_held_out_model_effect_claim") is not False
        or manifest.get("scientific_analysis", {}).get("completed") is not False
    ):
        raise ValueError("artifact/scientific scope summary failed")

    audit_path = RESULTS / "evaluation_audit_v3.json"
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("passed") is not True
        or audit.get("artifact_integrity", {}).get("passed") is not True
        or audit.get("scientific_results", {}).get("analysis_completed") is not False
    ):
        raise ValueError("evaluation audit did not preserve artifact/science separation")

    scheduler_path = RESULTS / "evidence/evaluation_v3/scheduler_evidence_v3.json"
    scheduler = json.loads(scheduler_path.read_text())
    if (
        scheduler.get("passed") is not True
        or scheduler.get("token_bound_backend_exit0") is not True
        or set(scheduler.get("tasks", {})) != {"t82450", "t82451"}
    ):
        raise ValueError("scheduler evidence failed")
    for task_id, item in scheduler["tasks"].items():
        status = RESULTS / f"evidence/evaluation_v3/backend_exit_{task_id}.status"
        fields = status.read_text().strip().split("\t")
        if (
            len(fields) != 3
            or fields[0] != "0"
            or fields[2] != item.get("exit_status_token")
            or item.get("queue_status") != "done"
            or item.get("exit_code") != 0
        ):
            raise ValueError(f"independent token-bound exit check failed: {task_id}")

    partition_signatures: set[str] = set()
    source_reference = None
    source_manifest_hashes: set[str] = set()
    raw_files = {
        str(path.resolve().relative_to(REPO))
        for path in PARTITIONS.rglob("*")
        if path.is_file()
    }
    if not raw_files.issubset(record_paths):
        raise ValueError("not every synchronized partition file is sealed")
    for shard_index, shard in enumerate(("shard0_jtl110gpu", "shard1_jtl110gpu")):
        root = PARTITIONS / shard
        host = json.loads((root / "host_execution.json").read_text())
        results = list(root.rglob("sumo_seed*.json"))
        matrices = sorted(root.rglob("matrix_manifest.shard-*.json"))
        if (
            host.get("status") != "completed"
            or host.get("returncode") != 0
            or host.get("shard_index") != shard_index
            or len(results) != 315
            or len(matrices) != 3
        ):
            raise ValueError(f"independent source partition check failed: {shard}")
        for matrix_path in matrices:
            matrix = json.loads(matrix_path.read_text())
            jobs = matrix.get("jobs", [])
            if (
                len(jobs) != 105
                or matrix.get("matrix_partition", {}).get("shard_index") != shard_index
                or matrix.get("matrix_partition", {}).get("shard_count") != 2
            ):
                raise ValueError(f"independent partition matrix check failed: {matrix_path}")
            source_manifest_hashes.add(sha256(matrix_path))
            for job in jobs:
                output = Path(job["output"]).resolve()
                if not output.is_file():
                    raise ValueError(f"missing partition output: {output}")
                source = _validate_result(job, output, source_reference)
                if source_reference is None:
                    source_reference = source
                signature = str(job["signature"])
                if signature in partition_signatures:
                    raise ValueError("duplicate result signature in source partitions")
                partition_signatures.add(signature)
    if (
        len(partition_signatures) != 630
        or len(source_manifest_hashes) != 6
        or source_reference is None
    ):
        raise ValueError("independent partition union check failed")

    canonical_signatures: set[str] = set()
    expected_methods = {
        f"{algorithm}-seed{seed}-events{events:09d}"
        for algorithm, budgets in BUDGETS.items()
        for seed in range(6201, 6206)
        for events in budgets
    }
    for index in range(1, 4):
        block = f"v8-fewshot-development-{index:02d}"
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
        ):
            raise ValueError(f"independent canonical matrix check failed: {block}")
        observed_sources = {
            descriptor.get("sha256") for descriptor in merge["source_manifests"]
        }
        if not observed_sources.issubset(source_manifest_hashes):
            raise ValueError(f"canonical source-manifest binding failed: {block}")
        for job in jobs:
            _validate_method(job["method"])
            cells.add((job["method"]["name"], float(job["scenario"]["demand_scale"])))
            signature = str(job["signature"])
            if signature in canonical_signatures:
                raise ValueError("duplicate canonical signature")
            canonical_signatures.add(signature)
        expected_cells = {
            (method, scale)
            for method in expected_methods
            for scale in (0.85, 1.0, 1.15)
        }
        if cells != expected_cells:
            raise ValueError(f"canonical Cartesian coverage failed: {block}")
    if canonical_signatures != partition_signatures:
        raise ValueError("canonical and synchronized source unions differ")

    return {
        "schema_version": (
            "transit-h2oplus-v8-fewshot-evaluation-file-manifest-verification-v3"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": {
            "file_records_aggregate_sha256": manifest[
                "file_records_aggregate_sha256"
            ],
            "path": str(manifest_path.relative_to(REPO)),
            "sha256": sha256(manifest_path),
        },
        "independent_artifact_audit": {
            "all_numeric_values_finite": True,
            "canonical_matrix_count": 3,
            "corrected_source_file_count": 41,
            "file_count": len(records),
            "passed": True,
            "result_count": 630,
            "source_manifest_count": 6,
            "source_partition_count": 2,
            "total_size_bytes": total,
            "unique_job_signature_count": 630,
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
