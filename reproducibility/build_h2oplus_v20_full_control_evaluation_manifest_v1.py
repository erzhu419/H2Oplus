#!/usr/bin/env python3
"""Build the file-level manifest for the verified v20 evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
RESULTS = REPRO / "h2oplus_v20_full_control_results"
PARTITION = RESULTS / "evaluation_partition_v1/single_jtl110gpu"
EVIDENCE = RESULTS / "evidence/evaluation_v1"
AUDIT = RESULTS / "evaluation_audit_v1.json"
SUBMISSION = REPRO / "h2oplus_v20_full_control_evaluation_submission_v1.json"
OUTPUT = RESULTS / "evaluation_file_manifest_v1.json"
EXPECTED_AUDIT_SHA256 = "2e516ab3ea0cff10dc88607d6fb8adf3e3888a822e32a14f84a86953a466f583"
EXPECTED_OBSERVED = {
    "scheduler": "63b5e39a9fa3edc2483f06ecb7685f41e3f24cd442fbeac899f63f2949d841dd",
    "source_sync": "02a63f240717b20d4e03d9fd6bd838ff13778f3d95340d8b8fb2ea3b83caa736",
    "host_execution": "043db2c914da08449f5545ee9cbdd3e4087d9190976c799b5e525a33bee5593c",
    "host_summary": "a78319246c8a5e2610b7e1b40990e657e95a1966b26178f0748e67a59dee661c",
    "matrix_01": "230e9ae6246bd4d6960f173c8072b4ac0d36ca9917f4dbad8e82168c11ed1d0e",
    "matrix_02": "4e0954df68ec72a7cf0a14bad73f53bf739703c2b38a5d4874cd8ed4a474ed64",
    "matrix_03": "4e08082078a24e1f32a7504bd68f7d1b979ed1bcdc8f3a57a7c0a54ec1509055",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved.relative_to(ROOT)),
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def expected_paths(audit: dict) -> list[Path]:
    paths = {
        path.resolve(strict=True) for path in PARTITION.rglob("*") if path.is_file()
    }
    paths.update(
        path.resolve(strict=True) for path in EVIDENCE.rglob("*") if path.is_file()
    )
    paths.add(AUDIT.resolve(strict=True))
    paths.add(SUBMISSION.resolve(strict=True))
    for item in audit["bindings"]["immutable_controls"].values():
        paths.add((ROOT / item["path"]).resolve(strict=True))
    return sorted(paths, key=lambda path: str(path.relative_to(ROOT)))


def aggregate(records: list[dict]) -> str:
    digest = hashlib.sha256()
    for item in records:
        digest.update(
            f"{item['path']}\t{item['bytes']}\t{item['sha256']}\n".encode()
        )
    return digest.hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite file manifest: {OUTPUT}")
    if sha256(AUDIT) != EXPECTED_AUDIT_SHA256:
        raise ValueError("v20 evaluation audit identity drifted")
    audit = json.loads(AUDIT.read_text())
    if (
        audit.get("passed") is not True
        or audit.get("formal_held_out_outcomes_read") is not False
        or audit.get("local_sumo_executed") is not False
        or audit.get("artifact_integrity", {}).get("physical_result_count") != 405
    ):
        raise ValueError("v20 audit did not authorize file manifest creation")
    observed_paths = {
        "scheduler": EVIDENCE / "scheduler_evidence_v1.json",
        "source_sync": EVIDENCE / "source_sync_v1.json",
        "host_execution": PARTITION / "host_execution.json",
        "host_summary": PARTITION / "host_summary.json",
        "matrix_01": PARTITION
        / "v20-full-control-development-01/matrix_manifest.json",
        "matrix_02": PARTITION
        / "v20-full-control-development-02/matrix_manifest.json",
        "matrix_03": PARTITION
        / "v20-full-control-development-03/matrix_manifest.json",
    }
    if any(
        sha256(path) != EXPECTED_OBSERVED[name]
        for name, path in observed_paths.items()
    ):
        raise ValueError("observed v20 scheduler, sync, host, or matrix hash drifted")
    files = [record(path) for path in expected_paths(audit)]
    result_count = sum(
        Path(item["path"]).name in {"sumo_seed7501.json", "sumo_seed7502.json", "sumo_seed7503.json"}
        for item in files
    )
    if result_count != 405:
        raise ValueError("file manifest does not cover exactly 405 result JSON files")
    payload = {
        "schema_version": "transit-h2oplus-v20-full-control-evaluation-file-manifest-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": "development",
        "matrix_type": "full-development-control-matrix",
        "passed": True,
        "physical_result_count": 405,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "aggregate_sha256": aggregate(files),
        "observed_bindings": {
            name: record(path) for name, path in observed_paths.items()
        },
        "evaluation_audit": record(AUDIT),
        "files": files,
        "prior_v14_through_v19_development_outcomes_used_for_design": True,
        "material_data_environment_validity_factors_present": True,
        "local_sumo_executed": False,
        "formal_held_out_outcomes_read": False,
        "supports_formal_held_out_effect_claim": False,
        "formal_confirmation_authorized_or_submitted": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    partial = OUTPUT.with_suffix(OUTPUT.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, OUTPUT)
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "sha256": sha256(OUTPUT),
                "file_count": len(files),
                "aggregate_sha256": payload["aggregate_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
