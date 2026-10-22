#!/usr/bin/env python3
"""Build the file-level manifest for the verified v19-v2 evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
RESULTS = REPRO / "h2oplus_v19_completion_safety_results"
PARTITION = RESULTS / "evaluation_partition_v2/single_jtl110gpu"
EVIDENCE = RESULTS / "evidence/evaluation_v2"
AUDIT = RESULTS / "evaluation_audit_v2.json"
OUTPUT = RESULTS / "evaluation_file_manifest_v2.json"
EXPECTED_AUDIT_SHA256 = "267370a31bed2bc8053842dc28e9223ae76c517ef3877603b3c9c81588b8636b"
EXPECTED_OBSERVED = {
    "scheduler": "fa02ece6ce209d672bd8b994cee615703b13fd57dd6a39855d0b1b2fef0264d0",
    "source_sync": "4d1d9fe6a6b92c83e8211b974d09980a5790c64ec3f169be140dc68266e5ff10",
    "host_execution": "3167f0872a9288be90b6cd5e5c270ae1ae7ff47506c3d627159d59dd7e5f2ce3",
    "host_summary": "aac9d94f8ad5cf518f6a3e77146b1b9f49e3b0cfb3a8ff12f7c179a0318c7cbd",
    "matrix_01": "09cee7c1c616a8b3740a854635716da0f74fff92e4b01d9968f677823de40683",
    "matrix_02": "5830e5595a350a64df94b88a99f87a4e684dac183d8278a39a51aa5adedf38b0",
    "matrix_03": "d06e017570c69cb5c432401a111ac680c1a7b38bce91b12ea95c65f5710be82c",
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
    paths.add(
        (
            REPRO / "h2oplus_v19_completion_safety_evaluation_submission_v2.json"
        ).resolve(strict=True)
    )
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
        raise ValueError("v19 evaluation audit identity drifted")
    audit = json.loads(AUDIT.read_text())
    if (
        audit.get("passed") is not True
        or audit.get("formal_held_out_outcomes_read") is not False
        or audit.get("v19_v1_results_reused") is not False
        or audit.get("artifact_integrity", {}).get("physical_result_count") != 135
    ):
        raise ValueError("v19 audit did not authorize file manifest creation")
    observed_paths = {
        "scheduler": EVIDENCE / "scheduler_evidence_v2.json",
        "source_sync": EVIDENCE / "source_sync_v2.json",
        "host_execution": PARTITION / "host_execution.json",
        "host_summary": PARTITION / "host_summary.json",
        "matrix_01": PARTITION
        / "v19-completion-safety-development-01/matrix_manifest.json",
        "matrix_02": PARTITION
        / "v19-completion-safety-development-02/matrix_manifest.json",
        "matrix_03": PARTITION
        / "v19-completion-safety-development-03/matrix_manifest.json",
    }
    if any(
        sha256(path) != EXPECTED_OBSERVED[name]
        for name, path in observed_paths.items()
    ):
        raise ValueError("observed v19 scheduler, sync, host, or matrix hash drifted")
    files = [record(path) for path in expected_paths(audit)]
    result_count = sum(
        any(item["path"].endswith(f"/sumo_seed{seed}.json") for seed in (7301, 7302, 7303))
        for item in files
    )
    if result_count != 135:
        raise ValueError("file manifest does not cover exactly 135 result JSON files")
    payload = {
        "schema_version": "transit-h2oplus-v19-completion-safety-evaluation-file-manifest-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": "development",
        "screen_type": "paired-mechanism-screen",
        "passed": True,
        "physical_result_count": 135,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "aggregate_sha256": aggregate(files),
        "observed_bindings": {
            name: record(path) for name, path in observed_paths.items()
        },
        "evaluation_audit": record(AUDIT),
        "files": files,
        "prior_v14_v15_v16_v17_v18_development_outcomes_used_for_design": True,
        "material_data_environment_validity_factors_present": True,
        "v19_v1_results_reused": False,
        "local_sumo_executed": False,
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_eligible": False,
        "formal_confirmation_authorized": False,
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
