#!/usr/bin/env python3
"""Build the file-level manifest for the verified v18-v3 evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
RESULTS = REPRO / "h2oplus_v18_empty_tail_reserve_results"
PARTITION = RESULTS / "evaluation_partition_v3/single_jtl110gpu"
EVIDENCE = RESULTS / "evidence/evaluation_v3"
AUDIT = RESULTS / "evaluation_audit_v3.json"
OUTPUT = RESULTS / "evaluation_file_manifest_v3.json"
EXPECTED_AUDIT_SHA256 = "053feec4df5d1ffdaaea4e494815a7378fa011956a2bc2703f14a5d3dd1ac64c"
EXPECTED_OBSERVED = {
    "scheduler": "b514e9e354d15b919e1d2ed2f8562b1a3e94160adfcecb5968e95adaaec8a8ad",
    "source_sync": "da38e79455ee5c8d78ac7386c17d0cd38d04396efd8d8dbf2f3ba2a050e2af15",
    "host_execution": "c7fbdc20a78feb72c956ae2d0f3b678a5a1f45a3f04350dc889689ae1b151bf7",
    "host_summary": "858dd0f0f6b4a6bca60f9cb156f9f0894b61cb499ce5a378d219da40f84d82d7",
    "matrix_01": "0b2b6d60c9ebbf623f7b8a617649d5ce873e076fcf089cb6161cbf03907aaa10",
    "matrix_02": "7f3a9490b6fb9c6c98e6dd1f4f013481d2012d6c42e1d2fa8717473d029eaf21",
    "matrix_03": "7de35e6642f844cdaff65f0dc29cb1ec9ba68abea5e0912d74dfd66e7dfb582f",
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
    paths = {path.resolve(strict=True) for path in PARTITION.rglob("*") if path.is_file()}
    paths.update(path.resolve(strict=True) for path in EVIDENCE.rglob("*") if path.is_file())
    paths.add(AUDIT.resolve(strict=True))
    paths.add((REPRO / "h2oplus_v18_empty_tail_reserve_evaluation_submission_v3.json").resolve(strict=True))
    for descriptor in audit["bindings"]["immutable_controls"].values():
        paths.add((ROOT / descriptor["path"]).resolve(strict=True))
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
        raise ValueError("v18 evaluation audit identity drifted")
    audit = json.loads(AUDIT.read_text())
    if (
        audit.get("passed") is not True
        or audit.get("formal_held_out_outcomes_read") is not False
        or audit.get("v18_v1_or_v2_results_reused") is not False
        or audit.get("artifact_integrity", {}).get("physical_result_count") != 135
    ):
        raise ValueError("v18 audit did not authorize file manifest creation")
    observed_paths = {
        "scheduler": EVIDENCE / "scheduler_evidence_v3.json",
        "source_sync": EVIDENCE / "source_sync_v3.json",
        "host_execution": PARTITION / "host_execution.json",
        "host_summary": PARTITION / "host_summary.json",
        "matrix_01": PARTITION / "v18-empty-tail-development-01/matrix_manifest.json",
        "matrix_02": PARTITION / "v18-empty-tail-development-02/matrix_manifest.json",
        "matrix_03": PARTITION / "v18-empty-tail-development-03/matrix_manifest.json",
    }
    if any(sha256(path) != EXPECTED_OBSERVED[name] for name, path in observed_paths.items()):
        raise ValueError("observed v18 scheduler, sync, host, or matrix hash drifted")
    files = [record(path) for path in expected_paths(audit)]
    if len([item for item in files if item["path"].endswith("/sumo_seed7101.json") or item["path"].endswith("/sumo_seed7102.json") or item["path"].endswith("/sumo_seed7103.json")]) != 135:
        raise ValueError("file manifest does not cover exactly 135 result JSON files")
    payload = {
        "schema_version": "transit-h2oplus-v18-empty-tail-reserve-evaluation-file-manifest-v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": "development",
        "screen_type": "paired-mechanism-screen",
        "passed": True,
        "physical_result_count": 135,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "aggregate_sha256": aggregate(files),
        "observed_bindings": {name: record(path) for name, path in observed_paths.items()},
        "evaluation_audit": record(AUDIT),
        "files": files,
        "prior_v14_v15_v16_v17_development_outcomes_used_for_design": True,
        "material_data_environment_validity_factors_present": True,
        "v18_v1_or_v2_results_reused": False,
        "local_sumo_executed": False,
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_eligible": False,
        "formal_confirmation_authorized": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    partial = OUTPUT.with_suffix(OUTPUT.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, OUTPUT)
    print(json.dumps({"output": str(OUTPUT), "sha256": sha256(OUTPUT), "file_count": len(files), "aggregate_sha256": payload["aggregate_sha256"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
