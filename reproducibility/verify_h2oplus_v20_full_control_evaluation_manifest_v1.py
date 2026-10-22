#!/usr/bin/env python3
"""Independently verify the v20 evaluation file manifest."""

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
MANIFEST = RESULTS / "evaluation_file_manifest_v1.json"
OUTPUT = RESULTS / "evaluation_file_manifest_verification_v1.json"
EXPECTED_AUDIT_SHA256 = "2e516ab3ea0cff10dc88607d6fb8adf3e3888a822e32a14f84a86953a466f583"


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


def aggregate(records: list[dict]) -> str:
    digest = hashlib.sha256()
    for item in records:
        digest.update(
            f"{item['path']}\t{item['bytes']}\t{item['sha256']}\n".encode()
        )
    return digest.hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite manifest verification: {OUTPUT}")
    if sha256(AUDIT) != EXPECTED_AUDIT_SHA256:
        raise ValueError("independent verifier observed audit drift")
    audit = json.loads(AUDIT.read_text())
    manifest = json.loads(MANIFEST.read_text())
    expected = {
        path.resolve(strict=True) for path in PARTITION.rglob("*") if path.is_file()
    }
    expected.update(
        path.resolve(strict=True) for path in EVIDENCE.rglob("*") if path.is_file()
    )
    expected.add(AUDIT.resolve(strict=True))
    expected.add(SUBMISSION.resolve(strict=True))
    for item in audit["bindings"]["immutable_controls"].values():
        expected.add((ROOT / item["path"]).resolve(strict=True))
    actual = [
        record(path)
        for path in sorted(expected, key=lambda path: str(path.relative_to(ROOT)))
    ]
    result_count = sum(
        Path(item["path"]).name in {"sumo_seed7501.json", "sumo_seed7502.json", "sumo_seed7503.json"}
        for item in actual
    )
    if (
        manifest.get("schema_version")
        != "transit-h2oplus-v20-full-control-evaluation-file-manifest-v1"
        or manifest.get("passed") is not True
        or manifest.get("formal_held_out_outcomes_read") is not False
        or manifest.get("local_sumo_executed") is not False
        or manifest.get("physical_result_count") != 405
        or result_count != 405
        or manifest.get("file_count") != len(actual)
        or manifest.get("total_bytes") != sum(item["bytes"] for item in actual)
        or manifest.get("aggregate_sha256") != aggregate(actual)
        or manifest.get("files") != actual
    ):
        raise ValueError("independent v20 file-manifest verification failed")
    payload = {
        "schema_version": "transit-h2oplus-v20-full-control-evaluation-file-manifest-verification-v1",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "manifest": record(MANIFEST),
        "audit": record(AUDIT),
        "independently_recomputed_file_count": len(actual),
        "independently_recomputed_total_bytes": sum(item["bytes"] for item in actual),
        "independently_recomputed_aggregate_sha256": aggregate(actual),
        "physical_result_count": result_count,
        "formal_held_out_outcomes_read": False,
        "supports_formal_held_out_effect_claim": False,
        "formal_confirmation_authorized_or_submitted": False,
    }
    partial = OUTPUT.with_suffix(OUTPUT.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, OUTPUT)
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "sha256": sha256(OUTPUT),
                "passed": True,
                "file_count": len(actual),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
