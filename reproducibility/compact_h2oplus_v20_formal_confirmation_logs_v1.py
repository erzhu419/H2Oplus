#!/usr/bin/env python3
"""Remove redundant per-job logs after the formal result audit passes."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
RESULTS = REPRO / "h2oplus_v20_formal_confirmation_results"
PARTITION = RESULTS / "evaluation_v4/single_jtl110gpu"
AUDIT = RESULTS / "evaluation_audit_v1.json"
OUTPUT = RESULTS / "evaluation_v4_log_compaction_v1.json"
AUDIT_SHA256 = "aa71cfa4060c22dc6413a500f0aa922b415a7fc09516231ff8b99401c15dd5d6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to repeat compaction: {OUTPUT}")
    if sha256(AUDIT) != AUDIT_SHA256:
        raise ValueError("formal audit identity drifted")
    audit = json.loads(AUDIT.read_text())
    if audit.get("passed") is not True or audit.get("scientific_effects_computed") is not False:
        raise ValueError("formal audit did not establish the required precondition")

    block_logs = sorted(PARTITION.glob("*.log"))
    nested_logs = sorted(
        path for path in PARTITION.rglob("*.log") if path.parent != PARTITION
    )
    if len(block_logs) != 10 or len(nested_logs) != 1350:
        raise ValueError(
            f"unexpected log layout: block={len(block_logs)} nested={len(nested_logs)}"
        )
    reclaimed = sum(path.stat().st_size for path in nested_logs)
    for path in nested_logs:
        path.unlink()

    result_count = len(list(PARTITION.rglob("sumo_seed*.json")))
    manifest_count = len(list(PARTITION.rglob("matrix_manifest.json")))
    remaining_block_logs = len(list(PARTITION.glob("*.log")))
    remaining_nested_logs = len(
        [path for path in PARTITION.rglob("*.log") if path.parent != PARTITION]
    )
    if (
        result_count != 1350
        or manifest_count != 10
        or remaining_block_logs != 10
        or remaining_nested_logs != 0
    ):
        raise ValueError("post-compaction formal evidence counts are invalid")

    payload = {
        "schema_version": "transit-h2oplus-v20-formal-confirmation-log-compaction-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "audit_sha256": AUDIT_SHA256,
        "deleted_redundant_per_job_log_count": 1350,
        "reclaimed_bytes": reclaimed,
        "retained_block_log_count": 10,
        "retained_scientific_result_count": result_count,
        "retained_matrix_manifest_count": manifest_count,
        "retained_scheduler_and_backend_evidence": True,
        "scientific_results_deleted": False,
        "checkpoints_datasets_optimizer_replay_or_workspace_files_synchronized": False,
    }
    partial = OUTPUT.with_suffix(OUTPUT.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, OUTPUT)
    print(json.dumps({"output": str(OUTPUT), **payload}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
