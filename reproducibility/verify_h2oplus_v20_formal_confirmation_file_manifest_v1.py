#!/usr/bin/env python3
"""Independently verify the compact v20 formal-confirmation file manifest."""

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
EVIDENCE = RESULTS / "evidence/evaluation_v4"
AUDIT = RESULTS / "evaluation_audit_v1.json"
COMPACTION = RESULTS / "evaluation_v4_log_compaction_v1.json"
SUBMISSION = REPRO / "h2oplus_v20_formal_confirmation_submission_v4.json"
SYNC = REPRO / "h2oplus_v20_formal_confirmation_minimal_sync_v4.json"
ANALYZER = REPRO / "analyze_h2oplus_v20_formal_confirmation.py"
SYNC_POLICY = REPRO / "h2oplus_minimal_remote_sync_policy_v1.json"
MANIFEST = RESULTS / "evaluation_file_manifest_v1.json"
OUTPUT = RESULTS / "evaluation_file_manifest_verification_v1.json"

AUDIT_SHA256 = "aa71cfa4060c22dc6413a500f0aa922b415a7fc09516231ff8b99401c15dd5d6"
COMPACTION_SHA256 = "639c29f8e9ddf0f094e32c08d9d1d587d5601c5264b57e800e5d5f10545bb3b0"
EXCLUDED_BINDINGS = {"dataset"}
LARGE_ARTIFACT_SUFFIXES = {".h5", ".pt", ".pth", ".ckpt", ".npy", ".npz"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def category(path: Path) -> str:
    if path.match("**/sumo_seed*.json"):
        return "scientific-result"
    if path.name == "matrix_manifest.json":
        return "matrix-manifest"
    if path.parent == PARTITION and path.suffix == ".log":
        return "block-log"
    if EVIDENCE in path.parents:
        return "terminal-evidence"
    if path == AUDIT:
        return "independent-audit"
    if path == COMPACTION:
        return "log-compaction"
    if path == SUBMISSION:
        return "submission"
    if path == SYNC:
        return "minimal-sync"
    if path == ANALYZER:
        return "frozen-analyzer"
    if path == SYNC_POLICY:
        return "sync-policy"
    if path.name in {"host_execution.json", "host_summary.json"}:
        return "host-closeout"
    return "immutable-control"


def record(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved.relative_to(ROOT)),
        "category": category(resolved),
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def aggregate(records: list[dict]) -> str:
    digest = hashlib.sha256()
    for item in records:
        digest.update(
            (
                f"{item['path']}\t{item['category']}\t{item['bytes']}\t"
                f"{item['sha256']}\n"
            ).encode()
        )
    return digest.hexdigest()


def expected_paths(audit: dict) -> list[Path]:
    paths = {path.resolve(strict=True) for path in PARTITION.rglob("*") if path.is_file()}
    paths.update(path.resolve(strict=True) for path in EVIDENCE.rglob("*") if path.is_file())
    paths.update(
        {
            AUDIT.resolve(strict=True),
            COMPACTION.resolve(strict=True),
            SUBMISSION.resolve(strict=True),
            SYNC.resolve(strict=True),
            ANALYZER.resolve(strict=True),
            SYNC_POLICY.resolve(strict=True),
        }
    )
    for name, item in audit["bindings"]["immutable_controls"].items():
        if name not in EXCLUDED_BINDINGS:
            paths.add((ROOT / item["path"]).resolve(strict=True))
    return sorted(paths, key=lambda path: str(path.relative_to(ROOT)))


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite verification: {OUTPUT}")
    if sha256(AUDIT) != AUDIT_SHA256 or sha256(COMPACTION) != COMPACTION_SHA256:
        raise ValueError("independent verifier observed audit or compaction drift")
    audit = json.loads(AUDIT.read_text())
    manifest = json.loads(MANIFEST.read_text())
    actual = [record(path) for path in expected_paths(audit)]
    categories = {}
    for item in actual:
        categories[item["category"]] = categories.get(item["category"], 0) + 1
    excluded_dataset = audit["bindings"]["immutable_controls"]["dataset"]
    failed = (
        manifest.get("schema_version")
        != "transit-h2oplus-v20-formal-confirmation-file-manifest-v1"
        or manifest.get("passed") is not True
        or manifest.get("physical_result_count") != 1350
        or manifest.get("matrix_manifest_count") != 10
        or manifest.get("file_count") != len(actual)
        or manifest.get("total_bytes") != sum(item["bytes"] for item in actual)
        or manifest.get("aggregate_sha256") != aggregate(actual)
        or manifest.get("category_counts") != categories
        or manifest.get("files") != actual
        or manifest.get("excluded_large_artifact_bindings", {}).get("dataset")
        != excluded_dataset
        or manifest.get(
            "checkpoint_optimizer_replay_dataset_or_workspace_files_included"
        )
        is not False
        or categories.get("scientific-result") != 1350
        or categories.get("matrix-manifest") != 10
        or categories.get("block-log") != 10
        or len(list(PARTITION.rglob("*.log"))) != 10
        or any(Path(item["path"]).suffix.lower() in LARGE_ARTIFACT_SUFFIXES for item in actual)
        or any("/workspace/" in f"/{item['path']}/" for item in actual)
    )
    if failed:
        raise ValueError("independent formal file-manifest verification failed")

    payload = {
        "schema_version": "transit-h2oplus-v20-formal-confirmation-file-manifest-verification-v1",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "manifest": record(MANIFEST),
        "audit": record(AUDIT),
        "independently_recomputed_file_count": len(actual),
        "independently_recomputed_total_bytes": sum(item["bytes"] for item in actual),
        "independently_recomputed_aggregate_sha256": aggregate(actual),
        "category_counts": categories,
        "physical_result_count": 1350,
        "matrix_manifest_count": 10,
        "large_artifact_file_count": 0,
        "workspace_file_count": 0,
        "formal_held_out_outcome_access_authorized": True,
        "formal_held_out_outcomes_read": True,
        "scientific_effects_computed": False,
        "supports_formal_held_out_effect_claim": False,
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
                "aggregate_sha256": payload[
                    "independently_recomputed_aggregate_sha256"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
