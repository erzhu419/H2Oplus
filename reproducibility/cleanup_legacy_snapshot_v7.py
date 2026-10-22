#!/usr/bin/env python3
"""Build and enforce an exact deletion manifest for legacy snapshot archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = ROOT / "bus_h2o" / "datasets_v2"
DEFAULT_MANIFEST = (
    ROOT
    / "reproducibility"
    / "cleanup_manifests"
    / "legacy_snapshot_cleanup_20260810_dry_run.json"
)
FORMAL_DATASET = (
    ROOT
    / "bus_h2o"
    / "datasets_v7"
    / "formal_v2_full"
    / "transit_smdp_v7_formal_v2.h5"
)
MIN_ARCHIVE_BYTES = 4_000_000_000
SNAPSHOT_KEYS = frozenset({"raw_snapshot", "snapshot_T1"})


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def allocated_bytes(path: Path) -> int:
    return path.stat().st_blocks * 512


def snapshot_metadata(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        keys = sorted(handle.keys())
        snapshot_keys = sorted(SNAPSHOT_KEYS.intersection(keys))
        attrs = {
            key: value.item() if hasattr(value, "item") else value
            for key, value in handle.attrs.items()
            if key in {"n_episodes", "n_transitions", "policy", "seed", "snapshot_fmt"}
        }
    return {
        "hdf5_keys": keys,
        "snapshot_keys": snapshot_keys,
        "selected_snapshot_attrs": attrs,
    }


def is_candidate(path: Path, target: Path) -> tuple[bool, str, dict[str, Any]]:
    if path.parent != target:
        return False, "not_direct_child", {}
    if path.is_symlink() or not path.is_file():
        return False, "not_regular_file", {}
    if not (path.name.startswith("sumo_") and ".h5" in path.name):
        return False, "name_not_legacy_sumo_hdf5", {}
    stat = path.stat()
    if stat.st_size < MIN_ARCHIVE_BYTES:
        return False, "below_archive_size_threshold", {}
    if stat.st_nlink != 1:
        return False, "hard_link_count_not_one", {}
    try:
        metadata = snapshot_metadata(path)
    except OSError as exc:
        return False, f"invalid_hdf5:{exc}", {}
    if not metadata["snapshot_keys"]:
        return False, "no_snapshot_dataset", metadata
    return True, "legacy_snapshot_archive", metadata


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def file_record(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": relative(path),
        "size_bytes": stat.st_size,
        "allocated_bytes": allocated_bytes(path),
        "inode": stat.st_ino,
        "hard_link_count": stat.st_nlink,
        "sha256": sha256_file(path),
        **metadata,
    }


def build_manifest(target: Path) -> dict[str, Any]:
    target = target.resolve()
    expected = DEFAULT_TARGET.resolve()
    if target != expected:
        raise SystemExit(f"refusing non-canonical target: {target} != {expected}")
    if target.is_symlink() or not target.is_dir():
        raise SystemExit(f"target must be a real directory: {target}")

    candidates: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    for path in sorted(target.iterdir(), key=lambda item: item.name):
        selected, reason, metadata = is_candidate(path, target)
        if selected:
            candidates.append(file_record(path, metadata))
        elif path.is_file() or path.is_symlink():
            stat = path.lstat()
            retained.append(
                {
                    "path": relative(path),
                    "size_bytes": stat.st_size,
                    "allocated_bytes": stat.st_blocks * 512,
                    "hard_link_count": stat.st_nlink,
                    "reason": reason,
                }
            )

    if not candidates:
        raise SystemExit("no legacy snapshot archives matched the strict selection rule")

    formal_stat = FORMAL_DATASET.stat()
    manifest: dict[str, Any] = {
        "schema": "h2oplus-legacy-snapshot-cleanup-v7-v1",
        "mode": "dry_run",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(ROOT.resolve()),
        "target_root": str(target),
        "selection_rule": {
            "direct_child_only": True,
            "filename": "sumo_*.h5*",
            "minimum_size_bytes": MIN_ARCHIVE_BYTES,
            "required_hdf5_keys_any": sorted(SNAPSHOT_KEYS),
            "required_hard_link_count": 1,
            "symlinks_allowed": False,
        },
        "candidate_summary": {
            "file_count": len(candidates),
            "logical_bytes": sum(item["size_bytes"] for item in candidates),
            "allocated_bytes": sum(item["allocated_bytes"] for item in candidates),
        },
        "candidate_files": candidates,
        "retained_files_in_target": retained,
        "protected_paths": [
            {
                "path": relative(FORMAL_DATASET),
                "size_bytes": formal_stat.st_size,
                "sha256": sha256_file(FORMAL_DATASET),
                "reason": "canonical formal v7 no-snapshot training dataset",
            },
            {"path": "bus_h2o/datasets_v7", "reason": "all v7 datasets protected"},
            {"path": "experiment_output_v7", "reason": "all v7 outputs and checkpoints protected"},
            {"path": "reproducibility", "reason": "audit and deployment evidence protected"},
        ],
        "execution_contract": {
            "requires_manifest_sha256": True,
            "rechecks_path_size_inode_hard_links_and_sha256": True,
            "refuses_new_or_missing_candidates": True,
            "deletes_directories": False,
        },
    }
    return manifest


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()


def write_manifest(manifest: dict[str, Any], output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(manifest)
    output.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def verify_execution_manifest(manifest_path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = manifest_path.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            f"manifest SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    payload = json.loads(raw)
    if payload.get("schema") != "h2oplus-legacy-snapshot-cleanup-v7-v1":
        raise SystemExit("unsupported cleanup manifest schema")
    if Path(payload["target_root"]).resolve() != DEFAULT_TARGET.resolve():
        raise SystemExit("manifest target is not the canonical legacy archive directory")

    fresh = build_manifest(DEFAULT_TARGET)
    expected_records = {item["path"]: item for item in payload["candidate_files"]}
    fresh_records = {item["path"]: item for item in fresh["candidate_files"]}
    if expected_records.keys() != fresh_records.keys():
        raise SystemExit("candidate set changed after dry run")
    checked_fields = ("size_bytes", "allocated_bytes", "inode", "hard_link_count", "sha256")
    for path, expected in expected_records.items():
        current = fresh_records[path]
        for field in checked_fields:
            if current[field] != expected[field]:
                raise SystemExit(f"candidate drift: {path} field {field}")
    return payload


def execute(manifest_path: Path, manifest_sha256: str, receipt: Path) -> None:
    payload = verify_execution_manifest(manifest_path, manifest_sha256)
    deleted: list[dict[str, Any]] = []
    for item in payload["candidate_files"]:
        path = ROOT / item["path"]
        path.unlink()
        deleted.append(
            {
                "path": item["path"],
                "size_bytes": item["size_bytes"],
                "allocated_bytes": item["allocated_bytes"],
                "sha256": item["sha256"],
            }
        )

    remaining = [item["path"] for item in payload["candidate_files"] if (ROOT / item["path"]).exists()]
    if remaining:
        raise SystemExit(f"deletion verification failed: {remaining}")
    protected = payload["protected_paths"][0]
    protected_path = ROOT / protected["path"]
    if sha256_file(protected_path) != protected["sha256"]:
        raise SystemExit("formal v7 dataset changed during cleanup")

    result = {
        "schema": "h2oplus-legacy-snapshot-cleanup-v7-receipt-v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": relative(manifest_path),
        "source_manifest_sha256": manifest_sha256,
        "deleted_file_count": len(deleted),
        "deleted_logical_bytes": sum(item["size_bytes"] for item in deleted),
        "deleted_allocated_bytes": sum(item["allocated_bytes"] for item in deleted),
        "deleted_files": deleted,
        "protected_formal_dataset_verified": protected,
    }
    write_manifest(result, receipt)
    print(json.dumps(result, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-manifest-sha256")
    parser.add_argument(
        "--receipt",
        type=Path,
        default=DEFAULT_MANIFEST.with_name("legacy_snapshot_cleanup_20260810_receipt.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.execute:
        if not args.confirm_manifest_sha256:
            raise SystemExit("--execute requires --confirm-manifest-sha256")
        execute(args.manifest.resolve(), args.confirm_manifest_sha256, args.receipt.resolve())
        return
    manifest = build_manifest(args.target)
    manifest_sha256 = write_manifest(manifest, args.manifest.resolve())
    summary = manifest["candidate_summary"]
    print(
        json.dumps(
            {
                "manifest": str(args.manifest.resolve()),
                "manifest_sha256": manifest_sha256,
                **summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
