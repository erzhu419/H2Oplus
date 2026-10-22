#!/usr/bin/env python3
"""Release baseline dependency gates from a verified v7 core manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GATE_SCHEMA = "transit-core-dependency-gate-v7-v1"
HOSTS = ("jtl110gpu", "jtl110gpu2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_manifest(manifest: dict[str, Any]) -> dict[str, str]:
    if manifest.get("status") != "verified" or manifest.get("errors"):
        raise RuntimeError("core manifest is not verified and error-free")
    local = manifest.get("initializer_audit") or {}
    local_hashes = {
        name: record.get("sha256")
        for name, record in (local.get("initializers") or {}).items()
    }
    if not local.get("passed") or len(local_hashes) != 5 or any(
        not value for value in local_hashes.values()
    ):
        raise RuntimeError("local canonical initializer set is incomplete")
    remote = manifest.get("remote_initializer_audits") or {}
    if set(remote) != set(HOSTS):
        raise RuntimeError("remote initializer audit host set is incomplete")
    for host in HOSTS:
        record = remote[host]
        if (
            not record.get("passed")
            or record.get("initializers_audited") != 5
            or record.get("source_sha256_count") != 37
            or record.get("initializer_sha256") != local_hashes
        ):
            raise RuntimeError(f"{host} initializer audit does not match canonical set")
    if not manifest.get("partition_runs_disjoint"):
        raise RuntimeError("core source partitions are not disjoint")
    matrix = manifest.get("independent_matrix_audit") or {}
    if not matrix.get("passed") or matrix.get("jobs_audited") != 10:
        raise RuntimeError("independent matrix audit is incomplete")
    return local_hashes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    gate_dir = args.gate_dir.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    initializer_hashes = validate_manifest(manifest)
    manifest_sha256 = sha256(manifest_path)
    created_utc = datetime.now(timezone.utc).isoformat()
    for host in HOSTS:
        remote_audit = manifest["remote_initializer_audits"][host]
        gate = {
            "schema_version": GATE_SCHEMA,
            "created_utc": created_utc,
            "status": "ready",
            "host": host,
            "manifest": {
                "path": str(manifest_path),
                "sha256": manifest_sha256,
            },
            "matrix_audit_sha256": manifest["independent_matrix_audit"]["sha256"],
            "initializer_audit_sha256": manifest["initializer_audit"]["sha256"],
            "remote_initializer_audit_sha256": remote_audit["sha256"],
            "initializer_sha256": initializer_hashes,
        }
        gate_path = gate_dir / f"core_verified_{host}.ready"
        _write_json(gate_path, gate)
        print(gate_path)


if __name__ == "__main__":
    main()
