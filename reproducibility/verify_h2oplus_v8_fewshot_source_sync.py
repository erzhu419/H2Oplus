#!/usr/bin/env python3
"""Verify one exact remote few-shot training partition against its local copy."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "transit-h2oplus-v8-fewshot-source-sync-v1"
RUN_NAME = re.compile(r"^(?:h2oplus|sac-online)_seed62\d{2}$")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def local_inventory(path: Path) -> tuple[int, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def remote_inventory(host: str, path: str) -> tuple[int, int]:
    command = f"find {shlex.quote(path)} -type f -printf '%s\\n'"
    result = subprocess.run(
        [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=15",
            host,
            command,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"remote inventory failed for {host}:{path}: {result.stderr.strip()}")
    sizes = [int(line) for line in result.stdout.splitlines() if line.strip()]
    return len(sizes), sum(sizes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-host", choices=("jtl110gpu", "jtl110gpu2"), required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--run-name", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    local_root = args.local_root.expanduser().resolve(strict=True)
    run_names = sorted(args.run_name)
    if len(run_names) != len(set(run_names)) or any(not RUN_NAME.fullmatch(name) for name in run_names):
        raise ValueError("run names must be unique registered few-shot run identifiers")

    errors: list[str] = []
    differences: list[dict] = []
    runs = []
    for run_name in run_names:
        local = (local_root / run_name).resolve(strict=True)
        if local.parent != local_root or not local.is_dir():
            raise ValueError(f"local run escapes the synchronized root: {local}")
        remote = f"{args.remote_root.rstrip('/')}/{run_name}"
        result = subprocess.run(
            [
                "rsync", "-rnc", "--delete", "--out-format=%i %n%L",
                f"{args.source_host}:{remote}/", f"{local}/",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=1800,
        )
        output_lines = [line for line in result.stdout.splitlines() if line.strip()]
        if result.returncode != 0:
            errors.append(
                f"checksum rsync failed for {run_name} with {result.returncode}: "
                f"{result.stderr.strip()}"
            )
        if output_lines:
            differences.append({"run_name": run_name, "items": output_lines})
        local_count, local_bytes = local_inventory(local)
        try:
            source_count, source_bytes = remote_inventory(args.source_host, remote)
        except Exception as error:
            errors.append(str(error))
            source_count, source_bytes = -1, -1
        if (source_count, source_bytes) != (local_count, local_bytes):
            errors.append(
                f"inventory mismatch for {run_name}: source={(source_count, source_bytes)} "
                f"local={(local_count, local_bytes)}"
            )
        runs.append(
            {
                "run_name": run_name,
                "source_file_count": source_count,
                "source_total_bytes": source_bytes,
                "local_file_count": local_count,
                "local_total_bytes": local_bytes,
            }
        )

    payload = {
        "schema_version": SCHEMA,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not errors and not differences,
        "errors": errors,
        "differences": differences,
        "checksum_mode": "rsync-content-checksum-dry-run",
        "source_host": args.source_host,
        "remote_root": args.remote_root,
        "local_root": str(local_root),
        "run_names": run_names,
        "source_file_count": sum(item["source_file_count"] for item in runs),
        "source_total_bytes": sum(item["source_total_bytes"] for item in runs),
        "local_file_count": sum(item["local_file_count"] for item in runs),
        "local_total_bytes": sum(item["local_total_bytes"] for item in runs),
        "runs": runs,
    }
    atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
