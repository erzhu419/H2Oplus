#!/usr/bin/env python3
"""Build the Python 3.8 SUMO 1.25 lock for the v8 Stage A screen."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import (
    FORMAL_V7_SUMO_BACKEND,
    FORMAL_V7_SUMO_VERSION,
    SUMO_RUNTIME_LOCK_SCHEMA,
    sha256,
    sumo_runtime_fingerprint,
)


REQUIREMENTS_SCHEMA = "transit-sumo-runtime-requirements-v7-v1"
EXPECTED_ABI = "cp38"
EXPECTED_DISTRIBUTIONS = {"libsumo", "sumolib", "traci"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_requirements(path: Path, wheel_dir: Path) -> tuple[dict, list[dict]]:
    requirements_path = path.expanduser().resolve()
    payload = json.loads(requirements_path.read_text())
    if payload.get("schema_version") != REQUIREMENTS_SCHEMA:
        raise ValueError("unsupported SUMO runtime requirements schema")
    if (
        payload.get("backend") != FORMAL_V7_SUMO_BACKEND
        or payload.get("sumo_version") != FORMAL_V7_SUMO_VERSION
        or payload.get("python_abi") != EXPECTED_ABI
    ):
        raise ValueError("v8 Stage A SUMO runtime requirements drifted")
    if sys.version_info[:2] != (3, 8):
        raise RuntimeError("v8 Stage A CPU evaluation requires Python 3.8")

    resolved_wheel_dir = wheel_dir.expanduser().resolve()
    wheels = []
    names = set()
    for raw in payload.get("wheels", []):
        name = raw.get("distribution")
        filename = raw.get("filename")
        expected = raw.get("sha256")
        if not all(isinstance(value, str) and value for value in (name, filename)):
            raise ValueError("SUMO runtime requirement has malformed wheel metadata")
        if name in names or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("SUMO runtime wheel requirements are not unique and valid")
        names.add(name)
        wheel = (resolved_wheel_dir / filename).resolve()
        if wheel.parent != resolved_wheel_dir or not wheel.is_file():
            raise FileNotFoundError(wheel)
        actual = sha256(wheel)
        if actual != expected:
            raise ValueError(
                f"wheel hash mismatch for {wheel}: expected {expected}, got {actual}"
            )
        wheels.append(
            {
                "distribution": name,
                "filename": filename,
                "path": str(wheel),
                "size_bytes": int(wheel.stat().st_size),
                "sha256": actual,
            }
        )
    if names != EXPECTED_DISTRIBUTIONS:
        raise ValueError("v8 Stage A requires exactly libsumo, sumolib, and traci")
    return {
        "path": str(requirements_path),
        "sha256": sha256(requirements_path),
        **payload,
    }, wheels


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite runtime lock {output}")
    requirements, wheels = load_requirements(args.requirements, args.wheel_dir)
    fingerprint = sumo_runtime_fingerprint(
        FORMAL_V7_SUMO_BACKEND,
        expected_version=FORMAL_V7_SUMO_VERSION,
    )
    executable = Path(sys.executable).resolve()
    payload = {
        "schema_version": SUMO_RUNTIME_LOCK_SCHEMA,
        "campaign": "H2O+ v8 disjoint-development Stage A",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "requirements": requirements,
        "wheels": wheels,
        "python_executable": {
            "path": str(executable),
            "size_bytes": int(executable.stat().st_size),
            "sha256": sha256(executable),
        },
        "runtime_fingerprint": fingerprint,
        "runtime_signature": fingerprint["portable_signature"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
