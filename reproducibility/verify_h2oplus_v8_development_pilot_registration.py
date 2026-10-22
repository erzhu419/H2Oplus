#!/usr/bin/env python3
"""Independently verify the H2O+ v8 development-pilot registration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproducibility.register_h2oplus_v8_development_pilot import (
    SCHEMA_VERSION as REGISTRATION_SCHEMA,
    _descriptor,
    register,
)


SCHEMA_VERSION = "transit-h2oplus-v8-development-pilot-verification-v1"


def _stable(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if key != "registered_utc"}


def verify(path: Path) -> dict:
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text())
    errors: list[str] = []
    if payload.get("schema_version") != REGISTRATION_SCHEMA:
        errors.append("unexpected v8 registration schema")
    evidence = payload.get("evidence", {})
    argument_names = {
        "mechanism_analysis": "complete_v7_mechanism_analysis",
        "sensitivity_manifest": "sensitivity_manifest",
        "sensitivity_verification": "sensitivity_manifest_verification",
        "sensitivity_stability": "sensitivity_stability_audit",
        "ablation_manifest": "ablation_manifest",
        "ablation_verification": "ablation_manifest_verification",
        "core_manifest": "core_manifest",
        "held_out_design": "frozen_held_out_environment_design",
    }
    arguments = {}
    for argument, key in argument_names.items():
        descriptor = evidence.get(key, {})
        candidate = Path(descriptor.get("path", "")).expanduser().resolve()
        if not candidate.is_file() or descriptor != _descriptor(candidate):
            errors.append(f"v8 registration evidence mismatch: {key}")
        arguments[argument] = candidate
    arguments["output"] = path
    if not errors:
        recomputed = register(argparse.Namespace(**arguments))
        if _stable(payload) != _stable(recomputed):
            errors.append("v8 registration differs from independent recomputation")
    if payload.get("status") != "registered-not-submitted":
        errors.append("v8 pilot is not in registered-not-submitted state")
    if payload.get("execution_authorized") is not False:
        errors.append("v8 pilot unexpectedly authorizes execution")
    if payload.get("scheduler_task_ids") != []:
        errors.append("v8 pilot unexpectedly contains scheduler tasks")
    if payload.get("submission_requires_explicit_user_approval") is not True:
        errors.append("v8 pilot lacks explicit submission approval gate")
    if payload.get("development_environment", {}).get(
        "formal_held_out_may_be_read_for_selection"
    ) is not False:
        errors.append("v8 pilot does not protect the formal held-out environment")
    return {
        "schema_version": SCHEMA_VERSION,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not errors,
        "errors": errors,
        "registration": _descriptor(path),
        "execution_authorized": payload.get("execution_authorized"),
        "scheduler_task_ids": payload.get("scheduler_task_ids"),
    }


def _write_json(path: Path, payload: dict) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.registration)
    _write_json(args.output, report)
    print(args.output.expanduser().resolve())
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
