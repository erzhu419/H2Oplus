#!/usr/bin/env python3
"""Independently verify the sealed v18-v2 namespace-binding incident."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
SCHEMA = "transit-h2oplus-v18-empty-tail-reserve-incident-verification-v2"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheduler-root", type=Path, default=ROOT.parent.parent / "scheduleurm"
    )
    parser.add_argument(
        "--closeout",
        type=Path,
        default=REPRO
        / (
            "h2oplus_v18_empty_tail_reserve_results/evaluation_incidents/"
            "v2_namespace_binding/incident_closeout_v2.json"
        ),
    )
    parser.add_argument("--expected-closeout-sha256", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPRO
        / (
            "h2oplus_v18_empty_tail_reserve_results/evaluation_incidents/"
            "v2_namespace_binding/incident_verification_v2.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    closeout_path = args.closeout.expanduser().resolve(strict=True)
    closeout = json.loads(closeout_path.read_text())
    if digest(closeout_path) != args.expected_closeout_sha256:
        raise ValueError("incident closeout identity drifted")
    inventory = closeout.get("remote_incident_inventory", {})
    if (
        closeout.get("status") != "sealed-invalid-zero-result-incident"
        or closeout.get("completed_scientific_result_count") != 0
        or closeout.get("scientific_numeric_outcomes_read") is not False
        or closeout.get("v18_v2_results_reusable") is not False
        or inventory.get("result_file_count") != 0
    ):
        raise ValueError("incident closeout scope drifted")

    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    sys.path.insert(0, str(scheduler_root))
    scheduler = importlib.import_module("skill.scheduler")
    remote_root = inventory["root"]
    expected_entries = inventory["entries"]
    status_path = closeout["backend_exit"]["path"]
    remote_code = r'''
import hashlib
import json
from pathlib import Path

root = Path(%r).resolve(strict=True)
status = Path(%r)

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()

entries = [
    {'path': str(path.relative_to(root)), 'sha256': digest(path), 'size_bytes': path.stat().st_size}
    for path in sorted(item for item in root.rglob('*') if item.is_file())
]
print(json.dumps({
    'entries': entries,
    'result_file_count': len(list(root.rglob('sumo_seed*.json'))),
    'evaluator_log_count': len(list(root.rglob('sumo_seed*.log'))),
    'status_text': status.read_text(errors='replace').strip(),
}, sort_keys=True))
''' % (remote_root, status_path)
    command = (
        "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python -c "
        + shlex.quote(remote_code)
    )
    returncode, stdout, stderr = scheduler.run_on(
        "jtl110gpu", command, timeout=120, check=False
    )
    observation = json.loads(stdout.strip()) if returncode == 0 else {}
    passed = bool(
        returncode == 0
        and observation.get("entries") == expected_entries
        and observation.get("result_file_count") == 0
        and observation.get("evaluator_log_count") == 45
        and observation.get("status_text") == closeout["backend_exit"]["observed"]
    )
    payload = {
        "schema_version": SCHEMA,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "closeout": {
            "path": str(closeout_path),
            "sha256": args.expected_closeout_sha256,
        },
        "formal_held_out_outcomes_read": False,
        "scientific_numeric_outcomes_read": False,
        "remote_returncode": returncode,
        "remote_stderr": stderr.strip(),
        "observed_entry_count": len(observation.get("entries", [])),
        "observed_result_file_count": observation.get("result_file_count"),
        "observed_evaluator_log_count": observation.get("evaluator_log_count"),
        "v18_v2_results_reusable": False,
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite incident verification: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
