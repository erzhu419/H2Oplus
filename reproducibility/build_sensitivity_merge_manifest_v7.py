#!/usr/bin/env python3
"""Build the fail-closed file manifest for formal v7 H2O+ sensitivities."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproducibility import build_ablation_merge_manifest_v7 as base
from reproducibility.audit_sensitivity_stability_v7 import audit


METHODS = (
    "h2oplus-lambda0",
    "h2oplus-lambda0p2",
    "h2oplus-lambda0p5",
    "h2oplus-lambda1",
)


def _configure() -> None:
    base.SCHEMA_VERSION = "transit-formal-sensitivity-merge-manifest-v7-v1"
    base.METHODS = METHODS
    base.ANALYSIS_METHODS = ("h2oplus",) + METHODS
    base.EXPECTED_JOBS = {
        f"{method}_seed{seed}" for method in METHODS for seed in base.SEEDS
    }
    base.TASK_IDS = ("t82055", "t82056")


def _stable_payload(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if key != "created_utc"}


def build(args):
    _configure()
    manifest = base.build(args)
    stability_path = args.stability_audit.expanduser().resolve()
    sync_path = args.partition_sync_evidence.expanduser().resolve()
    stability = json.loads(stability_path.read_text())
    sync = json.loads(sync_path.read_text())
    recomputed = audit(args.matrix_root, args.plan)
    errors = manifest["errors"]
    if _stable_payload(stability) != _stable_payload(recomputed):
        errors.append("sensitivity stability audit differs from the merged artifacts")
    if stability.get("artifact_integrity_implication") != "none":
        errors.append("sensitivity stability audit confuses stability and integrity")
    if stability.get("supports_held_out_model_effect_claim") is not False:
        errors.append("sensitivity stability audit lacks held-out-effect disclaimer")
    if sync.get("passed") is not True or set(sync.get("hosts", {})) != {
        "jtl110gpu",
        "jtl110gpu2",
    }:
        errors.append("sensitivity source partitions lack exact rsync evidence")
    for host, record in sync.get("hosts", {}).items():
        if record.get("result_tree_checksum_differences") != []:
            errors.append(f"{host} result-tree checksum dry-run found differences")
        if record.get("execution_tree_checksum_differences") != []:
            errors.append(f"{host} execution-tree checksum dry-run found differences")
        if record.get("backend_exit_exact") is not True:
            errors.append(f"{host} backend exit file differs from its source")
    manifest.update(
        {
            "campaign": "H2O+ v7 formal-v2 training-v7r1 registered sensitivities",
            "scientific_stability": {
                **base._descriptor(stability_path),
                "passed": stability.get("scientific_stability_passed"),
                "failed_runs": stability.get("failed_runs"),
                "artifact_integrity_implication": stability.get(
                    "artifact_integrity_implication"
                ),
            },
            "partition_sync_evidence": base._descriptor(sync_path),
        }
    )
    manifest["status"] = "verified" if not errors else "failed"
    manifest["artifact_integrity_verified"] = not errors
    return manifest


def main() -> None:
    parser = base.argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dispatch", type=Path, required=True)
    parser.add_argument("--matrix-audit", type=Path, required=True)
    parser.add_argument("--mechanism-analysis", type=Path, required=True)
    parser.add_argument("--scheduler-evidence", type=Path, required=True)
    parser.add_argument("--stability-audit", type=Path, required=True)
    parser.add_argument("--partition-sync-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args)
    base._write_json(args.output.expanduser().resolve(), manifest)
    print(args.output.expanduser().resolve())
    if manifest["status"] != "verified":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
