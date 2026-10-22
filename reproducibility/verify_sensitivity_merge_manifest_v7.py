#!/usr/bin/env python3
"""Independently verify the formal v7 H2O+ sensitivity merge manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproducibility import verify_ablation_merge_manifest_v7 as base
from reproducibility.audit_sensitivity_stability_v7 import audit


METHODS = (
    "h2oplus-lambda0",
    "h2oplus-lambda0p2",
    "h2oplus-lambda0p5",
    "h2oplus-lambda1",
)


def _configure() -> None:
    base.SCHEMA_VERSION = "transit-formal-sensitivity-manifest-verification-v7-v1"
    base.MANIFEST_SCHEMA = "transit-formal-sensitivity-merge-manifest-v7-v1"
    base.METHODS = METHODS
    base.ANALYSIS_METHODS = ("h2oplus",) + METHODS
    base.EXPECTED_RUNS = {
        f"{method}_seed{seed}" for method in METHODS for seed in base.SEEDS
    }
    base.TASKS = ("t82055", "t82056")


def _stable_payload(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if key != "created_utc"}


def verify(manifest_path: Path) -> dict:
    _configure()
    report = base.verify(manifest_path)
    manifest = json.loads(manifest_path.expanduser().resolve().read_text())
    errors = report["errors"]
    descriptor = manifest.get("scientific_stability", {})
    stability = base._load_descriptor("sensitivity stability audit", descriptor, errors)
    sync = base._load_descriptor(
        "sensitivity partition sync evidence",
        manifest.get("partition_sync_evidence", {}),
        errors,
    )
    if sync is not None:
        if sync.get("passed") is not True or set(sync.get("hosts", {})) != {
            "jtl110gpu",
            "jtl110gpu2",
        }:
            errors.append("partition sync evidence is incomplete")
        for host, record in sync.get("hosts", {}).items():
            if record.get("result_tree_checksum_differences") != []:
                errors.append(f"{host} result-tree checksum evidence is not exact")
            if record.get("execution_tree_checksum_differences") != []:
                errors.append(f"{host} execution-tree checksum evidence is not exact")
            if record.get("backend_exit_exact") is not True:
                errors.append(f"{host} backend-exit sync evidence is not exact")
    if stability is not None:
        frozen = manifest.get("frozen_inputs", {})
        plan_descriptor = frozen.get("plan", {})
        plan_path = Path(plan_descriptor.get("path", "")).expanduser().resolve()
        matrix_root = Path(
            manifest.get("merged_results", {}).get("root", "")
        ).expanduser().resolve()
        if plan_path.is_file() and matrix_root.is_dir():
            recomputed = audit(matrix_root, plan_path)
            if _stable_payload(stability) != _stable_payload(recomputed):
                errors.append(
                    "sensitivity stability audit differs from independent recomputation"
                )
        if descriptor.get("passed") != stability.get("scientific_stability_passed"):
            errors.append("manifest stability result differs from its audit")
        if descriptor.get("failed_runs") != stability.get("failed_runs"):
            errors.append("manifest stability failures differ from its audit")
        if descriptor.get("artifact_integrity_implication") != "none":
            errors.append("manifest confuses scientific stability with artifact integrity")
        if stability.get("supports_held_out_model_effect_claim") is not False:
            errors.append("stability audit lacks held-out-effect disclaimer")
    report["passed"] = not errors
    if errors:
        report["runs_verified"] = 0
        report["merged_files_verified"] = 0
        report["merged_bytes_verified"] = 0
        report["source_partitions_verified"] = []
    report["scientific_stability_verified_separately"] = not any(
        "stability" in error for error in errors
    )
    return report


def main() -> None:
    parser = base.argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.manifest)
    base._write_json(args.output, report)
    print(args.output.expanduser().resolve())
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
