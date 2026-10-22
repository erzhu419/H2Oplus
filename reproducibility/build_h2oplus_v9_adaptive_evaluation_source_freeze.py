#!/usr/bin/env python3
"""Freeze the v9 development evaluation implementation before any SUMO run."""

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

from bus_h2o.sumo_provenance import sha256


SCHEMA = "transit-h2oplus-v9-adaptive-evaluation-source-freeze-v1"
SOURCES = (
    "SimpleSAC/evaluate_transit_v9_adaptive.py",
    "reproducibility/run_evaluation_matrix_v9_adaptive.py",
    "reproducibility/run_h2oplus_v9_adaptive_evaluation_dispatch.py",
    "reproducibility/analyze_h2oplus_v9_adaptive.py",
)


def descriptor(path: Path) -> dict:
    resolved = path.expanduser().resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256(resolved), "bytes": resolved.stat().st_size}


def relative_descriptor(relative: str) -> dict:
    path = (ROOT / relative).resolve(strict=True)
    return {"path": relative, "sha256": sha256(path), "bytes": path.stat().st_size}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--environment-design", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--count-correction", type=Path, required=True)
    parser.add_argument("--training-closeout", type=Path, required=True)
    parser.add_argument("--training-verification", type=Path, required=True)
    parser.add_argument("--methods-manifest", type=Path, required=True)
    parser.add_argument("--source-correction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    controls = {
        "registration": descriptor(args.registration),
        "authorization": descriptor(args.authorization),
        "freeze": descriptor(args.freeze),
        "environment_design": descriptor(args.environment_design),
        "protocol": descriptor(args.protocol),
        "count_correction": descriptor(args.count_correction),
        "training_closeout": descriptor(args.training_closeout),
        "training_verification": descriptor(args.training_verification),
        "methods_manifest": descriptor(args.methods_manifest),
        "source_correction": descriptor(args.source_correction),
    }
    registration = json.loads(args.registration.expanduser().resolve().read_text())
    authorization = json.loads(args.authorization.expanduser().resolve().read_text())
    count = json.loads(args.count_correction.expanduser().resolve().read_text())
    closeout = json.loads(args.training_closeout.expanduser().resolve().read_text())
    verification = json.loads(args.training_verification.expanduser().resolve().read_text())
    methods = json.loads(args.methods_manifest.expanduser().resolve().read_text())
    source_correction = json.loads(args.source_correction.expanduser().resolve().read_text())
    if (
        registration.get("formal_held_out_outcomes_read") is not False
        or authorization.get("remote_development_evaluation_after_training_authorized") is not True
        or authorization.get("formal_held_out_access_authorized") is not False
        or count.get("performance_or_evaluation_outcomes_read") is not False
        or count.get("corrected_unique_actor_tensors") != 42
        or count.get("corrected_remote_sumo_episodes") != 252
        or closeout.get("artifact_integrity_passed") is not True
        or closeout.get("development_evaluation_may_proceed") is not True
        or verification.get("passed") is not True
        or methods.get("physical_method_count") != 42
        or methods.get("method_budget_alias_count") != 60
        or methods.get("formal_held_out_outcomes_read") is not False
        or source_correction.get("schema_version")
        != "transit-h2oplus-v9-adaptive-evaluation-source-correction-v2"
        or source_correction.get("formal_held_out_outcomes_read") is not False
        or source_correction.get("scientific_evaluation_outcomes_read") is not False
        or source_correction.get("changed_source_paths")
        != [
            "SimpleSAC/evaluate_transit_v9_adaptive.py",
            "reproducibility/run_evaluation_matrix_v9_adaptive.py",
        ]
    ):
        raise ValueError("v9 evaluation source freeze prerequisites drifted")
    payload = {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-development-evaluation",
        "study_stage": "development",
        "scope": "v9 adaptive development evaluation implementation and immutable inputs",
        "formal_held_out_outcomes_read": False,
        "evaluation_outcomes_read": False,
        "local_training_or_sumo_executed": False,
        "training_or_policy_artifacts_changed": False,
        "physical_actor_count": 42,
        "logical_method_budget_alias_count": 60,
        "remote_sumo_episode_count": 252,
        "workers_per_shard": 12,
        "shard_count": 2,
        "jobs_per_shard": 126,
        "sources": [relative_descriptor(relative) for relative in SOURCES],
        "source_correction": controls["source_correction"],
        "controls": controls,
        "interpretation": {
            "artifact_integrity_separate_from_scientific_result": True,
            "development_result_only": True,
            "supports_formal_held_out_model_effect_claim": False,
            "formal_confirmation_stage_authorized_or_submitted": False,
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "sha256": sha256(output), "sources": len(SOURCES)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
