#!/usr/bin/env python3
"""Merge verified local/server v7 policy exports into one formal method freeze."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproducibility.build_training_plan_v7 import FORMAL_SEEDS
from reproducibility.export_evaluation_policies_v7 import (
    EXPORT_SCHEMA,
)
from reproducibility.run_evaluation_matrix_v4 import (
    METHODS_SCHEMAS,
    V7_POLICY_SELECTION_RULE,
    load_methods,
)
from SimpleSAC.train_transit_v3 import sha256
from SimpleSAC.transit_v3.protocols import METHOD_PROFILES
from SimpleSAC.transit_v7.trainer import (
    POLICY_ARTIFACT_SCHEMA,
    load_actor_checkpoint_v7,
)


MERGE_SCHEMA = "transit-evaluation-methods-merge-v7-v1"
OPERATIONAL_REFERENCES = (
    {
        "name": "zero-hold",
        "algorithm": "zero-hold",
        "training_seed": None,
        "type": "zero",
    },
    *(
        {
            "name": f"headway-gain-{gain:g}",
            "algorithm": f"headway-gain-{gain:g}",
            "training_seed": None,
            "type": "headway-balance",
            "gain": gain,
        }
        for gain in (0.05, 0.1, 0.25, 0.5)
    ),
    {
        "name": "myopic-welfare-v2",
        "algorithm": "myopic-welfare-v2",
        "training_seed": None,
        "type": "myopic-welfare",
    },
)


def _export_method(
    export_root: Path,
    raw: dict[str, Any],
    output_parent: Path,
) -> dict[str, Any]:
    checkpoint_relative = Path(raw.get("checkpoint", ""))
    if checkpoint_relative.is_absolute() or ".." in checkpoint_relative.parts:
        raise ValueError("policy export contains an unsafe actor path")
    checkpoint = (export_root / checkpoint_relative).resolve()
    if export_root != checkpoint and export_root not in checkpoint.parents:
        raise ValueError("policy export actor escapes its export root")
    if (
        not checkpoint.is_file()
        or raw.get("checkpoint_sha256") != sha256(checkpoint)
    ):
        raise ValueError(f"exported actor is missing or stale: {checkpoint}")
    actor, extra = load_actor_checkpoint_v7(
        checkpoint, "cpu", require_policy_artifact=True
    )
    if not all(bool(torch.isfinite(parameter).all()) for parameter in actor.parameters()):
        raise ValueError(f"exported actor has non-finite parameters: {checkpoint}")
    expected = {
        "policy_artifact_schema": POLICY_ARTIFACT_SCHEMA,
        "phase": "complete",
        "method": raw.get("algorithm"),
        "training_seed": raw.get("training_seed"),
        "run_signature": raw.get("training_run_signature"),
        "training_checkpoint_sha256": raw.get("training_checkpoint_sha256"),
        "offline_initializer_consumed": raw.get(
            "offline_initializer_consumed"
        ),
        "offline_initializer_produced": raw.get(
            "offline_initializer_produced"
        ),
    }
    mismatches = {
        name: (extra.get(name), value)
        for name, value in expected.items()
        if extra.get(name) != value
    }
    if extra.get("training_admission", {}).get("sha256") != raw.get(
        "training_admission_sha256"
    ):
        mismatches["training_admission_sha256"] = (
            extra.get("training_admission", {}).get("sha256"),
            raw.get("training_admission_sha256"),
        )
    if mismatches:
        raise ValueError(f"exported actor provenance mismatch: {mismatches}")
    return {
        "name": raw["name"],
        "algorithm": raw["algorithm"],
        "training_seed": raw["training_seed"],
        "type": "checkpoint",
        "checkpoint": os.path.relpath(checkpoint, output_parent),
        "training_run_signature": raw["training_run_signature"],
        "training_admission_sha256": raw["training_admission_sha256"],
        "training_checkpoint_sha256": raw["training_checkpoint_sha256"],
        "training_plan_sha256": raw["training_plan_sha256"],
        "offline_initializer_consumed": raw.get(
            "offline_initializer_consumed"
        ),
        "offline_initializer_produced": raw.get(
            "offline_initializer_produced"
        ),
        "training_artifacts": raw.get("training_artifacts", {}),
    }


def merge_exports(
    export_manifests: Iterable[Path],
    output: Path,
    *,
    expected_methods: Iterable[str] = tuple(METHOD_PROFILES),
    expected_seeds: Iterable[int] = FORMAL_SEEDS,
) -> dict[str, Any]:
    destination = output.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to replace method freeze: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_identities = {
        (method, int(seed))
        for method in expected_methods
        for seed in expected_seeds
    }
    methods = []
    identities = set()
    admissions = set()
    evaluation_designs = set()
    export_descriptors = []
    for raw_manifest in export_manifests:
        manifest_path = raw_manifest.expanduser().resolve(strict=True)
        payload = json.loads(manifest_path.read_text())
        if (
            payload.get("schema_version") != EXPORT_SCHEMA
            or payload.get("selection_rule") != V7_POLICY_SELECTION_RULE
        ):
            raise ValueError(f"not a formal v7 policy export: {manifest_path}")
        export_root = manifest_path.parent
        admissions.add(payload.get("training_admission_sha256"))
        evaluation_designs.add(
            payload.get("evaluation_environment_design_sha256")
        )
        export_descriptors.append(
            {
                "path": os.path.relpath(manifest_path, destination.parent),
                "sha256": sha256(manifest_path),
                "bytes": manifest_path.stat().st_size,
                "hostname": payload.get("hostname"),
                "plans": payload.get("plans"),
            }
        )
        for raw_method in payload.get("methods", []):
            identity = (raw_method.get("algorithm"), raw_method.get("training_seed"))
            if identity in identities:
                raise ValueError(f"duplicate policy across exports: {identity}")
            identities.add(identity)
            methods.append(
                _export_method(export_root, raw_method, destination.parent)
            )
    if identities != expected_identities:
        missing = sorted(expected_identities - identities)
        extra = sorted(identities - expected_identities)
        raise ValueError(
            f"formal v7 policy set is not exact: missing={missing}, extra={extra}"
        )
    if len(admissions) != 1 or None in admissions:
        raise ValueError("policy exports do not share one formal admission")
    if len(evaluation_designs) != 1 or None in evaluation_designs:
        raise ValueError("policy exports do not share one evaluation design")
    methods.sort(key=lambda item: (item["algorithm"], item["training_seed"]))
    payload = {
        "schema_version": METHODS_SCHEMAS["v7"],
        "merge_schema_version": MERGE_SCHEMA,
        "study_stage": "formal",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": V7_POLICY_SELECTION_RULE,
        "training_admission_sha256": next(iter(admissions)),
        "evaluation_environment_design_sha256": next(iter(evaluation_designs)),
        "learned_policy_count": len(methods),
        "operational_reference_count": len(OPERATIONAL_REFERENCES),
        "operational_reference_rule": (
            "zero hold, every preregistered fixed gain, and the closed-form "
            "myopic policy are reported separately; held-out outcomes cannot "
            "select one fixed gain"
        ),
        "exports": export_descriptors,
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "methods": [*methods, *OPERATIONAL_REFERENCES],
    }
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    try:
        load_methods(temporary, METHODS_SCHEMAS["v7"])
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = merge_exports(args.export, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "sha256": sha256(args.output.expanduser().resolve()),
                "learned_policies": payload["learned_policy_count"],
                "operational_references": payload[
                    "operational_reference_count"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
