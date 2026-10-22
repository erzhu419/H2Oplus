#!/usr/bin/env python3
"""Compose frozen v7 collection manifests without moving their shards."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import sha256
from reproducibility.run_collection_matrix_v7 import MATRIX_SCHEMA, POLICIES


COMPOSITION_SCHEMA = "transit-smdp-v7-composed-collection-matrix-v1"
CANONICAL_POLICIES = {policy["policy_name"]: policy for policy in POLICIES}
CANONICAL_POLICY_ORDER = tuple(CANONICAL_POLICIES)
SHARED_FIELDS = (
    "schema_version",
    "study_stage",
    "environment_design",
    "collector",
    "collector_sha256",
    "runner_sha256",
    "source_sha256",
    "sumo_version",
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def validate_and_select_manifest(
    manifest: dict[str, Any], *, block_ids: set[int] | None = None
) -> dict[str, Any]:
    """Validate one component and optionally retain complete realization blocks."""

    if manifest.get("schema_version") != MATRIX_SCHEMA:
        raise ValueError("composition requires collection matrix v2 components")
    if manifest.get("study_stage") != "formal":
        raise ValueError("composition requires formal collection components")
    design = manifest.get("environment_design")
    if not isinstance(design, dict):
        raise ValueError("collection component lacks an environment design")
    demand_scales = {float(value) for value in design.get("demand_scales", ())}
    if not demand_scales:
        raise ValueError("collection component has no demand scales")
    mixture = manifest.get("behavior_mixture")
    if not isinstance(mixture, dict):
        raise ValueError("collection component lacks its behavior mixture")
    if mixture.get("policies") != list(POLICIES):
        raise ValueError("collection component policy definitions changed")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("collection component contains no jobs")

    outputs: set[str] = set()
    signatures: set[str] = set()
    cells: set[tuple[int, str]] = set()
    environments: dict[int, tuple[Any, ...]] = {}
    policy_names: set[str] = set()
    for job in jobs:
        block_id = int(job["block_id"])
        policy = job.get("policy")
        policy_name = policy.get("policy_name") if isinstance(policy, dict) else None
        if policy_name not in CANONICAL_POLICIES or policy != CANONICAL_POLICIES[policy_name]:
            raise ValueError("collection component contains an unknown policy definition")
        output = str(Path(job["output"]).expanduser().resolve())
        signature = str(job.get("signature", ""))
        cell = (block_id, policy_name)
        if output in outputs or not signature or signature in signatures or cell in cells:
            raise ValueError("collection component repeats an output, signature, or cell")
        outputs.add(output)
        signatures.add(signature)
        cells.add(cell)
        policy_names.add(policy_name)
        scenario = job["scenario"]
        environment = (
            int(job["realization_block_id"]),
            str(scenario["sha256"]),
            int(scenario["selection_seed"]),
            float(scenario["demand_scale"]),
            int(job["sumo_seed"]),
        )
        previous = environments.setdefault(block_id, environment)
        if previous != environment:
            raise ValueError("policies disagree about a registered environment cell")
        if job.get("source_sha256") != manifest.get("source_sha256"):
            raise ValueError("component job source hashes disagree with its manifest")
        if job.get("sumo_version") != manifest.get("sumo_version"):
            raise ValueError("component job SUMO version disagrees with its manifest")

    declared_policies = mixture.get("selected_policy_names")
    expected_policy_names = [
        name for name in CANONICAL_POLICY_ORDER if name in policy_names
    ]
    if declared_policies != expected_policy_names:
        raise ValueError("component selected-policy declaration is inconsistent")
    expected_cells = {
        (block_id, policy_name)
        for block_id in environments
        for policy_name in policy_names
    }
    if cells != expected_cells:
        raise ValueError("component does not contain every policy in every environment")
    if len(environments) != int(manifest.get("environment_cells", -1)):
        raise ValueError("component environment-cell count is inconsistent")

    realization_rows: dict[int, dict[float, int]] = {}
    realization_seeds: dict[int, tuple[int, int]] = {}
    for block_id, environment in environments.items():
        realization_id, _, demand_seed, demand_scale, sumo_seed = environment
        rows = realization_rows.setdefault(realization_id, {})
        if demand_scale in rows:
            raise ValueError("component duplicates a demand scale within a realization")
        rows[demand_scale] = block_id
        seeds = realization_seeds.setdefault(realization_id, (demand_seed, sumo_seed))
        if seeds != (demand_seed, sumo_seed):
            raise ValueError("component mixes demand or SUMO seeds within a realization")
    if len(realization_rows) != int(manifest.get("independent_realization_blocks", -1)):
        raise ValueError("component realization-block count is inconsistent")
    if any(set(rows) != demand_scales for rows in realization_rows.values()):
        raise ValueError("component contains an incomplete demand-scale realization")
    if len(set(realization_seeds.values())) != len(realization_seeds):
        raise ValueError("component reuses a demand/SUMO seed pair")

    if block_ids is None:
        selected_blocks = set(environments)
    else:
        selected_blocks = {int(value) for value in block_ids}
        if not selected_blocks or not selected_blocks <= set(environments):
            raise ValueError("requested block identifiers are absent from a component")
    selected_realizations = {
        environments[block_id][0] for block_id in selected_blocks
    }
    complete_selected_blocks = {
        block_id
        for realization_id in selected_realizations
        for block_id in realization_rows[realization_id].values()
    }
    if selected_blocks != complete_selected_blocks:
        raise ValueError("block selection must contain complete demand-scale realizations")

    selected = copy.deepcopy(manifest)
    selected["jobs"] = [
        job for job in selected["jobs"] if int(job["block_id"]) in selected_blocks
    ]
    selected["environment_cells"] = len(selected_blocks)
    selected["independent_realization_blocks"] = len(selected_realizations)
    selected["behavior_mixture"]["selected_policy_names"] = expected_policy_names
    if block_ids is not None:
        selected["selection"] = {
            "complete_realization_blocks": sorted(selected_realizations),
            "environment_cell_block_ids": sorted(selected_blocks),
        }
    return selected


def compose_payloads(
    components: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    block_ids: set[int] | None = None,
) -> dict[str, Any]:
    """Return one strict cross-root matrix from validated component payloads."""

    if len(components) < 2:
        raise ValueError("at least two frozen components are required")
    selected = [
        (validate_and_select_manifest(payload, block_ids=block_ids), descriptor)
        for payload, descriptor in components
    ]
    reference = selected[0][0]
    for manifest, _ in selected[1:]:
        for field in SHARED_FIELDS:
            if _canonical(manifest.get(field)) != _canonical(reference.get(field)):
                raise ValueError(f"collection components disagree on {field}")

    jobs: list[dict[str, Any]] = []
    seen_outputs: set[str] = set()
    seen_signatures: set[str] = set()
    seen_cells: set[tuple[int, str]] = set()
    block_environment: dict[int, tuple[Any, ...]] = {}
    policies: set[str] = set()
    for manifest, _ in selected:
        for job in manifest["jobs"]:
            block_id = int(job["block_id"])
            policy_name = job["policy"]["policy_name"]
            output = str(Path(job["output"]).expanduser().resolve())
            signature = str(job["signature"])
            cell = (block_id, policy_name)
            scenario = job["scenario"]
            environment = (
                int(job["realization_block_id"]),
                str(scenario["sha256"]),
                int(scenario["selection_seed"]),
                float(scenario["demand_scale"]),
                int(job["sumo_seed"]),
            )
            if output in seen_outputs or signature in seen_signatures or cell in seen_cells:
                raise ValueError("collection components overlap outputs, signatures, or cells")
            if block_environment.setdefault(block_id, environment) != environment:
                raise ValueError("collection components disagree on an environment cell")
            seen_outputs.add(output)
            seen_signatures.add(signature)
            seen_cells.add(cell)
            policies.add(policy_name)
            jobs.append(copy.deepcopy(job))

    expected_cells = {
        (block_id, policy_name)
        for block_id in block_environment
        for policy_name in policies
    }
    if seen_cells != expected_cells:
        raise ValueError("composed matrix is not a complete environment-policy product")
    realization_ids = {environment[0] for environment in block_environment.values()}
    policy_names = [name for name in CANONICAL_POLICY_ORDER if name in policies]
    payload = copy.deepcopy(reference)
    payload["created_utc"] = datetime.now(timezone.utc).isoformat()
    payload["environment_cells"] = len(block_environment)
    payload["independent_realization_blocks"] = len(realization_ids)
    payload["behavior_mixture"]["selected_policy_names"] = policy_names
    payload["jobs"] = sorted(
        jobs,
        key=lambda job: (
            CANONICAL_POLICY_ORDER.index(job["policy"]["policy_name"]),
            int(job["block_id"]),
        ),
    )
    payload.pop("workers", None)
    payload["composition"] = {
        "schema_version": COMPOSITION_SCHEMA,
        "components": [descriptor for _, descriptor in selected],
        "environment_cell_block_ids": sorted(block_environment),
        "complete_realization_blocks": sorted(realization_ids),
    }
    return payload


def _load_frozen_component(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.expanduser().resolve()
    audit_path = path.with_name(path.stem + ".audit.json")
    manifest = json.loads(path.read_text())
    audit = json.loads(audit_path.read_text())
    digest = sha256(path)
    frozen = audit.get("frozen_manifest", {})
    if (
        audit.get("passed") is not True
        or Path(frozen.get("path", "")).expanduser().resolve() != path
        or frozen.get("sha256") != digest
    ):
        raise ValueError(f"component lacks a matching passed freeze audit: {path}")
    return manifest, {
        "path": str(path),
        "sha256": digest,
        "audit_path": str(audit_path),
        "audit_sha256": sha256(audit_path),
        "audit_schema_version": audit.get("schema_version"),
    }


def compose(
    component_paths: list[Path], output: Path, *, block_ids: set[int] | None = None
) -> dict[str, Any]:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("composed v7 manifests are immutable")
    components = [_load_frozen_component(path) for path in component_paths]
    payload = compose_payloads(components, block_ids=block_ids)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with output.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-ids", type=int, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = compose(
        args.components,
        args.output,
        block_ids=set(args.block_ids) if args.block_ids is not None else None,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "sha256": sha256(args.output.expanduser().resolve()),
                "jobs": len(payload["jobs"]),
                "environment_cells": payload["environment_cells"],
                "independent_realization_blocks": payload[
                    "independent_realization_blocks"
                ],
                "policies": payload["behavior_mixture"]["selected_policy_names"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
