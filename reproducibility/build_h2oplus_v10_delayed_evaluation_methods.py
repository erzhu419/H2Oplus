#!/usr/bin/env python3
"""Build the frozen logical-to-physical policy map for v10 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
SEEDS = tuple(range(6201, 6206))
EVENTS = (0, 5_000, 10_000, 25_000)
V10_VARIANTS = (
    "unit-control",
    "delayed-replay",
    "delayed-actor",
    "delayed-order",
    "delayed-combined",
)
LOGICAL_ALGORITHMS = (
    "h2oplus",
    "sac-online",
    *(f"h2oplus-v10-{variant}" for variant in V10_VARIANTS),
)
SCHEMA = "transit-evaluation-methods-v10-delayed-v1"
SELECTION_RULE = (
    "all preregistered common-budget aliases fixed before evaluation; physical "
    "evaluation deduplicated only by exact actor tensor SHA256 within a seed"
)
V8_PROTOCOL_SHA256 = "26d4f5c4ca9f337c11886855658127ef93a35954ca58eb43710aa70350610294"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def actor_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
        digest.update(b"\n")
    return digest.hexdigest()


def descriptor(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def resolve(owner: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve(strict=True)


def bound_json(path: Path, expected: str, schema: str) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    if sha256(resolved) != expected:
        raise ValueError(f"identity drift: {resolved}")
    payload = json.loads(resolved.read_text())
    if payload.get("schema_version") != schema:
        raise ValueError(f"schema drift: {resolved}")
    return resolved, payload


def load_actor(path: Path, lineage: str) -> tuple[dict[str, Any], str]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("artifact_schema") != "transit-policy-artifact-v7-v1":
        raise ValueError(f"policy artifact schema drifted: {path}")
    extra = artifact.get("extra", {})
    digest = actor_digest(artifact.get("actor", {}))
    key = "fewshot_actor_tensor_sha256" if lineage == "v8-fewshot" else "v10_actor_tensor_sha256"
    if extra.get(key) != digest:
        raise ValueError(f"actor tensor digest drifted: {path}")
    return extra, digest


def v8_row(manifest: Path, raw: dict[str, Any]) -> dict[str, Any]:
    checkpoint = resolve(manifest, raw["checkpoint"])
    extra, tensor_hash = load_actor(checkpoint, "v8-fewshot")
    expected = {
        "method": raw["algorithm"],
        "training_seed": raw["training_seed"],
        "run_signature": raw["training_run_signature"],
        "training_checkpoint_sha256": raw["training_checkpoint_sha256"],
        "fewshot_online_events_nominal": raw["online_events"],
        "fewshot_online_events_actual": raw["actual_online_events"],
        "fewshot_learner_update_steps": raw["learner_update_steps"],
        "fewshot_ratio_optimizer_steps": raw["ratio_optimizer_steps"],
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
    }
    mismatches = {key: (extra.get(key), value) for key, value in expected.items() if extra.get(key) != value}
    admission = extra.get("training_admission", {})
    if (
        mismatches
        or admission.get("schema_version")
        != "transit-training-admission-v8-fewshot-development-v1"
        or admission.get("sha256") != raw["training_admission_sha256"]
        or extra.get("fewshot_protocol", {}).get("sha256") != V8_PROTOCOL_SHA256
    ):
        raise ValueError(f"v8 policy provenance drifted: {checkpoint}: {mismatches}")
    return {
        **raw,
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256(checkpoint),
        "actor_tensor_sha256": tensor_hash,
        "artifact_lineage": "v8-fewshot",
        "source_protocol_sha256": V8_PROTOCOL_SHA256,
    }


def v10_row(path: Path, algorithm: str, expected_variant: str, events: int, protocol_hash: str) -> dict[str, Any]:
    extra, tensor_hash = load_actor(path, "v10-delayed")
    admission = extra.get("training_admission", {})
    expected_stage = "prefix" if expected_variant == "shared-prefix" else "branch"
    expected = {
        "method": "h2oplus",
        "variant": expected_variant,
        "stage": expected_stage,
        "phase": "complete",
        "v10_online_events_nominal": events,
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
    }
    mismatches = {key: (extra.get(key), value) for key, value in expected.items() if extra.get(key) != value}
    if (
        mismatches
        or admission.get("schema_version")
        != "transit-training-admission-v10-delayed-development-v4"
        or extra.get("v10_protocol", {}).get("sha256") != protocol_hash
        or extra.get("v10_intervention", {}).get("variant") != expected_variant
    ):
        raise ValueError(f"v10 policy provenance drifted: {path}: {mismatches}")
    return {
        "name": f"{algorithm}-seed{extra['training_seed']}-events{events:09d}",
        "algorithm": algorithm,
        "variant": expected_variant,
        "training_seed": int(extra["training_seed"]),
        "type": "checkpoint",
        "checkpoint": path.resolve(strict=True),
        "checkpoint_sha256": sha256(path),
        "training_run_signature": extra["run_signature"],
        "training_admission_sha256": admission["sha256"],
        "training_checkpoint_sha256": extra["training_checkpoint_sha256"],
        "online_events": events,
        "actual_online_events": int(extra["v10_online_events_actual"]),
        "learner_update_steps": int(extra["v10_learner_update_steps"]),
        "ratio_optimizer_steps": int(extra["v10_ratio_optimizer_steps"]),
        "actor_tensor_sha256": tensor_hash,
        "artifact_lineage": "v10-delayed",
        "source_protocol_sha256": protocol_hash,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v8-methods",
        type=Path,
        default=REPRO / "h2oplus_v8_fewshot_results/evaluation_methods_v8_fewshot.json",
    )
    parser.add_argument(
        "--prefix-root",
        type=Path,
        default=REPRO / "h2oplus_v10_delayed_results/prefix_v4",
    )
    parser.add_argument(
        "--branch-root",
        type=Path,
        default=REPRO / "h2oplus_v10_delayed_results/branches_v4",
    )
    parser.add_argument(
        "--branch-closeout",
        type=Path,
        default=REPRO / "h2oplus_v10_delayed_results/branch_closeout_v4.json",
    )
    parser.add_argument(
        "--branch-verification",
        type=Path,
        default=REPRO / "h2oplus_v10_delayed_results/branch_closeout_verification_v4.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPRO / "h2oplus_v10_delayed_protocol_v4.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPRO / "h2oplus_v10_delayed_evaluation_methods_v1.json",
    )
    parser.add_argument("--expected-v8-methods-sha256", required=True)
    parser.add_argument("--expected-branch-closeout-sha256", required=True)
    parser.add_argument("--expected-branch-verification-sha256", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    v8_path, v8 = bound_json(
        args.v8_methods,
        args.expected_v8_methods_sha256,
        "transit-evaluation-methods-v8-fewshot-v1",
    )
    closeout_path, closeout = bound_json(
        args.branch_closeout,
        args.expected_branch_closeout_sha256,
        "transit-h2oplus-v10-delayed-branch-closeout-v4",
    )
    verification_path, verification = bound_json(
        args.branch_verification,
        args.expected_branch_verification_sha256,
        "transit-h2oplus-v10-delayed-branch-closeout-verification-v4",
    )
    protocol_path, protocol = bound_json(
        args.protocol,
        args.expected_protocol_sha256,
        "transit-h2oplus-v10-delayed-protocol-v4",
    )
    if (
        v8.get("fewshot_protocol_sha256") != V8_PROTOCOL_SHA256
        or v8.get("formal_held_out_outcomes_read") is not False
        or closeout.get("artifact_integrity_passed") is not True
        or closeout.get("development_evaluation_release_gate_open") is not True
        or closeout.get("development_evaluation_submitted") is not False
        or closeout.get("formal_held_out_outcomes_read") is not False
        or verification.get("passed") is not True
        or verification.get("artifact_integrity_verified") is not True
        or protocol.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("sealed training or protocol scope drifted")

    v8_sources: dict[tuple[str, int, int], dict[str, Any]] = {}
    for raw in v8.get("methods", []):
        key = (raw.get("algorithm"), raw.get("training_seed"), raw.get("online_events"))
        if key[0] in {"h2oplus", "sac-online"} and key[1] in SEEDS and key[2] in {0, 10_000, 25_000}:
            v8_sources[key] = v8_row(v8_path, raw)
    if len(v8_sources) != 30:
        raise ValueError(f"v8 control inventory drifted: {len(v8_sources)}")

    prefix_root = args.prefix_root.expanduser().resolve(strict=True)
    branch_root = args.branch_root.expanduser().resolve(strict=True)
    physical_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    aliases: list[dict[str, Any]] = []
    identity_evidence = []

    def add_physical(row: dict[str, Any]) -> dict[str, Any]:
        key = (int(row["training_seed"]), row["actor_tensor_sha256"])
        existing = physical_by_key.get(key)
        if existing is None:
            physical_by_key[key] = row
            return row
        return existing

    for seed in SEEDS:
        control_rows: dict[tuple[str, int], dict[str, Any]] = {}
        for algorithm in ("h2oplus", "sac-online"):
            for events in (0, 10_000, 25_000):
                control_rows[(algorithm, events)] = add_physical(v8_sources[(algorithm, seed, events)])
            control_rows[(algorithm, 5_000)] = control_rows[(algorithm, 0)]

        prefix_dir = prefix_root / f"h2oplus_v10c4_shared_prefix_seed{seed}" / "v10_policies"
        prefix_rows = {}
        for events in (0, 5_000, 10_000):
            path = prefix_dir / f"online_events_{events:09d}.pt"
            expected_phase = "complete" if events == 10_000 else (
                "offline_initializer_loaded" if events == 0 else "online_warmup"
            )
            extra, tensor_hash = load_actor(path, "v10-delayed")
            if (
                extra.get("training_seed") != seed
                or extra.get("variant") != "shared-prefix"
                or extra.get("stage") != "prefix"
                or extra.get("phase") != expected_phase
                or extra.get("v10_online_events_nominal") != events
                or extra.get("v10_protocol", {}).get("sha256")
                != args.expected_protocol_sha256
            ):
                raise ValueError(f"v10 prefix policy drifted: {path}")
            prefix_rows[events] = (path, tensor_hash)
        if not (
            prefix_rows[0][1]
            == prefix_rows[5_000][1]
            == control_rows[("h2oplus", 0)]["actor_tensor_sha256"]
        ):
            raise ValueError(f"v10/v8 shared initializer identity drifted for seed {seed}")
        shared_10k = add_physical(
            v10_row(
                prefix_rows[10_000][0],
                "h2oplus-v10-shared-prefix",
                "shared-prefix",
                10_000,
                args.expected_protocol_sha256,
            )
        )
        branch_rows = {}
        for variant in V10_VARIANTS:
            path = (
                branch_root
                / f"h2oplus_v10c4_{variant.replace('-', '_')}_seed{seed}"
                / "v10_policies/online_events_000025000.pt"
            )
            algorithm = f"h2oplus-v10-{variant}"
            branch_rows[algorithm] = add_physical(
                v10_row(path, algorithm, variant, 25_000, args.expected_protocol_sha256)
            )

        for algorithm in LOGICAL_ALGORITHMS:
            for events in EVENTS:
                if algorithm in {"h2oplus", "sac-online"}:
                    physical = control_rows[(algorithm, events)]
                elif events in (0, 5_000):
                    physical = control_rows[("h2oplus", 0)]
                elif events == 10_000:
                    physical = shared_10k
                else:
                    physical = branch_rows[algorithm]
                aliases.append(
                    {
                        "name": f"{algorithm}-seed{seed}-events{events:09d}",
                        "algorithm": algorithm,
                        "training_seed": seed,
                        "online_events": events,
                        "physical_method": physical["name"],
                        "actor_tensor_sha256": physical["actor_tensor_sha256"],
                    }
                )
        seed_aliases = [row for row in aliases if row["training_seed"] == seed]
        seed_physical = [row for (row_seed, _), row in physical_by_key.items() if row_seed == seed]
        identity_evidence.append(
            {
                "training_seed": seed,
                "logical_aliases": len(seed_aliases),
                "physical_actor_tensors": len(seed_physical),
                "v10_zero_and_5k_equal_v8_h2oplus_zero": True,
            }
        )

    physical = sorted(physical_by_key.values(), key=lambda row: row["name"])
    if len(aliases) != 140 or len(physical) != 60:
        raise ValueError(f"evaluation identity arithmetic drifted: {len(aliases)} aliases, {len(physical)} actors")
    if any(row["logical_aliases"] != 28 or row["physical_actor_tensors"] != 12 for row in identity_evidence):
        raise ValueError("per-seed identity arithmetic drifted")

    output = args.output.expanduser().resolve()
    payload = {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": "development",
        "selection_rule": SELECTION_RULE,
        "formal_held_out_outcomes_read": False,
        "development_evaluation_outcomes_read": False,
        "v8_protocol_sha256": V8_PROTOCOL_SHA256,
        "v10_protocol_sha256": args.expected_protocol_sha256,
        "logical_method_budget_alias_count": len(aliases),
        "physical_method_count": len(physical),
        "expected_remote_sumo_episodes": len(physical) * 3 * 3,
        "identity_evidence": identity_evidence,
        "bindings": {
            "v8_methods": descriptor(v8_path),
            "v10_protocol": descriptor(protocol_path),
            "v10_branch_closeout": descriptor(closeout_path),
            "v10_branch_closeout_verification": descriptor(verification_path),
        },
        "aliases": sorted(aliases, key=lambda row: row["name"]),
        "methods": [],
    }
    for raw in physical:
        row = {
            key: raw[key]
            for key in (
                "name",
                "algorithm",
                "variant",
                "training_seed",
                "type",
                "training_run_signature",
                "training_admission_sha256",
                "training_checkpoint_sha256",
                "online_events",
                "actual_online_events",
                "learner_update_steps",
                "ratio_optimizer_steps",
                "actor_tensor_sha256",
                "artifact_lineage",
                "source_protocol_sha256",
            )
            if key in raw
        }
        row["checkpoint"] = os.path.relpath(raw["checkpoint"], output.parent)
        row["checkpoint_sha256"] = raw["checkpoint_sha256"]
        payload["methods"].append(row)
    atomic_json(output, payload)
    print(
        json.dumps(
            {
                "methods_sha256": sha256(output),
                "logical_aliases": len(aliases),
                "physical_actors": len(physical),
                "remote_sumo_episodes": len(physical) * 9,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
