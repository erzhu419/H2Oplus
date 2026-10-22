#!/usr/bin/env python3
"""Build the frozen logical-to-physical policy map for v11 evaluation."""

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
LOGICAL_ALGORITHMS = (
    "h2oplus",
    "sac-online",
    "h2oplus-v10-delayed-actor",
    "h2oplus-v11-actor-hold",
)
SCHEMA = "transit-evaluation-methods-v11-actor-hold-v1"
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


def descriptor(path: Path, owner: Path | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    value = str(resolved)
    if owner is not None:
        value = os.path.relpath(resolved, owner.parent)
    return {
        "path": value,
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
    actual = sha256(resolved)
    if actual != expected:
        raise ValueError(f"identity drift: {resolved}: {actual} != {expected}")
    payload = json.loads(resolved.read_text())
    if payload.get("schema_version") != schema:
        raise ValueError(f"schema drift: {resolved}")
    return resolved, payload


def load_policy(path: Path, tensor_key: str) -> tuple[dict[str, Any], str]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("artifact_schema") != "transit-policy-artifact-v7-v1":
        raise ValueError(f"policy artifact schema drifted: {path}")
    extra = artifact.get("extra", {})
    digest = actor_digest(artifact.get("actor", {}))
    if extra.get(tensor_key) != digest:
        raise ValueError(f"actor tensor digest drifted: {path}")
    return extra, digest


def v8_row(manifest: Path, raw: dict[str, Any]) -> dict[str, Any]:
    checkpoint = resolve(manifest, raw["checkpoint"])
    extra, tensor_hash = load_policy(checkpoint, "fewshot_actor_tensor_sha256")
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
    mismatches = {
        key: (extra.get(key), value)
        for key, value in expected.items()
        if extra.get(key) != value
    }
    admission = extra.get("training_admission", {})
    if (
        mismatches
        or admission.get("schema_version")
        != "transit-training-admission-v8-fewshot-development-v1"
        or admission.get("sha256") != raw["training_admission_sha256"]
        or extra.get("fewshot_protocol", {}).get("sha256") != V8_PROTOCOL_SHA256
        or sha256(checkpoint) != raw.get("checkpoint_sha256", sha256(checkpoint))
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


def v10_row(manifest: Path, raw: dict[str, Any], protocol_hash: str) -> dict[str, Any]:
    checkpoint = resolve(manifest, raw["checkpoint"])
    extra, tensor_hash = load_policy(checkpoint, "v10_actor_tensor_sha256")
    algorithm = raw["algorithm"]
    variant = algorithm[len("h2oplus-v10-") :]
    expected = {
        "method": "h2oplus",
        "variant": variant,
        "training_seed": raw["training_seed"],
        "run_signature": raw["training_run_signature"],
        "training_checkpoint_sha256": raw["training_checkpoint_sha256"],
        "v10_online_events_nominal": raw["online_events"],
        "v10_online_events_actual": raw["actual_online_events"],
        "v10_learner_update_steps": raw["learner_update_steps"],
        "v10_ratio_optimizer_steps": raw["ratio_optimizer_steps"],
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
    }
    mismatches = {
        key: (extra.get(key), value)
        for key, value in expected.items()
        if extra.get(key) != value
    }
    admission = extra.get("training_admission", {})
    intervention = extra.get("v10_intervention", {})
    if (
        mismatches
        or admission.get("schema_version")
        != "transit-training-admission-v10-delayed-development-v4"
        or admission.get("sha256") != raw["training_admission_sha256"]
        or extra.get("v10_protocol", {}).get("sha256") != protocol_hash
        or intervention.get("variant") != variant
        or intervention.get("unit_ratio_weights") is not True
        or tensor_hash != raw["actor_tensor_sha256"]
        or sha256(checkpoint) != raw["checkpoint_sha256"]
    ):
        raise ValueError(f"v10 policy provenance drifted: {checkpoint}: {mismatches}")
    return {**raw, "checkpoint": checkpoint}


def v11_row(path: Path, events: int, protocol_hash: str) -> tuple[dict[str, Any], dict[str, Any]]:
    extra, tensor_hash = load_policy(path, "v11_actor_tensor_sha256")
    expected_phase = {
        0: "offline_initializer_loaded",
        5_000: "online_warmup",
        10_000: "online_training",
        25_000: "complete",
    }[events]
    expected_steps = {
        0: (50_000, 0),
        5_000: (50_000, 0),
        10_000: (55_000, 2_250),
        25_000: (70_000, 6_000),
    }[events]
    expected = {
        "method": "h2oplus",
        "variant": "actor-hold-online-only",
        "stage": "candidate",
        "phase": expected_phase,
        "v11_online_events_nominal": events,
        "v11_learner_update_steps": expected_steps[0],
        "v11_ratio_optimizer_steps": expected_steps[1],
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
    }
    mismatches = {
        key: (extra.get(key), value)
        for key, value in expected.items()
        if extra.get(key) != value
    }
    admission = extra.get("training_admission", {})
    intervention = extra.get("v11_intervention", {})
    expected_intervention = {
        "schema_version": "transit-h2oplus-v11-actor-hold-variant-v2",
        "variant": "actor-hold-online-only",
        "warmup_events": 5_000,
        "actor_and_temperature_hold_online_critic_updates": 5_000,
        "actor_and_temperature_hold_through_online_events": 10_000,
        "actor_state_source_after_release": "online-only",
        "online_update_order": "v7-actor-before-critic",
        "unit_ratio_weights": True,
        "checkpoint_selection_by_behavior": False,
    }
    intervention_mismatches = {
        key: (intervention.get(key), value)
        for key, value in expected_intervention.items()
        if intervention.get(key) != value
    }
    if (
        mismatches
        or intervention_mismatches
        or admission.get("schema_version")
        != "transit-training-admission-v11-actor-hold-development-v2"
        or extra.get("v11_protocol", {}).get("sha256") != protocol_hash
        or not isinstance(extra.get("v11_temperature_tensor_sha256"), str)
        or len(extra["v11_temperature_tensor_sha256"]) != 64
    ):
        raise ValueError(
            f"v11 policy provenance drifted: {path}: {mismatches}: {intervention_mismatches}"
        )
    row = {
        "name": (
            f"h2oplus-v11-actor-hold-seed{extra['training_seed']}"
            f"-events{events:09d}"
        ),
        "algorithm": "h2oplus-v11-actor-hold",
        "variant": "actor-hold-online-only",
        "training_seed": int(extra["training_seed"]),
        "type": "checkpoint",
        "checkpoint": path.resolve(strict=True),
        "checkpoint_sha256": sha256(path),
        "training_run_signature": extra["run_signature"],
        "training_admission_sha256": admission["sha256"],
        "training_checkpoint_sha256": extra["training_checkpoint_sha256"],
        "online_events": events,
        "actual_online_events": int(extra["v11_online_events_actual"]),
        "learner_update_steps": int(extra["v11_learner_update_steps"]),
        "ratio_optimizer_steps": int(extra["v11_ratio_optimizer_steps"]),
        "actor_tensor_sha256": tensor_hash,
        "temperature_tensor_sha256": extra["v11_temperature_tensor_sha256"],
        "artifact_lineage": "v11-actor-hold",
        "source_protocol_sha256": protocol_hash,
    }
    evidence = {
        "online_events": events,
        "actual_online_events": row["actual_online_events"],
        "actor_tensor_sha256": tensor_hash,
        "temperature_tensor_sha256": row["temperature_tensor_sha256"],
        "policy": descriptor(path),
    }
    return row, evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v8-methods",
        type=Path,
        default=REPRO / "h2oplus_v8_fewshot_results/evaluation_methods_v8_fewshot.json",
    )
    parser.add_argument(
        "--v10-methods",
        type=Path,
        default=REPRO / "h2oplus_v10_delayed_evaluation_methods_v1.json",
    )
    parser.add_argument(
        "--v10-evaluation-closeout",
        type=Path,
        default=REPRO / "h2oplus_v10_delayed_results/evaluation_closeout_v1.json",
    )
    parser.add_argument(
        "--v10-evaluation-verification",
        type=Path,
        default=REPRO / "h2oplus_v10_delayed_results/evaluation_closeout_verification_v1.json",
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=REPRO / "h2oplus_v11_actor_hold_results/training_v2",
    )
    parser.add_argument(
        "--training-closeout",
        type=Path,
        default=REPRO / "h2oplus_v11_actor_hold_results/training_closeout_v2.json",
    )
    parser.add_argument(
        "--training-verification",
        type=Path,
        default=REPRO / "h2oplus_v11_actor_hold_results/training_closeout_verification_v2.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPRO / "h2oplus_v11_actor_hold_protocol_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPRO / "h2oplus_v11_actor_hold_evaluation_methods_v1.json",
    )
    parser.add_argument("--expected-v8-methods-sha256", required=True)
    parser.add_argument("--expected-v10-methods-sha256", required=True)
    parser.add_argument("--expected-v10-evaluation-closeout-sha256", required=True)
    parser.add_argument("--expected-v10-evaluation-verification-sha256", required=True)
    parser.add_argument("--expected-training-closeout-sha256", required=True)
    parser.add_argument("--expected-training-verification-sha256", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    v8_path, v8 = bound_json(
        args.v8_methods,
        args.expected_v8_methods_sha256,
        "transit-evaluation-methods-v8-fewshot-v1",
    )
    v10_path, v10 = bound_json(
        args.v10_methods,
        args.expected_v10_methods_sha256,
        "transit-evaluation-methods-v10-delayed-v1",
    )
    v10_closeout_path, v10_closeout = bound_json(
        args.v10_evaluation_closeout,
        args.expected_v10_evaluation_closeout_sha256,
        "transit-h2oplus-v10-delayed-evaluation-closeout-v1",
    )
    v10_verification_path, v10_verification = bound_json(
        args.v10_evaluation_verification,
        args.expected_v10_evaluation_verification_sha256,
        "transit-h2oplus-v10-delayed-evaluation-closeout-verification-v1",
    )
    closeout_path, closeout = bound_json(
        args.training_closeout,
        args.expected_training_closeout_sha256,
        "transit-h2oplus-v11-actor-hold-training-closeout-v2",
    )
    verification_path, verification = bound_json(
        args.training_verification,
        args.expected_training_verification_sha256,
        "transit-h2oplus-v11-actor-hold-training-closeout-verification-v2",
    )
    protocol_path, protocol = bound_json(
        args.protocol,
        args.expected_protocol_sha256,
        "transit-h2oplus-v11-actor-hold-protocol-v2",
    )
    if (
        v8.get("fewshot_protocol_sha256") != V8_PROTOCOL_SHA256
        or v8.get("formal_held_out_outcomes_read") is not False
        or v10.get("physical_method_count") != 60
        or v10.get("logical_method_budget_alias_count") != 140
        or v10.get("formal_held_out_outcomes_read") is not False
        or v10.get("development_evaluation_outcomes_read") is not False
        or v10_closeout.get("passed") is not True
        or v10_closeout.get("artifact_integrity", {}).get("passed") is not True
        or v10_closeout.get("formal_held_out_outcomes_read") is not False
        or v10_verification.get("passed") is not True
        or v10_verification.get("artifact_integrity_verified") is not True
        or closeout.get("artifact_integrity_passed") is not True
        or closeout.get("all_5_training_runs_stability_gate_passed") is not True
        or closeout.get("development_evaluation_release_gate_open") is not True
        or closeout.get("development_evaluation_submitted") is not False
        or closeout.get("formal_held_out_outcomes_read") is not False
        or verification.get("passed") is not True
        or verification.get("artifact_integrity_verified") is not True
        or verification.get("scientific_stability_verified_separately") is not True
        or protocol.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("sealed training, controls, or protocol scope drifted")

    v8_sources: dict[tuple[str, int, int], dict[str, Any]] = {}
    for raw in v8.get("methods", []):
        key = (raw.get("algorithm"), raw.get("training_seed"), raw.get("online_events"))
        if key[0] in {"h2oplus", "sac-online"} and key[1] in SEEDS and key[2] in {
            0,
            10_000,
            25_000,
        }:
            v8_sources[key] = v8_row(v8_path, raw)
    if len(v8_sources) != 30:
        raise ValueError(f"v8 control inventory drifted: {len(v8_sources)}")

    v10_methods = {row["name"]: row for row in v10.get("methods", [])}
    v10_aliases = {
        (row["training_seed"], row["online_events"]): row
        for row in v10.get("aliases", [])
        if row.get("algorithm") == "h2oplus-v10-delayed-actor"
    }
    if len(v10_aliases) != 20 or len(v10_methods) != 60:
        raise ValueError("v10 delayed-actor inventory drifted")

    output = args.output.expanduser().resolve()
    training_root = args.training_root.expanduser().resolve(strict=True)
    physical_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    aliases: list[dict[str, Any]] = []
    identity_evidence: list[dict[str, Any]] = []

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
                control_rows[(algorithm, events)] = add_physical(
                    v8_sources[(algorithm, seed, events)]
                )
            control_rows[(algorithm, 5_000)] = control_rows[(algorithm, 0)]

        v10_rows = {}
        for events in EVENTS:
            alias = v10_aliases[(seed, events)]
            raw = v10_methods[alias["physical_method"]]
            if raw["actor_tensor_sha256"] != alias["actor_tensor_sha256"]:
                raise ValueError(f"v10 alias tensor identity drifted: {alias['name']}")
            if raw["artifact_lineage"] == "v8-fewshot":
                physical = control_rows[("h2oplus", 0)]
                if physical["actor_tensor_sha256"] != raw["actor_tensor_sha256"]:
                    raise ValueError(f"v10/v8 initializer identity drifted for seed {seed}")
            else:
                physical = add_physical(
                    v10_row(v10_path, raw, v10["v10_protocol_sha256"])
                )
            v10_rows[events] = physical

        candidate_dir = (
            training_root
            / f"h2oplus_v11a2_actor_hold_online_only_seed{seed}"
            / "v11_policies"
        )
        candidate_rows = {}
        candidate_evidence = []
        for events in EVENTS:
            source, evidence = v11_row(
                candidate_dir / f"online_events_{events:09d}.pt",
                events,
                args.expected_protocol_sha256,
            )
            if source["training_seed"] != seed:
                raise ValueError(f"v11 seed identity drifted: {source['name']}")
            candidate_rows[events] = add_physical(source)
            evidence["policy"] = descriptor(
                candidate_dir / f"online_events_{events:09d}.pt", output
            )
            candidate_evidence.append(evidence)

        held_actor_hashes = {
            candidate_evidence[index]["actor_tensor_sha256"] for index in range(3)
        }
        held_temperature_hashes = {
            candidate_evidence[index]["temperature_tensor_sha256"]
            for index in range(3)
        }
        initializer_hash = control_rows[("h2oplus", 0)]["actor_tensor_sha256"]
        if (
            len(held_actor_hashes) != 1
            or len(held_temperature_hashes) != 1
            or next(iter(held_actor_hashes)) != initializer_hash
            or any(
                candidate_rows[events]["name"]
                != control_rows[("h2oplus", 0)]["name"]
                for events in (0, 5_000, 10_000)
            )
        ):
            raise ValueError(f"v11 actor/temperature hold identity drifted for seed {seed}")

        source_rows = {
            "h2oplus": control_rows,
            "sac-online": control_rows,
            "h2oplus-v10-delayed-actor": v10_rows,
            "h2oplus-v11-actor-hold": candidate_rows,
        }
        for algorithm in LOGICAL_ALGORITHMS:
            for events in EVENTS:
                if algorithm in {"h2oplus", "sac-online"}:
                    physical = control_rows[(algorithm, events)]
                else:
                    physical = source_rows[algorithm][events]
                alias = {
                    "name": f"{algorithm}-seed{seed}-events{events:09d}",
                    "algorithm": algorithm,
                    "training_seed": seed,
                    "online_events": events,
                    "physical_method": physical["name"],
                    "actor_tensor_sha256": physical["actor_tensor_sha256"],
                }
                if algorithm == "h2oplus-v11-actor-hold":
                    evidence = candidate_evidence[EVENTS.index(events)]
                    alias.update(
                        {
                            "source_artifact_lineage": "v11-actor-hold",
                            "source_checkpoint_sha256": evidence["policy"]["sha256"],
                            "source_temperature_tensor_sha256": evidence[
                                "temperature_tensor_sha256"
                            ],
                        }
                    )
                aliases.append(alias)

        seed_aliases = [row for row in aliases if row["training_seed"] == seed]
        seed_physical = [
            row for (row_seed, _), row in physical_by_key.items() if row_seed == seed
        ]
        identity_evidence.append(
            {
                "training_seed": seed,
                "logical_aliases": len(seed_aliases),
                "physical_actor_tensors": len(seed_physical),
                "v10_zero_and_5k_equal_v8_h2oplus_zero": all(
                    v10_rows[event]["actor_tensor_sha256"] == initializer_hash
                    for event in (0, 5_000)
                ),
                "v11_zero_5k_10k_equal_v8_h2oplus_zero": True,
                "v11_zero_5k_10k_temperature_equal": True,
                "v11_source_policies": candidate_evidence,
            }
        )

    physical = sorted(physical_by_key.values(), key=lambda row: row["name"])
    if len(aliases) != 80 or len(physical) != 45:
        raise ValueError(
            f"evaluation identity arithmetic drifted: {len(aliases)} aliases, "
            f"{len(physical)} actors"
        )
    if any(
        row["logical_aliases"] != 16
        or row["physical_actor_tensors"] != 9
        or row["v10_zero_and_5k_equal_v8_h2oplus_zero"] is not True
        for row in identity_evidence
    ):
        raise ValueError("per-seed identity arithmetic drifted")

    payload = {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": "development",
        "selection_rule": SELECTION_RULE,
        "formal_held_out_outcomes_read": False,
        "development_evaluation_outcomes_read": False,
        "v8_protocol_sha256": V8_PROTOCOL_SHA256,
        "v10_protocol_sha256": v10["v10_protocol_sha256"],
        "v11_protocol_sha256": args.expected_protocol_sha256,
        "logical_method_budget_alias_count": len(aliases),
        "physical_method_count": len(physical),
        "expected_remote_sumo_episodes": len(physical) * 3 * 3,
        "identity_evidence": identity_evidence,
        "bindings": {
            "v8_methods": descriptor(v8_path, output),
            "v10_methods": descriptor(v10_path, output),
            "v10_evaluation_closeout": descriptor(v10_closeout_path, output),
            "v10_evaluation_closeout_verification": descriptor(
                v10_verification_path, output
            ),
            "v11_protocol": descriptor(protocol_path, output),
            "v11_training_closeout": descriptor(closeout_path, output),
            "v11_training_closeout_verification": descriptor(verification_path, output),
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
                "temperature_tensor_sha256",
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
