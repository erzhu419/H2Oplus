#!/usr/bin/env python3
"""Build preregistered v6 gate, core, ablation, and sensitivity plans."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REPRODUCIBILITY = ROOT / "reproducibility"
DATASET = ROOT / "bus_h2o" / "datasets_v6" / "transit_smdp_v6_formal_v1.h5"
DATASET_AUDIT = DATASET.with_suffix(".audit.json")
SIMULATOR = ROOT / "bus_h2o" / "calibrated_env"
METHOD_AUDIT = REPRODUCIBILITY / "METHOD_PROTOCOL_AUDIT_V6.md"
TRAINING_DESIGN = REPRODUCIBILITY / "training_environment_design_v6_formal_v1.json"
EVALUATION_DESIGN = REPRODUCIBILITY / "evaluation_environment_design_v4_formal_v1.json"

PLAN_SCHEMA = "transit-training-plan-v6"
REGISTRATION_SCHEMA = "transit-training-registration-v1"
FORMAL_SEEDS = (5101, 5102, 5103, 5104, 5105)
FORMAL_ONLINE_EVENTS = 100_000
FORMAL_EVENTS_PER_EPOCH = 500
GATE_SEEDS = (9151,)
GATE_ONLINE_EVENTS = 64
GATE_WARMUP_EVENTS = 16
GATE_EVENTS_PER_EPOCH = 16

CORE_METHODS = (
    "h2oplus",
    "h2oplus-source",
    "rlpd",
    "wsrl",
    "sac-mixed",
    "sac-online",
    "cql",
    "iql",
    "bc",
)
ABLATION_METHODS = (
    "h2oplus-no-ratio",
    "h2oplus-no-warmstart",
)
SENSITIVITY_METHODS = (
    "h2oplus-lambda0",
    "h2oplus-lambda0p2",
    "h2oplus-lambda0p5",
    "h2oplus-lambda1",
    "ratio-sac",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, output_dir: Path) -> str:
    return os.path.relpath(path.resolve(), output_dir.resolve())


def _descriptor(path: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "path": _relative(path, output_dir),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def _method_arguments(
    method: str,
    *,
    formal: bool,
    total_online_events: int,
    events_per_epoch: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    from SimpleSAC.transit_v3.protocols import METHOD_PROFILES

    profile = METHOD_PROFILES[method]
    uses_online = profile.online_update_mode != "none"
    if formal:
        warmup_events = profile.default_warmup_events if uses_online else 0
        offline_updates = profile.default_offline_updates
        ratio_pretrain_updates = profile.default_ratio_pretrain_updates
    else:
        warmup_events = GATE_WARMUP_EVENTS if uses_online else 0
        offline_updates = 8 if profile.offline_objective != "none" else 0
        ratio_pretrain_updates = 4 if profile.use_ratio_correction else 0

    training_events = total_online_events - warmup_events if uses_online else 0
    if training_events < 0 or training_events % events_per_epoch:
        raise ValueError(
            f"{method}: post-warmup events {training_events} are not a nonnegative "
            f"multiple of {events_per_epoch}"
        )
    epochs = training_events // events_per_epoch
    updates_per_epoch = events_per_epoch * profile.utd_ratio if uses_online else 0
    online_critic_updates = epochs * updates_per_epoch
    online_actor_updates = (
        online_critic_updates // profile.actor_update_interval if uses_online else 0
    )
    arguments = {
        "offline_updates": offline_updates,
        "warmup_events": warmup_events,
        "ratio_pretrain_updates": ratio_pretrain_updates,
        "epochs": epochs,
        "updates_per_epoch": updates_per_epoch,
        "utd_ratio": profile.utd_ratio,
        "batch_size": profile.batch_size,
        "ensemble_size": profile.ensemble_size,
        "online_fraction": profile.online_fraction,
        "target_entropy": profile.target_entropy,
        "initial_temperature": profile.initial_temperature,
        "iql_advantage_temperature": profile.iql_advantage_temperature,
        "offline_cql_alpha": profile.offline_cql_alpha,
        "ratio_update_interval": profile.ratio_update_interval,
    }
    ratio_interval = profile.ratio_update_interval
    ratio_start = offline_updates
    ratio_stop = offline_updates + online_critic_updates
    online_ratio_updates = (
        (ratio_stop - 1) // ratio_interval
        - (ratio_start - 1) // ratio_interval
        if uses_online and profile.use_ratio_correction and online_critic_updates
        else 0
    )
    budget = {
        "offline_gradient_updates": offline_updates,
        "warmup_simulator_events": warmup_events,
        "training_simulator_events": training_events,
        "total_simulator_events": warmup_events + training_events,
        "online_critic_updates": online_critic_updates,
        "online_actor_updates": online_actor_updates,
        "ratio_pretrain_updates": ratio_pretrain_updates,
        "online_ratio_updates": online_ratio_updates,
    }
    expected_total = total_online_events if uses_online else 0
    if budget["total_simulator_events"] != expected_total:
        raise AssertionError(f"{method}: interaction budget does not close")
    return arguments, budget


def build_jobs(
    methods: Iterable[str],
    seeds: Iterable[int],
    *,
    formal: bool,
    total_online_events: int,
    events_per_epoch: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    from SimpleSAC.transit_v3.protocols import METHOD_PROFILES

    jobs: list[dict[str, Any]] = []
    budgets: dict[str, dict[str, int]] = {}
    methods = tuple(methods)
    seeds = tuple(seeds)
    missing = set(methods) - set(METHOD_PROFILES)
    if missing:
        raise KeyError(f"unknown registered methods: {sorted(missing)}")
    for method in methods:
        arguments, budget = _method_arguments(
            method,
            formal=formal,
            total_online_events=total_online_events,
            events_per_epoch=events_per_epoch,
        )
        budgets[method] = budget
        for seed in seeds:
            jobs.append(
                {
                    "name": f"{method}_seed{seed}",
                    "method": method,
                    "seed": seed,
                    "arguments": dict(arguments),
                }
            )
    names = [job["name"] for job in jobs]
    if len(names) != len(set(names)):
        raise AssertionError("generated duplicate training job names")
    return jobs, budgets


def _registration(
    *,
    plan_id: str,
    role: str,
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
    budgets: dict[str, dict[str, int]],
    output_dir: Path,
    created_utc: str,
    formal: bool,
    generator_snapshot: Path,
) -> dict[str, Any]:
    from SimpleSAC.transit_v3.protocols import METHOD_PROFILES

    return {
        "schema_version": REGISTRATION_SCHEMA,
        "plan_id": plan_id,
        "status": "frozen-before-formal-outcomes" if formal else "execution-gate",
        "created_utc": created_utc,
        "scientific_role": role,
        "statistical_unit": "independent training seed evaluated on paired, disjoint SUMO scenario cells",
        "primary_outcome": "system passenger time divided by completed passengers over the fixed 24000-second horizon",
        "online_budget_semantics": (
            "total simulator decision events include the frozen-policy warmup; "
            "post-warmup critic UTD follows each registered source protocol"
        ),
        "comparison_disclosure": (
            "online interaction budgets are equal across online methods, but gradient "
            "budgets differ by registered UTD and pretraining protocol"
        ),
        "compute_policy": {
            "observation_cache_storage": "float16",
            "network_compute": "float32",
            "compiled_repeated_mlp_kernels": True,
            "shared_gpu_allowed": False,
        },
        "seeds": list(seeds),
        "methods": {
            method: {
                "description": METHOD_PROFILES[method].description,
                "profile": asdict(METHOD_PROFILES[method]),
                "budget": budgets[method],
            }
            for method in methods
        },
        "evidence": {
            "dataset_audit": _descriptor(DATASET_AUDIT, output_dir),
            "method_protocol_audit": _descriptor(METHOD_AUDIT, output_dir),
            "training_environment_design": _descriptor(TRAINING_DESIGN, output_dir),
            "evaluation_environment_design": _descriptor(EVALUATION_DESIGN, output_dir),
            "generator": _descriptor(generator_snapshot, output_dir),
        },
    }


def build_plan(
    *,
    plan_id: str,
    role: str,
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
    output_dir: Path,
    output_root: Path,
    device: str,
    formal: bool,
    created_utc: str,
    generator_snapshot: Path,
) -> dict[str, Any]:
    total_online_events = FORMAL_ONLINE_EVENTS if formal else GATE_ONLINE_EVENTS
    events_per_epoch = FORMAL_EVENTS_PER_EPOCH if formal else GATE_EVENTS_PER_EPOCH
    jobs, budgets = build_jobs(
        methods,
        seeds,
        formal=formal,
        total_online_events=total_online_events,
        events_per_epoch=events_per_epoch,
    )
    shared_arguments: dict[str, Any] = {
        "cpu_threads": 8,
        "max_offline_rows": 0 if formal else 4096,
        "online_capacity": 120_000 if formal else 256,
        "events_per_epoch": events_per_epoch,
        "interaction_chunk_events": 16,
        "offline_checkpoint_every": 10_000 if formal else 8,
        "checkpoint_every": 5 if formal else 1,
        "log_every": 250 if formal else 1,
        "resource_interval": 5 if formal else 1,
        "device_cache_observation_dtype": "float16" if device.startswith("cuda") else "float32",
        "compile_networks": device.startswith("cuda"),
    }
    return {
        "schema_version": PLAN_SCHEMA,
        "mdp_version": "v6",
        "plan_id": plan_id,
        "dataset": _relative(DATASET, output_dir),
        "sim_env": _relative(SIMULATOR, output_dir),
        "output_root": _relative(output_root, output_dir),
        "device": device,
        "allow_shared_gpu": False,
        "registration": _registration(
            plan_id=plan_id,
            role=role,
            methods=methods,
            seeds=seeds,
            budgets=budgets,
            output_dir=output_dir,
            created_utc=created_utc,
            formal=formal,
            generator_snapshot=generator_snapshot,
        ),
        "shared_arguments": shared_arguments,
        "jobs": jobs,
    }


def _validate_inputs() -> None:
    required = (
        DATASET,
        DATASET_AUDIT,
        SIMULATOR,
        METHOD_AUDIT,
        TRAINING_DESIGN,
        EVALUATION_DESIGN,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"formal v6 inputs are missing: {missing}")
    audit = json.loads(DATASET_AUDIT.read_text())
    if audit.get("passed") is not True:
        raise RuntimeError("formal v6 dataset audit has not passed")
    actual_dataset_sha256 = sha256(DATASET)
    if audit.get("dataset_sha256") != actual_dataset_sha256:
        raise RuntimeError("formal v6 dataset no longer matches its audit")
    training_design = json.loads(TRAINING_DESIGN.read_text())
    disjointness = training_design.get("evaluation_disjointness", {})
    if disjointness.get("verified_disjoint") is not True:
        raise RuntimeError("training/evaluation environment seeds are not verified disjoint")
    if sha256(EVALUATION_DESIGN) != disjointness.get("sha256"):
        raise RuntimeError("evaluation design no longer matches the training-design lock")


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_plans(output_dir: Path, *, device: str, revision: str) -> list[Path]:
    from SimpleSAC.transit_v3.protocols import METHOD_PROFILES

    if not re.fullmatch(r"v[1-9][0-9]*", revision):
        raise ValueError("revision must have the form v1, v2, ...")
    _validate_inputs()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    created_utc = datetime.now(timezone.utc).isoformat()
    generator_snapshot = (
        output_dir / "frozen_sources" / f"build_training_plan_v6_{revision}.py"
    )
    _write_new(generator_snapshot, Path(__file__).read_bytes())
    gate_methods = tuple(METHOD_PROFILES)
    definitions = (
        (
            f"training_plan_v6_gate_cpu_{revision}.json",
            f"v6-gate-cpu-{revision}",
            "CPU execution gate for every registered method profile",
            gate_methods,
            GATE_SEEDS,
            "cpu",
            False,
            ROOT / "experiment_output_v7" / f"gate_cpu_{revision}",
        ),
        (
            f"training_plan_v6_gate_gpu_{revision}.json",
            f"v6-gate-gpu-{revision}",
            "CUDA compilation and execution gate for every registered method profile",
            gate_methods,
            GATE_SEEDS,
            device,
            False,
            ROOT / "experiment_output_v7" / f"gate_gpu_{revision}",
        ),
        (
            f"training_plan_v6_formal_core_{revision}.json",
            f"v6-formal-core-{revision}",
            "primary method and baseline comparison",
            CORE_METHODS,
            FORMAL_SEEDS,
            device,
            True,
            ROOT / "experiment_output_v7" / f"formal_{revision}" / "core",
        ),
        (
            f"training_plan_v6_formal_ablation_{revision}.json",
            f"v6-formal-ablation-{revision}",
            "component ablation of dynamics-ratio correction and IQL warm start",
            ABLATION_METHODS,
            FORMAL_SEEDS,
            device,
            True,
            ROOT / "experiment_output_v7" / f"formal_{revision}" / "ablation",
        ),
        (
            f"training_plan_v6_formal_sensitivity_{revision}.json",
            f"v6-formal-sensitivity-{revision}",
            "published lambda grid and non-H2O ratio-SAC mechanism control",
            SENSITIVITY_METHODS,
            FORMAL_SEEDS,
            device,
            True,
            ROOT / "experiment_output_v7" / f"formal_{revision}" / "sensitivity",
        ),
    )
    written = []
    for filename, plan_id, role, methods, seeds, plan_device, formal, output_root in definitions:
        path = output_dir / filename
        payload = build_plan(
            plan_id=plan_id,
            role=role,
            methods=methods,
            seeds=seeds,
            output_dir=output_dir,
            output_root=output_root,
            device=plan_device,
            formal=formal,
            created_utc=created_utc,
            generator_snapshot=generator_snapshot,
        )
        _write_new(
            path,
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
        )
        written.append(path)
    registry_path = output_dir / f"training_plan_v6_registry_{revision}.json"
    registry = {
        "schema_version": "transit-training-plan-registry-v1",
        "revision": revision,
        "created_utc": created_utc,
        "generator": _descriptor(generator_snapshot, output_dir),
        "plans": [
            {
                **_descriptor(path, output_dir),
                "plan_id": json.loads(path.read_text())["plan_id"],
                "jobs": len(json.loads(path.read_text())["jobs"]),
            }
            for path in written
        ],
    }
    _write_new(
        registry_path,
        (json.dumps(registry, indent=2, sort_keys=True) + "\n").encode(),
    )
    return [*written, registry_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPRODUCIBILITY)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--revision", default="v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in write_plans(
        args.output_dir, device=args.device, revision=args.revision
    ):
        print(path)


if __name__ == "__main__":
    main()
