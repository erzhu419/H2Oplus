#!/usr/bin/env python3
"""Build admission-locked v7 server training plans and CPU validation gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
EVALUATION_DESIGN = (
    REPRODUCIBILITY / "evaluation_environment_design_v7_formal_v1.json"
)

from bus_h2o.sumo_provenance import sha256
from reproducibility.audit_gpu_capacity_v7 import (
    DEFAULT_MINIMUM_AGGREGATE_SPEEDUP as CAPACITY_MINIMUM_AGGREGATE_SPEEDUP,
    DEFAULT_MINIMUM_FREE_BYTES as CAPACITY_MINIMUM_FREE_BYTES,
    DEFAULT_MINIMUM_WORKER_RATE_FRACTION as CAPACITY_MINIMUM_WORKER_RATE_FRACTION,
    SCHEMA_VERSION as CAPACITY_AUDIT_SCHEMA,
)
from reproducibility.build_host_gpu_concurrency_audit_v7 import (
    SCHEMA_VERSION as HOST_CONCURRENCY_AUDIT_SCHEMA,
)
from reproducibility.build_training_admission_v7 import (
    RUNTIME_LOCK_FIELDS,
    SCHEMA_VERSION as ADMISSION_SCHEMA,
)
from reproducibility.build_training_plan_v6 import (
    ABLATION_METHODS,
    CORE_METHODS,
    FORMAL_EVENTS_PER_EPOCH,
    FORMAL_ONLINE_EVENTS,
    FORMAL_SEEDS,
    GATE_EVENTS_PER_EPOCH,
    GATE_ONLINE_EVENTS,
    GATE_SEEDS,
    LOCAL_CORE_METHODS,
    LOCAL_H2OPLUS_METHODS,
    LOCAL_SENSITIVITY_METHODS,
    SERVER_CORE_METHODS,
    SERVER_SENSITIVITY_METHODS,
    TOOLCHAIN_HEADER,
    TOOLCHAIN_INCLUDE,
    TOOLCHAIN_PROVENANCE,
    TOOLCHAIN_ROOT,
    build_jobs,
)
from SimpleSAC.train_transit_v3 import simulator_input_hashes, source_hashes
from SimpleSAC.transit_v3.protocols import METHOD_PROFILES
from SimpleSAC.transit_v7.trainer import TransitH2OTrainerV7


PLAN_SCHEMA = "transit-training-plan-v7"
REGISTRATION_SCHEMA = "transit-training-registration-v2"
REGISTRY_SCHEMA = "transit-training-plan-registry-v2"
INITIALIZER_DEPENDENCY_SCHEMA = "transit-offline-initializer-dependency-v7-v1"
EVALUATION_DESIGN_SCHEMA = "transit-evaluation-environment-design-v1"
SERVER_H2OPLUS_EXECUTION_CLASS = "server-h2oplus"
SERVER_EXECUTION_CLASS = "server-non-h2oplus"
H2OPLUS_METHODS = tuple(LOCAL_H2OPLUS_METHODS)
H2OPLUS_CORE_METHODS = tuple(LOCAL_CORE_METHODS)
H2OPLUS_SENSITIVITY_METHODS = tuple(LOCAL_SENSITIVITY_METHODS)
SERVER_METHODS = (
    "rlpd",
    "wsrl",
    "sac-mixed",
    "sac-online",
    "cql",
    "iql",
    "bc",
    "ratio-sac",
)
DEFAULT_COMPILER = Path(
    "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/"
    "x86_64-conda-linux-gnu-gcc"
)
SERVER_LIBCURL_PRELOAD = Path("/usr/lib/x86_64-linux-gnu/libcurl.so.4")
EXPECTED_EVIDENCE = {
    "dataset_audit",
    "serialized_credit_audit",
    "reward_audit",
    "reward_loss_pilot",
    "entropy_pilot",
    "device_precision_audit",
    "zero_fidelity_audit",
    "intervention_fidelity_audit",
    "method_protocol_audit",
}
EXPECTED_SELECTED_ARGUMENTS = {
    "reward_scale",
    "critic_loss",
    "huber_delta",
    "target_entropy",
    "device_cache_observation_dtype",
    "sim_demand_scales",
}
SENSITIVITY_METHODS = (
    "h2oplus-lambda0",
    "h2oplus-lambda0p2",
    "h2oplus-lambda0p5",
    "h2oplus-lambda1",
    "ratio-sac",
)
SHARED_IQL_CONSUMERS = {
    "h2oplus-no-ratio",
    "h2oplus-lambda0",
    "h2oplus-lambda0p2",
    "h2oplus-lambda0p5",
    "h2oplus-lambda1",
}
FORMAL_SERVER_GPU_CONCURRENCY: dict[str, int] = {}
FORMAL_SERVER_CPU_THREADS = 6
FORMAL_SERVER_JOBS_PER_HOST = 2
FORMAL_SERVER_HOSTS = ("jtl110gpu", "jtl110gpu2")
FORMAL_HOST_CONCURRENCY_AUDITS = tuple(
    REPRODUCIBILITY
    / "capacity_v7_formal_v2_v1"
    / host
    / "host_gpu_concurrency_audit.json"
    for host in FORMAL_SERVER_HOSTS
)


def _relative(path: Path, output_dir: Path) -> str:
    return os.path.relpath(path.resolve(), output_dir.resolve())


def _descriptor(path: Path, output_dir: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    return {
        "path": _relative(path, output_dir),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def _host_concurrency_evidence(
    paths: Iterable[Path],
    *,
    output_dir: Path,
    dataset_sha256: str,
) -> list[dict[str, Any]]:
    evidence = []
    observed_hosts = set()
    expected_sources = {
        "reproducibility/build_host_gpu_concurrency_audit_v7.py": sha256(
            REPRODUCIBILITY / "build_host_gpu_concurrency_audit_v7.py"
        ),
        "reproducibility/benchmark_training_kernel_v4.py": sha256(
            REPRODUCIBILITY / "benchmark_training_kernel_v4.py"
        ),
    }
    for raw_path in paths:
        path = raw_path.expanduser().resolve(strict=True)
        payload = json.loads(path.read_text())
        host = payload.get("host_label")
        if host in observed_hosts:
            raise ValueError(f"duplicate host concurrency audit for {host}")
        observed_hosts.add(host)
        expected = {
            "schema_version": HOST_CONCURRENCY_AUDIT_SCHEMA,
            "passed": True,
            "status": "passed",
            "method": "rlpd",
            "phase": "online",
            "dataset_sha256": dataset_sha256,
            "device_cache_mode": "float32",
            "compile_networks": True,
            "compile_mode": "default",
            "max_canonical_processes_per_gpu": 1,
            "canonical_processes_per_host": FORMAL_SERVER_JOBS_PER_HOST,
            "cpu_threads_per_process": FORMAL_SERVER_CPU_THREADS,
            "physical_gpus": ["0", "1"],
        }
        mismatches = {
            name: {"observed": payload.get(name), "expected": value}
            for name, value in expected.items()
            if payload.get(name) != value
        }
        if payload.get("source_sha256") != expected_sources:
            mismatches["source_sha256"] = "audit or benchmark source is stale"
        if payload.get("decision", {}).get("passed") is not True:
            mismatches["decision.passed"] = payload.get("decision", {}).get(
                "passed"
            )
        if mismatches:
            raise ValueError(
                f"host concurrency audit is incompatible with v7: {mismatches}"
            )
        evidence.append(
            {
                **_descriptor(path, output_dir),
                "host_label": host,
                "aggregate_updates_per_second": payload.get(
                    "aggregate_updates_per_second"
                ),
                "aggregate_speedup": payload.get("aggregate_speedup"),
                "minimum_observed_worker_rate_fraction": payload.get(
                    "minimum_observed_worker_rate_fraction"
                ),
            }
        )
    if observed_hosts != set(FORMAL_SERVER_HOSTS):
        raise ValueError(
            "formal v7 plans require passed host concurrency audits for "
            f"{list(FORMAL_SERVER_HOSTS)}"
        )
    return sorted(evidence, key=lambda item: item["host_label"])


def _evaluation_design_descriptor(output_dir: Path) -> dict[str, Any]:
    """Fail closed if the held-out v7 environment freeze has drifted."""

    design_path = EVALUATION_DESIGN.resolve(strict=True)
    payload = json.loads(design_path.read_text())
    if (
        payload.get("schema_version") != EVALUATION_DESIGN_SCHEMA
        or payload.get("mdp_version") != "v7"
        or payload.get("study_stage") != "formal"
        or payload.get("environment_cells") != 30
        or payload.get("independent_realization_blocks") != 10
        or payload.get("demand_scales") != [0.75, 1.0, 1.25]
    ):
        raise ValueError("v7 evaluation environment design is not the formal freeze")
    blocks = payload.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != 10 or any(
        len(block.get("scenario_manifests", [])) != 3 for block in blocks
    ):
        raise ValueError("v7 evaluation environment design has invalid block coverage")
    disjointness = payload.get("training_disjointness", {})
    if not (
        disjointness.get("verified_disjoint") is True
        and disjointness.get("scenario_manifest_hash_overlap") == []
        and disjointness.get("scenario_runtime_hash_overlap") == []
    ):
        raise ValueError("v7 training/evaluation environments are not hash-disjoint")
    generator = Path(payload.get("generator", "")).expanduser().resolve()
    if not generator.is_file() or payload.get("generator_sha256") != sha256(
        generator
    ):
        raise ValueError("v7 evaluation-design generator is missing or stale")
    for label, descriptor in (
        ("training design", disjointness.get("training_design", {})),
        ("physical evaluation design", payload.get("physical_scenario_reuse", {})),
    ):
        artifact = Path(descriptor.get("path", "")).expanduser().resolve()
        if not artifact.is_file() or descriptor.get("sha256") != sha256(artifact):
            raise ValueError(f"v7 {label} is missing or stale")
    return _descriptor(design_path, output_dir)


def _resolve_relative(
    base: Path,
    raw: Any,
    *,
    label: str,
    allowed_root: Path,
) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} path is missing")
    relative = Path(raw).expanduser()
    if relative.is_absolute():
        raise ValueError(f"{label} path must be a safe relative path")
    resolved = (base / relative).resolve()
    root = allowed_root.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} path escapes the artifact root")
    return resolved


def _same_float(first: Any, second: Any) -> bool:
    try:
        return math.isclose(float(first), float(second), rel_tol=0.0, abs_tol=0.0)
    except (TypeError, ValueError):
        return False


def load_admission(path: Path) -> dict[str, Any]:
    """Resolve and revalidate the formal certificate before plan generation."""

    admission_path = path.expanduser().resolve(strict=True)
    artifact_root = (
        ROOT
        if admission_path == ROOT or ROOT in admission_path.parents
        else admission_path.parent
    )
    payload = json.loads(admission_path.read_text())
    if (
        payload.get("schema_version") != ADMISSION_SCHEMA
        or payload.get("study_stage") != "formal"
        or payload.get("passed") is not True
    ):
        raise ValueError("v7 training admission is not a passed formal certificate")

    builder = ROOT / "reproducibility" / "build_training_admission_v7.py"
    if payload.get("builder_sha256") != sha256(builder):
        raise ValueError("v7 training admission was built by stale code")

    dataset_descriptor = payload.get("dataset", {})
    dataset = _resolve_relative(
        admission_path.parent,
        dataset_descriptor.get("path"),
        label="admission dataset",
        allowed_root=artifact_root,
    )
    if (
        not dataset.is_file()
        or dataset_descriptor.get("sha256") != sha256(dataset)
        or dataset_descriptor.get("bytes") != dataset.stat().st_size
    ):
        raise ValueError("v7 training admission dataset is missing or stale")

    simulator_descriptor = payload.get("simulator", {})
    simulator_root = _resolve_relative(
        admission_path.parent,
        simulator_descriptor.get("path"),
        label="admission simulator",
        allowed_root=artifact_root,
    )
    if not simulator_root.is_dir():
        raise ValueError("v7 training admission simulator is missing")
    simulator = simulator_input_hashes(
        simulator_root, runtime_calibration_version="v7_v4"
    )
    if (
        simulator_descriptor.get("aggregate_sha256")
        != simulator["aggregate_sha256"]
        or simulator_descriptor.get("runtime_sha256")
        != simulator["runtime_sha256"]
    ):
        raise ValueError("v7 training admission simulator is stale")

    if payload.get("training_source_sha256") != source_hashes("v7"):
        raise ValueError("v7 training admission source map is stale")
    runtime_lock = payload.get("training_runtime_lock")
    if not isinstance(runtime_lock, dict) or set(runtime_lock) != RUNTIME_LOCK_FIELDS:
        raise ValueError("v7 training admission runtime lock is incomplete")

    evidence = payload.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != EXPECTED_EVIDENCE:
        raise ValueError("v7 training admission evidence set is incomplete")
    for name, descriptor in evidence.items():
        if not isinstance(descriptor, dict):
            raise ValueError(f"v7 admission evidence {name} is invalid")
        evidence_path = _resolve_relative(
            admission_path.parent,
            descriptor.get("path"),
            label=f"admission evidence {name}",
            allowed_root=artifact_root,
        )
        if (
            not evidence_path.is_file()
            or descriptor.get("sha256") != sha256(evidence_path)
            or descriptor.get("bytes") != evidence_path.stat().st_size
        ):
            raise ValueError(f"v7 admission evidence {name} is missing or stale")
        evidence_payload = json.loads(evidence_path.read_text())
        if evidence_payload.get("schema_version") != descriptor.get(
            "schema_version"
        ):
            raise ValueError(f"v7 admission evidence {name} schema drifted")

    selected = payload.get("selected_arguments")
    if not isinstance(selected, dict) or set(selected) != EXPECTED_SELECTED_ARGUMENTS:
        raise ValueError("v7 training admission selected arguments are incomplete")
    scales = selected.get("sim_demand_scales")
    if scales != [0.75, 1.0, 1.25]:
        raise ValueError("v7 formal demand schedule must be [0.75, 1.0, 1.25]")
    if selected.get("critic_loss") not in {"mse", "huber"}:
        raise ValueError("v7 critic loss selection is invalid")
    if selected.get("device_cache_observation_dtype") not in {
        "float32",
        "float16",
    }:
        raise ValueError("v7 observation cache selection is invalid")
    for name in ("reward_scale", "huber_delta", "target_entropy"):
        try:
            value = float(selected[name])
        except (TypeError, ValueError) as error:
            raise ValueError(f"v7 selected argument {name} is invalid") from error
        if not math.isfinite(value):
            raise ValueError(f"v7 selected argument {name} is not finite")
    if float(selected["reward_scale"]) <= 0.0 or float(selected["huber_delta"]) <= 0.0:
        raise ValueError("v7 reward scale and Huber delta must be positive")

    return {
        "path": admission_path,
        "payload": payload,
        "dataset": dataset,
        "simulator_root": simulator_root,
        "simulator": simulator,
        "selected_arguments": dict(selected),
        "training_runtime_lock": dict(runtime_lock),
    }


def _selected_job_arguments(selected: dict[str, Any]) -> dict[str, Any]:
    return {
        "reward_scale": float(selected["reward_scale"]),
        "critic_loss": str(selected["critic_loss"]),
        "huber_delta": float(selected["huber_delta"]),
        "target_entropy": float(selected["target_entropy"]),
        "device_cache_observation_dtype": str(
            selected["device_cache_observation_dtype"]
        ),
        "sim_demand_scales": [
            float(value) for value in selected["sim_demand_scales"]
        ],
    }


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
    compiler: Path,
    execution_class: str,
    allow_external_gpu_processes: bool,
    max_canonical_processes_per_gpu_by_method: dict[str, int],
    admission: dict[str, Any],
    offline_initializer_dependencies: dict[int, dict[str, Any]],
    host_concurrency_audits: Iterable[Path],
) -> dict[str, Any]:
    concurrency = {
        method: max_canonical_processes_per_gpu_by_method.get(method, 1)
        for method in methods
    }
    formal_server = formal and execution_class in {
        SERVER_H2OPLUS_EXECUTION_CLASS,
        SERVER_EXECUTION_CLASS,
    }
    timing_claimable = {
        method: (
            concurrency[method] == 1
            and not allow_external_gpu_processes
            and not formal_server
        )
        for method in methods
    }
    host_concurrency_evidence = (
        _host_concurrency_evidence(
            host_concurrency_audits,
            output_dir=output_dir,
            dataset_sha256=sha256(admission["dataset"]),
        )
        if formal_server
        else []
    )
    selected = admission["selected_arguments"]
    return {
        "schema_version": REGISTRATION_SCHEMA,
        "plan_id": plan_id,
        "status": "frozen-before-formal-outcomes" if formal else "execution-gate",
        "created_utc": created_utc,
        "scientific_role": role,
        "statistical_unit": (
            "independent training seed evaluated on paired, disjoint SUMO "
            "scenario cells"
        ),
        "primary_outcome": (
            "accrued generalized passenger seconds divided by departed "
            "passengers over the fixed 24000-second horizon"
        ),
        "online_budget_semantics": (
            "the simulator decision-event hard cap includes frozen-policy warmup; "
            "simultaneous decision batches remain atomic and a final batch that "
            "would exceed the cap is not run"
        ),
        "comparison_disclosure": (
            "online methods share one decision-event cap; gradient budgets differ "
            "only through registered method UTD and pretraining protocols"
        ),
        "execution_class": execution_class,
        "checkpoint_version": TransitH2OTrainerV7.CHECKPOINT_VERSION,
        "formal_policy_artifact": {
            "schema_version": "transit-policy-artifact-v7-v1",
            "relative_path": "checkpoints/final_actor.pt",
            "selection_rule": (
                "the completed final actor from every registered training seed; "
                "no held-out evaluation outcome may select a checkpoint"
            ),
        },
        "legacy_snapshot_dataset_required": False,
        "placement_disclosure": (
            "Every canonical v7 CUDA job executes only on registered GPU servers. "
            "H2O+ and its direct branches use the server-h2oplus execution class; "
            "RLPD, WSRL, SAC controls, offline controls, and ratio-SAC use the "
            "server-non-h2oplus execution class. Every method consumes the same "
            "admitted HDF5 and calibrated simulator; no formal v7 method reads "
            "the legacy snapshot corpus."
        ),
        "deterministic_offline_prefix_reuse": {
            "enabled": bool(
                formal
                and (
                    "h2oplus" in methods
                    or set(methods) & SHARED_IQL_CONSUMERS
                )
            ),
            "artifact_schema": "transit-offline-initializer-v7-v1",
            "dependency_schema": INITIALIZER_DEPENDENCY_SCHEMA,
            "producer_method": "h2oplus",
            "producer_relative_path": "checkpoints/offline_initializer.pt",
            "producer_offline_updates": 50_000,
            "producer_seeds": (
                list(seeds) if formal and "h2oplus" in methods else []
            ),
            "consumer_methods": sorted(
                set(methods) & SHARED_IQL_CONSUMERS if formal else set()
            ),
            "source_dependencies": {
                str(seed): dependency
                for seed, dependency in sorted(
                    offline_initializer_dependencies.items()
                )
            },
            "scientific_accounting": (
                "each consumer inherits the exact same-seed 50000-update IQL "
                "trajectory registered for that branch; the prefix is physically "
                "executed once per seed and no evaluation outcome selects or "
                "alters the shared state"
            ),
        },
        "compute_policy": {
            "training_runtime_lock": admission["training_runtime_lock"],
            "observation_cache_storage": selected[
                "device_cache_observation_dtype"
            ],
            "network_compute": "float32",
            "optimizer": {
                "name": "Adam",
                "cuda_backend": "fused",
                "capturable_device_step_counters": True,
            },
            "compiled_repeated_mlp_kernels": True,
            "external_gpu_processes_allowed": allow_external_gpu_processes,
            "canonical_jobs_per_physical_gpu": (
                next(iter(set(concurrency.values())))
                if len(set(concurrency.values())) == 1
                else "method-specific"
            ),
            "canonical_jobs_per_physical_gpu_by_method": concurrency,
            "canonical_jobs_per_host": (
                FORMAL_SERVER_JOBS_PER_HOST if formal_server else 1
            ),
            "cpu_threads_per_canonical_job": (
                FORMAL_SERVER_CPU_THREADS if formal_server else 8
            ),
            "shared_gpu_allowed_methods": sorted(
                method for method, value in concurrency.items() if value > 1
            ),
            "host_gpu_concurrency": {
                "physical_gpus_per_host": 2 if formal_server else 1,
                "canonical_jobs_per_host": (
                    FORMAL_SERVER_JOBS_PER_HOST if formal_server else 1
                ),
                "canonical_jobs_per_physical_gpu": 1,
                "cpu_threads_per_job": (
                    FORMAL_SERVER_CPU_THREADS if formal_server else 8
                ),
                "scheduling_rule": (
                    "at most one canonical process per physical GPU; both GPUs "
                    "on a registered server may run concurrently"
                    if formal_server
                    else "one execution-gate process at a time"
                ),
                "evidence": host_concurrency_evidence,
            },
            "same_gpu_capacity_gate": {
                "required_for_methods": sorted(
                    method for method, value in concurrency.items() if value > 1
                ),
                "schema_version": CAPACITY_AUDIT_SCHEMA,
                "protocol": _descriptor(
                    ROOT / "reproducibility" / "audit_gpu_capacity_v7.py",
                    output_dir,
                ),
                "benchmark_protocol": _descriptor(
                    ROOT
                    / "reproducibility"
                    / "benchmark_training_kernel_v4.py",
                    output_dir,
                ),
                "full_registered_dataset_required": True,
                "minimum_free_memory_bytes": CAPACITY_MINIMUM_FREE_BYTES,
                "minimum_warmup_updates_per_process": 100,
                "minimum_measured_updates_per_process": 5_000,
                "single_process_reference_required": True,
                "minimum_aggregate_speedup": (
                    CAPACITY_MINIMUM_AGGREGATE_SPEEDUP
                ),
                "minimum_worker_rate_fraction": (
                    CAPACITY_MINIMUM_WORKER_RATE_FRACTION
                ),
                "admission_rule": (
                    "same-GPU sharing is disabled for current formal v7 plans; "
                    "any future quota above one must pass this gate before "
                    "formal training"
                ),
            },
            "runtime_timing_claimable": all(timing_claimable.values()),
            "runtime_timing_claimable_by_method": timing_claimable,
            "compiler": _descriptor(compiler, output_dir),
            "compiler_compatibility_header": _descriptor(
                TOOLCHAIN_HEADER, output_dir
            ),
            "compiler_compatibility_provenance": _descriptor(
                TOOLCHAIN_PROVENANCE, output_dir
            ),
        },
        "task_adaptation_freeze": _selected_job_arguments(selected),
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
            "training_admission": _descriptor(admission["path"], output_dir),
            "evaluation_environment_design": _evaluation_design_descriptor(
                output_dir
            ),
            "generator": _descriptor(generator_snapshot, output_dir),
        },
    }


def _validate_execution_boundary(
    methods: tuple[str, ...], device: str, execution_class: str
) -> None:
    h2oplus_methods = set(H2OPLUS_METHODS)
    baseline_methods = set(SERVER_METHODS)
    registered_methods = set(METHOD_PROFILES)
    if (
        not h2oplus_methods.isdisjoint(baseline_methods)
        or h2oplus_methods | baseline_methods != registered_methods
        or any(not method.startswith("h2oplus") for method in h2oplus_methods)
    ):
        raise RuntimeError("v7 server execution-class partition is invalid")
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("training plans require unique registered methods")
    unknown = set(methods) - registered_methods
    if unknown:
        raise ValueError(f"training plan contains unknown methods: {sorted(unknown)}")

    if execution_class == "cpu-validation":
        if device != "cpu":
            raise ValueError("CPU validation plans must use the CPU device")
        return

    if execution_class == "local-h2oplus":
        raise ValueError("local H2O+ execution is disabled; use server-h2oplus")
    if not device.startswith("cuda"):
        raise ValueError("server training plans must use a CUDA device")
    if execution_class == SERVER_H2OPLUS_EXECUTION_CLASS:
        misplaced = set(methods) - h2oplus_methods
        if misplaced:
            raise ValueError(
                "only H2O+ family methods may use server-h2oplus: "
                f"{sorted(misplaced)}"
            )
        return
    if execution_class == SERVER_EXECUTION_CLASS:
        misplaced = set(methods) - baseline_methods
        if misplaced:
            raise ValueError(
                "H2O+ family methods require server-h2oplus: "
                f"{sorted(misplaced)}"
            )
        return
    raise ValueError(f"unknown v7 execution class: {execution_class}")


def build_plan(
    *,
    plan_id: str,
    role: str,
    methods: Iterable[str],
    seeds: Iterable[int],
    output_dir: Path,
    output_root: Path,
    device: str,
    formal: bool,
    created_utc: str,
    generator_snapshot: Path,
    compiler: Path,
    execution_class: str,
    admission: dict[str, Any],
    allow_external_gpu_processes: bool = False,
    max_canonical_processes_per_gpu_by_method: dict[str, int] | None = None,
    offline_initializer_dependencies: dict[int, dict[str, Any]] | None = None,
    host_concurrency_audits: Iterable[Path] = (),
) -> dict[str, Any]:
    methods = tuple(methods)
    seeds = tuple(seeds)
    _validate_execution_boundary(methods, device, execution_class)
    concurrency = dict(max_canonical_processes_per_gpu_by_method or {})
    initializer_dependencies = dict(offline_initializer_dependencies or {})
    active_consumers = set(methods) & SHARED_IQL_CONSUMERS
    expected_dependency_seeds = set(seeds) if formal and active_consumers else set()
    if set(initializer_dependencies) != expected_dependency_seeds:
        raise ValueError(
            "formal shared-IQL plans require one frozen source dependency per seed"
        )
    if any(
        not isinstance(dependency, dict)
        or dependency.get("schema_version") != INITIALIZER_DEPENDENCY_SCHEMA
        for dependency in initializer_dependencies.values()
    ):
        raise ValueError("offline initializer dependency schema is invalid")
    total_online_events = FORMAL_ONLINE_EVENTS if formal else GATE_ONLINE_EVENTS
    events_per_epoch = FORMAL_EVENTS_PER_EPOCH if formal else GATE_EVENTS_PER_EPOCH
    jobs, budgets = build_jobs(
        methods,
        seeds,
        formal=formal,
        total_online_events=total_online_events,
        events_per_epoch=events_per_epoch,
        max_canonical_processes_per_gpu_by_method=concurrency,
    )
    selected_arguments = _selected_job_arguments(
        admission["selected_arguments"]
    )
    for job in jobs:
        job["arguments"].update(selected_arguments)
        if job["method"] in active_consumers and formal:
            job["offline_initializer"] = dict(
                initializer_dependencies[job["seed"]]
            )

    formal_server = formal and execution_class in {
        SERVER_H2OPLUS_EXECUTION_CLASS,
        SERVER_EXECUTION_CLASS,
    }
    shared_arguments: dict[str, Any] = {
        "cpu_threads": FORMAL_SERVER_CPU_THREADS if formal_server else 8,
        "max_offline_rows": 0 if formal else 4096,
        "online_capacity": 120_000 if formal else 256,
        "events_per_epoch": events_per_epoch,
        "interaction_chunk_events": 16,
        "offline_checkpoint_every": 10_000 if formal else 8,
        "checkpoint_every": 5 if formal else 1,
        "log_every": 250 if formal else 1,
        "resource_interval": 5 if formal else 1,
        "compile_networks": device.startswith("cuda"),
        "compile_mode": "default",
        **selected_arguments,
    }
    required_environment = {
        "CC": str(compiler.expanduser().resolve()),
        "TMPDIR": "~/.cache/h2oplus-v7/tmp",
    }
    if device.startswith("cuda"):
        required_environment["CPATH"] = _relative(TOOLCHAIN_INCLUDE, output_dir)
    if execution_class in {
        SERVER_H2OPLUS_EXECUTION_CLASS,
        SERVER_EXECUTION_CLASS,
    }:
        required_environment["LD_PRELOAD"] = str(SERVER_LIBCURL_PRELOAD)

    return {
        "schema_version": PLAN_SCHEMA,
        "mdp_version": "v7",
        "plan_id": plan_id,
        "execution_class": execution_class,
        "dataset": _relative(admission["dataset"], output_dir),
        "sim_env": _relative(admission["simulator_root"], output_dir),
        "training_admission": _relative(admission["path"], output_dir),
        "output_root": _relative(output_root, output_dir),
        "device": device,
        "allow_shared_gpu": False,
        "max_canonical_processes_per_gpu": 1,
        "allow_external_gpu_processes": allow_external_gpu_processes,
        "required_environment": required_environment,
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
            compiler=compiler,
            execution_class=execution_class,
            allow_external_gpu_processes=allow_external_gpu_processes,
            max_canonical_processes_per_gpu_by_method=concurrency,
            admission=admission,
            offline_initializer_dependencies=initializer_dependencies,
            host_concurrency_audits=host_concurrency_audits,
        ),
        "shared_arguments": shared_arguments,
        "jobs": jobs,
    }


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _validate_toolchain(compiler: Path) -> None:
    compiler = compiler.expanduser().resolve(strict=True)
    if not compiler.is_file() or not os.access(str(compiler), os.X_OK):
        raise FileNotFoundError(f"formal compiler is not executable: {compiler}")
    provenance = json.loads(TOOLCHAIN_PROVENANCE.read_text())
    if provenance.get("schema_version") != "transit-vendored-toolchain-input-v1":
        raise ValueError("compiler compatibility provenance schema is invalid")
    for relative, expected in provenance.get("files_sha256", {}).items():
        artifact = TOOLCHAIN_ROOT / relative
        if not artifact.is_file() or sha256(artifact) != expected:
            raise ValueError(
                f"compiler compatibility artifact is stale: {artifact}"
            )


def build_offline_initializer_dependencies(
    source_plan_path: Path,
    *,
    output_dir: Path,
    seeds: Iterable[int],
) -> dict[int, dict[str, Any]]:
    """Bind downstream branches to signed same-seed H2O+ core jobs."""

    from reproducibility.run_training_matrix_v4 import load_plan

    source_plan_path = source_plan_path.expanduser().resolve(strict=True)
    source_payload = json.loads(source_plan_path.read_text())
    if (
        source_payload.get("schema_version") != PLAN_SCHEMA
        or source_payload.get("execution_class")
        != SERVER_H2OPLUS_EXECUTION_CLASS
        or source_payload.get("registration", {}).get("status")
        != "frozen-before-formal-outcomes"
    ):
        raise ValueError("shared IQL source must be the formal server H2O+ core plan")
    source_jobs, _ = load_plan(source_plan_path)
    by_seed = {
        job["seed"]: job for job in source_jobs if job["method"] == "h2oplus"
    }
    requested_seeds = tuple(seeds)
    if set(by_seed) != set(requested_seeds):
        raise ValueError("formal H2O+ source plan does not cover the target seeds")
    return {
        seed: {
            "schema_version": INITIALIZER_DEPENDENCY_SCHEMA,
            "path": _relative(
                by_seed[seed]["run_dir"]
                / "checkpoints"
                / "offline_initializer.pt",
                output_dir,
            ),
            "source_plan": _relative(source_plan_path, output_dir),
            "source_plan_sha256": sha256(source_plan_path),
            "source_job_name": by_seed[seed]["name"],
            "source_run_signature": by_seed[seed]["signature"],
        }
        for seed in requested_seeds
    }


def write_plans(
    output_dir: Path,
    *,
    admission_path: Path,
    device: str,
    revision: str,
    compiler: Path = DEFAULT_COMPILER,
    experiment_output_root: Path | None = None,
    host_concurrency_audits: Iterable[Path] = FORMAL_HOST_CONCURRENCY_AUDITS,
) -> list[Path]:
    if not re.fullmatch(r"v[1-9][0-9]*", revision):
        raise ValueError("revision must have the form v1, v2, ...")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    compiler = compiler.expanduser().resolve(strict=True)
    _validate_toolchain(compiler)
    admission = load_admission(admission_path)
    experiment_output_root = (
        experiment_output_root.expanduser().resolve()
        if experiment_output_root is not None
        else ROOT / "experiment_output_smdp_v7"
    )
    created_utc = datetime.now(timezone.utc).isoformat()
    generator_snapshot = (
        output_dir / "frozen_sources" / f"build_training_plan_v7_{revision}.py"
    )
    _write_new(generator_snapshot, Path(__file__).read_bytes())

    definitions = (
        (
            f"training_plan_v7_gate_cpu_{revision}.json",
            f"v7-gate-cpu-{revision}",
            "CPU execution gate for every registered method profile",
            tuple(METHOD_PROFILES),
            GATE_SEEDS,
            "cpu",
            False,
            experiment_output_root / f"gate_cpu_{revision}",
            "cpu-validation",
            {},
        ),
        (
            f"training_plan_v7_gate_gpu_server_h2oplus_{revision}.json",
            f"v7-gate-gpu-server-h2oplus-{revision}",
            "server CUDA execution gate for the H2O+ family",
            H2OPLUS_METHODS,
            GATE_SEEDS,
            device,
            False,
            experiment_output_root / f"gate_gpu_{revision}" / "server_h2oplus",
            SERVER_H2OPLUS_EXECUTION_CLASS,
            {},
        ),
        (
            f"training_plan_v7_gate_gpu_server_baselines_{revision}.json",
            f"v7-gate-gpu-server-baselines-{revision}",
            "server CUDA execution gate for every non-H2O+ method",
            SERVER_METHODS,
            GATE_SEEDS,
            device,
            False,
            experiment_output_root / f"gate_gpu_{revision}" / "server_baselines",
            SERVER_EXECUTION_CLASS,
            {},
        ),
        (
            f"training_plan_v7_formal_core_server_h2oplus_{revision}.json",
            f"v7-formal-core-server-h2oplus-{revision}",
            "server H2O+ portion of the primary comparison",
            H2OPLUS_CORE_METHODS,
            FORMAL_SEEDS,
            device,
            True,
            experiment_output_root / f"formal_{revision}" / "server" / "h2oplus_core",
            SERVER_H2OPLUS_EXECUTION_CLASS,
            {},
        ),
        (
            f"training_plan_v7_formal_core_server_baselines_{revision}.json",
            f"v7-formal-core-server-baselines-{revision}",
            "server baseline portion of the primary comparison",
            SERVER_CORE_METHODS,
            FORMAL_SEEDS,
            device,
            True,
            experiment_output_root / f"formal_{revision}" / "server" / "core",
            SERVER_EXECUTION_CLASS,
            FORMAL_SERVER_GPU_CONCURRENCY,
        ),
        (
            f"training_plan_v7_formal_ablation_server_h2oplus_{revision}.json",
            f"v7-formal-ablation-server-h2oplus-{revision}",
            "server H2O+ component ablations",
            ABLATION_METHODS,
            FORMAL_SEEDS,
            device,
            True,
            experiment_output_root / f"formal_{revision}" / "server" / "h2oplus_ablation",
            SERVER_H2OPLUS_EXECUTION_CLASS,
            {},
        ),
        (
            f"training_plan_v7_formal_sensitivity_server_h2oplus_{revision}.json",
            f"v7-formal-sensitivity-server-h2oplus-{revision}",
            "server H2O+ mixed-backup sensitivity",
            H2OPLUS_SENSITIVITY_METHODS,
            FORMAL_SEEDS,
            device,
            True,
            experiment_output_root / f"formal_{revision}" / "server" / "h2oplus_sensitivity",
            SERVER_H2OPLUS_EXECUTION_CLASS,
            {},
        ),
        (
            f"training_plan_v7_formal_sensitivity_server_ratio_sac_{revision}.json",
            f"v7-formal-sensitivity-server-ratio-sac-{revision}",
            "server non-H2O ratio-SAC mechanism control",
            SERVER_SENSITIVITY_METHODS,
            FORMAL_SEEDS,
            device,
            True,
            experiment_output_root / f"formal_{revision}" / "server" / "sensitivity",
            SERVER_EXECUTION_CLASS,
            {},
        ),
    )
    written: list[Path] = []
    formal_h2oplus_core_plan = (
        output_dir
        / f"training_plan_v7_formal_core_server_h2oplus_{revision}.json"
    )
    for (
        filename,
        plan_id,
        role,
        methods,
        seeds,
        plan_device,
        formal,
        output_root,
        execution_class,
        concurrency,
    ) in definitions:
        path = output_dir / filename
        initializer_dependencies = (
            build_offline_initializer_dependencies(
                formal_h2oplus_core_plan,
                output_dir=output_dir,
                seeds=seeds,
            )
            if formal and set(methods) & SHARED_IQL_CONSUMERS
            else {}
        )
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
            compiler=compiler,
            execution_class=execution_class,
            admission=admission,
            allow_external_gpu_processes=False,
            max_canonical_processes_per_gpu_by_method=concurrency,
            offline_initializer_dependencies=initializer_dependencies,
            host_concurrency_audits=(
                host_concurrency_audits
                if formal
                and execution_class
                in {SERVER_H2OPLUS_EXECUTION_CLASS, SERVER_EXECUTION_CLASS}
                else ()
            ),
        )
        _write_new(
            path,
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
        )
        written.append(path)

    registry_path = output_dir / f"training_plan_v7_registry_{revision}.json"
    registry = {
        "schema_version": REGISTRY_SCHEMA,
        "revision": revision,
        "created_utc": created_utc,
        "admission": _descriptor(admission["path"], output_dir),
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
    parser.add_argument("--training-admission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--revision", default="v1")
    parser.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER)
    parser.add_argument("--experiment-output-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = write_plans(
        args.output_dir,
        admission_path=args.training_admission,
        device=args.device,
        revision=args.revision,
        compiler=args.compiler,
        experiment_output_root=args.experiment_output_root,
    )
    print(
        json.dumps(
            {
                "files": [str(path) for path in paths],
                "sha256": {str(path): sha256(path) for path in paths},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
