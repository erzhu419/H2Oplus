#!/usr/bin/env python3
"""Run a resumable, provenance-locked transit v4/v5/v6/v7 matrix or job shard."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "SimpleSAC" / "train_transit_v4.py"
TRAINERS = {
    "v4": TRAINER,
    "v5": ROOT / "SimpleSAC" / "train_transit_v5.py",
    "v6": ROOT / "SimpleSAC" / "train_transit_v6.py",
    "v7": ROOT / "SimpleSAC" / "train_transit_v7.py",
}
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SimpleSAC.train_transit_v3 import (
    MAX_CANONICAL_GPU_PROCESSES,
    V7_SHARED_IQL_BRANCH_METHODS,
    collect_runtime_fingerprint,
    sha256,
    simulator_input_hashes,
    source_hashes,
)
from SimpleSAC.transit_v3.protocols import METHOD_PROFILES
from reproducibility.build_training_admission_v7 import (
    SCHEMA_VERSION as V7_ADMISSION_SCHEMA,
)


PLAN_SCHEMA = "transit-training-plan-v4"
MATRIX_SCHEMA = "transit-training-matrix-v4"
SUMMARY_SCHEMA = "transit-training-summary-v5"
SCHEMA_CONTRACTS = {
    "v4": {
        "plan": PLAN_SCHEMA,
        "matrix": MATRIX_SCHEMA,
        "summary": SUMMARY_SCHEMA,
    },
    "v5": {
        "plan": "transit-training-plan-v5",
        "matrix": "transit-training-matrix-v5",
        "summary": "transit-training-summary-v6",
    },
    "v6": {
        "plan": "transit-training-plan-v6",
        "matrix": "transit-training-matrix-v6",
        "summary": "transit-training-summary-v7",
    },
    "v7": {
        "plan": "transit-training-plan-v7",
        "matrix": "transit-training-matrix-v7",
        "summary": "transit-training-summary-v8",
    },
}
PLAN_SCHEMA_TO_MDP_VERSION = {
    contract["plan"]: mdp_version
    for mdp_version, contract in SCHEMA_CONTRACTS.items()
}
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
RESERVED_ARGUMENTS = {
    "method",
    "seed",
    "dataset",
    "sim_env",
    "output_root",
    "device",
    "resume",
    "run_name",
    "run_signature",
    "allow_shared_gpu",
    "allow_external_gpu_processes",
    "max_canonical_processes_per_gpu",
    "training_admission",
    "offline_initializer",
    "offline_initializer_source_signature",
}
ALLOWED_ARGUMENTS = {
    "cpu_threads",
    "sim_demand_scale",
    "sim_demand_scales",
    "max_offline_rows",
    "online_capacity",
    "offline_updates",
    "warmup_events",
    "ratio_pretrain_updates",
    "epochs",
    "events_per_epoch",
    "interaction_chunk_events",
    "updates_per_epoch",
    "utd_ratio",
    "batch_size",
    "reward_scale",
    "critic_loss",
    "huber_delta",
    "ensemble_size",
    "hidden_dim",
    "online_fraction",
    "target_entropy",
    "initial_temperature",
    "iql_advantage_temperature",
    "offline_cql_alpha",
    "cql_n_actions",
    "ratio_update_interval",
    "offline_checkpoint_every",
    "checkpoint_every",
    "log_every",
    "resource_interval",
    "no_device_cache",
    "device_cache_observation_dtype",
    "compile_networks",
    "compile_mode",
}
ALLOWED_ENVIRONMENT = {"CC", "CPATH", "LD_PRELOAD", "TMPDIR"}
SYSTEM_LIBCURL_PRELOAD = Path("/usr/lib/x86_64-linux-gnu/libcurl.so.4")
SYSTEM_LIBCURL_CONTRACT = "system-libcurl-soname-v4"
V7_ADMISSION_BUILDER = ROOT / "reproducibility" / "build_training_admission_v7.py"
V7_INITIALIZER_DEPENDENCY_SCHEMA = "transit-offline-initializer-dependency-v7-v1"


def _validate_v7_admission(
    path: Path,
    *,
    dataset: Path,
    sim_env: Path,
    simulator: dict[str, Any],
    sources: dict[str, str],
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    admission_path = path.expanduser().resolve()
    if not admission_path.is_file():
        raise FileNotFoundError(f"v7 training admission is missing: {admission_path}")
    payload = json.loads(admission_path.read_text())
    if (
        payload.get("schema_version") != V7_ADMISSION_SCHEMA
        or payload.get("study_stage") != "formal"
        or payload.get("passed") is not True
    ):
        raise ValueError("v7 training admission is not a passed formal certificate")
    if payload.get("builder_sha256") != sha256(V7_ADMISSION_BUILDER):
        raise ValueError("v7 training admission was built by stale admission code")
    declared_dataset = payload.get("dataset", {})
    declared_dataset_path = (
        admission_path.parent / declared_dataset.get("path", "")
    ).resolve()
    if (
        declared_dataset_path != dataset
        or declared_dataset.get("sha256") != sha256(dataset)
        or declared_dataset.get("bytes") != dataset.stat().st_size
    ):
        raise ValueError("v7 training admission is bound to another dataset")
    declared_simulator = payload.get("simulator", {})
    declared_simulator_path = (
        admission_path.parent / declared_simulator.get("path", "")
    ).resolve()
    if (
        declared_simulator_path != sim_env
        or declared_simulator.get("aggregate_sha256")
        != simulator.get("aggregate_sha256")
        or declared_simulator.get("runtime_sha256")
        != simulator.get("runtime_sha256")
    ):
        raise ValueError("v7 training admission is bound to another simulator")
    if payload.get("training_source_sha256") != sources:
        raise ValueError("v7 training admission is stale relative to training source")
    runtime_lock = payload.get("training_runtime_lock")
    if not isinstance(runtime_lock, dict) or not runtime_lock:
        raise ValueError("v7 training admission lacks a runtime lock")

    evidence = payload.get("evidence", {})
    expected_evidence = {
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
    if not isinstance(evidence, dict) or set(evidence) != expected_evidence:
        raise ValueError("v7 training admission evidence set is incomplete")
    for name, descriptor in evidence.items():
        relative_evidence = Path(descriptor.get("path", ""))
        if relative_evidence.is_absolute() or not relative_evidence.parts:
            raise ValueError(f"v7 admission evidence path is invalid: {name}")
        evidence_path = (
            admission_path.parent / relative_evidence
        ).resolve()
        if (
            not evidence_path.is_file()
            or descriptor.get("sha256") != sha256(evidence_path)
            or descriptor.get("bytes") != evidence_path.stat().st_size
        ):
            raise ValueError(f"v7 admission evidence is missing or stale: {name}")
        evidence_payload = json.loads(evidence_path.read_text())
        if evidence_payload.get("schema_version") != descriptor.get(
            "schema_version"
        ):
            raise ValueError(f"v7 admission evidence schema drifted: {name}")

    selected = payload.get("selected_arguments", {})
    expected_arguments = {
        "reward_scale",
        "critic_loss",
        "huber_delta",
        "target_entropy",
        "device_cache_observation_dtype",
        "sim_demand_scales",
    }
    if not isinstance(selected, dict) or set(selected) != expected_arguments:
        raise ValueError("v7 training admission selected arguments are incomplete")
    for job in jobs:
        arguments = job["arguments"]
        mismatches = {
            name: (arguments.get(name), selected[name])
            for name in expected_arguments
            if arguments.get(name) != selected[name]
        }
        if arguments.get("no_device_cache", False):
            mismatches["no_device_cache"] = (True, False)
        if mismatches:
            raise ValueError(
                f"v7 job {job['name']} violates training admission: {mismatches}"
            )
    return {
        "path": str(admission_path),
        "sha256": sha256(admission_path),
        "bytes": admission_path.stat().st_size,
        "schema_version": payload["schema_version"],
        "selected_arguments": selected,
        "runtime_lock": runtime_lock,
    }


def signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _training_admission_artifact_descriptor(
    evidence: dict[str, Any],
) -> dict[str, Any] | None:
    """Return the certificate identity recorded by the trainer artifacts.

    The matrix additionally carries the certificate's expanded runtime lock so it
    can validate the launch environment.  The trainer records the certificate
    path and digest, which already commit to that lock, but does not duplicate the
    expanded lock in every checkpoint and policy artifact.
    """

    admission = evidence.get("training_admission")
    if admission is None:
        return None
    return {
        name: value
        for name, value in admission.items()
        if name != "runtime_lock"
    }


def _declared_path_has_identity(
    declared: Any,
    *,
    recorded_root: Any | None,
    expected_relative: Path,
) -> bool:
    """Match a hashed artifact by logical location after workspace relocation."""

    if not isinstance(declared, str) or not declared:
        return False
    path = Path(declared).expanduser()
    if path.is_absolute():
        if not isinstance(recorded_root, str) or not recorded_root:
            return False
        try:
            path = path.relative_to(Path(recorded_root).expanduser())
        except ValueError:
            return False
    return path == expected_relative


def _initializer_descriptor_matches(
    recorded: Any, expected: dict[str, Any]
) -> bool:
    """Validate initializer identity while permitting a relocated workspace root."""

    if not isinstance(recorded, dict):
        return False
    if {
        name: value for name, value in recorded.items() if name != "path"
    } != {name: value for name, value in expected.items() if name != "path"}:
        return False
    recorded_path = recorded.get("path")
    if not isinstance(recorded_path, str) or not recorded_path:
        return False
    if recorded_path == expected.get("path"):
        return True
    source_run_name = expected.get("source_run_name")
    if not isinstance(source_run_name, str) or not source_run_name:
        return False
    suffix = (source_run_name, "checkpoints", "offline_initializer.pt")
    return Path(recorded_path).is_absolute() and Path(recorded_path).parts[-3:] == suffix


def _recorded_training_admission_descriptor(
    evidence: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any] | None:
    """Validate and retain the trainer's relocatable certificate descriptor."""

    expected = _training_admission_artifact_descriptor(evidence)
    recorded = metadata.get("training_admission")
    if expected is None:
        if recorded is not None:
            raise ValueError("unexpected training admission in run metadata")
        return None
    if not isinstance(recorded, dict):
        raise ValueError("run metadata lacks the training admission descriptor")
    if {
        name: value for name, value in recorded.items() if name != "path"
    } != {name: value for name, value in expected.items() if name != "path"}:
        raise ValueError("run metadata records a different training admission")
    recorded_path = recorded.get("path")
    expected_path = expected.get("path")
    if not all(isinstance(value, str) and value for value in (recorded_path, expected_path)):
        raise ValueError("training admission descriptor path is invalid")
    if recorded_path != expected_path and not (
        Path(recorded_path).is_absolute()
        and Path(expected_path).is_absolute()
        and Path(recorded_path).name == Path(expected_path).name
    ):
        raise ValueError("training admission descriptor path identity changed")
    return recorded


def _resolve_path(value: str, plan_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = plan_path.parent / path
    return path.resolve()


def _validate_arguments(raw: Any, *, location: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{location} must be an object")
    unknown = set(raw) - ALLOWED_ARGUMENTS
    reserved = set(raw) & RESERVED_ARGUMENTS
    if unknown or reserved:
        raise ValueError(
            f"{location} contains unsupported arguments: "
            f"unknown={sorted(unknown)}, reserved={sorted(reserved)}"
        )
    for name, value in raw.items():
        if name == "sim_demand_scales":
            if (
                not isinstance(value, list)
                or not value
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    for item in value
                )
            ):
                raise ValueError(
                    f"{location}.{name} must be a non-empty numeric list"
                )
            continue
        if value is None or isinstance(value, (dict, list)):
            raise ValueError(f"{location}.{name} must be a scalar")
        if name in {"no_device_cache", "compile_networks"} and not isinstance(
            value, bool
        ):
            raise ValueError(f"{location}.{name} must be boolean")
        if name == "device_cache_observation_dtype" and value not in {
            "float32",
            "float16",
        }:
            raise ValueError(
                f"{location}.{name} must be 'float32' or 'float16'"
            )
        if name == "compile_mode" and value not in {
            "default",
            "max-autotune-no-cudagraphs",
        }:
            raise ValueError(
                f"{location}.{name} must be 'default' or "
                "'max-autotune-no-cudagraphs'"
            )
        if name not in {"no_device_cache", "compile_networks"} and isinstance(
            value, bool
        ):
            raise ValueError(f"{location}.{name} cannot be boolean")
    return dict(raw)


def _environment_path(value: str, plan_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = plan_path.parent / path
    return path.resolve()


def _resolve_required_environment(
    raw: Any, *, plan_path: Path
) -> tuple[dict[str, str], dict[str, Any]]:
    if raw is None:
        return {}, {}
    if not isinstance(raw, dict):
        raise ValueError("required_environment must be an object")
    unknown = set(raw) - ALLOWED_ENVIRONMENT
    if unknown:
        raise ValueError(
            "required_environment contains unsupported variables: "
            + ", ".join(sorted(unknown))
        )
    if any(not isinstance(value, str) or not value for value in raw.values()):
        raise ValueError("required_environment values must be non-empty strings")

    values: dict[str, str] = {}
    fingerprint: dict[str, Any] = {}
    if "CC" in raw:
        configured = raw["CC"]
        if os.sep not in configured:
            resolved_command = shutil.which(configured)
            if resolved_command is None:
                raise FileNotFoundError(
                    f"required compiler is unavailable on PATH: {configured}"
                )
            compiler = Path(resolved_command).resolve()
        else:
            compiler = _environment_path(configured, plan_path)
        if not compiler.is_file() or not os.access(str(compiler), os.X_OK):
            raise FileNotFoundError(
                f"required compiler is not an executable file: {compiler}"
            )
        values["CC"] = str(compiler)
        fingerprint["CC"] = {
            "sha256": sha256(compiler),
        }
    if "CPATH" in raw:
        resolved_entries = []
        descriptors = []
        for configured in raw["CPATH"].split(os.pathsep):
            if not configured:
                raise ValueError("required_environment.CPATH contains an empty entry")
            include_dir = _environment_path(configured, plan_path)
            if not include_dir.is_dir():
                raise FileNotFoundError(
                    f"required compiler include directory is missing: {include_dir}"
                )
            crypt_header = include_dir / "crypt.h"
            if not crypt_header.is_file():
                raise FileNotFoundError(
                    f"required compiler include lacks crypt.h: {include_dir}"
                )
            resolved_entries.append(str(include_dir))
            descriptors.append(
                {
                    "crypt_h_sha256": sha256(crypt_header),
                }
            )
        values["CPATH"] = os.pathsep.join(resolved_entries)
        fingerprint["CPATH"] = descriptors
    if "LD_PRELOAD" in raw:
        configured_preload = Path(raw["LD_PRELOAD"]).expanduser()
        if configured_preload != SYSTEM_LIBCURL_PRELOAD:
            raise ValueError(
                "required_environment.LD_PRELOAD is restricted to "
                f"{SYSTEM_LIBCURL_PRELOAD}"
            )
        preload = configured_preload.resolve(strict=True)
        if not preload.is_file():
            raise FileNotFoundError(f"required preload library is missing: {preload}")
        values["LD_PRELOAD"] = str(configured_preload)
        fingerprint["LD_PRELOAD"] = {
            "contract": SYSTEM_LIBCURL_CONTRACT,
            "sha256": sha256(preload),
        }
    if "TMPDIR" in raw:
        configured_temporary = raw["TMPDIR"]
        temporary = _environment_path(configured_temporary, plan_path)
        if not temporary.exists() and configured_temporary.startswith("~/"):
            temporary.mkdir(parents=True, exist_ok=True)
        if not temporary.is_dir() or not os.access(str(temporary), os.W_OK | os.X_OK):
            raise FileNotFoundError(
                f"required temporary directory is not writable: {temporary}"
            )
        values["TMPDIR"] = str(temporary)
        fingerprint["TMPDIR"] = {"contract": "writable-executable-directory"}
    return values, fingerprint


def _required_environment_identity(fingerprint: dict[str, Any]) -> dict[str, Any]:
    """Return host-neutral fields used in signed job identities."""

    identity = dict(fingerprint)
    if "LD_PRELOAD" in identity:
        identity["LD_PRELOAD"] = {"contract": SYSTEM_LIBCURL_CONTRACT}
    return identity


def load_plan(
    path: Path,
    *,
    _dependency_stack: frozenset[Path] = frozenset(),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan_path = path.expanduser().resolve()
    if plan_path in _dependency_stack:
        raise ValueError(f"cyclic training-plan dependency: {plan_path}")
    dependency_stack = _dependency_stack | {plan_path}
    payload = json.loads(plan_path.read_text())
    plan_schema = payload.get("schema_version")
    if plan_schema not in PLAN_SCHEMA_TO_MDP_VERSION:
        raise ValueError(
            "training plan must use one of "
            f"{sorted(PLAN_SCHEMA_TO_MDP_VERSION)}"
        )
    mdp_version = PLAN_SCHEMA_TO_MDP_VERSION[plan_schema]
    declared_mdp_version = payload.get("mdp_version", mdp_version)
    if declared_mdp_version != mdp_version:
        raise ValueError(
            f"training plan schema {plan_schema} implies {mdp_version}, but "
            f"mdp_version is {declared_mdp_version!r}"
        )
    schema_contract = SCHEMA_CONTRACTS[mdp_version]
    trainer = TRAINERS[mdp_version]
    if not trainer.is_file():
        raise FileNotFoundError(f"training entrypoint is missing: {trainer}")
    for name in ("dataset", "sim_env", "output_root", "device", "jobs"):
        if name not in payload:
            raise ValueError(f"training plan lacks {name!r}")
    dataset = _resolve_path(payload["dataset"], plan_path)
    sim_env = _resolve_path(payload["sim_env"], plan_path)
    output_root = _resolve_path(payload["output_root"], plan_path)
    if not dataset.is_file():
        raise FileNotFoundError(f"training dataset is missing: {dataset}")
    if not sim_env.is_dir():
        raise FileNotFoundError(f"training simulator is missing: {sim_env}")
    device = payload["device"]
    if not isinstance(device, str) or not device:
        raise ValueError("training-plan device must be a non-empty string")
    allow_shared_gpu = payload.get("allow_shared_gpu", False)
    if not isinstance(allow_shared_gpu, bool):
        raise ValueError("training-plan allow_shared_gpu must be boolean")
    max_canonical_processes_per_gpu = payload.get(
        "max_canonical_processes_per_gpu", 1
    )
    if (
        isinstance(max_canonical_processes_per_gpu, bool)
        or not isinstance(max_canonical_processes_per_gpu, int)
        or not 1
        <= max_canonical_processes_per_gpu
        <= MAX_CANONICAL_GPU_PROCESSES
    ):
        raise ValueError(
            "training-plan max_canonical_processes_per_gpu must lie in "
            f"[1, {MAX_CANONICAL_GPU_PROCESSES}]"
        )
    if allow_shared_gpu and "max_canonical_processes_per_gpu" in payload:
        raise ValueError(
            "training-plan allow_shared_gpu is incompatible with an explicit "
            "canonical-process quota"
        )
    allow_external_gpu_processes = payload.get(
        "allow_external_gpu_processes", False
    )
    if not isinstance(allow_external_gpu_processes, bool):
        raise ValueError(
            "training-plan allow_external_gpu_processes must be boolean"
        )
    shared_arguments = _validate_arguments(
        payload.get("shared_arguments"), location="shared_arguments"
    )
    required_environment, environment_fingerprint = _resolve_required_environment(
        payload.get("required_environment"), plan_path=plan_path
    )
    environment_identity = _required_environment_identity(environment_fingerprint)
    raw_jobs = payload["jobs"]
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError("training plan jobs must be a non-empty list")

    names: set[str] = set()
    jobs = []
    for index, raw in enumerate(raw_jobs):
        if not isinstance(raw, dict):
            raise ValueError(f"jobs[{index}] must be an object")
        name = raw.get("name")
        method = raw.get("method")
        seed = raw.get("seed")
        if not isinstance(name, str) or not SAFE_NAME.fullmatch(name):
            raise ValueError(f"jobs[{index}] has an unsafe name")
        if name in names:
            raise ValueError(f"duplicate training job name {name!r}")
        names.add(name)
        if method not in METHOD_PROFILES:
            raise ValueError(f"jobs[{index}] has unknown method {method!r}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"jobs[{index}].seed must be an integer")
        job_max_processes = raw.get(
            "max_canonical_processes_per_gpu",
            max_canonical_processes_per_gpu,
        )
        if (
            isinstance(job_max_processes, bool)
            or not isinstance(job_max_processes, int)
            or not 1 <= job_max_processes <= MAX_CANONICAL_GPU_PROCESSES
        ):
            raise ValueError(
                f"jobs[{index}].max_canonical_processes_per_gpu must be a "
                f"value in [1, {MAX_CANONICAL_GPU_PROCESSES}]"
            )
        if allow_shared_gpu and "max_canonical_processes_per_gpu" in raw:
            raise ValueError(
                f"jobs[{index}] cannot combine legacy allow_shared_gpu with an "
                "explicit canonical-process quota"
            )
        arguments = dict(shared_arguments)
        arguments.update(
            _validate_arguments(raw.get("arguments"), location=f"jobs[{index}].arguments")
        )
        jobs.append(
            {
                "name": name,
                "method": method,
                "seed": seed,
                "arguments": arguments,
                "max_canonical_processes_per_gpu": job_max_processes,
                "run_dir": output_root / name,
                "offline_initializer_dependency_raw": raw.get(
                    "offline_initializer"
                ),
            }
        )

    simulator = simulator_input_hashes(
        sim_env,
        runtime_calibration_version=("v7_v4" if mdp_version == "v7" else "v4"),
    )
    sources = source_hashes(mdp_version)
    admission = None
    admission_path = None
    if mdp_version == "v7":
        raw_admission = payload.get("training_admission")
        if not isinstance(raw_admission, str) or not raw_admission:
            raise ValueError("v7 training plan requires training_admission")
        admission_path = _resolve_path(raw_admission, plan_path)
        admission = _validate_v7_admission(
            admission_path,
            dataset=dataset,
            sim_env=sim_env,
            simulator=simulator,
            sources=sources,
            jobs=jobs,
        )

    evidence = {
        "mdp_version": mdp_version,
        "plan_schema": schema_contract["plan"],
        "matrix_schema": schema_contract["matrix"],
        "summary_schema": schema_contract["summary"],
        "plan": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset),
        "sim_env": str(sim_env),
        "simulator": simulator,
        "output_root": str(output_root),
        "device": device,
        "allow_shared_gpu": allow_shared_gpu,
        "max_canonical_processes_per_gpu": max_canonical_processes_per_gpu,
        "allow_external_gpu_processes": allow_external_gpu_processes,
        "source_sha256": sources,
        "runner_sha256": sha256(Path(__file__).resolve()),
        "trainer_entrypoint": str(trainer),
        "trainer_entrypoint_sha256": sha256(trainer),
        "required_environment": environment_fingerprint,
        "required_environment_identity": environment_identity,
        "resolved_required_environment": dict(required_environment),
    }
    if admission is not None:
        evidence["training_admission"] = admission

    dependency_source_cache: dict[
        Path, tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]
    ] = {}
    for index, job in enumerate(jobs):
        raw_dependency = job.pop("offline_initializer_dependency_raw")
        if raw_dependency is None:
            continue
        if mdp_version != "v7" or job["method"] not in V7_SHARED_IQL_BRANCH_METHODS:
            raise ValueError(
                f"jobs[{index}] has an unregistered offline initializer dependency"
            )
        if not isinstance(raw_dependency, dict) or set(raw_dependency) != {
            "schema_version",
            "path",
            "source_plan",
            "source_plan_sha256",
            "source_job_name",
            "source_run_signature",
        }:
            raise ValueError(
                f"jobs[{index}].offline_initializer has an invalid contract"
            )
        if raw_dependency.get("schema_version") != V7_INITIALIZER_DEPENDENCY_SCHEMA:
            raise ValueError(
                f"jobs[{index}].offline_initializer has an invalid schema"
            )
        source_plan_raw = raw_dependency.get("source_plan")
        initializer_raw = raw_dependency.get("path")
        if not all(
            isinstance(value, str) and value
            for value in (source_plan_raw, initializer_raw)
        ):
            raise ValueError(
                f"jobs[{index}].offline_initializer paths must be non-empty strings"
            )
        source_plan_path = _resolve_path(source_plan_raw, plan_path)
        initializer_path = _resolve_path(initializer_raw, plan_path)
        if source_plan_path not in dependency_source_cache:
            source_jobs, source_evidence = load_plan(
                source_plan_path,
                _dependency_stack=dependency_stack,
            )
            source_payload = json.loads(source_plan_path.read_text())
            dependency_source_cache[source_plan_path] = (
                source_jobs,
                source_evidence,
                source_payload,
            )
        source_jobs, source_evidence, source_payload = dependency_source_cache[
            source_plan_path
        ]
        if (
            raw_dependency.get("source_plan_sha256") != sha256(source_plan_path)
            or source_payload.get("schema_version") != SCHEMA_CONTRACTS["v7"]["plan"]
            or source_payload.get("execution_class") != "server-h2oplus"
            or source_payload.get("registration", {}).get("status")
            != "frozen-before-formal-outcomes"
        ):
            raise ValueError(
                f"jobs[{index}] is bound to a stale or non-formal source plan"
            )
        source_job_name = raw_dependency.get("source_job_name")
        source_job = next(
            (candidate for candidate in source_jobs if candidate["name"] == source_job_name),
            None,
        )
        if (
            source_job is None
            or source_job["method"] != "h2oplus"
            or source_job["seed"] != job["seed"]
            or source_job["signature"]
            != raw_dependency.get("source_run_signature")
            or initializer_path
            != source_job["run_dir"] / "checkpoints" / "offline_initializer.pt"
        ):
            raise ValueError(
                f"jobs[{index}] offline initializer is not the same-seed H2O+ source"
            )
        current_admission = evidence.get("training_admission", {}).get("sha256")
        source_admission = source_evidence.get("training_admission", {}).get(
            "sha256"
        )
        if not (
            source_evidence["dataset_sha256"] == evidence["dataset_sha256"]
            and source_evidence["simulator"]["runtime_sha256"]
            == evidence["simulator"]["runtime_sha256"]
            and source_evidence["simulator"]["aggregate_sha256"]
            == evidence["simulator"]["aggregate_sha256"]
            and source_evidence["source_sha256"] == evidence["source_sha256"]
            and source_admission == current_admission
        ):
            raise ValueError(
                f"jobs[{index}] offline initializer source uses different evidence"
            )
        job["offline_initializer"] = initializer_path
        job["offline_initializer_source_signature"] = source_job["signature"]
        job["offline_initializer_dependency"] = dict(raw_dependency)

    evidence["offline_initializer_source_plans"] = [
        {
            "path": str(source_plan),
            "sha256": source_evidence["plan_sha256"],
        }
        for source_plan, (_, source_evidence, _) in sorted(
            dependency_source_cache.items(), key=lambda item: str(item[0])
        )
    ]
    for job in jobs:
        job["dataset"] = dataset
        job["sim_env"] = sim_env
        job["output_root"] = output_root
        job["device"] = device
        job["allow_shared_gpu"] = allow_shared_gpu
        job["allow_external_gpu_processes"] = allow_external_gpu_processes
        job["mdp_version"] = mdp_version
        job["trainer"] = trainer
        job["required_environment"] = dict(required_environment)
        if admission_path is not None:
            job["training_admission"] = admission_path
        job["signature"] = signature(
            {
                "mdp_version": mdp_version,
                "plan_sha256": evidence["plan_sha256"],
                "dataset_sha256": evidence["dataset_sha256"],
                "simulator_aggregate_sha256": evidence["simulator"][
                    "aggregate_sha256"
                ],
                "source_sha256": evidence["source_sha256"],
                "runner_sha256": evidence["runner_sha256"],
                "trainer_entrypoint_sha256": evidence["trainer_entrypoint_sha256"],
                "training_admission_sha256": (
                    admission["sha256"] if admission is not None else None
                ),
                "method": job["method"],
                "method_profile": asdict(METHOD_PROFILES[job["method"]]),
                "seed": job["seed"],
                "device": device,
                "allow_shared_gpu": allow_shared_gpu,
                "max_canonical_processes_per_gpu": job[
                    "max_canonical_processes_per_gpu"
                ],
                "allow_external_gpu_processes": allow_external_gpu_processes,
                "arguments": job["arguments"],
                "offline_initializer_dependency": job.get(
                    "offline_initializer_dependency"
                ),
                "required_environment": environment_identity,
            }
        )
    return jobs, evidence


def _argument_tokens(arguments: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    for name, value in sorted(arguments.items()):
        option = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                tokens.append(option)
        elif isinstance(value, list):
            tokens.append(option)
            tokens.extend(str(item) for item in value)
        else:
            tokens.extend((option, str(value)))
    return tokens


def job_command(job: dict[str, Any], *, resume: Path | None = None) -> list[str]:
    command = [
        sys.executable,
        str(job.get("trainer", TRAINER)),
        "--method",
        job["method"],
        "--seed",
        str(job["seed"]),
        "--run-name",
        job["name"],
        "--run-signature",
        job["signature"],
        "--dataset",
        str(job["dataset"]),
        "--sim-env",
        str(job["sim_env"]),
        "--output-root",
        str(job["output_root"]),
        "--device",
        job["device"],
        *_argument_tokens(job["arguments"]),
    ]
    if job.get("mdp_version") == "v7":
        command.extend(
            ("--training-admission", str(job["training_admission"]))
        )
        if job.get("offline_initializer") is not None:
            command.extend(
                (
                    "--offline-initializer",
                    str(job["offline_initializer"]),
                    "--offline-initializer-source-signature",
                    job["offline_initializer_source_signature"],
                )
            )
    if resume is not None:
        command.extend(("--resume", str(resume)))
    if job["allow_shared_gpu"]:
        command.append("--allow-shared-gpu")
    else:
        command.extend(
            (
                "--max-canonical-processes-per-gpu",
                str(job["max_canonical_processes_per_gpu"]),
            )
        )
    if job.get("allow_external_gpu_processes", False):
        command.append("--allow-external-gpu-processes")
    return command


def job_environment(job: dict[str, Any]) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(job.get("required_environment", {}))
    return environment


def validate_training_runtime(evidence: dict[str, Any]) -> dict[str, Any] | None:
    """Reject v7 execution under software other than the precision-gate runtime."""

    if evidence.get("mdp_version") != "v7":
        return None
    expected = evidence.get("training_admission", {}).get("runtime_lock")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("v7 execution evidence lacks the admitted runtime lock")
    overrides = evidence.get("resolved_required_environment", {})
    previous = {name: os.environ.get(name) for name in overrides}
    os.environ.update(overrides)
    try:
        import torch

        observed = collect_runtime_fingerprint(torch, evidence["device"])
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    if observed["required"] != expected:
        mismatches = {
            name: {
                "expected": expected.get(name),
                "observed": observed["required"].get(name),
            }
            for name in sorted(set(expected) | set(observed["required"]))
            if expected.get(name) != observed["required"].get(name)
        }
        raise RuntimeError(
            "v7 training runtime differs from the precision-gate runtime: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return observed


def _v7_initializer_records(
    job: dict[str, Any],
    evidence: dict[str, Any],
    summary: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    from SimpleSAC.transit_v7.trainer import (
        OFFLINE_INITIALIZER_IGNORED_CONFIG_FIELDS,
        OFFLINE_INITIALIZER_SCHEMA,
        inspect_offline_initializer_v7,
    )

    training_admission = _recorded_training_admission_descriptor(evidence, metadata)
    consumes = job.get("offline_initializer") is not None
    produces = (
        job["method"] == "h2oplus"
        and int(job["arguments"].get("offline_updates", -1)) == 50_000
        and evidence.get("training_admission") is not None
    )
    if consumes and produces:
        raise ValueError("a v7 run cannot both consume and produce an initializer")
    path = None
    expected_signature = None
    exact_source_config = False
    if consumes:
        path = Path(job["offline_initializer"])
        expected_signature = job["offline_initializer_source_signature"]
    elif produces:
        path = job["run_dir"] / "checkpoints" / "offline_initializer.pt"
        expected_signature = job["signature"]
        exact_source_config = True

    expected_consumed = None
    expected_produced = None
    if path is not None:
        if not path.is_file():
            raise FileNotFoundError(path)
        inspected = inspect_offline_initializer_v7(path)
        extra = inspected["extra"]
        expected_extra = {
            "phase": "offline_pretraining",
            "source_method": "h2oplus",
            "source_run_name": f"h2oplus_seed{job['seed']}",
            "source_run_signature": expected_signature,
            "source_training_seed": job["seed"],
            "offline_updates_completed": 50_000,
            "trainer_total_updates": 50_000,
            "dataset_sha256": evidence["dataset_sha256"],
            "simulator_runtime_sha256": evidence["simulator"]["runtime_sha256"],
            "simulator_aggregate_sha256": evidence["simulator"][
                "aggregate_sha256"
            ],
            "source_sha256": evidence["source_sha256"],
            "contract_config": metadata.get("contract_config"),
            "mdp_version": "v7",
            "training_admission": training_admission,
            "offline_metrics": summary.get("offline_metrics"),
        }
        if any(extra.get(name) != value for name, value in expected_extra.items()):
            raise ValueError("v7 offline initializer provenance is incomplete")
        if inspected["total_updates"] != 50_000:
            raise ValueError("v7 offline initializer has the wrong update count")
        trainer_config = metadata.get("trainer_config")
        compatibility = {
            name: value
            for name, value in trainer_config.items()
            if name not in OFFLINE_INITIALIZER_IGNORED_CONFIG_FIELDS
        }
        if inspected["offline_compatibility_config"] != compatibility:
            raise ValueError("v7 offline initializer config is incompatible")
        if exact_source_config and inspected["source_config"] != trainer_config:
            raise ValueError("v7 source initializer config drifted")
        descriptor = {
            "path": str(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "schema_version": OFFLINE_INITIALIZER_SCHEMA,
            "source_method": "h2oplus",
            "source_run_name": f"h2oplus_seed{job['seed']}",
            "source_run_signature": expected_signature,
            "source_training_seed": job["seed"],
            "offline_updates": 50_000,
        }
        if consumes:
            expected_consumed = descriptor
        else:
            expected_produced = descriptor

    validated = {}
    for name, expected in (
        ("offline_initializer_consumed", expected_consumed),
        ("offline_initializer_produced", expected_produced),
    ):
        records = [document.get(name) for document in (summary, metadata)]
        if records[0] != records[1]:
            raise ValueError("v7 initializer records disagree across run artifacts")
        if expected is None:
            if records[0] is not None:
                raise ValueError("v7 initializer records disagree across run artifacts")
            validated[name] = None
            continue
        if not _initializer_descriptor_matches(records[0], expected):
            raise ValueError("v7 initializer records disagree across run artifacts")
        validated[name] = records[0]
    consumed = validated["offline_initializer_consumed"]
    produced = validated["offline_initializer_produced"]
    return consumed, produced


def _v7_final_policy_is_complete(
    job: dict[str, Any],
    evidence: dict[str, Any],
    summary: dict[str, Any],
    metadata: dict[str, Any],
    checkpoint_sha256: str,
) -> bool:
    import torch

    from SimpleSAC.transit_v7.trainer import (
        POLICY_ARTIFACT_SCHEMA,
        load_actor_checkpoint_v7,
    )

    final_policy_path = job["run_dir"] / "checkpoints" / "final_actor.pt"
    descriptor = summary.get("final_policy")
    if not isinstance(descriptor, dict) or not final_policy_path.is_file():
        return False
    policy_sha256 = sha256(final_policy_path)
    if not (
        _declared_path_has_identity(
            descriptor.get("path"),
            recorded_root=summary.get("run_dir"),
            expected_relative=Path("checkpoints") / "final_actor.pt",
        )
        and descriptor.get("sha256") == policy_sha256
        and descriptor.get("source_training_checkpoint_sha256")
        == checkpoint_sha256
    ):
        return False
    actor, extra = load_actor_checkpoint_v7(
        final_policy_path,
        "cpu",
        require_policy_artifact=True,
    )
    if not all(bool(torch.isfinite(parameter).all()) for parameter in actor.parameters()):
        return False
    consumed, produced = _v7_initializer_records(
        job, evidence, summary, metadata
    )
    training_admission = _recorded_training_admission_descriptor(evidence, metadata)
    expected = {
        "phase": "complete",
        "run_signature": job["signature"],
        "method": job["method"],
        "method_profile": asdict(METHOD_PROFILES[job["method"]]),
        "training_seed": job["seed"],
        "dataset_sha256": evidence["dataset_sha256"],
        "simulator_runtime_sha256": evidence["simulator"]["runtime_sha256"],
        "simulator_aggregate_sha256": evidence["simulator"]["aggregate_sha256"],
        "source_sha256": evidence["source_sha256"],
        "contract_config": metadata.get("contract_config"),
        "mdp_version": "v7",
        "trainer_config": metadata.get("trainer_config"),
        "training_checkpoint_sha256": checkpoint_sha256,
        "training_admission": training_admission,
        "offline_initializer_consumed": consumed,
        "offline_initializer_produced": produced,
        "policy_artifact_schema": POLICY_ARTIFACT_SCHEMA,
        "total_updates": summary.get("total_updates"),
        "simulator_decision_events": summary.get("simulator_decision_events"),
    }
    return all(extra.get(name) == value for name, value in expected.items())


def _is_complete(job: dict[str, Any], evidence: dict[str, Any]) -> bool:
    run_dir = job["run_dir"]
    summary_path = run_dir / "summary.json"
    metadata_path = run_dir / "metadata.json"
    checkpoint = run_dir / "checkpoints" / "latest.pt"
    if not summary_path.is_file() or not metadata_path.is_file() or not checkpoint.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text())
        metadata = json.loads(metadata_path.read_text())
        expected_summary_schema = evidence.get("summary_schema", SUMMARY_SCHEMA)
        checkpoint_sha256 = sha256(checkpoint)
        complete = (
            summary.get("schema_version") == expected_summary_schema
            and metadata.get("mdp_version") == job.get("mdp_version", "v4")
            and summary.get("run_signature") == job["signature"]
            and summary.get("method") == job["method"]
            and summary.get("seed") == job["seed"]
            and summary.get("checkpoint", {}).get("sha256") == checkpoint_sha256
            and metadata.get("run_signature") == job["signature"]
            and metadata.get("dataset", {}).get("sha256")
            == evidence["dataset_sha256"]
            and metadata.get("simulator", {}).get("aggregate_sha256")
            == evidence["simulator"]["aggregate_sha256"]
            and metadata.get("source_sha256") == evidence["source_sha256"]
            and (
                job.get("mdp_version") != "v7"
                or metadata.get("training_admission", {}).get("sha256")
                == evidence.get("training_admission", {}).get("sha256")
            )
        )
        if not complete:
            return False
        return job.get("mdp_version") != "v7" or _v7_final_policy_is_complete(
            job,
            evidence,
            summary,
            metadata,
            checkpoint_sha256,
        )
    except Exception:
        return False


def _existing_signature(run_dir: Path) -> str | None:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        value = json.loads(metadata_path.read_text()).get("run_signature")
        return value if isinstance(value, str) else None
    except Exception:
        return None


def _prepare_existing_run(
    job: dict[str, Any],
) -> tuple[Path | None, Path | None]:
    """Return a checkpoint or preserve an interrupted pre-checkpoint run."""

    run_dir = job["run_dir"]
    if not run_dir.exists():
        return None, None
    existing_signature = _existing_signature(run_dir)
    if existing_signature != job["signature"]:
        raise RuntimeError(
            f"existing run {run_dir} has a different or missing signature; "
            "use a new job name instead of overwriting evidence"
        )
    checkpoint = run_dir / "checkpoints" / "latest.pt"
    if checkpoint.is_file():
        return checkpoint, None

    archive_root = run_dir.parent / ".incomplete_runs"
    archive_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    archive = archive_root / f"{run_dir.name}.{timestamp}"
    os.replace(run_dir, archive)
    return None, archive


def run_job(job: dict[str, Any], evidence: dict[str, Any], log_dir: Path) -> dict[str, Any]:
    if _is_complete(job, evidence):
        return {
            "status": "completed",
            "run_dir": str(job["run_dir"]),
            "reused_existing_run": True,
        }
    run_dir = job["run_dir"]
    resume, archived_run = _prepare_existing_run(job)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job['name']}.log"
    with log_path.open("a") as log:
        log.write(
            f"\n[{datetime.now(timezone.utc).isoformat()}] "
            f"signature={job['signature']} resume={resume}\n"
        )
        log.flush()
        result = subprocess.run(
            job_command(job, resume=resume),
            env=job_environment(job),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0 or not _is_complete(job, evidence):
        return {
            "status": "failed",
            "returncode": result.returncode,
            "run_dir": str(run_dir),
            "log": str(log_path),
            "archived_incomplete_run": (
                str(archived_run) if archived_run is not None else None
            ),
        }
    return {
        "status": "completed",
        "run_dir": str(run_dir),
        "log": str(log_path),
        "reused_existing_run": False,
        "archived_incomplete_run": (
            str(archived_run) if archived_run is not None else None
        ),
    }


def _serializable_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": job["name"],
        "method": job["method"],
        "seed": job["seed"],
        "arguments": job["arguments"],
        "run_dir": str(job["run_dir"]),
        "signature": job["signature"],
        "allow_shared_gpu": job["allow_shared_gpu"],
        "max_canonical_processes_per_gpu": job[
            "max_canonical_processes_per_gpu"
        ],
        "allow_external_gpu_processes": job.get(
            "allow_external_gpu_processes", False
        ),
        "mdp_version": job.get("mdp_version", "v4"),
        "required_environment": job.get("required_environment", {}),
        "offline_initializer_dependency": job.get(
            "offline_initializer_dependency"
        ),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _archive_existing_manifest(path: Path) -> Path | None:
    """Move an earlier matrix attempt into immutable timestamped history."""

    if not path.is_file():
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    digest = sha256(path)
    archive_root = path.parent / "matrix_manifest_history"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive = archive_root / f"matrix_manifest.{timestamp}.{digest[:16]}.json"
    os.replace(path, archive)
    return archive


def _select_jobs(
    jobs: list[dict[str, Any]], requested_names: list[str] | None
) -> list[dict[str, Any]]:
    if not requested_names:
        return jobs
    if len(requested_names) != len(set(requested_names)):
        raise ValueError("--job-name cannot contain duplicates")
    by_name = {job["name"]: job for job in jobs}
    missing = [name for name in requested_names if name not in by_name]
    if missing:
        raise ValueError(f"training plan has no requested jobs: {missing}")
    return [by_name[name] for name in requested_names]


def _run_job_shard(
    jobs: list[dict[str, Any]], evidence: dict[str, Any], output_root: Path
) -> None:
    """Run an explicitly assigned serial shard without locking the whole matrix."""

    worker_root = output_root / "launcher_workers"
    worker_root.mkdir(parents=True, exist_ok=True)
    lock_root = output_root / ".job-locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    hostname = re.sub(r"[^A-Za-z0-9._-]", "_", socket.gethostname())
    worker_path = worker_root / f"{timestamp}.{hostname}.pid{os.getpid()}.json"
    manifest = {
        "schema_version": "transit-training-worker-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "launcher": {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "plan": evidence["plan"],
        "plan_sha256": evidence["plan_sha256"],
        "output_root": str(output_root),
        "jobs": [_serializable_job(job) for job in jobs],
        "results": {},
    }
    _write_json(worker_path, manifest)
    failures = []
    for index, job in enumerate(jobs, start=1):
        job_lock = (lock_root / f"{job['name']}.lock").open("a+")
        try:
            try:
                fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                result = {
                    "status": "busy",
                    "run_dir": str(job["run_dir"]),
                    "error": "another launcher on this host holds the job lock",
                }
            else:
                result = run_job(job, evidence, output_root / "launcher_logs")
        finally:
            try:
                fcntl.flock(job_lock, fcntl.LOCK_UN)
            finally:
                job_lock.close()
        manifest["results"][job["name"]] = {
            **result,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(worker_path, manifest)
        print(
            f"[{index}/{len(jobs)}] {result['status']} "
            f"{job['name']} method={job['method']} seed={job['seed']}",
            flush=True,
        )
        if result["status"] in {"failed", "busy"}:
            failures.append((job, result))
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["status"] = "failed" if failures else "completed"
    _write_json(worker_path, manifest)
    if failures:
        details = "\n".join(
            f"{job['name']}: {result.get('log') or result.get('error')}"
            for job, result in failures
        )
        raise RuntimeError(f"{len(failures)} shard jobs did not complete:\n{details}")
    print(
        f"DONE training matrix shard jobs={len(jobs)} worker={worker_path}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--job-name",
        action="append",
        help=(
            "run only this registered job; repeat to form one serial worker shard. "
            "Run the plan once without selectors after all shards to audit/aggregate."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs, evidence = load_plan(args.plan)
    evidence["training_runtime"] = validate_training_runtime(evidence)
    output_root = Path(evidence["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    if args.job_name:
        _run_job_shard(_select_jobs(jobs, args.job_name), evidence, output_root)
        return
    lock = (output_root / ".training-matrix.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(
            f"training matrix already running under {output_root}"
        ) from error

    manifest_path = output_root / "matrix_manifest.json"
    previous_manifest = _archive_existing_manifest(manifest_path)
    manifest = {
        "schema_version": evidence.get("matrix_schema", MATRIX_SCHEMA),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **evidence,
        "jobs": [_serializable_job(job) for job in jobs],
        "results": {},
        "previous_manifest": (
            str(previous_manifest) if previous_manifest is not None else None
        ),
    }
    _write_json(manifest_path, manifest)
    failures = []
    try:
        for index, job in enumerate(jobs, start=1):
            result = run_job(job, evidence, output_root / "launcher_logs")
            manifest["results"][job["name"]] = {
                **result,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            }
            _write_json(manifest_path, manifest)
            print(
                f"[{index}/{len(jobs)}] {result['status']} "
                f"{job['name']} method={job['method']} seed={job['seed']}",
                flush=True,
            )
            if result["status"] == "failed":
                failures.append((job, result))
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["status"] = "failed" if failures else "completed"
    _write_json(manifest_path, manifest)
    if failures:
        details = "\n".join(
            f"{job['name']}: {result.get('log')}" for job, result in failures
        )
        raise RuntimeError(f"{len(failures)} training jobs failed:\n{details}")
    print(
        f"DONE training matrix jobs={len(jobs)} manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
