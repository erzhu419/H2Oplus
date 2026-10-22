#!/usr/bin/env python3
"""Build a fail-closed, relocatable admission certificate for v7 training."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.mdp_contract_v7 import SCHEMA_VERSION as DATASET_SCHEMA
from bus_h2o.sumo_provenance import sha256
from reproducibility.audit_device_cache_precision_v6 import (
    MINIMUM_ONLINE_ROWS as PRECISION_MINIMUM_ONLINE_ROWS,
    MINIMUM_PAIRED_UPDATES,
    MINIMUM_PAIRED_OFFLINE_ROWS,
    MINIMUM_SAMPLE_ROWS,
    SCHEMA_VERSION as PRECISION_SCHEMA,
    precision_audit_source_hashes,
)
from reproducibility.audit_reward_conditioning_v7 import (
    AUDIT_SCHEMA as REWARD_AUDIT_SCHEMA,
    reward_audit_source_hashes,
)
from reproducibility.audit_transit_v7_dataset import (
    AUDIT_SCHEMA as DATASET_AUDIT_SCHEMA,
    dataset_audit_source_hashes,
)
from reproducibility.audit_serialized_credit_v7 import (
    SCHEMA_VERSION as CREDIT_AUDIT_SCHEMA,
    credit_audit_source_hashes,
)
from reproducibility.audit_simulator_fidelity_v7 import (
    SCHEMA_VERSION as ZERO_FIDELITY_SCHEMA,
    fidelity_source_hashes,
)
from reproducibility.audit_simulator_intervention_fidelity_v7 import (
    DEMAND_SCALES,
    POLICIES,
    SCHEMA_VERSION as INTERVENTION_SCHEMA,
    SIMULATOR_SEEDS,
    intervention_source_hashes,
)
from reproducibility.run_hybrid_entropy_pilot_v7 import (
    MINIMUM_OFFLINE_UPDATES as ENTROPY_MINIMUM_OFFLINE_UPDATES,
    MINIMUM_ONLINE_ROWS as ENTROPY_MINIMUM_ONLINE_ROWS,
    MINIMUM_ONLINE_UPDATES as ENTROPY_MINIMUM_ONLINE_UPDATES,
    PILOT_SEEDS as ENTROPY_PILOT_SEEDS,
    SCHEMA_VERSION as ENTROPY_PILOT_SCHEMA,
    TARGET_ENTROPY_CANDIDATES,
    entropy_target_gates,
    entropy_pilot_source_hashes,
)
from reproducibility.run_reward_loss_pilot_v7 import (
    CRITIC_LOSSES,
    DEFAULT_REWARD_SCALES,
    MINIMUM_UPDATES as REWARD_LOSS_MINIMUM_UPDATES,
    PILOT_SEEDS as REWARD_LOSS_PILOT_SEEDS,
    SCHEMA_VERSION as REWARD_LOSS_PILOT_SCHEMA,
    configuration_gates,
    reward_loss_source_hashes,
    select_reward_loss_configuration,
)
from SimpleSAC.train_transit_v3 import simulator_input_hashes, source_hashes


SCHEMA_VERSION = "transit-training-admission-v7-v2"
METHOD_AUDIT_SCHEMA = "transit-method-protocol-audit-v7-v2"
FORMAL_DEMAND_SCALES = (0.75, 1.0, 1.25)
RUNTIME_LOCK_FIELDS = {
    "python_version",
    "python_implementation",
    "python_cache_tag",
    "python_executable_sha256",
    "packages",
    "torch_cuda_version",
    "torch_binary_sha256",
    "compiler",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--pilot-dataset", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--serialized-credit-audit", type=Path, required=True)
    parser.add_argument("--sim-env", type=Path, required=True)
    parser.add_argument("--zero-fidelity-audit", type=Path, required=True)
    parser.add_argument("--intervention-fidelity-audit", type=Path, required=True)
    parser.add_argument("--reward-audit", type=Path, required=True)
    parser.add_argument("--reward-loss-pilot", type=Path, required=True)
    parser.add_argument("--entropy-pilot", type=Path, required=True)
    parser.add_argument("--device-precision-audit", type=Path, required=True)
    parser.add_argument("--method-protocol-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _same_float(first: Any, second: Any) -> bool:
    try:
        return math.isclose(float(first), float(second), rel_tol=0.0, abs_tol=0.0)
    except (TypeError, ValueError):
        return False


def validated_pilot_dataset(
    pilot_dataset: Path, reward_loss: dict[str, Any]
) -> Path:
    """Bind remote pilot evidence to an explicitly supplied local artifact."""

    pilot_dataset = pilot_dataset.expanduser().resolve()
    descriptor = reward_loss.get("dataset", {})
    _require(
        pilot_dataset.is_file()
        and isinstance(descriptor, dict)
        and descriptor.get("sha256") == sha256(pilot_dataset),
        "reward/loss pilot dataset is missing or stale",
    )
    return pilot_dataset


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def portable_simulator_identity(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the complete path-independent identity of simulator inputs."""

    _require(
        isinstance(payload, dict)
        and payload.get("schema_version") == "transit-simulator-inputs-v1",
        "simulator input identity has an invalid schema",
    )
    line_ids = payload.get("line_ids")
    runtime_files = payload.get("runtime_files_sha256")
    report = payload.get("calibration_report")
    _require(
        isinstance(line_ids, list)
        and line_ids
        and all(isinstance(value, str) and value for value in line_ids)
        and len(line_ids) == len(set(line_ids)),
        "simulator input identity has invalid line identifiers",
    )
    _require(
        isinstance(runtime_files, dict)
        and runtime_files
        and all(
            isinstance(name, str)
            and name
            and _is_sha256(digest)
            for name, digest in runtime_files.items()
        ),
        "simulator input identity has invalid runtime file hashes",
    )
    _require(
        isinstance(report, dict)
        and all(
            _is_sha256(report.get(name))
            for name in (
                "sha256",
                "source_manifest_sha256",
                "source_bus_objects_sha256",
            )
        )
        and _is_sha256(payload.get("runtime_sha256"))
        and _is_sha256(payload.get("aggregate_sha256")),
        "simulator input identity has invalid aggregate hashes",
    )
    return {
        "schema_version": payload["schema_version"],
        "line_ids": list(line_ids),
        "runtime_files_sha256": dict(runtime_files),
        "runtime_sha256": payload["runtime_sha256"],
        "calibration_report": {
            name: report[name]
            for name in (
                "sha256",
                "source_manifest_sha256",
                "source_bus_objects_sha256",
            )
        },
        "aggregate_sha256": payload["aggregate_sha256"],
    }


def validated_training_runtime_lock(precision: dict[str, Any]) -> dict[str, Any]:
    fingerprint = precision.get("runtime_fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError("device precision audit lacks a runtime fingerprint")
    required = fingerprint.get("required")
    host = fingerprint.get("host")
    if (
        not isinstance(required, dict)
        or set(required) != RUNTIME_LOCK_FIELDS
        or not isinstance(host, dict)
        or not str(host.get("device", "")).startswith("cuda")
        or not isinstance(host.get("gpu"), str)
        or not host["gpu"]
    ):
        raise ValueError("device precision runtime fingerprint is incomplete")
    packages = required.get("packages")
    torch_binaries = required.get("torch_binary_sha256")
    compiler = required.get("compiler")
    if (
        not _is_sha256(required.get("python_executable_sha256"))
        or not isinstance(packages, dict)
        or not packages
        or any(not isinstance(value, str) or not value for value in packages.values())
        or not isinstance(torch_binaries, dict)
        or set(torch_binaries) != {"python_extension", "cuda_library"}
        or any(not _is_sha256(value) for value in torch_binaries.values())
        or not isinstance(compiler, dict)
        or not _is_sha256(compiler.get("sha256"))
        or not isinstance(compiler.get("version"), str)
        or not compiler["version"]
    ):
        raise ValueError("device precision runtime lock has invalid identities")
    return dict(required)


def _descriptor(
    path: Path, output_dir: Path, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    result = {
        "path": os.path.relpath(path, output_dir.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }
    if payload is not None:
        result["schema_version"] = payload.get("schema_version")
    return result


def _collection_jobs(dataset: Path) -> tuple[set[str], set[str]]:
    with h5py.File(dataset, "r") as handle:
        _require(
            handle.attrs.get("schema_version") == DATASET_SCHEMA,
            f"dataset {dataset} is not transit-smdp-v7",
        )
        raw = handle.attrs.get("collection_matrix")
        _require(raw is not None, f"dataset {dataset} lacks collection_matrix")
        matrix = json.loads(raw)
        _require(
            isinstance(matrix, dict),
            f"dataset {dataset} has invalid collection_matrix metadata",
        )
        declared_jobs = matrix.get("jobs")
        _require(
            isinstance(declared_jobs, int)
            and not isinstance(declared_jobs, bool)
            and declared_jobs > 0,
            f"dataset {dataset} has invalid collection job count",
        )

        raw_shards = handle.attrs.get("source_shards")
        _require(raw_shards is not None, f"dataset {dataset} lacks source_shards")
        shards = json.loads(raw_shards)
        _require(
            isinstance(shards, list)
            and len(shards) == declared_jobs
            and all(isinstance(shard, dict) for shard in shards),
            f"dataset {dataset} source shards disagree with the collection job count",
        )
        signatures = [shard.get("job_signature") for shard in shards]

        raw_split_names = handle.attrs.get("split_names")
        _require(
            raw_split_names is not None,
            f"dataset {dataset} lacks split_names",
        )
        split_names = json.loads(raw_split_names)
        _require(
            isinstance(split_names, dict)
            and split_names.get("train") is not None
            and all(
                isinstance(name, str)
                and isinstance(value, int)
                and not isinstance(value, bool)
                for name, value in split_names.items()
            )
            and len(set(split_names.values())) == len(split_names),
            f"dataset {dataset} has invalid split_names",
        )
        split_ids = handle.get("split_id")
        _require(
            split_ids is not None and len(split_ids) > 0,
            f"dataset {dataset} lacks nonempty split_id rows",
        )
        observed_split_ids: set[int] = set()
        chunk_rows = 1_000_000
        for start in range(0, len(split_ids), chunk_rows):
            observed_split_ids.update(
                int(value)
                for value in np.unique(split_ids[start : start + chunk_rows])
            )
        names_by_id = {value: name for name, value in split_names.items()}
        _require(
            observed_split_ids <= set(names_by_id),
            f"dataset {dataset} contains an unregistered split identifier",
        )

    _require(
        signatures and all(_is_sha256(value) for value in signatures),
        f"dataset {dataset} has invalid collection job signatures",
    )
    _require(
        len(signatures) == len(set(signatures)),
        f"dataset {dataset} repeats collection job signatures",
    )
    splits = {names_by_id[value] for value in observed_split_ids}
    return set(signatures), splits


def validate_pilot_subset(pilot_dataset: Path, formal_dataset: Path) -> dict[str, Any]:
    pilot_signatures, pilot_splits = _collection_jobs(pilot_dataset)
    formal_signatures, formal_splits = _collection_jobs(formal_dataset)
    _require(pilot_splits == {"train"}, "pilot dataset contains a non-training split")
    _require(formal_splits == {"train"}, "formal dataset contains a non-training split")
    _require(
        pilot_signatures < formal_signatures,
        "numerical pilot jobs must be a strict subset of formal training jobs",
    )
    return {
        "passed": True,
        "pilot_jobs": len(pilot_signatures),
        "formal_jobs": len(formal_signatures),
        "shared_jobs": len(pilot_signatures & formal_signatures),
        "pilot_is_strict_subset": True,
        "job_signature_source": "HDF5 source_shards[].job_signature",
        "split_source": "HDF5 split_id mapped by split_names",
    }


def _validate_current_file_map(files: dict[str, str], label: str) -> None:
    _require(isinstance(files, dict) and files, f"{label} file map is empty")
    for relative, expected in files.items():
        path = Path(relative)
        _require(not path.is_absolute() and ".." not in path.parts, f"{label} path is unsafe")
        current = ROOT / path
        _require(current.is_file(), f"{label} source is missing: {relative}")
        _require(sha256(current) == expected, f"{label} source is stale: {relative}")


def _validate_reward_loss_execution(payload: dict[str, Any]) -> None:
    design = payload.get("design", {})
    updates = int(design.get("updates_per_run", -1))
    _require(
        payload.get("study_stage") == "numerical-pilot"
        and design.get("reward_scales") == list(DEFAULT_REWARD_SCALES)
        and design.get("critic_losses") == list(CRITIC_LOSSES)
        and design.get("seeds") == list(REWARD_LOSS_PILOT_SEEDS)
        and design.get("minimum_updates_per_run")
        == REWARD_LOSS_MINIMUM_UPDATES
        and updates >= REWARD_LOSS_MINIMUM_UPDATES,
        "reward/loss pilot design is incomplete or below its minimum horizon",
    )
    runs = payload.get("runs", [])
    expected = {
        (float(scale), loss, int(seed))
        for scale in DEFAULT_REWARD_SCALES
        for loss in CRITIC_LOSSES
        for seed in REWARD_LOSS_PILOT_SEEDS
    }
    observed = set()
    for run in runs:
        key = (
            float(run.get("reward_scale", math.nan)),
            run.get("critic_loss"),
            int(run.get("seed", -1)),
        )
        observed.add(key)
        config = run.get("trainer_config", {})
        _require(
            int(run.get("updates", -1)) == updates
            and _same_float(config.get("reward_scale"), key[0])
            and config.get("critic_loss") == key[1]
            and isinstance(run.get("numerical_gate", {}).get("eligible"), bool),
            "reward/loss pilot run does not match its registered configuration",
        )
    _require(
        len(runs) == len(expected) and observed == expected,
        "reward/loss pilot does not cover every registered configuration and seed",
    )
    paired = payload.get("paired_initialization_checks", {})
    expected_paired = {
        f"seed{seed}/{module}"
        for seed in REWARD_LOSS_PILOT_SEEDS
        for module in ("actor", "critic", "target_critic", "value", "ratio")
    }
    _require(
        set(paired) == expected_paired and all(value is True for value in paired.values()),
        "reward/loss pilot initialization was not fully paired",
    )
    recomputed_gates = configuration_gates(runs)
    _require(
        payload.get("configuration_gates") == recomputed_gates,
        "reward/loss pilot gates do not reproduce from its runs",
    )


def _validate_entropy_execution(payload: dict[str, Any]) -> None:
    design = payload.get("design", {})
    offline_updates = int(design.get("offline_updates", -1))
    online_updates = int(design.get("online_updates", -1))
    online_rows = int(design.get("online_rows", -1))
    _require(
        payload.get("study_stage") == "numerical-pilot"
        and design.get("target_entropy_candidates")
        == list(TARGET_ENTROPY_CANDIDATES)
        and design.get("seeds") == list(ENTROPY_PILOT_SEEDS)
        and design.get("minimum_offline_updates")
        == ENTROPY_MINIMUM_OFFLINE_UPDATES
        and design.get("minimum_online_updates")
        == ENTROPY_MINIMUM_ONLINE_UPDATES
        and design.get("minimum_online_rows") == ENTROPY_MINIMUM_ONLINE_ROWS
        and offline_updates >= ENTROPY_MINIMUM_OFFLINE_UPDATES
        and online_updates >= ENTROPY_MINIMUM_ONLINE_UPDATES
        and online_rows >= ENTROPY_MINIMUM_ONLINE_ROWS,
        "entropy pilot design is incomplete or below its minimum horizon",
    )
    runs = payload.get("runs", [])
    expected = {
        (float(target), int(seed))
        for target in TARGET_ENTROPY_CANDIDATES
        for seed in ENTROPY_PILOT_SEEDS
    }
    observed = set()
    for run in runs:
        key = (
            float(run.get("target_entropy", math.nan)),
            int(run.get("seed", -1)),
        )
        observed.add(key)
        config = run.get("trainer_config", {})
        _require(
            int(run.get("offline_updates", -1)) == offline_updates
            and int(run.get("online_updates", -1)) == online_updates
            and _same_float(config.get("target_entropy"), key[0])
            and isinstance(run.get("numerical_gate", {}).get("eligible"), bool),
            "entropy pilot run does not match its registered configuration",
        )
    _require(
        len(runs) == len(expected) and observed == expected,
        "entropy pilot does not cover every registered target and seed",
    )
    expected_paired = {
        f"seed{seed}/{module}"
        for seed in ENTROPY_PILOT_SEEDS
        for module in ("actor", "critic", "target_critic", "value", "ratio")
    }
    for name in ("paired_initialization_checks", "paired_online_start_checks"):
        paired = payload.get(name, {})
        _require(
            set(paired) == expected_paired
            and all(value is True for value in paired.values()),
            f"entropy pilot {name} was not fully paired",
        )
    _require(
        payload.get("target_entropy_gate") == entropy_target_gates(runs),
        "entropy pilot gate does not reproduce from its runs",
    )


def build_admission(
    *,
    dataset: Path,
    pilot_dataset: Path,
    dataset_audit_path: Path,
    credit_audit_path: Path,
    sim_env: Path,
    zero_fidelity_path: Path,
    intervention_path: Path,
    reward_audit_path: Path,
    reward_loss_pilot_path: Path,
    entropy_pilot_path: Path,
    precision_path: Path,
    method_audit_path: Path,
    output: Path,
) -> dict[str, Any]:
    dataset = dataset.expanduser().resolve()
    sim_env = sim_env.expanduser().resolve()
    output = output.expanduser().resolve()
    _require(dataset.is_file(), f"formal dataset is missing: {dataset}")
    _require(sim_env.is_dir(), f"simulator environment is missing: {sim_env}")
    dataset_digest = sha256(dataset)
    simulator = simulator_input_hashes(
        sim_env, runtime_calibration_version="v7_v4"
    )

    dataset_audit = _load(dataset_audit_path, "dataset audit")
    _require(
        dataset_audit.get("schema_version") == DATASET_AUDIT_SCHEMA
        and dataset_audit.get("passed") is True
        and dataset_audit.get("dataset_sha256") == dataset_digest
        and dataset_audit.get("source_sha256") == dataset_audit_source_hashes(),
        "formal dataset audit is stale or failed",
    )

    credit_audit = _load(credit_audit_path, "serialized credit audit")
    credit_counters = credit_audit.get("counters", {})
    _require(
        credit_audit.get("schema_version") == CREDIT_AUDIT_SCHEMA
        and credit_audit.get("passed") is True
        and credit_audit.get("dataset_sha256") == dataset_digest
        and int(credit_audit.get("multi_action_batches", 0)) > 0
        and int(
            credit_audit.get("maximum_simultaneous_decisions_observed", 0)
        )
        > 1
        and credit_audit.get("bellman_credit", {}).get(
            "maximum_absolute_slot_target_error"
        )
        == 0.0
        and isinstance(credit_counters, dict)
        and credit_counters
        and all(int(value) == 0 for value in credit_counters.values())
        and credit_audit.get("source_sha256") == credit_audit_source_hashes(),
        "serialized credit audit is stale, uncovered, or failed",
    )

    reward_loss = _load(reward_loss_pilot_path, "reward/loss pilot")
    _require(
        reward_loss.get("schema_version") == REWARD_LOSS_PILOT_SCHEMA,
        "reward/loss pilot schema is invalid",
    )
    pilot_dataset = validated_pilot_dataset(pilot_dataset, reward_loss)
    reward_audit = _load(reward_audit_path, "reward audit")
    _require(
        reward_audit.get("schema_version") == REWARD_AUDIT_SCHEMA
        and reward_audit.get("dataset_sha256") == sha256(pilot_dataset)
        and reward_audit.get("source_sha256") == reward_audit_source_hashes(),
        "reward conditioning audit is stale or bound to another pilot dataset",
    )
    _require(
        reward_loss.get("reward_audit", {}).get("sha256")
        == sha256(reward_audit_path.expanduser().resolve())
        and reward_loss.get("source_sha256") == reward_loss_source_hashes(),
        "reward/loss pilot evidence or source is stale",
    )
    _validate_reward_loss_execution(reward_loss)
    recomputed_selection = select_reward_loss_configuration(
        reward_audit, reward_loss.get("configuration_gates", {})
    )
    _require(
        recomputed_selection.get("passed") is True
        and reward_loss.get("configuration_selection") == recomputed_selection,
        "reward/loss selection does not reproduce from registered evidence",
    )
    selected = dict(recomputed_selection["selected"])

    entropy = _load(entropy_pilot_path, "entropy pilot")
    target_entropy = entropy.get("target_entropy_gate", {}).get(
        "selected_target_entropy"
    )
    entropy_config = entropy.get("reward_loss_configuration", {})
    _validate_entropy_execution(entropy)
    _require(
        entropy.get("schema_version") == ENTROPY_PILOT_SCHEMA
        and entropy.get("dataset", {}).get("sha256") == sha256(pilot_dataset)
        and entropy.get("target_entropy_gate", {}).get("passed") is True
        and target_entropy is not None
        and _same_float(entropy_config.get("reward_scale"), selected["reward_scale"])
        and entropy_config.get("critic_loss") == selected["critic_loss"]
        and _same_float(entropy_config.get("huber_delta"), selected["huber_delta"])
        and entropy_config.get("pilot_sha256")
        == sha256(reward_loss_pilot_path.expanduser().resolve())
        and entropy.get("source_sha256") == entropy_pilot_source_hashes(),
        "hybrid entropy pilot is stale, failed, or bound to another configuration",
    )
    selected["target_entropy"] = float(target_entropy)

    subset = validate_pilot_subset(pilot_dataset, dataset)

    precision = _load(precision_path, "device precision audit")
    precision_config = precision.get("trainer_config", {})
    precision_decision = precision.get("precision_decision", {})
    precision_selection = precision.get("selection_evidence", {})
    precision_minimums = precision.get("registered_minimums", {})
    full_storage = precision.get("full_dataset_storage_error", {})
    full_arrays = full_storage.get("arrays", {})
    _require(
        precision.get("schema_version") == PRECISION_SCHEMA
        and precision.get("dataset_sha256") == dataset_digest
        and precision.get("mdp_version") == "v7"
        and precision.get("method") == "h2oplus"
        and precision.get("phase") == "online"
        and int(precision.get("max_offline_rows", -1))
        >= MINIMUM_PAIRED_OFFLINE_ROWS
        and int(precision.get("sample_rows", -1)) >= MINIMUM_SAMPLE_ROWS
        and int(precision.get("online_rows", -1))
        >= PRECISION_MINIMUM_ONLINE_ROWS
        and int(precision.get("paired_updates", -1)) >= MINIMUM_PAIRED_UPDATES
        and precision_minimums
        == {
            "paired_offline_rows": MINIMUM_PAIRED_OFFLINE_ROWS,
            "sample_rows": MINIMUM_SAMPLE_ROWS,
            "online_rows": PRECISION_MINIMUM_ONLINE_ROWS,
            "paired_updates": MINIMUM_PAIRED_UPDATES,
        }
        and int(full_storage.get("training_rows", -1))
        == int(dataset_audit.get("rows", -2))
        and set(full_arrays) == {"observations", "next_observations"}
        and all(
            row.get("all_finite") is True
            and int(row.get("rows", -1)) == int(dataset_audit.get("rows", -2))
            for row in full_arrays.values()
        )
        and precision_decision.get("passed") is True
        and _same_float(precision_config.get("reward_scale"), selected["reward_scale"])
        and precision_config.get("critic_loss") == selected["critic_loss"]
        and _same_float(precision_config.get("huber_delta"), selected["huber_delta"])
        and _same_float(precision_config.get("target_entropy"), selected["target_entropy"])
        and precision_selection.get("reward_loss_pilot", {}).get("sha256")
        == sha256(reward_loss_pilot_path.expanduser().resolve())
        and precision_selection.get("reward_loss_pilot", {}).get("schema_version")
        == REWARD_LOSS_PILOT_SCHEMA
        and precision_selection.get("entropy_pilot", {}).get("sha256")
        == sha256(entropy_pilot_path.expanduser().resolve())
        and precision_selection.get("entropy_pilot", {}).get("schema_version")
        == ENTROPY_PILOT_SCHEMA
        and precision_selection.get("selected_arguments")
        == {
            "reward_scale": selected["reward_scale"],
            "critic_loss": selected["critic_loss"],
            "huber_delta": selected["huber_delta"],
            "target_entropy": selected["target_entropy"],
        }
        and precision.get("source_sha256") == precision_audit_source_hashes("v7"),
        "device precision audit is stale, incomplete, or bound to another configuration",
    )
    training_runtime_lock = validated_training_runtime_lock(precision)
    selected["device_cache_observation_dtype"] = precision_decision[
        "selected_observation_storage_dtype"
    ]
    selected["sim_demand_scales"] = list(FORMAL_DEMAND_SCALES)

    zero_fidelity = _load(zero_fidelity_path, "zero-policy fidelity audit")
    _require(
        zero_fidelity.get("schema_version") == ZERO_FIDELITY_SCHEMA
        and zero_fidelity.get("study_stage") == "formal"
        and zero_fidelity.get("passed") is True
        and portable_simulator_identity(zero_fidelity.get("simulator_inputs", {}))
        == portable_simulator_identity(simulator)
        and zero_fidelity.get("source_sha256") == fidelity_source_hashes(),
        "zero-policy simulator fidelity evidence is stale or failed",
    )

    intervention = _load(intervention_path, "intervention fidelity audit")
    expected_intervention_rows = (
        len(POLICIES) * len(DEMAND_SCALES) * len(SIMULATOR_SEEDS)
    )
    _require(
        intervention.get("schema_version") == INTERVENTION_SCHEMA
        and intervention.get("study_stage") == "formal"
        and intervention.get("passed") is True
        and intervention.get("invariants_passed") is True
        and intervention.get("matrix_manifest", {}).get("jobs")
        == expected_intervention_rows
        and len(intervention.get("sumo_rows", [])) == expected_intervention_rows
        and len(intervention.get("simulator_episodes", []))
        == expected_intervention_rows
        and portable_simulator_identity(
            intervention.get("simulator", {}).get("inputs", {})
        )
        == portable_simulator_identity(simulator)
        and intervention.get("source_sha256") == intervention_source_hashes(),
        "intervention simulator fidelity evidence is stale, partial, or failed",
    )

    method_audit = _load(method_audit_path, "method protocol audit")
    frozen = method_audit.get("task_adaptation_freeze", {})
    expected_freeze = {
        key: selected[key]
        for key in (
            "reward_scale",
            "critic_loss",
            "huber_delta",
            "target_entropy",
            "device_cache_observation_dtype",
        )
    }
    _require(
        method_audit.get("schema_version") == METHOD_AUDIT_SCHEMA
        and method_audit.get("verdict") == "passed"
        and method_audit.get("freeze_complete") is True
        and method_audit.get("unresolved_before_formal_training") == []
        and frozen == expected_freeze,
        "method protocol audit is unresolved or bound to another adaptation freeze",
    )
    _validate_current_file_map(
        method_audit.get("local_files", {}), "method protocol audit"
    )

    output_dir = output.parent
    evidence_paths = {
        "dataset_audit": (dataset_audit_path, dataset_audit),
        "serialized_credit_audit": (credit_audit_path, credit_audit),
        "reward_audit": (reward_audit_path, reward_audit),
        "reward_loss_pilot": (reward_loss_pilot_path, reward_loss),
        "entropy_pilot": (entropy_pilot_path, entropy),
        "device_precision_audit": (precision_path, precision),
        "zero_fidelity_audit": (zero_fidelity_path, zero_fidelity),
        "intervention_fidelity_audit": (intervention_path, intervention),
        "method_protocol_audit": (method_audit_path, method_audit),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": "formal",
        "passed": True,
        "selection_boundary": (
            "reward, loss, and entropy are frozen from a strict training-only "
            "pilot subset; no reserved evaluation cell is read"
        ),
        "dataset": {
            "path": os.path.relpath(dataset, output_dir),
            "sha256": dataset_digest,
            "bytes": dataset.stat().st_size,
            "schema_version": DATASET_SCHEMA,
        },
        "pilot_dataset": {
            "path": os.path.relpath(pilot_dataset, output_dir),
            "sha256": sha256(pilot_dataset),
            "bytes": pilot_dataset.stat().st_size,
            "subset_audit": subset,
        },
        "simulator": {
            "path": os.path.relpath(sim_env, output_dir),
            "aggregate_sha256": simulator["aggregate_sha256"],
            "runtime_sha256": simulator["runtime_sha256"],
        },
        "selected_arguments": selected,
        "training_runtime_lock": training_runtime_lock,
        "evidence": {
            name: _descriptor(path, output_dir, payload)
            for name, (path, payload) in evidence_paths.items()
        },
        "training_source_sha256": source_hashes("v7"),
        "builder_sha256": sha256(Path(__file__).resolve()),
    }


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite training admission {output}")
    payload = build_admission(
        dataset=args.dataset,
        pilot_dataset=args.pilot_dataset,
        dataset_audit_path=args.dataset_audit,
        credit_audit_path=args.serialized_credit_audit,
        sim_env=args.sim_env,
        zero_fidelity_path=args.zero_fidelity_audit,
        intervention_path=args.intervention_fidelity_audit,
        reward_audit_path=args.reward_audit,
        reward_loss_pilot_path=args.reward_loss_pilot,
        entropy_pilot_path=args.entropy_pilot,
        precision_path=args.device_precision_audit,
        method_audit_path=args.method_protocol_audit,
        output=output,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "passed": payload["passed"],
                "selected_arguments": payload["selected_arguments"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
