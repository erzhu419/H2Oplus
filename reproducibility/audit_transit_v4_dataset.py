#!/usr/bin/env python3
"""Independently audit a derived transit-smdp-v4 HDF5 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.mdp_contract_v4 import (
    CANONICAL_STATE_INDEX,
    CANONICAL_STATE_NAMES,
    DEFAULT_CONFIG_V4,
    RAW_EVENT_STATE_INDEX,
    RAW_EVENT_STATE_NAMES,
    SCHEMA_VERSION,
    continuous_time_discount_v4,
    raw_event_to_canonical_v4,
    transition_cost_components_v4,
    transition_duration_seconds_v4,
)


PERCENTILES = (0, 1, 5, 25, 50, 75, 95, 99, 100)
REQUIRED_DATASETS = (
    "observations",
    "actions",
    "rewards",
    "next_observations",
    "discounts",
    "terminals",
    "raw_observations",
    "raw_next_observations",
    "waiting_cost",
    "holding_cost",
    "delta_t_seconds",
    "policy_id",
    "policy_seed",
    "sumo_seed",
    "split_id",
    "episode_id",
    "trajectory_id",
    "scenario_id",
    "demand_scale",
    "source_shard_id",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_summary(values: np.ndarray) -> dict[str, Any]:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = flattened[np.isfinite(flattened)]
    if not finite.size:
        return {"n": int(flattened.size), "finite": 0}
    quantiles = np.percentile(finite, PERCENTILES)
    return {
        "n": int(flattened.size),
        "finite": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "percentiles": {
            str(percentile): float(value)
            for percentile, value in zip(PERCENTILES, quantiles)
        },
    }


def _maximum_absolute_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(actual, dtype=np.float64)
                - np.asarray(expected, dtype=np.float64)
            )
        )
    )


def logical_content_sha256(
    handle: h5py.File, metadata: dict[str, Any], chunk_rows: int = 65_536
) -> str:
    """Recompute the builder's container-independent logical content hash."""

    digest = hashlib.sha256()
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    )
    for name in sorted(handle.keys()):
        dataset = handle[name]
        header = {
            "name": name,
            "dtype": dataset.dtype.str,
            "shape": dataset.shape,
        }
        digest.update(
            json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        )
        for start in range(0, dataset.shape[0], chunk_rows):
            values = np.ascontiguousarray(dataset[start : start + chunk_rows])
            digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def _source_integrity(
    source_descriptors: list[dict[str, Any]],
    correction_provenance: dict[str, Any],
    *,
    verify_source_files: bool,
) -> dict[str, Any]:
    missing = []
    mismatched = []
    metadata_mismatched = []
    matrix_cells = []
    source_correction = Counter()
    source_minima: dict[str, float] = {}

    if not verify_source_files:
        return {
            "verified": False,
            "declared_shards": len(source_descriptors),
            "reason": "source-file verification disabled",
        }

    for descriptor in source_descriptors:
        path = Path(descriptor["path"])
        if not path.is_file():
            missing.append(str(path))
            continue
        actual_hash = sha256(path)
        if actual_hash != descriptor["sha256"]:
            mismatched.append(
                {
                    "path": str(path),
                    "declared": descriptor["sha256"],
                    "actual": actual_hash,
                }
            )
            continue
        summary_path = path.with_suffix(path.suffix + ".json")
        if not summary_path.is_file():
            metadata_mismatched.append({"path": str(path), "reason": "missing summary"})
            continue
        summary = json.loads(summary_path.read_text())
        if summary.get("shard_sha256") != actual_hash:
            metadata_mismatched.append(
                {"path": str(path), "reason": "summary hash mismatch"}
            )
        matrix_cells.append(
            (
                summary.get("policy_name"),
                summary.get("scenario", {}).get("manifest_sha256"),
                summary.get("sumo_seed"),
            )
        )
        with np.load(path, allow_pickle=False) as source:
            for endpoint in ("observations", "next_observations"):
                observations = np.asarray(source[endpoint])
                for field in (
                    "forward_headway_seconds",
                    "backward_headway_seconds",
                ):
                    values = observations[:, RAW_EVENT_STATE_INDEX[field]]
                    key = f"{endpoint}.{field}"
                    source_correction[key] += int(np.sum(values < 0.0))
                    source_minima[key] = min(
                        source_minima.get(key, float("inf")), float(np.min(values))
                    )

    declared_fields = correction_provenance.get("fields", {})
    correction_matches = all(
        source_correction.get(name, 0) == int(details["corrected_values"])
        and np.isclose(
            source_minima.get(name, float("nan")),
            float(details["minimum_source_seconds"]),
        )
        for name, details in declared_fields.items()
    )
    return {
        "verified": not missing and not mismatched and not metadata_mismatched,
        "declared_shards": len(source_descriptors),
        "verified_shards": len(source_descriptors) - len(missing) - len(mismatched),
        "missing": missing,
        "hash_mismatches": mismatched,
        "metadata_mismatches": metadata_mismatched,
        "unique_matrix_cells": len(set(matrix_cells)),
        "duplicate_matrix_cells": len(matrix_cells) - len(set(matrix_cells)),
        "measurement_correction_matches_sources": bool(correction_matches),
        "source_negative_headway_counts": dict(sorted(source_correction.items())),
        "source_minimum_headways_seconds": dict(sorted(source_minima.items())),
    }


def audit(path: Path, *, verify_source_files: bool = True) -> dict[str, Any]:
    path = path.expanduser().resolve()
    errors: list[str] = []
    disclosures: list[str] = []

    with h5py.File(path, "r") as handle:
        attrs = {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in handle.attrs.items()
        }
        missing = sorted(set(REQUIRED_DATASETS).difference(handle.keys()))
        extra = sorted(set(handle.keys()).difference(REQUIRED_DATASETS))
        if missing:
            raise KeyError(f"dataset is missing required arrays: {missing}")
        arrays = {name: handle[name][:] for name in REQUIRED_DATASETS}
        shapes = {
            name: {"shape": list(dataset.shape), "dtype": str(dataset.dtype)}
            for name, dataset in sorted(handle.items())
        }
        config = json.loads(attrs["config"])
        state_names = json.loads(attrs["state_names"])
        raw_state_names = json.loads(attrs["raw_state_names"])
        policy_name_to_id = json.loads(attrs["policy_names"])
        split_names = json.loads(attrs["split_names"])
        scenario_to_id = json.loads(attrs["scenario_manifest_hashes"])
        station_max = json.loads(attrs["station_max_by_line"])
        content_metadata = {
            "schema_version": attrs["schema_version"],
            "config": config,
            "state_names": state_names,
            "raw_state_names": raw_state_names,
            "policy_names": policy_name_to_id,
            "split_names": split_names,
            "scenario_manifest_hashes": scenario_to_id,
            "station_max_by_line": station_max,
        }
        recomputed_content_signature = logical_content_sha256(handle, content_metadata)

    n_rows = arrays["rewards"].shape[0]
    for name, values in arrays.items():
        if values.shape[0] != n_rows:
            errors.append(f"{name} has {values.shape[0]} rows instead of {n_rows}")
        if not np.isfinite(values).all():
            errors.append(f"{name} contains non-finite values")
    if extra:
        errors.append(f"unexpected HDF5 arrays are not covered by the content hash: {extra}")
    if attrs.get("schema_version") != SCHEMA_VERSION:
        errors.append("HDF5 schema version does not match the executable v4 contract")
    if config != DEFAULT_CONFIG_V4.to_dict():
        errors.append("HDF5 contract configuration does not match the executable default")
    if state_names != list(CANONICAL_STATE_NAMES):
        errors.append("canonical state names or order do not match the contract")
    if raw_state_names != list(RAW_EVENT_STATE_NAMES):
        errors.append("raw state names or order do not match the contract")
    if attrs.get("content_signature") != recomputed_content_signature:
        errors.append("logical content signature does not match HDF5 arrays and semantics")

    raw = arrays["raw_observations"]
    raw_next = arrays["raw_next_observations"]
    actions = arrays["actions"]
    if actions.shape != (n_rows, 1) or np.any(np.abs(actions) > 1.00001):
        errors.append("actions do not have shape (N, 1) within [-1, 1]")
    for endpoint, observations in (("raw", raw), ("raw_next", raw_next)):
        for field in ("forward_headway_seconds", "backward_headway_seconds"):
            values = observations[:, RAW_EVENT_STATE_INDEX[field]]
            if np.any(values < 0.0):
                errors.append(f"{endpoint}.{field} remains negative after derivation")
        for field in (
            "target_forward_headway_seconds",
            "target_backward_headway_seconds",
            "bus_capacity",
        ):
            if np.any(observations[:, RAW_EVENT_STATE_INDEX[field]] <= 0.0):
                errors.append(f"{endpoint}.{field} contains non-positive values")
        for field in ("forward_bus_present", "backward_bus_present"):
            values = observations[:, RAW_EVENT_STATE_INDEX[field]]
            if np.any((values != 0.0) & (values != 1.0)):
                errors.append(f"{endpoint}.{field} is not binary")

    canonical = raw_event_to_canonical_v4(raw, station_max)
    canonical_next = raw_event_to_canonical_v4(raw_next, station_max)
    components = transition_cost_components_v4(raw, actions, raw_next)
    durations = transition_duration_seconds_v4(raw, raw_next)
    discounts = continuous_time_discount_v4(durations)
    reconstruction_errors = {
        "observations": _maximum_absolute_error(arrays["observations"], canonical),
        "next_observations": _maximum_absolute_error(
            arrays["next_observations"], canonical_next
        ),
        "waiting_cost": _maximum_absolute_error(
            arrays["waiting_cost"], components["waiting_cost"]
        ),
        "holding_cost": _maximum_absolute_error(
            arrays["holding_cost"], components["holding_cost"]
        ),
        "reward_identity": _maximum_absolute_error(
            arrays["rewards"],
            -(arrays["waiting_cost"] + arrays["holding_cost"]),
        ),
        "reward_from_raw": _maximum_absolute_error(
            arrays["rewards"], -np.asarray(components["total_cost"])
        ),
        "delta_t_seconds": _maximum_absolute_error(
            arrays["delta_t_seconds"], durations
        ),
        "discounts": _maximum_absolute_error(arrays["discounts"], discounts),
    }
    for name, error in reconstruction_errors.items():
        if error != 0.0:
            errors.append(f"{name} reconstruction error is {error}")

    raw_line = np.rint(raw[:, RAW_EVENT_STATE_INDEX["line_id"]]).astype(np.int64)
    next_line = np.rint(raw_next[:, RAW_EVENT_STATE_INDEX["line_id"]]).astype(np.int64)
    raw_station = raw[:, RAW_EVENT_STATE_INDEX["station_index"]]
    next_station = raw_next[:, RAW_EVENT_STATE_INDEX["station_index"]]
    if np.any(raw_line != next_line):
        errors.append("one or more transitions cross directional lines")
    if np.any(next_station <= raw_station):
        errors.append("one or more transitions do not move strictly downstream")
    if np.any(arrays["terminals"] != 0):
        errors.append("continuing same-trip transitions contain terminal flags")

    trajectory_ids = arrays["trajectory_id"]
    episode_ids = arrays["episode_id"]
    policy_ids = arrays["policy_id"]
    source_ids = arrays["source_shard_id"]
    order = np.argsort(trajectory_ids, kind="stable")
    ordered_trajectory = trajectory_ids[order]
    starts = np.r_[
        0,
        np.flatnonzero(ordered_trajectory[1:] != ordered_trajectory[:-1]) + 1,
    ]
    first_episode = episode_ids[order][starts]
    first_policy = policy_ids[order][starts]
    first_source = source_ids[order][starts]
    group = np.searchsorted(ordered_trajectory[starts], ordered_trajectory)
    trajectory_group_conflicts = int(
        np.sum(episode_ids[order] != first_episode[group])
        + np.sum(policy_ids[order] != first_policy[group])
        + np.sum(source_ids[order] != first_source[group])
    )
    if trajectory_group_conflicts:
        errors.append("trajectory IDs cross episode, policy, or source-shard groups")

    build_provenance = json.loads(attrs["build_provenance"])
    source_descriptors = json.loads(attrs["source_shards"])
    source_integrity = _source_integrity(
        source_descriptors,
        build_provenance["measurement_resolution_correction"],
        verify_source_files=verify_source_files,
    )
    if verify_source_files and not source_integrity["verified"]:
        errors.append("one or more immutable source shards failed hash verification")
    if verify_source_files and source_integrity["duplicate_matrix_cells"]:
        errors.append("source shards repeat policy-scenario-SUMO-seed matrix cells")
    if verify_source_files and not source_integrity[
        "measurement_correction_matches_sources"
    ]:
        errors.append("measurement correction provenance disagrees with source shards")

    unique_splits = sorted(map(int, np.unique(arrays["split_id"])))
    if unique_splits == [int(split_names["train"])]:
        disclosures.append(
            "The offline dataset is training-only; model selection and final testing "
            "must use fresh, disjoint simulator and SUMO seeds."
        )
    corrected = int(
        build_provenance["measurement_resolution_correction"][
            "total_corrected_values"
        ]
    )
    if corrected:
        disclosures.append(
            f"The documented one-second measurement-resolution rule corrected "
            f"{corrected} endpoint headway values from immutable source shards."
        )

    policy_id_to_name = {
        int(identifier): name for name, identifier in policy_name_to_id.items()
    }
    by_policy = {}
    for policy_id in sorted(np.unique(policy_ids)):
        selected = policy_ids == policy_id
        policy_actions = actions[selected, 0]
        by_policy[policy_id_to_name[int(policy_id)]] = {
            "rows": int(np.sum(selected)),
            "episodes": int(np.unique(episode_ids[selected]).size),
            "sumo_seeds": sorted(map(int, np.unique(arrays["sumo_seed"][selected]))),
            "exact_no_hold_rate": float(np.mean(policy_actions == -1.0)),
            "hold_seconds": numeric_summary(30.0 * (policy_actions + 1.0)),
            "reward": numeric_summary(arrays["rewards"][selected]),
            "waiting_cost": numeric_summary(arrays["waiting_cost"][selected]),
            "holding_cost": numeric_summary(arrays["holding_cost"][selected]),
        }

    state = {}
    for index, name in enumerate(state_names):
        values = arrays["observations"][:, index]
        state[name] = {
            **numeric_summary(values),
            "exact_zero_rate": float(np.mean(values == 0.0)),
        }
    for name in ("forward_headway_ratio", "backward_headway_ratio"):
        values = arrays["observations"][:, CANONICAL_STATE_INDEX[name]]
        state[name]["at_configured_cap"] = int(
            np.sum(values == DEFAULT_CONFIG_V4.max_headway_ratio)
        )
        if state[name]["at_configured_cap"]:
            disclosures.append(
                f"Canonical feature {name} reaches its configured cap in "
                f"{state[name]['at_configured_cap']} rows."
            )

    report = {
        "schema_version": "transit-smdp-v4-audit-v1",
        "dataset": str(path),
        "dataset_sha256": sha256(path),
        "logical_content_signature": recomputed_content_signature,
        "logical_content_signature_matches": (
            attrs.get("content_signature") == recomputed_content_signature
        ),
        "rows": n_rows,
        "episodes": int(np.unique(episode_ids).size),
        "trajectories": int(np.unique(trajectory_ids).size),
        "trajectory_group_conflicts": trajectory_group_conflicts,
        "attributes": attrs,
        "datasets": shapes,
        "reconstruction_max_absolute_error": reconstruction_errors,
        "source_integrity": source_integrity,
        "matrix": {
            "policies": len(policy_name_to_id),
            "scenarios": len(scenario_to_id),
            "sumo_seeds": sorted(map(int, np.unique(arrays["sumo_seed"]))),
            "splits": unique_splits,
        },
        "actions": {
            "exact_no_hold_rate": float(np.mean(actions[:, 0] == -1.0)),
            "exact_max_hold_rate": float(np.mean(actions[:, 0] == 1.0)),
            "hold_seconds": numeric_summary(30.0 * (actions[:, 0] + 1.0)),
        },
        "rewards": numeric_summary(arrays["rewards"]),
        "waiting_cost": numeric_summary(arrays["waiting_cost"]),
        "holding_cost": numeric_summary(arrays["holding_cost"]),
        "delta_t_seconds": numeric_summary(arrays["delta_t_seconds"]),
        "state": state,
        "by_policy": by_policy,
        "errors": errors,
        "disclosures": disclosures,
        "passed": not errors,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-source-files", action="store_true")
    args = parser.parse_args()
    report = audit(
        args.dataset,
        verify_source_files=not args.skip_source_files,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({
        "output": str(output),
        "passed": report["passed"],
        "errors": report["errors"],
        "disclosures": report["disclosures"],
    }, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
