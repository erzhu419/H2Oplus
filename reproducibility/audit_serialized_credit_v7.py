#!/usr/bin/env python3
"""Audit Bellman credit flow through serialized simultaneous v7 decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.mdp_contract_v7 import (
    MAX_SIMULTANEOUS_DECISIONS,
    RAW_PREFIX_NAMES,
    RAW_SLOT_WIDTH,
    RAW_STATE_INDEX,
    SCHEMA_VERSION as DATASET_SCHEMA,
)


SCHEMA_VERSION = "transit-smdp-v7-serialized-credit-audit-v1"
BOOTSTRAP_PROBE = 0.73125
REQUIRED_DATASETS = (
    "raw_observations",
    "raw_next_observations",
    "actions",
    "rewards",
    "discounts",
    "terminals",
    "delta_t_seconds",
    "episode_id",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def credit_audit_source_hashes() -> dict[str, str]:
    paths = (
        ROOT / "bus_h2o" / "mdp_contract_v7.py",
        Path(__file__).resolve(),
    )
    return {
        str(path.relative_to(ROOT)): sha256(path)
        for path in paths
    }


def _slot_start(slot: int) -> int:
    return len(RAW_PREFIX_NAMES) + int(slot) * RAW_SLOT_WIDTH


def _count_mismatches(actual: np.ndarray, expected: np.ndarray | float) -> int:
    return int(np.count_nonzero(actual != expected))


def _episode_spans(episode_ids: np.ndarray) -> tuple[list[tuple[int, int]], list[str]]:
    errors: list[str] = []
    if episode_ids.ndim != 1 or not len(episode_ids):
        return [], ["episode_id must be a non-empty vector"]
    boundaries = np.flatnonzero(episode_ids[1:] != episode_ids[:-1]) + 1
    starts = np.concatenate((np.asarray([0]), boundaries))
    ends = np.concatenate((boundaries, np.asarray([len(episode_ids)])))
    labels = episode_ids[starts]
    if len(np.unique(labels)) != len(labels):
        errors.append("an episode_id occurs in multiple disjoint spans")
    if not np.array_equal(labels, np.arange(len(labels), dtype=labels.dtype)):
        errors.append("episode_id spans are not consecutively numbered from zero")
    return list(zip(starts.tolist(), ends.tolist())), errors


def audit(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    errors: list[str] = []
    counters = {
        "chain_errors": 0,
        "index_errors": 0,
        "clock_errors": 0,
        "intermediate_duration_errors": 0,
        "intermediate_reward_errors": 0,
        "intermediate_discount_errors": 0,
        "intermediate_terminal_errors": 0,
        "final_duration_errors": 0,
        "final_discount_errors": 0,
        "batch_reward_ownership_errors": 0,
        "action_history_errors": 0,
        "next_action_history_errors": 0,
        "terminal_structure_errors": 0,
    }
    slot_rows = np.zeros(MAX_SIMULTANEOUS_DECISIONS, dtype=np.int64)
    physical_batches = 0
    multi_action_batches = 0
    maximum_batch_size = 0
    zero_duration_rows = 0
    maximum_slot_target_error = 0.0

    with h5py.File(path, "r") as handle:
        missing = sorted(set(REQUIRED_DATASETS).difference(handle.keys()))
        if missing:
            raise KeyError(f"dataset is missing arrays {missing}")
        if handle.attrs.get("schema_version") != DATASET_SCHEMA:
            errors.append("dataset schema is not transit-smdp-v7")
        row_count = int(handle["rewards"].shape[0])
        for name in REQUIRED_DATASETS:
            if int(handle[name].shape[0]) != row_count:
                errors.append(f"{name} has a row-count mismatch")
        episode_ids = np.asarray(handle["episode_id"], dtype=np.int64)
        spans, span_errors = _episode_spans(episode_ids)
        errors.extend(span_errors)

        for episode_number, (start, end) in enumerate(spans):
            raw = np.asarray(handle["raw_observations"][start:end])
            raw_next = np.asarray(handle["raw_next_observations"][start:end])
            actions = np.asarray(handle["actions"][start:end, 0])
            rewards = np.asarray(handle["rewards"][start:end], dtype=np.float64)
            discounts = np.asarray(
                handle["discounts"][start:end], dtype=np.float64
            )
            terminals = np.asarray(handle["terminals"][start:end], dtype=bool)
            durations = np.asarray(
                handle["delta_t_seconds"][start:end], dtype=np.float64
            )
            size = end - start
            rows = np.arange(size, dtype=np.int64)

            counts_raw = raw[
                :, RAW_STATE_INDEX["simultaneous_decision_count"]
            ]
            indices_raw = raw[:, RAW_STATE_INDEX["current_decision_index"]]
            counts = np.rint(counts_raw).astype(np.int64)
            indices = np.rint(indices_raw).astype(np.int64)
            valid = (
                (counts_raw == counts)
                & (indices_raw == indices)
                & (counts >= 1)
                & (counts <= MAX_SIMULTANEOUS_DECISIONS)
                & (indices >= 0)
                & (indices < counts)
            )
            counters["index_errors"] += int(np.count_nonzero(~valid))
            if not np.all(valid):
                continue

            slot_rows += np.bincount(
                indices, minlength=MAX_SIMULTANEOUS_DECISIONS
            )[:MAX_SIMULTANEOUS_DECISIONS]
            maximum_batch_size = max(maximum_batch_size, int(counts.max()))
            starts = indices == 0
            finals = indices == counts - 1
            intermediate = ~finals
            final_nonterminal = finals & ~terminals
            start_rows = np.flatnonzero(starts)
            final_rows = np.flatnonzero(finals)
            physical_batches += int(len(start_rows))
            multi_action_batches += int(np.count_nonzero(counts[start_rows] > 1))
            zero_duration_rows += int(np.count_nonzero(durations == 0.0))

            counters["index_errors"] += int(not starts[0])
            counters["index_errors"] += int(not finals[-1])
            counters["index_errors"] += abs(len(start_rows) - len(final_rows))
            if size > 1:
                counters["chain_errors"] += int(
                    np.count_nonzero(np.any(raw_next[:-1] != raw[1:], axis=1))
                )

            intermediate_rows = np.flatnonzero(intermediate)
            if len(intermediate_rows):
                next_rows = intermediate_rows + 1
                counters["index_errors"] += _count_mismatches(
                    indices[next_rows], indices[intermediate_rows] + 1
                )
                counters["index_errors"] += _count_mismatches(
                    counts[next_rows], counts[intermediate_rows]
                )
                times = raw[
                    :, RAW_STATE_INDEX["decision_time_seconds"]
                ]
                counters["clock_errors"] += _count_mismatches(
                    times[next_rows], times[intermediate_rows]
                )
                counters["intermediate_duration_errors"] += _count_mismatches(
                    durations[intermediate_rows], 0.0
                )
                counters["intermediate_reward_errors"] += _count_mismatches(
                    rewards[intermediate_rows], 0.0
                )
                counters["intermediate_discount_errors"] += _count_mismatches(
                    discounts[intermediate_rows], 1.0
                )
                counters["intermediate_terminal_errors"] += int(
                    np.count_nonzero(terminals[intermediate_rows])
                )
                action_columns = (
                    len(RAW_PREFIX_NAMES)
                    + indices[intermediate_rows] * RAW_SLOT_WIDTH
                    + RAW_SLOT_WIDTH
                    - 1
                )
                counters["next_action_history_errors"] += _count_mismatches(
                    raw_next[intermediate_rows, action_columns],
                    actions[intermediate_rows],
                )

            counters["final_duration_errors"] += int(
                np.count_nonzero(durations[final_nonterminal] <= 0.0)
            )
            counters["final_discount_errors"] += _count_mismatches(
                discounts[final_nonterminal], 1.0
            )
            counters["final_discount_errors"] += _count_mismatches(
                discounts[terminals], 0.0
            )
            counters["terminal_structure_errors"] += int(
                np.count_nonzero(terminals) != 1
            )
            counters["terminal_structure_errors"] += int(not terminals[-1])
            counters["terminal_structure_errors"] += int(not finals[-1])

            if len(final_rows) == len(start_rows) and len(start_rows):
                batch_ids = np.cumsum(starts, dtype=np.int64) - 1
                reward_sums = np.bincount(
                    batch_ids, weights=rewards, minlength=len(start_rows)
                )
                counters["batch_reward_ownership_errors"] += _count_mismatches(
                    reward_sums, rewards[final_rows]
                )
                batch_final_rows = rows - indices + counts - 1
                if np.any(batch_final_rows < 0) or np.any(batch_final_rows >= size):
                    counters["index_errors"] += int(
                        np.count_nonzero(
                            (batch_final_rows < 0) | (batch_final_rows >= size)
                        )
                    )
                else:
                    final_targets = (
                        rewards[batch_final_rows]
                        + discounts[batch_final_rows] * BOOTSTRAP_PROBE
                    )
                    propagated = rewards + discounts * final_targets
                    if len(intermediate_rows):
                        maximum_slot_target_error = max(
                            maximum_slot_target_error,
                            float(
                                np.max(
                                    np.abs(
                                        propagated[intermediate_rows]
                                        - final_targets[intermediate_rows]
                                    )
                                )
                            ),
                        )

            batch_starts = rows - indices
            for slot in range(MAX_SIMULTANEOUS_DECISIONS):
                column = _slot_start(slot) + RAW_SLOT_WIDTH - 1
                expected = np.full(size, -1.0, dtype=actions.dtype)
                selected = slot < indices
                if np.any(selected):
                    expected[selected] = actions[batch_starts[selected] + slot]
                counters["action_history_errors"] += _count_mismatches(
                    raw[:, column], expected
                )

            if np.any(episode_ids[start:end] != episode_number):
                counters["terminal_structure_errors"] += 1

    if multi_action_batches == 0:
        errors.append("dataset contains no multi-action physical batch")
    if maximum_slot_target_error != 0.0:
        errors.append(
            "serialized Bellman recursion does not preserve the final physical target"
        )
    for name, count in counters.items():
        if count:
            errors.append(f"{name}: {count}")

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": str(path),
        "dataset_sha256": sha256(path),
        "passed": not errors,
        "errors": errors,
        "source_sha256": credit_audit_source_hashes(),
        "rows": row_count,
        "episodes": len(spans),
        "physical_batches": physical_batches,
        "multi_action_batches": multi_action_batches,
        "multi_action_batch_fraction": (
            multi_action_batches / physical_batches if physical_batches else 0.0
        ),
        "zero_duration_rows": zero_duration_rows,
        "maximum_simultaneous_decisions_observed": maximum_batch_size,
        "slot_row_counts": slot_rows.tolist(),
        "bellman_credit": {
            "bootstrap_probe": BOOTSTRAP_PROBE,
            "maximum_absolute_slot_target_error": maximum_slot_target_error,
            "semantics": (
                "each zero-time token has reward 0 and continuation 1, so the "
                "last token's physical reward and post-batch bootstrap pass "
                "unchanged through every earlier token in the same batch"
            ),
        },
        "counters": counters,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.dataset)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".partial")
        temporary.write_text(encoded)
        os.replace(temporary, output)
    print(encoded, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
