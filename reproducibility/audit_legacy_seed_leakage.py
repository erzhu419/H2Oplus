#!/usr/bin/env python3
"""Prove whether legacy policy seeds produced duplicated target trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


FIELDS = ("observations", "actions", "next_observations")
DETERMINISTIC_POLICIES = ("heuristic_best", "sac", "zero")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(path: Path, chunk_rows: int = 100_000) -> dict:
    with h5py.File(path, "r") as handle:
        policy_names = json.loads(handle.attrs["policy_names"])
        policy_ids = {name: int(policy_names[name]) for name in DETERMINISTIC_POLICIES}
        seeds = sorted(map(int, np.unique(handle["seed"][:]).tolist()))
        digests = {
            (name, seed, field): hashlib.sha256()
            for name in DETERMINISTIC_POLICIES
            for seed in seeds
            for field in FIELDS
        }
        counts = {
            (name, seed): 0 for name in DETERMINISTIC_POLICIES for seed in seeds
        }
        n_rows = len(handle["policy_id"])
        for start in range(0, n_rows, chunk_rows):
            stop = min(start + chunk_rows, n_rows)
            policies = handle["policy_id"][start:stop]
            policy_seeds = handle["seed"][start:stop]
            arrays = {field: handle[field][start:stop] for field in FIELDS}
            for name, policy_id in policy_ids.items():
                for seed in seeds:
                    selected = (policies == policy_id) & (policy_seeds == seed)
                    count = int(selected.sum())
                    if not count:
                        continue
                    counts[(name, seed)] += count
                    for field in FIELDS:
                        digests[(name, seed, field)].update(
                            np.ascontiguousarray(arrays[field][selected]).tobytes()
                        )

    policies_report = {}
    all_duplicates = True
    for name in DETERMINISTIC_POLICIES:
        by_seed = {
            str(seed): {
                "rows": counts[(name, seed)],
                "field_sha256": {
                    field: digests[(name, seed, field)].hexdigest()
                    for field in FIELDS
                },
            }
            for seed in seeds
        }
        signatures = {
            (entry["rows"], json.dumps(entry["field_sha256"], sort_keys=True))
            for entry in by_seed.values()
        }
        duplicated = len(signatures) == 1
        all_duplicates &= duplicated
        policies_report[name] = {
            "exactly_duplicated_across_policy_seeds": duplicated,
            "by_policy_seed": by_seed,
        }

    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(path.resolve()),
        "source_sha256": file_sha256(path),
        "finding": "critical-data-split-leakage" if all_duplicates else "mixed",
        "all_deterministic_policies_exactly_duplicated": all_duplicates,
        "cause": (
            "legacy --seed was recorded and used for policy RNG but was not passed "
            "to SumoRLBridge/SUMO"
        ),
        "required_action": (
            "deduplicate deterministic-policy copies; mark all legacy transitions "
            "training-only; evaluate on fresh, explicitly propagated SUMO seeds"
        ),
        "policies": policies_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    args = parser.parse_args()
    report = audit(args.input, args.chunk_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(args.output.resolve())
    if not report["all_deterministic_policies_exactly_duplicated"]:
        raise SystemExit("legacy duplicate assumption was not verified")


if __name__ == "__main__":
    main()
