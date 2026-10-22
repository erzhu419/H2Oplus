#!/usr/bin/env python3
"""Audit that every v6 action-bearing transition covers the declared services."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.mdp_contract_v6 import (
    CANONICAL_PREFIX_NAMES,
    CANONICAL_SLOT_NAMES,
    CANONICAL_SLOT_WIDTH,
    RAW_PREFIX_NAMES,
    RAW_SLOT_NAMES,
    RAW_SLOT_WIDTH,
    SCHEMA_VERSION,
)
from bus_h2o.sumo_provenance import sha256


AUDIT_SCHEMA = "transit-v6-control-scope-audit-v1"


def _schedule_scope(schedule: Path) -> dict[str, Any]:
    rows = []
    missing_fields = 0
    for element in ET.parse(schedule).getroot().findall(".//bus_obj"):
        line_id = element.get("belong_line_id_s")
        bus_id = element.get("bus_id_s")
        if not line_id or not bus_id:
            missing_fields += 1
            continue
        rows.append((line_id, bus_id))

    line_counts = Counter(line_id for line_id, _ in rows)
    bus_counts = Counter(bus_id for _, bus_id in rows)
    line_ids = sorted(line_counts)
    return {
        "scheduled_vehicle_trips": len(rows),
        "unique_bus_ids": len(bus_counts),
        "duplicate_bus_ids": sorted(
            bus_id for bus_id, count in bus_counts.items() if count > 1
        ),
        "missing_line_or_bus_fields": missing_fields,
        "line_index": {line_id: index for index, line_id in enumerate(line_ids)},
        "scheduled_trips_by_line": {
            line_id: int(line_counts[line_id]) for line_id in line_ids
        },
    }


def audit_control_scope(
    dataset: Path, schedule: Path, *, chunk_size: int = 100_000
) -> dict[str, Any]:
    dataset = dataset.expanduser().resolve()
    schedule = schedule.expanduser().resolve()
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    schedule_scope = _schedule_scope(schedule)
    expected_line_index = schedule_scope["line_index"]
    raw_line_offset = RAW_SLOT_NAMES.index("line_id")
    canonical_line_offset = CANONICAL_SLOT_NAMES.index("line_id")
    errors = []
    line_counts: Counter[int] = Counter()
    malformed_current_index = 0
    inactive_current_raw_slot = 0
    inactive_current_canonical_slot = 0
    nonintegral_raw_line_id = 0
    raw_canonical_line_mismatch = 0

    with h5py.File(dataset, "r") as handle:
        schema = handle.attrs.get("schema_version", "")
        if isinstance(schema, bytes):
            schema = schema.decode("utf-8")
        if schema != SCHEMA_VERSION:
            errors.append(
                f"dataset schema is {schema!r}; expected {SCHEMA_VERSION!r}"
            )
        for name in ("raw_observations", "observations"):
            if name not in handle:
                errors.append(f"dataset is missing {name}")
        if errors:
            row_count = 0
        else:
            raw_dataset = handle["raw_observations"]
            canonical_dataset = handle["observations"]
            row_count = len(raw_dataset)
            if len(canonical_dataset) != row_count:
                errors.append("raw and canonical observation row counts differ")

            for start in range(0, row_count, chunk_size):
                stop = min(start + chunk_size, row_count)
                raw = np.asarray(raw_dataset[start:stop])
                canonical = np.asarray(canonical_dataset[start:stop])
                current_values = raw[:, 2]
                current = np.rint(current_values).astype(np.int64)
                valid_current = (
                    np.isfinite(current_values)
                    & (np.abs(current_values - current) <= 1e-6)
                    & (current >= 0)
                    & (current < 16)
                )
                malformed_current_index += int(np.count_nonzero(~valid_current))
                if not np.all(valid_current):
                    continue

                rows = np.arange(stop - start)
                raw_base = len(RAW_PREFIX_NAMES) + current * RAW_SLOT_WIDTH
                canonical_base = (
                    len(CANONICAL_PREFIX_NAMES) + current * CANONICAL_SLOT_WIDTH
                )
                raw_active = raw[rows, raw_base]
                canonical_active = canonical[rows, canonical_base]
                inactive_current_raw_slot += int(np.count_nonzero(raw_active != 1.0))
                inactive_current_canonical_slot += int(
                    np.count_nonzero(canonical_active != 1.0)
                )

                raw_lines = raw[rows, raw_base + raw_line_offset]
                canonical_lines = canonical[
                    rows, canonical_base + canonical_line_offset
                ]
                rounded_lines = np.rint(raw_lines).astype(np.int64)
                nonintegral_raw_line_id += int(
                    np.count_nonzero(
                        ~np.isfinite(raw_lines)
                        | (np.abs(raw_lines - rounded_lines) > 1e-6)
                    )
                )
                raw_canonical_line_mismatch += int(
                    np.count_nonzero(np.abs(raw_lines - canonical_lines) > 1e-6)
                )
                unique, counts = np.unique(rounded_lines, return_counts=True)
                line_counts.update(
                    {int(line_id): int(count) for line_id, count in zip(unique, counts)}
                )

    expected_indices = set(expected_line_index.values())
    observed_indices = set(line_counts)
    if schedule_scope["missing_line_or_bus_fields"]:
        errors.append("schedule contains bus records with missing identifiers")
    if schedule_scope["duplicate_bus_ids"]:
        errors.append("schedule bus identifiers are not globally unique")
    if observed_indices != expected_indices:
        errors.append(
            "action-bearing line indices differ from the scheduled line index: "
            f"observed={sorted(observed_indices)}, expected={sorted(expected_indices)}"
        )
    checks = {
        "malformed_current_decision_indices": malformed_current_index,
        "inactive_current_raw_slots": inactive_current_raw_slot,
        "inactive_current_canonical_slots": inactive_current_canonical_slot,
        "nonintegral_raw_line_ids": nonintegral_raw_line_id,
        "raw_canonical_line_mismatches": raw_canonical_line_mismatch,
    }
    for name, count in checks.items():
        if count:
            errors.append(f"{name}={count}")

    counts_by_line = {
        line_id: int(line_counts[index])
        for line_id, index in expected_line_index.items()
    }
    return {
        "schema_version": AUDIT_SCHEMA,
        "passed": not errors,
        "errors": errors,
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset),
        "dataset_rows": int(row_count),
        "schedule": str(schedule),
        "schedule_sha256": sha256(schedule),
        "schedule_scope": schedule_scope,
        "action_bearing_transitions_by_line": counts_by_line,
        "action_bearing_line_fractions": {
            line_id: (
                float(count) / float(row_count) if row_count else 0.0
            )
            for line_id, count in counts_by_line.items()
        },
        "encoding_checks": checks,
        "source_sha256": sha256(Path(__file__)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "bus_h2o/datasets_v6/transit_smdp_v6_formal_v1.h5",
    )
    parser.add_argument(
        "--schedule",
        type=Path,
        default=ROOT / "bus_h2o/sumo_env/initialize_obj/save_obj_bus.add.xml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reproducibility/control_scope_audit_v6_formal_v1.json",
    )
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {output}; pass --force")
    audit = audit_control_scope(
        args.dataset, args.schedule, chunk_size=args.chunk_size
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({
        "output": str(output),
        "passed": audit["passed"],
        "dataset_rows": audit["dataset_rows"],
        "action_bearing_transitions_by_line": (
            audit["action_bearing_transitions_by_line"]
        ),
        "errors": audit["errors"],
    }, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
