#!/usr/bin/env python3
"""Audit causal holding, headway, and open-door boarding semantics in SUMO."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if "SUMO_HOME" not in os.environ and Path("/usr/share/sumo").is_dir():
    os.environ["SUMO_HOME"] = "/usr/share/sumo"

from bus_h2o.envs.all_lines_sumo_env import AllLinesSumoEnv


SCHEMA_VERSION = "transit-transition-semantics-audit-v7-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return (completed.stdout or completed.stderr).strip() or None


def _source_descriptor(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def _event_identity(event: Any) -> dict[str, Any]:
    return {
        "line_id": str(event.line_id),
        "bus_id": str(event.bus_id),
        "stop_id": str(event.stop_id),
        "stop_index": int(event.stop_idx),
    }


def _event_record(event: Any) -> dict[str, Any]:
    return {
        **_event_identity(event),
        "event_time_seconds": float(event.sim_time),
        "arrival_time_seconds": float(event.metadata["arrive_time"]),
        "service_completion_time_seconds": float(
            event.metadata["service_completion_time"]
        ),
        "base_dwell_seconds": float(event.base_stop_duration),
        "forward_headway_seconds": float(event.forward_headway),
        "backward_headway_seconds": float(event.backward_headway),
        "target_forward_headway_seconds": float(event.target_forward_headway),
        "target_backward_headway_seconds": float(event.target_backward_headway),
        "forward_bus_present": bool(event.forward_bus_present),
        "backward_bus_present": bool(event.backward_bus_present),
        "segment_mean_speed_mps": float(event.segment_mean_speed),
        "onboard_passengers": int(event.onboard_passengers),
        "eligible_stranded_passengers": int(
            event.eligible_stranded_passengers
        ),
    }


def _append_violation(
    audit: dict[str, Any], event: Any, check: str, observed: Any, expected: Any
) -> None:
    if len(audit["violations"]) >= 100:
        return
    audit["violations"].append(
        {
            "event": _event_identity(event),
            "event_time_seconds": float(event.sim_time),
            "check": check,
            "observed": observed,
            "expected": expected,
        }
    )


def audit_event_semantics(
    bridge: Any, event: Any, audit: dict[str, Any], *, tolerance: float = 1e-6
) -> None:
    """Check one emitted event against the bridge's causal source records."""

    audit["checked_events"] += 1
    bus = bridge.bus_obj_dic[event.bus_id]
    forward, backward = bridge._find_neighbors(event.line_id, event.bus_id)
    service_completion = float(event.metadata["service_completion_time"])
    arrival = float(event.metadata["arrive_time"])

    errors = audit["maximum_absolute_error_seconds"]
    clock_error = abs(float(event.sim_time) - service_completion)
    errors["event_clock"] = max(errors["event_clock"], clock_error)
    clock_delay = float(event.sim_time) - service_completion
    if clock_delay < -tolerance or clock_delay > bridge.time_step + tolerance:
        _append_violation(
            audit,
            event,
            "event_clock_is_first_grid_point_after_service_completion",
            event.sim_time,
            {
                "minimum": service_completion,
                "maximum": service_completion + bridge.time_step,
            },
        )

    base_error = abs((service_completion - arrival) - event.base_stop_duration)
    errors["base_dwell"] = max(errors["base_dwell"], base_error)
    if base_error > tolerance:
        _append_violation(
            audit, event, "base_dwell_definition", service_completion - arrival,
            event.base_stop_duration,
        )

    own_history = bus.service_completion_history.get(event.stop_id, [])
    if not own_history or abs(float(own_history[-1]) - service_completion) > tolerance:
        _append_violation(
            audit,
            event,
            "own_service_completion_history",
            None if not own_history else own_history[-1],
            service_completion,
        )

    default_target = float(
        bridge.line_headways.get(event.line_id, bridge.default_headway)
    )
    expected_forward_target = (
        abs(float(bus.start_time_n) - float(forward.start_time_n))
        if forward is not None
        else default_target
    )
    expected_backward_target = (
        abs(float(backward.start_time_n) - float(bus.start_time_n))
        if backward is not None
        else default_target
    )
    forward_target_error = abs(
        float(event.target_forward_headway) - expected_forward_target
    )
    backward_target_error = abs(
        float(event.target_backward_headway) - expected_backward_target
    )
    errors["target_forward_headway"] = max(
        errors["target_forward_headway"], forward_target_error
    )
    errors["target_backward_headway"] = max(
        errors["target_backward_headway"], backward_target_error
    )
    if forward_target_error > tolerance:
        _append_violation(
            audit, event, "target_forward_schedule_gap",
            event.target_forward_headway, expected_forward_target,
        )
    if backward_target_error > tolerance:
        _append_violation(
            audit, event, "target_backward_schedule_gap",
            event.target_backward_headway, expected_backward_target,
        )

    forward_history = (
        forward.service_completion_history.get(event.stop_id, [])
        if forward is not None
        else []
    )
    expected_forward_present = bool(forward_history)
    if bool(event.forward_bus_present) != expected_forward_present:
        _append_violation(
            audit, event, "forward_presence_requires_observed_predecessor",
            bool(event.forward_bus_present), expected_forward_present,
        )
    expected_forward_headway = (
        max(service_completion - float(forward_history[-1]), 0.0)
        if expected_forward_present
        else expected_forward_target
    )
    forward_error = abs(
        float(event.forward_headway) - expected_forward_headway
    )
    errors["forward_headway"] = max(errors["forward_headway"], forward_error)
    if forward_error > tolerance:
        _append_violation(
            audit, event, "forward_service_completion_headway",
            event.forward_headway, expected_forward_headway,
        )

    expected_backward_present = bool(
        backward is not None and backward.bus_id_s in bridge.active_bus_ids
    )
    if bool(event.backward_bus_present) != expected_backward_present:
        _append_violation(
            audit, event, "backward_presence_requires_active_follower",
            bool(event.backward_bus_present), expected_backward_present,
        )
    if expected_backward_present:
        distance_gap = max(
            float(bus.distance_n) - float(backward.distance_n), 0.0
        )
        cap = max(2400.0, 3.0 * expected_backward_target)
        expected_backward_headway = min(
            distance_gap / max(float(event.segment_mean_speed), 1.0), cap
        )
    else:
        expected_backward_headway = expected_backward_target
    backward_error = abs(
        float(event.backward_headway) - expected_backward_headway
    )
    errors["backward_headway"] = max(
        errors["backward_headway"], backward_error
    )
    if backward_error > tolerance:
        _append_violation(
            audit, event, "backward_spatial_time_gap",
            event.backward_headway, expected_backward_headway,
        )

    numeric_values = (
        event.forward_headway,
        event.backward_headway,
        event.target_forward_headway,
        event.target_backward_headway,
        event.base_stop_duration,
        event.segment_mean_speed,
    )
    if not all(np.isfinite(value) and value >= 0.0 for value in numeric_values):
        _append_violation(
            audit, event, "finite_nonnegative_event_values",
            [float(value) for value in numeric_values], "all finite and nonnegative",
        )


def _passenger_index(bridge: Any) -> dict[str, dict[str, Any]]:
    by_stop: dict[str, list[tuple[float, Any]]] = defaultdict(list)
    for passenger in bridge.passenger_obj_dic.values():
        by_stop[str(passenger.start_stop_id_s)].append(
            (float(passenger.start_time_n), passenger)
        )
    result = {}
    for stop_id, rows in by_stop.items():
        rows.sort(key=lambda row: row[0])
        result[stop_id] = {
            "times": [row[0] for row in rows],
            "rows": rows,
        }
    return result


def _future_eligible_passengers(
    bridge: Any,
    index: dict[str, dict[str, Any]],
    event: Any,
    window_seconds: float,
) -> list[dict[str, Any]]:
    indexed = index.get(event.stop_id)
    if indexed is None:
        return []
    start = float(event.sim_time)
    stop = start + float(window_seconds)
    lo = bisect.bisect_right(indexed["times"], start)
    line_stops = set(bridge.line_obj_dic[event.line_id].stop_id_l)
    candidates = []
    for appearance, passenger in indexed["rows"][lo:]:
        if appearance > stop:
            break
        next_stop = passenger.transfer_stop_id_s or passenger.end_stop_id_s
        if next_stop not in line_stops:
            continue
        candidates.append(
            {
                "passenger_id": str(passenger.passenger_id_s),
                "scheduled_appearance_time_seconds": float(appearance),
                "next_stop_id": str(next_stop),
            }
        )
    return candidates


def _new_semantic_audit() -> dict[str, Any]:
    return {
        "checked_events": 0,
        "violations": [],
        "maximum_absolute_error_seconds": {
            "event_clock": 0.0,
            "base_dwell": 0.0,
            "target_forward_headway": 0.0,
            "target_backward_headway": 0.0,
            "forward_headway": 0.0,
            "backward_headway": 0.0,
        },
    }


def _matching_identity(event: Any, identity: dict[str, Any]) -> bool:
    return _event_identity(event) == identity


def run_sumo_probe(
    *,
    sumo_root: Path,
    edge_xml: Path,
    seed: int,
    episode_seconds: int,
    min_selection_time: float,
    selection_window_seconds: float,
    applied_hold_seconds: float,
    target_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one deterministic trajectory through the intervention's next stop."""

    env = AllLinesSumoEnv(
        str(sumo_root),
        str(edge_xml),
        episode_seconds=episode_seconds,
        sumo_seed=seed,
        observation_schema="v5",
    )
    started = time.perf_counter()
    env.reset()
    bridge = env._bridge
    passenger_index = _passenger_index(bridge)
    semantic_audit = _new_semantic_audit()
    intervention = None
    candidates: list[dict[str, Any]] = []
    batches = 0

    try:
        while not bridge.done:
            events, done, _ = bridge.fetch_events()
            if done:
                break
            if not events:
                continue
            batches += 1

            for event in events:
                audit_event_semantics(bridge, event, semantic_audit)

            if intervention is not None:
                next_index = intervention["stop_index"] + 1
                next_events = [
                    event
                    for event in events
                    if event.line_id == intervention["line_id"]
                    and event.bus_id == intervention["bus_id"]
                    and int(event.stop_idx) == next_index
                ]
                if next_events:
                    next_event = next_events[0]
                    bus = bridge.bus_obj_dic[intervention["bus_id"]]
                    departure = float(
                        bus.depart_stop_time_d[intervention["stop_id"]]
                    )
                    passenger_outcomes = []
                    for candidate in candidates:
                        passenger = bridge.passenger_obj_dic[
                            candidate["passenger_id"]
                        ]
                        passenger_outcomes.append(
                            {
                                **candidate,
                                "state_at_next_stop_event": str(
                                    passenger.passenger_state_s
                                ),
                                "boarded_bus_id": str(passenger.take_bus_id_s),
                                "boarding_time_seconds": (
                                    float(passenger.boarding_time_n)
                                    if passenger.boarding_time_n
                                    else None
                                ),
                            }
                        )
                    return {
                        "applied_hold_seconds": float(applied_hold_seconds),
                        "intervention": intervention,
                        "actual_departure_time_seconds": departure,
                        "actual_postdecision_stop_seconds": (
                            departure
                            - intervention["service_completion_time_seconds"]
                        ),
                        "actual_total_stop_seconds": (
                            departure - intervention["arrival_time_seconds"]
                        ),
                        "next_event": _event_record(next_event),
                        "future_passenger_outcomes": passenger_outcomes,
                        "semantic_event_audit": semantic_audit,
                        "runtime": {
                            "event_batches": batches,
                            "wall_time_seconds": time.perf_counter() - started,
                        },
                    }

            selected_event = None
            selected_candidates: list[dict[str, Any]] = []
            if intervention is None:
                if target_identity is not None:
                    selected_event = next(
                        (
                            event
                            for event in events
                            if _matching_identity(event, target_identity)
                        ),
                        None,
                    )
                else:
                    for event in events:
                        line = bridge.line_obj_dic[event.line_id]
                        future = _future_eligible_passengers(
                            bridge,
                            passenger_index,
                            event,
                            selection_window_seconds,
                        )
                        if (
                            float(event.sim_time) >= min_selection_time
                            and event.forward_bus_present
                            and event.backward_bus_present
                            and int(event.stop_idx) + 1 < len(line.stop_id_l)
                            and future
                        ):
                            selected_event = event
                            selected_candidates = future
                            break

                if selected_event is not None:
                    if target_identity is not None:
                        selected_candidates = _future_eligible_passengers(
                            bridge,
                            passenger_index,
                            selected_event,
                            selection_window_seconds,
                        )
                    intervention = _event_record(selected_event)
                    candidates = selected_candidates
                    intervention["future_eligible_passengers"] = candidates

            for event in events:
                hold = (
                    applied_hold_seconds
                    if selected_event is event
                    else 0.0
                )
                bridge.apply_action(event, [hold, 1.0])
                if selected_event is event:
                    intervention["predicted_departure_time_seconds"] = float(
                        event.metadata["predicted_departure_time_seconds"]
                    )

            if intervention is None and events[-1].sim_time > episode_seconds:
                break
    finally:
        env.close()

    raise RuntimeError(
        "SUMO probe did not reach a qualifying intervention and its next stop"
    )


def run_open_door_boarding_probe(
    *,
    sumo_root: Path,
    edge_xml: Path,
    seed: int,
    episode_seconds: int,
    line_id: str,
    stop_id: str,
    min_selection_time: float,
    hold_seconds: float,
) -> dict[str, Any]:
    """Verify late boarding from TraCI vehicle membership at a held origin."""

    env = AllLinesSumoEnv(
        str(sumo_root),
        str(edge_xml),
        episode_seconds=episode_seconds,
        sumo_seed=seed,
        observation_schema="v5",
    )
    started = time.perf_counter()
    env.reset()
    bridge = env._bridge
    bridge_module = sys.modules[bridge.__class__.__module__]
    passenger_index = _passenger_index(bridge)
    semantic_audit = _new_semantic_audit()
    intervention = None
    candidates: list[dict[str, Any]] = []
    onboard_before: set[str] = set()
    batches = 0

    try:
        while not bridge.done:
            events, done, _ = bridge.fetch_events()
            if done:
                break
            if not events:
                continue
            batches += 1
            for event in events:
                audit_event_semantics(bridge, event, semantic_audit)

            if intervention is not None:
                bus = bridge.bus_obj_dic[intervention["bus_id"]]
                if intervention["stop_id"] in bus.depart_stop_time_d:
                    departure = float(
                        bus.depart_stop_time_d[intervention["stop_id"]]
                    )
                    onboard_after = set(
                        bridge_module.traci.vehicle.getPersonIDList(
                            intervention["bus_id"]
                        )
                    )
                    candidate_ids = {
                        row["passenger_id"] for row in candidates
                    }
                    newly_boarded = sorted(onboard_after - onboard_before)
                    return {
                        "intervention": intervention,
                        "actual_departure_time_seconds": departure,
                        "actual_postdecision_stop_seconds": (
                            departure
                            - intervention["service_completion_time_seconds"]
                        ),
                        "onboard_before": sorted(onboard_before),
                        "onboard_after": sorted(onboard_after),
                        "newly_boarded_passenger_ids": newly_boarded,
                        "late_candidate_passenger_ids": sorted(candidate_ids),
                        "late_candidates_boarded": sorted(
                            candidate_ids.intersection(newly_boarded)
                        ),
                        "boarding_service_seconds_per_passenger": float(
                            bridge.AveBoardingTime
                        ),
                        "alighting_service_seconds_per_passenger": float(
                            bridge.AveAlightingTime
                        ),
                        "semantic_event_audit": semantic_audit,
                        "runtime": {
                            "event_batches": batches,
                            "wall_time_seconds": time.perf_counter() - started,
                        },
                    }

            selected_event = None
            if intervention is None:
                selected_event = next(
                    (
                        event
                        for event in events
                        if event.line_id == line_id
                        and event.stop_id == stop_id
                        and float(event.sim_time) >= min_selection_time
                    ),
                    None,
                )
                if selected_event is not None:
                    candidates = _future_eligible_passengers(
                        bridge,
                        passenger_index,
                        selected_event,
                        hold_seconds,
                    )
                    if not candidates:
                        selected_event = None
                    else:
                        intervention = _event_record(selected_event)
                        intervention["future_eligible_passengers"] = candidates
                        onboard_before = set(
                            bridge_module.traci.vehicle.getPersonIDList(
                                selected_event.bus_id
                            )
                        )

            for event in events:
                hold = hold_seconds if event is selected_event else 0.0
                bridge.apply_action(event, [hold, 1.0])
                if event is selected_event:
                    intervention["predicted_departure_time_seconds"] = float(
                        event.metadata["predicted_departure_time_seconds"]
                    )
    finally:
        env.close()

    raise RuntimeError("SUMO boarding probe did not reach the selected departure")


def evaluate_pair(
    zero: dict[str, Any],
    held: dict[str, Any],
    *,
    requested_hold_seconds: float,
    step_tolerance_seconds: float,
) -> dict[str, Any]:
    """Evaluate paired probe records without requiring a live SUMO runtime."""

    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, passed: bool, observed: Any, requirement: str) -> None:
        checks[name] = {
            "passed": bool(passed),
            "observed": observed,
            "requirement": requirement,
        }

    zero_identity = {
        key: zero["intervention"][key]
        for key in ("line_id", "bus_id", "stop_id", "stop_index")
    }
    held_identity = {
        key: held["intervention"][key]
        for key in ("line_id", "bus_id", "stop_id", "stop_index")
    }
    intervention_time_error = abs(
        held["intervention"]["event_time_seconds"]
        - zero["intervention"]["event_time_seconds"]
    )
    add(
        "paired_intervention_identity",
        zero_identity == held_identity
        and intervention_time_error <= step_tolerance_seconds,
        {
            "zero": zero_identity,
            "held": held_identity,
            "event_time_error_seconds": intervention_time_error,
        },
        "same trip, stop, and pre-intervention event time",
    )

    zero_residual = float(zero["actual_postdecision_stop_seconds"])
    held_residual = float(held["actual_postdecision_stop_seconds"])
    residual_delta = held_residual - zero_residual
    add(
        "additional_hold_duration",
        abs(residual_delta - requested_hold_seconds) <= step_tolerance_seconds,
        {
            "zero_postdecision_seconds": zero_residual,
            "held_postdecision_seconds": held_residual,
            "paired_delta_seconds": residual_delta,
        },
        "paired postdecision stop-time delta equals the requested hold within one simulation-step tolerance",
    )
    add(
        "zero_action_does_not_repeat_base_dwell",
        zero["actual_total_stop_seconds"]
        <= zero["intervention"]["base_dwell_seconds"]
        + step_tolerance_seconds,
        {
            "actual_total_stop_seconds": zero["actual_total_stop_seconds"],
            "base_dwell_seconds": zero["intervention"]["base_dwell_seconds"],
        },
        "zero-action total stop is no more than base dwell plus step tolerance",
    )
    departure_prediction_errors = {
        label: abs(
            row["actual_departure_time_seconds"]
            - row["intervention"]["predicted_departure_time_seconds"]
        )
        for label, row in (("zero", zero), ("held", held))
    }
    add(
        "departure_prediction",
        max(departure_prediction_errors.values()) <= step_tolerance_seconds,
        departure_prediction_errors,
        "actual departure matches service completion plus additional hold within step tolerance",
    )

    next_time_delta = (
        held["next_event"]["event_time_seconds"]
        - zero["next_event"]["event_time_seconds"]
    )
    next_forward_delta = (
        held["next_event"]["forward_headway_seconds"]
        - zero["next_event"]["forward_headway_seconds"]
    )
    add(
        "downstream_action_response",
        next_time_delta > step_tolerance_seconds
        and abs(next_time_delta - next_forward_delta) <= step_tolerance_seconds,
        {
            "next_event_time_delta_seconds": next_time_delta,
            "next_forward_headway_delta_seconds": next_forward_delta,
        },
        "holding changes the next-stop event and its service-completion forward headway consistently",
    )

    violation_counts = {
        label: len(row["semantic_event_audit"]["violations"])
        for label, row in (("zero", zero), ("held", held))
    }
    add(
        "event_semantic_invariants",
        sum(violation_counts.values()) == 0,
        violation_counts,
        "no event violates schedule, presence, service-completion, or spatial-gap definitions",
    )

    return {
        "status": "passed" if all(row["passed"] for row in checks.values()) else "failed",
        "checks": checks,
        "derived_effects": {
            "actual_postdecision_stop_delta_seconds": residual_delta,
            "next_event_time_delta_seconds": next_time_delta,
            "next_forward_headway_delta_seconds": next_forward_delta,
        },
    }


def evaluate_boarding_probe(
    probe: dict[str, Any], *, requested_hold_seconds: float,
    step_tolerance_seconds: float,
) -> dict[str, Any]:
    candidates = set(probe["late_candidate_passenger_ids"])
    boarded = set(probe["late_candidates_boarded"])
    actual_stop = float(probe["actual_postdecision_stop_seconds"])
    maximum_boarding_overrun = (
        len(boarded)
        * float(probe["boarding_service_seconds_per_passenger"])
    )
    duration_excess = actual_stop - requested_hold_seconds
    unexplained_overrun = max(duration_excess - maximum_boarding_overrun, 0.0)
    checks = {
        "additional_hold_executed": {
            "passed": (
                actual_stop >= requested_hold_seconds - step_tolerance_seconds
                and unexplained_overrun <= step_tolerance_seconds
            ),
            "observed": {
                "requested_control_hold_seconds": requested_hold_seconds,
                "actual_postdecision_stop_seconds": actual_stop,
                "maximum_nonoverlapping_boarding_service_seconds": (
                    maximum_boarding_overrun
                ),
                "unexplained_overrun_seconds": unexplained_overrun,
            },
            "requirement": "control hold is the earliest departure bound; only nonoverlapping mandatory service may extend it",
        },
        "open_door_boarding_during_hold": {
            "passed": bool(boarded) and boarded.issubset(candidates),
            "observed": {
                "eligible_late_candidates": sorted(candidates),
                "late_candidates_boarded": sorted(boarded),
                "onboard_before": probe["onboard_before"],
                "onboard_after": probe["onboard_after"],
            },
            "requirement": "at least one route-eligible passenger scheduled after the decision enters the held bus",
        },
        "event_semantic_invariants": {
            "passed": not probe["semantic_event_audit"]["violations"],
            "observed": {
                "checked_events": probe["semantic_event_audit"][
                    "checked_events"
                ],
                "violation_count": len(
                    probe["semantic_event_audit"]["violations"]
                ),
            },
            "requirement": "no event before the boarding probe violates the causal state contract",
        },
    }
    return {
        "status": (
            "passed"
            if all(row["passed"] for row in checks.values())
            else "failed"
        ),
        "checks": checks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sumo-root",
        type=Path,
        default=ROOT.parent / "SUMO_ruiguang" / "online_control",
    )
    parser.add_argument(
        "--edge-xml",
        type=Path,
        default=ROOT / "bus_h2o" / "network_data" / "a_sorted_busline_edge.xml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "reproducibility"
        / "transition_semantics_audit_v7_formal_v6.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-seconds", type=int, default=7800)
    parser.add_argument("--min-selection-time", type=float, default=1200.0)
    parser.add_argument("--hold-seconds", type=float, default=60.0)
    parser.add_argument("--step-tolerance-seconds", type=float, default=2.0)
    parser.add_argument("--boarding-line", default="705S")
    parser.add_argument("--boarding-stop", default="705S01")
    parser.add_argument("--boarding-min-time", type=float, default=7200.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.hold_seconds <= 0.0:
        raise ValueError("hold-seconds must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen audit {args.output}")

    sumo_root = args.sumo_root.expanduser().resolve()
    edge_xml = args.edge_xml.expanduser().resolve()
    started = time.perf_counter()
    zero = run_sumo_probe(
        sumo_root=sumo_root,
        edge_xml=edge_xml,
        seed=args.seed,
        episode_seconds=args.episode_seconds,
        min_selection_time=args.min_selection_time,
        selection_window_seconds=args.hold_seconds,
        applied_hold_seconds=0.0,
    )
    target_identity = {
        key: zero["intervention"][key]
        for key in ("line_id", "bus_id", "stop_id", "stop_index")
    }
    held = run_sumo_probe(
        sumo_root=sumo_root,
        edge_xml=edge_xml,
        seed=args.seed,
        episode_seconds=args.episode_seconds,
        min_selection_time=args.min_selection_time,
        selection_window_seconds=args.hold_seconds,
        applied_hold_seconds=args.hold_seconds,
        target_identity=target_identity,
    )
    pair_evaluation = evaluate_pair(
        zero,
        held,
        requested_hold_seconds=args.hold_seconds,
        step_tolerance_seconds=args.step_tolerance_seconds,
    )
    boarding_probe = run_open_door_boarding_probe(
        sumo_root=sumo_root,
        edge_xml=edge_xml,
        seed=args.seed,
        episode_seconds=args.episode_seconds,
        line_id=args.boarding_line,
        stop_id=args.boarding_stop,
        min_selection_time=args.boarding_min_time,
        hold_seconds=args.hold_seconds,
    )
    boarding_evaluation = evaluate_boarding_probe(
        boarding_probe,
        requested_hold_seconds=args.hold_seconds,
        step_tolerance_seconds=args.step_tolerance_seconds,
    )
    overall_status = (
        "passed"
        if pair_evaluation["status"] == "passed"
        and boarding_evaluation["status"] == "passed"
        else "failed"
    )

    provenance_paths = [
        Path(__file__),
        ROOT / "bus_h2o" / "sumo_env" / "rl_bridge.py",
        ROOT
        / "bus_h2o"
        / "sumo_env"
        / "case"
        / "e_8_gurobi_test_considerbusnum_V3.py",
        ROOT / "bus_h2o" / "sumo_env" / "case" / "f_8_create_obj.py",
        edge_xml,
        sumo_root / "control_sim_traci_period.sumocfg",
        sumo_root / "initialize_obj" / "save_obj_bus.add.xml",
        sumo_root / "initialize_obj" / "save_obj_line.add.xml",
        sumo_root / "initialize_obj" / "save_obj_passenger.add.xml",
        sumo_root / "initialize_obj" / "save_obj_stop.add.xml",
        sumo_root / "sim_obj" / "bus.py",
        sumo_root / "sim_obj" / "passenger.py",
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": overall_status,
        "protocol": {
            "seed": args.seed,
            "episode_seconds": args.episode_seconds,
            "minimum_selection_time_seconds": args.min_selection_time,
            "requested_additional_hold_seconds": args.hold_seconds,
            "simulation_step_tolerance_seconds": args.step_tolerance_seconds,
            "decision_event_time_semantics": (
                "the first one-second SUMO grid point not earlier than the "
                "possibly fractional base-service completion time"
            ),
            "postdecision_stop_decomposition": (
                "commanded hold sets the earliest departure; passenger service "
                "runs concurrently and extends the stop only when unfinished"
            ),
            "passenger_service_parameters": {
                "alighting_seconds_per_passenger": (
                    boarding_probe["alighting_service_seconds_per_passenger"]
                ),
                "boarding_seconds_per_passenger": (
                    boarding_probe["boarding_service_seconds_per_passenger"]
                ),
                "source": (
                    "canonical g_8_SUMO.py control override; legacy Gurobi "
                    "static constants are intentionally ignored"
                ),
            },
            "selection_rule": (
                "first nonterminal event after the minimum time with observed "
                "predecessor and active follower plus an eligible passenger "
                "scheduled to appear within the hold window"
            ),
            "boarding_probe_selector": {
                "line_id": args.boarding_line,
                "stop_id": args.boarding_stop,
                "minimum_event_time_seconds": args.boarding_min_time,
            },
        },
        "zero_action_trajectory": zero,
        "held_trajectory": held,
        "open_door_boarding_probe": boarding_probe,
        "evaluation": {
            "status": overall_status,
            "paired_action_response": pair_evaluation,
            "open_door_boarding": boarding_evaluation,
        },
        "supersedes": [
            {
                "path": str(
                    ROOT
                    / "reproducibility"
                    / "transition_semantics_audit_v7_formal_v1.json"
                ),
                "reason": "failed provisional combined boarding selector",
            },
            {
                "path": str(
                    ROOT
                    / "reproducibility"
                    / "transition_semantics_audit_v7_formal_v2.json"
                ),
                "reason": "failed provisional combined boarding selector",
            },
            {
                "path": str(
                    ROOT
                    / "reproducibility"
                    / "transition_semantics_audit_v7_formal_v3.json"
                ),
                "reason": "used an exact event-clock check and legacy service constants",
            },
            {
                "path": str(
                    ROOT
                    / "reproducibility"
                    / "transition_semantics_audit_v7_formal_v4.json"
                ),
                "reason": "passed under legacy 1.2/1.5-second service constants that do not match canonical control",
            },
            {
                "path": str(
                    ROOT
                    / "reproducibility"
                    / "transition_semantics_audit_v7_formal_v5.json"
                ),
                "reason": "required every scheduled candidate to reach the stop and board before departure",
            },
        ],
        "provenance": {
            "git_head": _command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
            "git_status_porcelain": _command_output(
                ["git", "-C", str(ROOT), "status", "--short"]
            ),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sumo_version": _command_output(["sumo", "--version"]),
            "inputs": [_source_descriptor(path) for path in provenance_paths],
        },
        "runtime": {"wall_time_seconds": time.perf_counter() - started},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "status": overall_status,
                "paired_action_response": pair_evaluation,
                "open_door_boarding": boarding_evaluation,
            },
            indent=2,
        )
    )
    return 0 if overall_status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
