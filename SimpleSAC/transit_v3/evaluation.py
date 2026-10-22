"""Submission metrics for the canonical transit evaluation protocol."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

import numpy as np

from bus_h2o.mdp_contract import (
    CANONICAL_STATE_DIM,
    CANONICAL_STATE_INDEX,
    TransitMDPConfig,
    hold_seconds_to_policy_action,
)


def numeric_summary(values: Iterable[float]) -> dict[str, float | int] | None:
    array = np.asarray(tuple(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return None
    # Stable ordering makes floating reductions bit-reproducible even when the
    # source values came from sets or dictionaries with process-specific hashes.
    array.sort(kind="mergesort")
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def headway_balance_actions(
    observations: np.ndarray,
    gain: float,
    config: TransitMDPConfig,
) -> np.ndarray:
    """Return the two-sided proportional holding rule in policy units.

    Holding by ``u`` seconds approximately increases the forward headway by
    ``u`` and decreases the backward headway by ``u``.  Half their difference
    is therefore the one-step equalizing correction before gain and clipping.
    """

    if gain < 0:
        raise ValueError("headway-balance gain must be non-negative")
    states = np.asarray(observations, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != CANONICAL_STATE_DIM:
        raise ValueError("canonical observations have an invalid shape")
    scheduled = (
        states[:, CANONICAL_STATE_INDEX["scheduled_headway_fraction"]]
        * config.max_scheduled_headway_seconds
    )
    forward = states[:, 3] * scheduled
    backward = states[:, 4] * scheduled
    hold = np.clip(
        0.5 * gain * (backward - forward),
        0.0,
        config.max_hold_seconds,
    )
    actions = hold_seconds_to_policy_action(hold, config)
    return np.asarray(actions, dtype=np.float32)[:, None]


def _headway_summary(records: list[dict]) -> dict:
    forward = np.asarray(
        [row["forward_ratio"] for row in records if row["forward_ratio"] is not None],
        dtype=np.float64,
    )
    backward = np.asarray(
        [row["backward_ratio"] for row in records if row["backward_ratio"] is not None],
        dtype=np.float64,
    )
    errors = []
    if len(forward):
        errors.append(np.abs(forward - 1.0))
    if len(backward):
        errors.append(np.abs(backward - 1.0))
    absolute_errors = np.concatenate(errors) if errors else np.empty(0)
    return {
        "valid_forward_events": int(len(forward)),
        "valid_backward_events": int(len(backward)),
        "forward_ratio": numeric_summary(forward),
        "backward_ratio": numeric_summary(backward),
        "normalized_absolute_error": numeric_summary(absolute_errors),
        "bunching_rate": float((forward < 0.5).mean()) if len(forward) else None,
        "large_gap_rate": float((forward > 1.5).mean()) if len(forward) else None,
    }


def summarize_operations(event_records: list[dict], transition_records: list[dict]) -> dict:
    by_line: dict[str, list[dict]] = defaultdict(list)
    for row in event_records:
        by_line[row["line_id"]].append(row)
    line_summary = {}
    for line_id, rows in sorted(by_line.items()):
        holds = [row["hold_seconds"] for row in rows]
        line_summary[line_id] = {
            "decision_events": len(rows),
            "headway": _headway_summary(rows),
            "hold_seconds": numeric_summary(holds),
            "total_hold_seconds": float(np.sum(holds)),
        }

    rewards = [row["reward"] for row in transition_records]
    durations = [row["delta_t_seconds"] for row in transition_records]
    discounts = [row["discount"] for row in transition_records]
    holds = [row["hold_seconds"] for row in event_records]
    return {
        "decision_events": len(event_records),
        "same_trip_transitions": len(transition_records),
        "smdp_reward": numeric_summary(rewards),
        "cumulative_smdp_reward": float(np.sum(rewards)),
        "delta_t_seconds": numeric_summary(durations),
        "discount": numeric_summary(discounts),
        "headway": _headway_summary(event_records),
        "hold_seconds": numeric_summary(holds),
        "total_hold_seconds": float(np.sum(holds)),
        "nonzero_hold_rate": (
            float((np.asarray(holds) > 1e-6).mean()) if holds else None
        ),
        "per_line": line_summary,
    }


def summarize_passengers(bridge, horizon_seconds: float) -> dict:
    completed_leg_waits = []
    completed_leg_in_vehicle = []
    completed_leg_total = []
    accrued_wait = 0.0
    accrued_in_vehicle = 0.0
    ongoing_leg_count = 0
    invalid_legs = 0
    state_counts: Counter[str] = Counter()
    completed_ids = set(bridge.completed_passenger_times)

    for passenger_id in sorted(bridge.departed_passenger_ids):
        passenger = bridge.passenger_obj_dic[passenger_id]
        for leg in passenger.travel_data_l:
            if len(leg) < 6:
                invalid_legs += 1
                continue
            arrive, board, alight, wait, total, _ = leg
            in_vehicle = float(alight) - float(board)
            values = (float(wait), in_vehicle, float(total))
            if min(values) < 0 or not np.isfinite(values).all():
                invalid_legs += 1
                continue
            completed_leg_waits.append(values[0])
            completed_leg_in_vehicle.append(values[1])
            completed_leg_total.append(values[2])
            accrued_wait += values[0]
            accrued_in_vehicle += values[1]

        if passenger_id in completed_ids:
            continue
        state_counts[str(passenger.passenger_state_s)] += 1
        state = passenger.passenger_state_s
        if state == "Stop" and passenger.arriver_time_n > 0:
            accrued_wait += max(horizon_seconds - passenger.arriver_time_n, 0.0)
            ongoing_leg_count += 1
        elif state == "Bus" and passenger.boarding_time_n >= passenger.arriver_time_n:
            accrued_wait += passenger.boarding_time_n - passenger.arriver_time_n
            accrued_in_vehicle += max(
                horizon_seconds - passenger.boarding_time_n, 0.0
            )
            ongoing_leg_count += 1

    departed = len(bridge.departed_passenger_ids)
    completed = len(completed_ids)
    journey_times = [
        completion - bridge.passenger_obj_dic[passenger_id].start_time_n
        for passenger_id, completion in sorted(bridge.completed_passenger_times.items())
        if completion >= bridge.passenger_obj_dic[passenger_id].start_time_n
    ]
    return {
        "departed_passengers": departed,
        "completed_passengers": completed,
        "completion_rate": float(completed / departed) if departed else None,
        "unfinished_passengers": departed - completed,
        "state_counts_at_horizon": dict(sorted(state_counts.items())),
        "completed_legs": len(completed_leg_waits),
        "ongoing_legs": ongoing_leg_count,
        "invalid_leg_records": invalid_legs,
        "completed_leg_wait_seconds": numeric_summary(completed_leg_waits),
        "completed_leg_in_vehicle_seconds": numeric_summary(completed_leg_in_vehicle),
        "completed_leg_total_seconds": numeric_summary(completed_leg_total),
        "completed_journey_seconds": numeric_summary(journey_times),
        "accrued_wait_seconds_total": float(accrued_wait),
        "accrued_wait_seconds_per_departed_passenger": (
            float(accrued_wait / departed) if departed else None
        ),
        "accrued_in_vehicle_seconds_total": float(accrued_in_vehicle),
        "accrued_in_vehicle_seconds_per_departed_passenger": (
            float(accrued_in_vehicle / departed) if departed else None
        ),
        "accrued_wait_plus_in_vehicle_seconds_total": float(
            accrued_wait + accrued_in_vehicle
        ),
        "accrued_wait_plus_in_vehicle_seconds_per_departed_passenger": (
            float((accrued_wait + accrued_in_vehicle) / departed)
            if departed
            else None
        ),
    }


def summarize_service(bridge) -> dict:
    departed = len(bridge.departed_trip_ids)
    completed = len(bridge.completed_trip_times)
    trip_times = []
    for trip_id, completion in sorted(bridge.completed_trip_times.items()):
        bus = bridge.bus_obj_dic[trip_id]
        if completion >= bus.start_time_n:
            trip_times.append(completion - bus.start_time_n)
    return {
        "departed_trips": departed,
        "completed_trips": completed,
        "completion_rate": float(completed / departed) if departed else None,
        "active_trips_at_horizon": len(bridge.active_bus_ids),
        "completed_trip_duration_seconds": numeric_summary(trip_times),
    }
