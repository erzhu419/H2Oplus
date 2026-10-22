"""Evaluation summaries for the exact passenger-time v5 contract."""

from __future__ import annotations

from bus_h2o.passenger_time import PassengerTimeSnapshot
from SimpleSAC.transit_v3.evaluation import numeric_summary


def summarize_passengers_v5(
    bridge, snapshot: PassengerTimeSnapshot
) -> dict:
    """Summarize welfare without using the legacy sampled passenger histories."""

    departed_times = dict(bridge.departed_passenger_times)
    completed_times = dict(bridge.completed_passenger_times)
    departed = len(departed_times)
    completed = len(completed_times)
    unfinished = departed - completed
    other_active = (
        unfinished
        - snapshot.waiting_passengers
        - snapshot.in_vehicle_passengers
    )
    journey_times = [
        completion - departed_times[passenger_id]
        for passenger_id, completion in sorted(completed_times.items())
        if passenger_id in departed_times
        and completion >= departed_times[passenger_id]
    ]
    unfinished_ids = sorted(set(departed_times).difference(completed_times))
    unfinished_departures = [
        departed_times[passenger_id] for passenger_id in unfinished_ids
    ]
    unfinished_elapsed = [
        snapshot.time_seconds - departure
        for departure in unfinished_departures
        if snapshot.time_seconds >= departure
    ]
    return {
        "accounting_semantics": (
            "one-second integration of SUMO bus-stop waiting counts and "
            "transit-vehicle person counts"
        ),
        "legacy_sampled_trip_histories_used_for_welfare": False,
        "departed_passengers": departed,
        "completed_passengers": completed,
        "completion_rate": float(completed / departed) if departed else None,
        "unfinished_passengers": unfinished,
        "state_counts_at_horizon": {
            "waiting": snapshot.waiting_passengers,
            "in_vehicle": snapshot.in_vehicle_passengers,
            "other_active": other_active,
        },
        "active_state_count_balance_error": min(other_active, 0),
        "unfinished_departure_seconds": numeric_summary(
            unfinished_departures
        ),
        "unfinished_elapsed_seconds_at_horizon": numeric_summary(
            unfinished_elapsed
        ),
        "completed_journey_seconds": numeric_summary(journey_times),
        "accrued_wait_seconds_total": snapshot.accrued_wait_seconds,
        "accrued_wait_seconds_per_departed_passenger": (
            snapshot.accrued_wait_seconds / departed if departed else None
        ),
        "accrued_in_vehicle_seconds_total": (
            snapshot.accrued_in_vehicle_seconds
        ),
        "accrued_in_vehicle_seconds_per_departed_passenger": (
            snapshot.accrued_in_vehicle_seconds / departed if departed else None
        ),
        "accrued_wait_plus_in_vehicle_seconds_total": (
            snapshot.accrued_wait_seconds
            + snapshot.accrued_in_vehicle_seconds
        ),
        "accrued_wait_plus_in_vehicle_seconds_per_departed_passenger": (
            (
                snapshot.accrued_wait_seconds
                + snapshot.accrued_in_vehicle_seconds
            )
            / departed
            if departed
            else None
        ),
    }


__all__ = ["summarize_passengers_v5"]
