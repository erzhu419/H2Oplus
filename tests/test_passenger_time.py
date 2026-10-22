from __future__ import annotations

import pytest

from bus_h2o.passenger_time import (
    PassengerTimeLedger,
    aggregate_passenger_time_snapshots,
    generalized_passenger_time_delta,
)


def test_state_transitions_integrate_waiting_and_in_vehicle_time() -> None:
    ledger = PassengerTimeLedger()
    ledger.transition(2.0, "Lane", "Stop")
    ledger.transition(5.0, "Stop", "Bus")
    ledger.transition(9.0, "Bus", "completed")
    ledger.advance_to(12.0)

    snapshot = ledger.snapshot()
    assert snapshot.time_seconds == 12.0
    assert snapshot.waiting_passengers == 0
    assert snapshot.in_vehicle_passengers == 0
    assert snapshot.accrued_wait_seconds == 3.0
    assert snapshot.accrued_in_vehicle_seconds == 4.0


def test_count_observations_integrate_before_replacing_counts() -> None:
    ledger = PassengerTimeLedger()
    ledger.set_counts(0.0, waiting_passengers=3, in_vehicle_passengers=2)
    ledger.advance_to(4.0)
    ledger.set_counts(4.0, waiting_passengers=1, in_vehicle_passengers=5)
    ledger.advance_to(6.0)

    snapshot = ledger.snapshot()
    assert snapshot.accrued_wait_seconds == 14.0
    assert snapshot.accrued_in_vehicle_seconds == 18.0


def test_aggregate_and_generalized_delta_are_auditable() -> None:
    first = PassengerTimeLedger()
    second = PassengerTimeLedger()
    first.set_counts(0.0, waiting_passengers=2, in_vehicle_passengers=1)
    second.set_counts(0.0, waiting_passengers=1, in_vehicle_passengers=4)
    start = aggregate_passenger_time_snapshots(
        (first.snapshot(), second.snapshot())
    )
    first.advance_to(10.0)
    second.advance_to(10.0)
    end = aggregate_passenger_time_snapshots(
        (first.snapshot(), second.snapshot())
    )

    components = generalized_passenger_time_delta(
        start,
        end,
        waiting_time_value=2.0,
        in_vehicle_time_value=1.0,
        reward_unit_passenger_seconds=10.0,
    )
    assert components == {
        "wait_passenger_seconds": 30.0,
        "in_vehicle_passenger_seconds": 50.0,
        "waiting_cost": 6.0,
        "in_vehicle_cost": 5.0,
        "total_cost": 11.0,
    }


def test_ledger_rejects_backward_time_and_unbalanced_transitions() -> None:
    ledger = PassengerTimeLedger(5.0)
    with pytest.raises(ValueError, match="backwards"):
        ledger.advance_to(4.0)
    with pytest.raises(RuntimeError, match="underflow"):
        ledger.transition(5.0, "Stop", "Bus")
