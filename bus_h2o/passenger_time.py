"""Incremental accounting for accrued passenger waiting and in-vehicle time."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PassengerTimeSnapshot:
    """Cumulative passenger-time state at one simulation instant."""

    time_seconds: float
    waiting_passengers: int
    in_vehicle_passengers: int
    accrued_wait_seconds: float
    accrued_in_vehicle_seconds: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class PassengerTimeLedger:
    """Integrate passenger counts without rescanning complete trip histories."""

    _WAITING_STATES = frozenset({"Stop", "waiting"})
    _IN_VEHICLE_STATES = frozenset({"Bus", "onboard", "in_vehicle"})

    def __init__(self, time_seconds: float = 0.0) -> None:
        self.reset(time_seconds)

    @staticmethod
    def _finite_time(value: float, name: str) -> float:
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    @classmethod
    def _bucket(cls, state: str | None) -> str | None:
        if state in cls._WAITING_STATES:
            return "waiting"
        if state in cls._IN_VEHICLE_STATES:
            return "in_vehicle"
        return None

    def reset(
        self,
        time_seconds: float = 0.0,
        *,
        waiting_passengers: int = 0,
        in_vehicle_passengers: int = 0,
    ) -> None:
        time_seconds = self._finite_time(time_seconds, "time_seconds")
        waiting_passengers = int(waiting_passengers)
        in_vehicle_passengers = int(in_vehicle_passengers)
        if waiting_passengers < 0 or in_vehicle_passengers < 0:
            raise ValueError("passenger counts must be non-negative")
        self.time_seconds = time_seconds
        self.waiting_passengers = waiting_passengers
        self.in_vehicle_passengers = in_vehicle_passengers
        self.accrued_wait_seconds = 0.0
        self.accrued_in_vehicle_seconds = 0.0

    def advance_to(self, time_seconds: float) -> None:
        time_seconds = self._finite_time(time_seconds, "time_seconds")
        elapsed = time_seconds - self.time_seconds
        if elapsed < -1e-9:
            raise ValueError(
                f"passenger-time clock moved backwards by {-elapsed:g} seconds"
            )
        if elapsed <= 0.0:
            self.time_seconds = max(self.time_seconds, time_seconds)
            return
        self.accrued_wait_seconds += self.waiting_passengers * elapsed
        self.accrued_in_vehicle_seconds += self.in_vehicle_passengers * elapsed
        self.time_seconds = time_seconds

    def transition(
        self,
        time_seconds: float,
        previous_state: str | None,
        next_state: str | None,
    ) -> None:
        """Apply one passenger state change at ``time_seconds``."""

        self.advance_to(time_seconds)
        previous_bucket = self._bucket(previous_state)
        next_bucket = self._bucket(next_state)
        if previous_bucket == next_bucket:
            return
        if previous_bucket == "waiting":
            self.waiting_passengers -= 1
        elif previous_bucket == "in_vehicle":
            self.in_vehicle_passengers -= 1
        if next_bucket == "waiting":
            self.waiting_passengers += 1
        elif next_bucket == "in_vehicle":
            self.in_vehicle_passengers += 1
        if self.waiting_passengers < 0 or self.in_vehicle_passengers < 0:
            raise RuntimeError(
                "passenger-time ledger underflow; a state transition was not balanced"
            )

    def set_counts(
        self,
        time_seconds: float,
        *,
        waiting_passengers: int,
        in_vehicle_passengers: int,
    ) -> None:
        """Advance to a time and replace counts with an independently observed state."""

        self.advance_to(time_seconds)
        waiting_passengers = int(waiting_passengers)
        in_vehicle_passengers = int(in_vehicle_passengers)
        if waiting_passengers < 0 or in_vehicle_passengers < 0:
            raise ValueError("passenger counts must be non-negative")
        self.waiting_passengers = waiting_passengers
        self.in_vehicle_passengers = in_vehicle_passengers

    def snapshot(self) -> PassengerTimeSnapshot:
        return PassengerTimeSnapshot(
            time_seconds=float(self.time_seconds),
            waiting_passengers=int(self.waiting_passengers),
            in_vehicle_passengers=int(self.in_vehicle_passengers),
            accrued_wait_seconds=float(self.accrued_wait_seconds),
            accrued_in_vehicle_seconds=float(self.accrued_in_vehicle_seconds),
        )


def aggregate_passenger_time_snapshots(
    snapshots: Iterable[PassengerTimeSnapshot],
) -> PassengerTimeSnapshot:
    """Sum synchronized per-line snapshots into one system-wide snapshot."""

    values = tuple(snapshots)
    if not values:
        return PassengerTimeSnapshot(0.0, 0, 0, 0.0, 0.0)
    reference_time = values[0].time_seconds
    if any(abs(value.time_seconds - reference_time) > 1e-9 for value in values):
        raise ValueError("cannot aggregate passenger-time snapshots from different times")
    return PassengerTimeSnapshot(
        time_seconds=float(reference_time),
        waiting_passengers=sum(value.waiting_passengers for value in values),
        in_vehicle_passengers=sum(
            value.in_vehicle_passengers for value in values
        ),
        accrued_wait_seconds=sum(value.accrued_wait_seconds for value in values),
        accrued_in_vehicle_seconds=sum(
            value.accrued_in_vehicle_seconds for value in values
        ),
    )


def generalized_passenger_time_delta(
    start: PassengerTimeSnapshot,
    end: PassengerTimeSnapshot,
    *,
    waiting_time_value: float,
    in_vehicle_time_value: float,
    reward_unit_passenger_seconds: float,
) -> dict[str, float]:
    """Return auditable scaled cost components between two ledger snapshots."""

    waiting_time_value = float(waiting_time_value)
    in_vehicle_time_value = float(in_vehicle_time_value)
    reward_unit_passenger_seconds = float(reward_unit_passenger_seconds)
    parameters = (
        waiting_time_value,
        in_vehicle_time_value,
        reward_unit_passenger_seconds,
    )
    if not all(np.isfinite(value) and value > 0.0 for value in parameters):
        raise ValueError("passenger-time values and reward unit must be positive")
    if end.time_seconds < start.time_seconds - 1e-9:
        raise ValueError("passenger-time snapshot order is reversed")
    wait_seconds = end.accrued_wait_seconds - start.accrued_wait_seconds
    in_vehicle_seconds = (
        end.accrued_in_vehicle_seconds - start.accrued_in_vehicle_seconds
    )
    if wait_seconds < -1e-6 or in_vehicle_seconds < -1e-6:
        raise ValueError("cumulative passenger time decreased between snapshots")
    waiting_cost = waiting_time_value * max(wait_seconds, 0.0)
    in_vehicle_cost = in_vehicle_time_value * max(in_vehicle_seconds, 0.0)
    return {
        "wait_passenger_seconds": float(max(wait_seconds, 0.0)),
        "in_vehicle_passenger_seconds": float(max(in_vehicle_seconds, 0.0)),
        "waiting_cost": float(waiting_cost / reward_unit_passenger_seconds),
        "in_vehicle_cost": float(
            in_vehicle_cost / reward_unit_passenger_seconds
        ),
        "total_cost": float(
            (waiting_cost + in_vehicle_cost) / reward_unit_passenger_seconds
        ),
    }


__all__ = [
    "PassengerTimeLedger",
    "PassengerTimeSnapshot",
    "aggregate_passenger_time_snapshots",
    "generalized_passenger_time_delta",
]
