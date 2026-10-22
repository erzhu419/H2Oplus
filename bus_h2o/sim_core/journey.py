"""Causal passenger-journey state for the calibrated simulator."""

from __future__ import annotations

try:
    from bus_h2o.sim_core.passenger import Passenger
except ModuleNotFoundError:
    from sim_core.passenger import Passenger


class JourneyPassenger(Passenger):
    """One active ride leg with an ordered continuation to later legs."""

    __slots__ = (
        "journey_id",
        "leg_index",
        "remaining_legs",
        "source_departure_time",
    )

    def __init__(
        self,
        t,
        boarding_station,
        destination_station,
        *,
        journey_id: str,
        leg_index: int,
        remaining_legs: tuple[dict, ...],
        source_departure_time: float,
    ):
        super().__init__(t, boarding_station, destination_station)
        self.journey_id = str(journey_id)
        self.leg_index = int(leg_index)
        self.remaining_legs = tuple(remaining_legs)
        self.source_departure_time = float(source_departure_time)


__all__ = ["JourneyPassenger"]
