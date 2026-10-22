from types import SimpleNamespace

from SimpleSAC.evaluate_simulator_v4 import (
    simulator_passengers,
    simulator_service,
)


def _passenger(appear, board=None, arrive=None):
    return SimpleNamespace(
        appear_time=appear,
        boarding_time=board,
        arrive_time=arrive,
    )


def test_simulator_terminal_summaries_include_censored_passenger_time():
    line = SimpleNamespace(
        last_episode_passengers=(
            _passenger(0, 10, 30),
            _passenger(5),
            _passenger(2, 12),
        ),
        last_episode_bus_summaries=(
            {
                "completed": True,
                "launch_time_seconds": 10.0,
                "terminal_arrival_seconds": 70.0,
            },
            {
                "completed": False,
                "launch_time_seconds": 20.0,
                "terminal_arrival_seconds": None,
            },
        ),
    )
    env = SimpleNamespace(line_map={"line": line})

    passengers = simulator_passengers(env, horizon=100.0)
    assert passengers["generated_ride_legs"] == 3
    assert passengers["completed_ride_legs"] == 1
    assert passengers["accrued_wait_seconds_total"] == 115.0
    assert passengers["accrued_in_vehicle_seconds_total"] == 108.0
    assert passengers["state_counts_at_horizon"] == {
        "completed": 1,
        "onboard": 1,
        "waiting": 1,
    }

    service = simulator_service(env)
    assert service["departed_trips"] == 2
    assert service["completed_trips"] == 1
    assert service["completed_trip_duration_seconds"]["mean"] == 60.0
