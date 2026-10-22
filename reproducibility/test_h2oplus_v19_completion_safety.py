from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.mdp_contract_v7 import (  # noqa: E402
    RAW_EVENT_STATE_DIM,
    RAW_EVENT_STATE_INDEX,
    TransitMDPConfigV7,
    hold_seconds_to_policy_action_v7,
    pack_event_batch_v7,
    policy_action_to_hold_seconds_v7,
)
from SimpleSAC.evaluate_transit_v17_tail_service_reserve import (  # noqa: E402
    tail_service_hold_action,
)
from SimpleSAC.evaluate_transit_v19_completion_safety_reserve import (  # noqa: E402
    completion_safety_action,
)


CONFIG = TransitMDPConfigV7()


def _state(
    progress: float,
    *,
    onboard: float = 10.0,
    capacity: float = 50.0,
    arrival_rate: float = 0.01,
    forward_present: float = 1.0,
    backward_present: float = 0.0,
) -> np.ndarray:
    event = np.zeros(RAW_EVENT_STATE_DIM, dtype=np.float32)
    values = {
        "line_id": 0.0,
        "station_index": 5.0,
        "sim_time_seconds": progress * CONFIG.horizon_seconds,
        "forward_headway_seconds": 720.0,
        "backward_headway_seconds": 720.0,
        "target_forward_headway_seconds": 720.0,
        "target_backward_headway_seconds": 720.0,
        "eligible_stranded_passengers": 0.0,
        "onboard_passengers": onboard,
        "eligible_arrival_rate_per_second": arrival_rate,
        "base_dwell_seconds": 10.0,
        "forward_bus_present": forward_present,
        "backward_bus_present": backward_present,
        "bus_capacity": capacity,
        "system_waiting_passengers": 10.0,
        "system_in_vehicle_passengers": onboard,
    }
    for name, value in values.items():
        event[RAW_EVENT_STATE_INDEX[name]] = value
    return pack_event_batch_v7(event[None, :], 0, config=CONFIG)


def _hold(state: np.ndarray, variant: str, parent_hold: float = 10.0) -> float:
    action = hold_seconds_to_policy_action_v7(parent_hold, CONFIG)
    transformed = completion_safety_action(state, action, variant, CONFIG)
    return float(policy_action_to_hold_seconds_v7(transformed[0], CONFIG))


def test_baseline_is_exact_v17_twenty_second_parent() -> None:
    action = hold_seconds_to_policy_action_v7(10.0, CONFIG)
    for progress in (0.1, 0.15, 0.5, 0.749, 0.75, 0.9):
        state = _state(progress)
        expected = tail_service_hold_action(state, action, 20.0, CONFIG)
        observed = completion_safety_action(state, action, "baseline", CONFIG)
        np.testing.assert_array_equal(observed, expected)


def test_candidate_uses_registered_window_and_quarter_load_boundary() -> None:
    assert _hold(_state(0.149, onboard=12.5), "quarter-load-sixty") < 60.0
    assert _hold(_state(0.15, onboard=12.5), "quarter-load-sixty") == 60.0
    assert _hold(_state(0.749, onboard=12.5), "quarter-load-sixty") == 60.0
    assert _hold(_state(0.75, onboard=12.5), "quarter-load-sixty") < 60.0
    assert _hold(_state(0.4, onboard=12.51), "quarter-load-sixty") < 60.0


def test_candidate_requires_tail_service_signal() -> None:
    assert _hold(_state(0.4, forward_present=0.0), "quarter-load-sixty") < 60.0
    assert _hold(_state(0.4, backward_present=1.0), "quarter-load-sixty") < 60.0
    assert _hold(_state(0.4, arrival_rate=0.0), "quarter-load-sixty") < 60.0
    assert _hold(_state(0.4, capacity=0.0), "quarter-load-sixty") < 60.0


def test_fifty_second_variant_is_diagnostic_bracket() -> None:
    state = _state(0.4, onboard=12.5)
    assert _hold(state, "quarter-load-fifty") == pytest.approx(50.0)
    assert _hold(state, "quarter-load-sixty") == pytest.approx(60.0)


def test_transform_rejects_unregistered_or_nonfinite_actions() -> None:
    state = _state(0.4)
    with pytest.raises(ValueError, match="unregistered"):
        completion_safety_action(state, 0.0, "other", CONFIG)
    with pytest.raises(ValueError, match="finite scalar"):
        completion_safety_action(state, np.nan, "quarter-load-sixty", CONFIG)
