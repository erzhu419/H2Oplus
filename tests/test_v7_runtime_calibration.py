import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from bus_h2o.sim_core.bus import Bus, BusState
from bus_h2o.sim_core.route import Route
from bus_h2o.sim_core.station import Station
from reproducibility.build_simulator_runtime_calibration_v7 import (
    FORMAL_CALIBRATION_REALIZATION_BLOCKS,
    FORMAL_FIDELITY_REALIZATION_BLOCKS,
    MATRIX_SCHEMA,
    _load_frozen_zero_audit,
    _zero_jobs,
)
from reproducibility.build_simulator_runtime_calibration_v7_v4 import (
    CALIBRATION_SCHEMA_V4 as CALIBRATION_SCHEMA,
)
from reproducibility.audit_simulator_fidelity_v7 import (
    _reserved_jobs,
    fidelity_source_hashes,
)


def test_v7_fidelity_source_hash_keys_are_workspace_relative():
    source = fidelity_source_hashes()
    assert source
    assert all(not Path(path).is_absolute() for path in source)
    assert "reproducibility/audit_simulator_fidelity_v7.py" in source


def _matrix(tmp_path):
    jobs = []
    for block in range(10):
        for scale_index, scale in enumerate((0.75, 1.0, 1.25)):
            jobs.append(
                {
                    "realization_block_id": block,
                    "output": str(tmp_path / f"block{block}_scale{scale}.npz"),
                    "signature": f"{block:02d}{scale_index:02d}",
                    "policy": {"policy_name": "zero"},
                    "scenario": {"demand_scale": scale},
                }
            )
    path = tmp_path / "matrix.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": MATRIX_SCHEMA,
                "study_stage": "formal",
                "independent_realization_blocks": 10,
                "jobs": jobs,
            }
        )
    )
    return path


def test_formal_runtime_calibration_reserves_disjoint_fidelity_blocks(tmp_path):
    _, jobs, partition = _zero_jobs(_matrix(tmp_path), allow_partial=False)
    assert len(jobs) == 15
    assert sorted({job["realization_block_id"] for job in jobs}) == list(
        FORMAL_CALIBRATION_REALIZATION_BLOCKS
    )
    assert partition["reserved_fidelity_realization_blocks"] == list(
        FORMAL_FIDELITY_REALIZATION_BLOCKS
    )
    assert set(partition["calibration_realization_blocks"]).isdisjoint(
        partition["reserved_fidelity_realization_blocks"]
    )


def test_formal_runtime_calibration_rejects_missing_registered_block(tmp_path):
    path = _matrix(tmp_path)
    matrix = json.loads(path.read_text())
    matrix["jobs"] = [
        job for job in matrix["jobs"] if job["realization_block_id"] != 9
    ]
    path.write_text(json.dumps(matrix))
    with pytest.raises(ValueError, match="registered blocks"):
        _zero_jobs(path, allow_partial=False)


def test_v7_fidelity_uses_only_blocks_excluded_from_runtime_fit(tmp_path):
    path = _matrix(tmp_path)
    matrix = json.loads(path.read_text())
    calibration = {
        "schema_version": CALIBRATION_SCHEMA,
        "coverage_gate": {"passed": True},
        "data_partition": {
            "calibration_realization_blocks": list(
                FORMAL_CALIBRATION_REALIZATION_BLOCKS
            ),
            "reserved_fidelity_realization_blocks": list(
                FORMAL_FIDELITY_REALIZATION_BLOCKS
            ),
        },
    }
    jobs = _reserved_jobs(matrix, calibration)
    assert len(jobs) == 15
    assert sorted({job["realization_block_id"] for job in jobs}) == list(
        FORMAL_FIDELITY_REALIZATION_BLOCKS
    )


def test_v7_fidelity_rejects_unregistered_runtime_partition(tmp_path):
    path = _matrix(tmp_path)
    matrix = json.loads(path.read_text())
    calibration = {
        "schema_version": CALIBRATION_SCHEMA,
        "coverage_gate": {"passed": True},
        "data_partition": {
            "calibration_realization_blocks": list(range(6)),
            "reserved_fidelity_realization_blocks": list(range(6, 10)),
        },
    }
    with pytest.raises(ValueError, match="not registered"):
        _reserved_jobs(matrix, calibration)


def test_formal_runtime_calibration_requires_bound_freeze_audit(tmp_path):
    matrix_path = _matrix(tmp_path)
    audit_path = matrix_path.with_name(matrix_path.stem + ".audit.json")
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": "transit-smdp-v7-frozen-zero-matrix-v2",
                "passed": True,
                "frozen_manifest": {
                    "path": str(matrix_path.resolve()),
                    "sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
                },
                "topology": {
                    "calibration_realization_blocks": list(
                        FORMAL_CALIBRATION_REALIZATION_BLOCKS
                    ),
                    "reserved_fidelity_realization_blocks": list(
                        FORMAL_FIDELITY_REALIZATION_BLOCKS
                    ),
                },
                "artifact_set": {"exact_match": True},
                "shards": [{} for _ in range(30)],
            }
        )
    )
    resolved_path, audit = _load_frozen_zero_audit(matrix_path)
    assert resolved_path == audit_path.resolve()
    assert audit["passed"] is True


def test_formal_runtime_calibration_rejects_stale_freeze_audit(tmp_path):
    matrix_path = _matrix(tmp_path)
    audit_path = matrix_path.with_name(matrix_path.stem + ".audit.json")
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": "transit-smdp-v7-frozen-zero-matrix-v2",
                "passed": True,
                "frozen_manifest": {
                    "path": str(matrix_path.resolve()),
                    "sha256": "stale",
                },
                "topology": {
                    "calibration_realization_blocks": list(
                        FORMAL_CALIBRATION_REALIZATION_BLOCKS
                    ),
                    "reserved_fidelity_realization_blocks": list(
                        FORMAL_FIDELITY_REALIZATION_BLOCKS
                    ),
                },
                "artifact_set": {"exact_match": True},
                "shards": [{} for _ in range(30)],
            }
        )
    )
    with pytest.raises(ValueError, match="does not bind"):
        _load_frozen_zero_audit(matrix_path)


def test_runtime_sampling_conditions_on_registered_demand_scale(monkeypatch):
    route = Route("A-B", "A", "B", 100.0, 15.0, None)
    route.install_runtime_calibration(
        {
            "all_periods": {"n": 1, "samples_seconds": [100.0]},
            "by_service_hour": {},
            "by_demand_scale": {
                "0.75": {
                    "all_periods": {"n": 1, "samples_seconds": [75.0]},
                    "by_service_hour": {},
                },
                "1.25": {
                    "all_periods": {"n": 1, "samples_seconds": [125.0]},
                    "by_service_hour": {},
                },
            },
        },
        fixed_overhead_seconds=3,
    )
    monkeypatch.setattr("bus_h2o.sim_core.route.random.choice", lambda values: values[0])

    assert route.sample_runtime_seconds(0.0, demand_scale=0.75) == 75.0
    assert route.sample_runtime_seconds(0.0, demand_scale=1.25) == 125.0
    assert route.sample_runtime_seconds(0.0, demand_scale=0.9) == 100.0


def _low_fidelity_departure_tick(hold_seconds):
    bus = Bus.__new__(Bus)
    bus.on_route = True
    bus.state = BusState.WAITING_ACTION
    bus.obs = []
    bus.trip_id = 2
    bus.in_station = True
    bus._stop_start_time = None
    bus._stop_station = None
    bus.stop_records = []
    bus._calibrated_travel_steps_remaining = None
    bus._segment_start_time = None
    bus._board_during_open_stop = lambda _current_time: 0

    # The decision was emitted at t=0; its action is consumed on the next tick.
    bus.drive(1, hold_seconds, [], False, None)
    for current_time in range(2, 100):
        bus.drive(current_time, None, [], False, None)
        if bus.state is BusState.TRAVEL:
            return current_time
    raise AssertionError("low-fidelity bus did not depart")


def test_low_fidelity_hold_uses_the_same_one_second_discretization_as_sumo():
    zero_departure = _low_fidelity_departure_tick(0.0)
    assert _low_fidelity_departure_tick(1.0) - zero_departure == 0
    assert _low_fidelity_departure_tick(30.0) - zero_departure == 29
    assert _low_fidelity_departure_tick(60.0) - zero_departure == 59


def test_observation_speed_profile_uses_nearest_event_with_earlier_tie():
    station = Station(1, 3, "A04", True, None)
    station.requires_dynamic_observation_speed = True
    station.backward_headway_observation_speed_profile = (
        np.asarray([10.0, 20.0, 40.0]),
        np.asarray([2.0, 4.0, 8.0], dtype=np.float32),
    )

    assert station.backward_headway_observation_speed(9.0) == 2.0
    assert station.backward_headway_observation_speed(15.0) == 2.0
    assert station.backward_headway_observation_speed(16.0) == 4.0
    assert station.backward_headway_observation_speed(100.0) == 8.0


def _three_stop_bus():
    stations = [
        Station(0, 0, "A01", True, None),
        Station(1, 1, "A02", True, None),
        Station(0, 2, "A03", True, None),
    ]
    for index, station in enumerate(stations):
        station.backward_headway_observation_speed_profile = (
            np.asarray([0.0]),
            np.asarray([11.0 + index], dtype=np.float32),
        )
        station.requires_dynamic_observation_speed = index < 2
    routes = [
        Route("A01-A02", "A01", "A02", 100.0, 15.0, None),
        Route("A02-A03", "A02", "A03", 100.0, 15.0, None),
    ]
    bus = Bus(0, 0, 0.0, True, routes, stations, one_directional=True)
    bus.line_idx = 0
    bus.sumo_trip_index = 0
    bus.line_headway = 300.0
    bus.target_forward_headway = 300.0
    bus.target_backward_headway = 300.0
    bus.forward_headway = 310.0
    bus.backward_headway = 300.0
    bus.forward_bus_present = True
    bus.backward_bus_present = False
    bus._base_stop_duration = 4.0
    return bus, stations


def test_low_fidelity_origin_event_uses_sumo_station_zero_and_profile_speed():
    bus, stations = _three_stop_bus()

    bus._prepare_for_action(25.0, [bus], False)

    assert bus.state is BusState.WAITING_ACTION
    assert bus.obs[2] == 0.0
    assert bus.obs[14] == 11.0

    bus.state = BusState.HOLDING
    bus.obs = []
    bus.last_station = stations[1]
    bus.next_station = stations[2]
    bus._prepare_for_action(50.0, [bus], False)
    assert bus.obs[2] == 1.0
    assert bus.obs[14] == 12.0


def test_low_fidelity_terminal_does_not_emit_a_control_event():
    bus, stations = _three_stop_bus()
    bus.last_station = stations[-1]
    bus.next_station = stations[-1]
    bus.obs = []

    bus._prepare_for_action(75.0, [bus], False)

    assert bus.obs == []
