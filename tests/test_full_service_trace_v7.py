import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from reproducibility.collect_full_service_trace_v7 import (
    TRACE_MATRIX_SCHEMA,
    _load_trace_registration,
    canonical_signature,
    verify_control_replay,
)
from reproducibility.run_full_service_trace_matrix_v7 import register_matrix
from reproducibility.build_simulator_runtime_calibration_v7_v2 import (
    validate_trace_arrays,
)
from bus_h2o.full_service_trace_v7 import (
    TRACE_ARRAY_NAMES,
    decision_ready_time,
    extract_full_service_arrays,
)
from bus_h2o.sumo_provenance import sha256


def _bridge(*, omit_terminal_departure=False):
    stops = ["A", "B", "C"]
    departures = {"A": 16.0, "B": 34.0, "C": 56.0}
    if omit_terminal_departure:
        departures.pop("C")
    bus = SimpleNamespace(
        belong_line_id_s="7X",
        arriver_stop_time_d={"A": 10.0, "B": 28.0, "C": 49.0},
        service_completion_history=defaultdict(
            list,
            {"A": [14.5], "B": [32.0], "C": [54.5]},
        ),
        depart_stop_time_d=departures,
    )
    return SimpleNamespace(
        departed_trip_ids={"trip-1"},
        completed_trip_times={"trip-1": 60.0},
        bus_obj_dic={"trip-1": bus},
        line_obj_dic={"7X": SimpleNamespace(stop_id_l=stops)},
    )


def test_decision_ready_time_matches_integer_sumo_event_clock():
    assert decision_ready_time(14.0) == 14.0
    assert decision_ready_time(14.5) == 15.0


def test_full_service_trace_includes_terminal_segment():
    arrays, diagnostics = extract_full_service_arrays(_bridge(), ["7X"])

    assert arrays["stop_station_index"].tolist() == [0, 1, 2]
    assert arrays["stop_decision_ready_time_seconds"].tolist() == [15.0, 32.0, 55.0]
    assert arrays["segment_origin_station_index"].tolist() == [0, 1]
    assert arrays["segment_decision_clock_runtime_seconds"].tolist() == [13.0, 17.5]
    assert arrays["segment_road_runtime_seconds"].tolist() == [12.0, 15.0]
    assert arrays["segment_event_clock_minus_road_runtime_seconds"].tolist() == [
        1.0,
        2.5,
    ]
    assert diagnostics["completed_trips"] == 1
    assert diagnostics["per_line_segments"] == {"7X": 2}

    samples, rebuilt = validate_trace_arrays(
        {"full_service_trace": diagnostics}, arrays, {"7X": ["A", "B", "C"]}
    )
    assert samples == [
        ("7X", 0, 15.0, 13.0, 12.0),
        ("7X", 1, 32.0, 17.5, 15.0),
    ]
    assert rebuilt["maximum_decision_runtime_reconstruction_error_seconds"] == 0.0
    assert rebuilt["maximum_road_runtime_reconstruction_error_seconds"] == 0.0
    assert rebuilt["maximum_clock_difference_reconstruction_error_seconds"] == 0.0
    assert rebuilt["low_fidelity_event_clock_mapping_error_seconds"] == {
        "minimum": 0.0,
        "maximum": 1.0,
        "mean": 0.5,
        "mean_absolute": 0.5,
        "p95_absolute": 0.95,
        "maximum_absolute": 1.0,
    }


def test_v2_calibration_rejects_incomplete_segment_sequence():
    arrays, diagnostics = extract_full_service_arrays(_bridge(), ["7X"])
    for name in TRACE_ARRAY_NAMES:
        if name.startswith("segment_"):
            arrays[name] = arrays[name][:-1]

    with pytest.raises(ValueError, match="segment sequence is incomplete"):
        validate_trace_arrays(
            {"full_service_trace": diagnostics},
            arrays,
            {"7X": ["A", "B", "C"]},
        )


def test_full_service_trace_rejects_missing_terminal_departure():
    with pytest.raises(ValueError, match="departures=.*C"):
        extract_full_service_arrays(_bridge(omit_terminal_departure=True), ["7X"])


def test_control_replay_requires_all_identity_and_timing_fields_exact():
    summary = {
        "event_identity": {
            "bus_code_to_id": ["trip-1"],
            "line_index_to_id": ["7X"],
        }
    }
    arrays = {
        "event_bus_code": np.asarray([0, 0], dtype=np.int32),
        "event_line_index": np.asarray([0, 0], dtype=np.int16),
        "event_station_index": np.asarray([0, 1], dtype=np.int16),
        "event_decision_time_seconds": np.asarray([15.0, 32.0]),
        "event_base_dwell_seconds": np.asarray([4.5, 4.0]),
    }
    observed = {
        "bus_id": ["trip-1", "trip-1"],
        "line_id": ["7X", "7X"],
        "station_index": [0, 1],
        "decision_time_seconds": [15.0, 32.0],
        "base_dwell_seconds": [4.5, 4.0],
    }

    assert verify_control_replay(observed, summary, arrays)["passed"] is True
    observed["decision_time_seconds"][1] = 33.0
    with pytest.raises(ValueError, match="decision_time_seconds"):
        verify_control_replay(observed, summary, arrays)


def test_control_replay_rejects_length_mismatch():
    summary = {
        "event_identity": {
            "bus_code_to_id": ["trip-1"],
            "line_index_to_id": ["7X"],
        }
    }
    arrays = {
        "event_bus_code": np.asarray([0], dtype=np.int32),
        "event_line_index": np.asarray([0], dtype=np.int16),
        "event_station_index": np.asarray([0], dtype=np.int16),
        "event_decision_time_seconds": np.asarray([15.0]),
        "event_base_dwell_seconds": np.asarray([4.5]),
    }
    observed = {
        "bus_id": [],
        "line_id": [],
        "station_index": [],
        "decision_time_seconds": [],
        "base_dwell_seconds": [],
    }

    with pytest.raises(ValueError, match="observed_rows.*0"):
        verify_control_replay(observed, summary, arrays)


def test_trace_collector_requires_exact_registered_invocation(tmp_path):
    frozen = tmp_path / "zero_matrix.json"
    frozen.write_text("{}\n")
    output = (tmp_path / "trace.npz").resolve()
    collector = Path(
        "reproducibility/collect_full_service_trace_v7.py"
    ).resolve()
    registration = {
        "frozen_zero_matrix": {
            "path": str(frozen.resolve()),
            "sha256": sha256(frozen),
        },
        "trace_collector": {
            "path": str(collector),
            "sha256": sha256(collector),
        },
        "trace_runner": {"sha256": "runner-hash"},
        "jobs": [
            {
                "block_id": 4,
                "output": str(output),
                "trace_signature": "trace-signature",
            }
        ],
    }
    manifest_path = tmp_path / "trace_matrix.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": TRACE_MATRIX_SCHEMA,
                "registration": registration,
                "registration_sha256": canonical_signature(registration),
            }
        )
    )

    _, job = _load_trace_registration(
        manifest_path,
        matrix_manifest=frozen,
        output=output,
        block_id=4,
        trace_signature="trace-signature",
        trace_runner_sha256="runner-hash",
    )
    assert job["block_id"] == 4
    with pytest.raises(ValueError, match="differs from registration"):
        _load_trace_registration(
            manifest_path,
            matrix_manifest=frozen,
            output=tmp_path / "wrong.npz",
            block_id=4,
            trace_signature="trace-signature",
            trace_runner_sha256="runner-hash",
        )


def test_trace_matrix_registration_is_immutable(tmp_path, monkeypatch):
    registration = {"jobs": [{"trace_signature": "one"}]}
    jobs = [{"trace_signature": "one"}]
    monkeypatch.setattr(
        "reproducibility.run_full_service_trace_matrix_v7.build_registration",
        lambda matrix, output: (registration, jobs),
    )
    matrix = tmp_path / "frozen.json"
    matrix.write_text("{}\n")
    output_root = tmp_path / "traces"

    path, first, first_jobs = register_matrix(matrix, output_root)
    first_bytes = path.read_bytes()
    _, second, second_jobs = register_matrix(matrix, output_root)
    assert first == second
    assert first_jobs == second_jobs == jobs
    assert path.read_bytes() == first_bytes

    changed = {"jobs": [{"trace_signature": "changed"}]}
    monkeypatch.setattr(
        "reproducibility.run_full_service_trace_matrix_v7.build_registration",
        lambda matrix, output: (changed, jobs),
    )
    with pytest.raises(ValueError, match="differs from this run"):
        register_matrix(matrix, output_root)
