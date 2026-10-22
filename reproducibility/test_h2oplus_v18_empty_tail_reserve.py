from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.mdp_contract_v7 import (
    RAW_EVENT_STATE_DIM,
    RAW_EVENT_STATE_INDEX,
    TransitMDPConfigV7,
    hold_seconds_to_policy_action_v7,
    pack_event_batch_v7,
    policy_action_to_hold_seconds_v7,
)
from SimpleSAC.evaluate_transit_v17_tail_service_reserve import (
    tail_service_hold_action,
)
from SimpleSAC.evaluate_transit_v18_empty_tail_reserve import (
    empty_tail_reserve_action,
)
from reproducibility.analyze_h2oplus_v18_empty_tail_reserve import (
    _ledger_and_service_pass,
)
from reproducibility import run_evaluation_matrix_v18_empty_tail_reserve_v2 as matrix_v2
from reproducibility import run_evaluation_matrix_v18_empty_tail_reserve_v3 as matrix_v3


CONFIG = TransitMDPConfigV7()


def _state(progress: float, onboard: float = 0.0) -> np.ndarray:
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
        "eligible_arrival_rate_per_second": 0.01,
        "base_dwell_seconds": 10.0,
        "forward_bus_present": 1.0,
        "backward_bus_present": 0.0,
        "bus_capacity": 50.0,
        "system_waiting_passengers": 10.0,
        "system_in_vehicle_passengers": onboard,
    }
    for name, value in values.items():
        event[RAW_EVENT_STATE_INDEX[name]] = value
    return pack_event_batch_v7(event[None, :], 0, config=CONFIG)


def _hold(state: np.ndarray, variant: str) -> float:
    action = hold_seconds_to_policy_action_v7(10.0, CONFIG)
    transformed = empty_tail_reserve_action(state, action, variant, CONFIG)
    return float(policy_action_to_hold_seconds_v7(transformed[0], CONFIG))


def test_baseline_is_exact_v17_twenty_second_parent() -> None:
    action = hold_seconds_to_policy_action_v7(10.0, CONFIG)
    for progress in (0.2, 0.3, 0.5, 0.74, 0.75, 0.9):
        state = _state(progress)
        expected = tail_service_hold_action(state, action, 20.0, CONFIG)
        observed = empty_tail_reserve_action(state, action, "baseline", CONFIG)
        np.testing.assert_array_equal(observed, expected)


def test_empty_candidate_uses_only_registered_mid_horizon_guard() -> None:
    assert _hold(_state(0.24), "empty-forty") < 40.0
    assert _hold(_state(0.25), "empty-forty") == pytest.approx(40.0)
    assert _hold(_state(0.50), "empty-forty") == pytest.approx(40.0)
    assert _hold(_state(0.749), "empty-forty") == pytest.approx(40.0)
    assert _hold(_state(0.75), "empty-forty") < 40.0


def test_one_seat_is_diagnostic_only() -> None:
    one_seat = _state(0.4, onboard=1.0)
    assert _hold(one_seat, "empty-forty") < 40.0
    assert _hold(one_seat, "one-seat-forty") == pytest.approx(40.0)


def test_transform_rejects_unregistered_or_nonfinite_actions() -> None:
    state = _state(0.4)
    with pytest.raises(ValueError, match="unregistered"):
        empty_tail_reserve_action(state, 0.0, "other", CONFIG)
    with pytest.raises(ValueError, match="finite scalar"):
        empty_tail_reserve_action(state, np.nan, "empty-forty", CONFIG)


def _valid_accounting_payload() -> dict:
    return {
        "bus_service_semantics_audit": {
            "passed": True,
            "require_all_departed_trips_completed": True,
            "violations": {"missing_stops": [], "extra_stops": {}},
        },
        "service": {
            "active_trips_at_horizon": 0,
            "completed_trips": 24,
            "departed_trips": 24,
            "completion_rate": 1.0,
        },
        "ledger_audit": {
            "legacy_sampled_trip_histories_used_as_ground_truth": False,
            "count_source_audit": {
                "max_in_vehicle_count_error": 0,
                "max_waiting_count_error": 0,
                "subscription_fallback_queries": 0,
            },
        },
        "passengers": {
            "legacy_sampled_trip_histories_used_for_welfare": False,
            "active_state_count_balance_error": 0,
        },
        "passenger_welfare": {
            "primary": True,
            "chronological_coverage_errors": {"wait_seconds": 0.0},
            "maximum_reward_reconstruction_error": 0.0,
        },
    }


def test_analysis_accepts_structured_ledger_without_synthetic_pass_field() -> None:
    assert _ledger_and_service_pass(_valid_accounting_payload())


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("service", "active_trips_at_horizon", 1),
        ("passengers", "active_state_count_balance_error", 1),
        ("passenger_welfare", "maximum_reward_reconstruction_error", 1e-4),
    ),
)
def test_analysis_rejects_accounting_drift(
    section: str, field: str, value: float
) -> None:
    payload = _valid_accounting_payload()
    payload[section][field] = value
    assert not _ledger_and_service_pass(payload)


def _parser_smoke_job(tmp_path: Path) -> dict:
    methods_path = (
        ROOT
        / "reproducibility/h2oplus_v18_empty_tail_reserve_evaluation_methods_v1.json"
    )
    method = matrix_v2.load_development_methods(
        methods_path, matrix_v2.METHODS_SCHEMA
    )[0]
    dataset = (
        ROOT / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5"
    ).resolve(strict=True)
    runtime_lock = (
        ROOT / "reproducibility/h2oplus_v8_stage_a_sumo_runtime_lock_py38.json"
    ).resolve(strict=True)
    scenario = next(
        (ROOT / "bus_h2o/development_scenarios_v18/empty_tail_reserve_v1").rglob(
            "manifest_portable_v8.json"
        )
    ).resolve(strict=True)
    stat = dataset.stat()
    return {
        "evaluator": ROOT
        / "SimpleSAC/evaluate_transit_v18_empty_tail_reserve_v2.py",
        "method": method,
        "sumo_seed": 7101,
        "scenario": {"path": scenario},
        "dataset": dataset,
        "output": tmp_path / "unused.json",
        "signature": "v18-v2-parser-smoke",
        "device": "cpu",
        "max_events": 0,
        "max_sim_time_seconds": 0,
        "sumo_dir": ROOT.parent / "SUMO_ruiguang/online_control",
        "edge_xml": ROOT / "bus_h2o/network_data/a_sorted_busline_edge.xml",
        "mdp_version": "v7",
        "sumo_backend": "libsumo",
        "sumo_runtime_version": "1.25.0",
        "sumo_runtime_signature": "0" * 64,
        "sumo_runtime_lock": runtime_lock,
        "sumo_runtime_lock_sha256": hashlib.sha256(
            runtime_lock.read_bytes()
        ).hexdigest(),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "dataset_size_bytes": stat.st_size,
        "dataset_mtime_ns": stat.st_mtime_ns,
        "event_log": None,
    }


def test_v2_command_keeps_parent_and_empty_guard_flags_distinct(tmp_path: Path) -> None:
    command = matrix_v2.job_command(_parser_smoke_job(tmp_path))
    assert command.count("--expected-activation-start-fraction") == 1
    assert command.count("--expected-empty-activation-start-fraction") == 1
    assert command.count("--expected-activation-end-fraction") == 1
    assert command.count("--expected-empty-activation-end-fraction") == 1
    assert command.count("--expected-max-onboard-capacity-fraction") == 1
    assert command.count("--expected-empty-max-onboard-capacity-fraction") == 1


def test_v2_real_nested_parser_stack_without_sumo(tmp_path: Path) -> None:
    command = [*matrix_v2.job_command(_parser_smoke_job(tmp_path)), "--help"]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout
    assert "usage:" in completed.stdout
    assert not (tmp_path / "unused.json").exists()


def test_v3_rebinds_after_every_parent_parser_without_sumo(tmp_path: Path) -> None:
    job = _parser_smoke_job(tmp_path)
    job["evaluator"] = (
        ROOT / "SimpleSAC/evaluate_transit_v18_empty_tail_reserve_v3.py"
    )
    command = matrix_v3.job_command(job)
    probe = r'''
import json
import sys
from SimpleSAC import evaluate_transit_v18_empty_tail_reserve_v3 as target

def frozen_validator_probe(args, _config, _station_max, _dataset_hash):
    observed = {
        "parent_floor": args.expected_parent_tail_service_floor_seconds,
        "enabled": args.expected_empty_reserve_enabled,
        "empty_floor": args.expected_empty_reserve_floor_seconds,
        "start": args.expected_activation_start_fraction,
        "end": args.expected_activation_end_fraction,
        "max_onboard": args.expected_max_onboard_capacity_fraction,
        "parent_method": args.expected_v17_parent_method_name,
    }
    expected = {
        "parent_floor": 20.0,
        "enabled": False,
        "empty_floor": 40.0,
        "start": 0.25,
        "end": 0.75,
        "max_onboard": 0.0,
        "parent_method": (
            "h2oplus-v17-tail-service-reserve-twenty-"
            "seed6201-events000025000"
        ),
    }
    if observed != expected:
        raise RuntimeError(json.dumps({"observed": observed, "expected": expected}))
    print(json.dumps({"v18_final_binding": observed}, sort_keys=True))
    return None, {}

def no_sumo_base_main():
    args = target.base.parse_args()
    target.base.build_policy(args, None, None, None)

target.v1.build_development_policy = frozen_validator_probe
target.base.main = no_sumo_base_main
sys.argv = [str(target.__file__), *sys.argv[1:]]
target.main()
'''
    completed = subprocess.run(
        [sys.executable, "-c", probe, *command[2:]],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout
    assert '"start": 0.25' in completed.stdout
    assert '"max_onboard": 0.0' in completed.stdout
    assert not (tmp_path / "unused.json").exists()
