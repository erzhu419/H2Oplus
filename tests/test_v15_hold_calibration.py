from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.mdp_contract_v7 import DEFAULT_CONFIG_V7, policy_action_to_hold_seconds_v7
from SimpleSAC.evaluate_transit_v15_hold_calibration import calibrated_hold_action
from reproducibility import run_evaluation_matrix_v15_hold_calibration as matrix


METHODS = ROOT / "reproducibility/h2oplus_v15_hold_calibration_evaluation_methods_v1.json"


def test_hold_calibration_scales_physical_seconds_and_preserves_zero() -> None:
    zero = calibrated_hold_action(np.asarray([-1.0], np.float32), 0.25, DEFAULT_CONFIG_V7)
    maximum = calibrated_hold_action(np.asarray([1.0], np.float32), 0.25, DEFAULT_CONFIG_V7)

    assert zero.dtype == np.float32
    assert zero.tolist() == [-1.0]
    assert float(policy_action_to_hold_seconds_v7(maximum[0], DEFAULT_CONFIG_V7)) == pytest.approx(
        DEFAULT_CONFIG_V7.max_hold_seconds * 0.25
    )
    assert maximum.tolist() == pytest.approx([-0.5])


def test_registered_scales_are_monotone_and_unregistered_scale_fails_closed() -> None:
    holds = []
    for scale in (0.125, 0.25, 0.375):
        action = calibrated_hold_action(np.asarray([0.2], np.float32), scale, DEFAULT_CONFIG_V7)
        holds.append(float(policy_action_to_hold_seconds_v7(action[0], DEFAULT_CONFIG_V7)))
    assert holds[0] < holds[1] < holds[2]
    with pytest.raises(ValueError, match="unregistered v15 hold scale"):
        calibrated_hold_action(np.asarray([0.2], np.float32), 0.5, DEFAULT_CONFIG_V7)


def test_methods_manifest_has_15_distinct_physical_behavior_identities() -> None:
    payload = json.loads(METHODS.read_text())
    methods = matrix.load_development_methods(METHODS, matrix.METHODS_SCHEMA)
    identities = {
        (row["training_seed"], row["actor_tensor_sha256"], row["hold_scale"])
        for row in methods
    }

    assert payload["v15_development_evaluation_outcomes_read"] is False
    assert payload["formal_held_out_outcomes_read"] is False
    assert len(methods) == 15
    assert len(identities) == 15
    assert sum(row["algorithm"] == "h2oplus-v15-hold-quarter" for row in methods) == 5


def test_job_command_binds_parent_temperature_and_hold_scale_once() -> None:
    method = matrix.load_development_methods(METHODS, matrix.METHODS_SCHEMA)[0]
    job = {
        "method": method,
        "evaluator": ROOT / "SimpleSAC/evaluate_transit_v15_hold_calibration.py",
        "device": "cpu",
        "dataset": ROOT / "unused-dataset.h5",
        "sumo_dir": ROOT / "unused-sumo",
        "edge_xml": ROOT / "unused-edge.xml",
        "output": ROOT / "unused-output.json",
        "event_log": ROOT / "unused-events.jsonl",
        "sumo_seed": 1,
        "scenario": {"demand_scale": 1.0, "path": ROOT / "unused-scenario.json"},
        "environment_block": "v12-staged-release-development-01",
        "signature": "0" * 64,
        "source_sha256": {},
        "dataset_sha256": "1" * 64,
        "dataset_size_bytes": 1,
        "dataset_mtime_ns": 1,
        "sumo_runtime_lock": ROOT / "unused-runtime.json",
        "sumo_runtime_lock_sha256": "2" * 64,
        "sumo_runtime_signature": "3" * 64,
        "sumo_runtime_version": "1.25.0",
        "sumo_backend": "libsumo",
        "max_sim_time_seconds": 1,
        "max_events": 1,
        "mdp_version": "v7",
        "result_schema": "transit-evaluation-v7",
    }
    command = matrix.job_command(job)

    assert command.count("--expected-temperature-tensor-sha256") == 1
    assert command.count("--expected-hold-scale") == 1
    assert command.count("--expected-parent-method-name") == 1
    assert command[command.index("--expected-hold-scale") + 1] == str(method["hold_scale"])
