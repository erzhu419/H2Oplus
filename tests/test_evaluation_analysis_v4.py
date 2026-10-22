import json

import pytest

from reproducibility.analyze_evaluation_matrix_v4 import (
    analyze,
    build_cubes,
    load_rows,
)


def _metrics(welfare):
    return {
        "generalized_seconds_per_passenger": float(welfare),
        "wait_seconds_per_passenger": float(welfare / 4),
        "in_vehicle_seconds_per_passenger": float(welfare / 2),
        "passenger_completion_rate": 0.98,
        "headway_normalized_absolute_error": 0.1,
        "mean_hold_seconds": 1.0,
        "nonzero_hold_rate": 0.25,
        "proxy_cost_passenger_minutes": 100.0,
    }


def _row(algorithm, replicate, seed, cell_index, welfare):
    return {
        "algorithm": algorithm,
        "method": f"{algorithm}-{replicate}",
        "training_seed": seed,
        "replicate": replicate,
        "scenario": "demand-1",
        "demand_scale": 1.0,
        "sumo_seed": 100 + cell_index,
        "cell": ("demand-1", 100 + cell_index),
        "metrics": _metrics(welfare),
        "output": "synthetic",
    }


def test_analysis_preserves_training_replicates_and_pairs_environment_cells():
    rows = []
    for cell, welfare in enumerate((100.0, 110.0, 120.0)):
        rows.append(_row("zero", "method-zero", None, cell, welfare))
    for cell, welfare in enumerate((90.0, 100.0, 110.0)):
        rows.append(_row("h2oplus", "seed-1", 1, cell, welfare))
    for cell, welfare in enumerate((94.0, 104.0, 114.0)):
        rows.append(_row("h2oplus", "seed-2", 2, cell, welfare))

    result = analyze(
        rows,
        reference="zero",
        bootstrap_replicates=500,
        permutation_replicates=500,
        seed=7,
    )
    h2o = result["algorithms"]["h2oplus"]
    assert h2o["training_replicates"] == 2
    assert h2o["environment_cells"] == 3
    assert h2o["evaluation_seeds"] == 3
    assert h2o["runs"] == 6
    assert h2o["metrics"]["generalized_seconds_per_passenger"]["mean"] == 102.0
    comparison = result["comparisons_to_reference"]["h2oplus"]
    assert comparison["improvement_seconds_per_passenger"] == 8.0
    assert comparison["paired_environment_block_win_rate"] == 1.0
    assert comparison["test"] == "exact-sign-flip-clustered-by-environment-block"


def test_analysis_clusters_demand_scenarios_within_sumo_seed():
    rows = []
    for algorithm, offset in (("zero", 0.0), ("candidate", -5.0)):
        for seed_index in range(3):
            for scenario, demand_shift in (("low", -10.0), ("high", 10.0)):
                row = _row(
                    algorithm,
                    "method-zero" if algorithm == "zero" else "seed-1",
                    None if algorithm == "zero" else 1,
                    seed_index,
                    100.0 + offset + demand_shift + seed_index,
                )
                row["scenario"] = scenario
                row["demand_scale"] = 0.75 if scenario == "low" else 1.25
                row["cell"] = (scenario, row["sumo_seed"])
                rows.append(row)

    result = analyze(
        rows,
        reference="zero",
        bootstrap_replicates=100,
        permutation_replicates=100,
        seed=11,
    )
    assert result["evaluation_seeds"] == [100, 101, 102]
    assert len(result["environment_cells"]) == 6
    assert result["comparisons_to_reference"]["candidate"][
        "paired_environment_block_win_rate"
    ] == 1.0


def test_analysis_accepts_distinct_scenarios_in_explicit_environment_blocks():
    rows = []
    for algorithm, offset in (("zero", 0.0), ("candidate", -3.0)):
        for block_index in range(3):
            for scenario, scale in (("low", 0.75), ("high", 1.25)):
                row = _row(
                    algorithm,
                    "method-zero" if algorithm == "zero" else "seed-1",
                    None if algorithm == "zero" else 1,
                    block_index,
                    100.0 + offset + block_index,
                )
                row["scenario"] = f"{scenario}-realization-{block_index}"
                row["demand_scale"] = scale
                row["environment_block"] = f"block-{block_index}"
                row["cell"] = (row["scenario"], row["sumo_seed"])
                rows.append(row)

    result = analyze(
        rows,
        reference="zero",
        bootstrap_replicates=100,
        permutation_replicates=100,
        seed=19,
    )
    assert result["environment_blocks"] == ["block-0", "block-1", "block-2"]
    assert result["algorithms"]["candidate"]["environment_blocks"] == 3


def test_v6_analysis_keeps_correlated_demand_scales_in_realization_blocks():
    rows = []
    for algorithm, offset in (("zero", 0.0), ("candidate", -4.0)):
        for block_index in range(3):
            for scenario, scale in (("low", 0.75), ("nominal", 1.0), ("high", 1.25)):
                row = _row(
                    algorithm,
                    "method-zero" if algorithm == "zero" else "seed-1",
                    None if algorithm == "zero" else 1,
                    block_index,
                    100.0 + offset + block_index,
                )
                row["scenario"] = f"{scenario}-realization-{block_index}"
                row["demand_scale"] = scale
                row["demand_realization_seed"] = 3101 + block_index
                row["environment_block"] = f"block-{block_index}"
                row["cell"] = (row["scenario"], row["sumo_seed"])
                rows.append(row)

    result = analyze(
        rows,
        reference="zero",
        bootstrap_replicates=100,
        permutation_replicates=100,
        seed=23,
        mdp_version="v6",
    )
    assert result["mdp_version"] == "v6"
    assert len(result["environment_cells"]) == 9
    assert result["algorithms"]["candidate"]["environment_blocks"] == 3

    for row in rows:
        if row["environment_block"] == "block-2":
            row["demand_realization_seed"] = 3101
    with pytest.raises(ValueError, match="reused across environment blocks"):
        analyze(
            rows,
            reference="zero",
            bootstrap_replicates=10,
            permutation_replicates=10,
            seed=23,
            mdp_version="v6",
        )


def test_cube_builder_rejects_missing_paired_cell():
    rows = [
        _row("zero", "method-zero", None, 0, 100),
        _row("zero", "method-zero", None, 1, 110),
        _row("candidate", "seed-1", 1, 0, 90),
    ]
    with pytest.raises(ValueError, match="incomplete cells"):
        build_cubes(rows)


def test_v7_loader_rechecks_output_provenance_and_accounting(tmp_path):
    output = tmp_path / "zero" / "scenario" / "sumo_seed2101.json"
    output.parent.mkdir(parents=True)
    source = {"evaluate_transit_v7.py": "source"}
    runtime_inputs = {"scenario.xml": "runtime"}
    payload = {
        "schema_version": "transit-evaluation-v7",
        "method": "zero",
        "sumo_seed": 2101,
        "evaluation_job": {"signature": "signature"},
        "dataset": {
            "sha256": "dataset",
            "size_bytes": 123,
            "mtime_ns": 456,
            "verification": "parent-matrix-sha256-plus-child-stat-lock",
        },
        "source_sha256": source,
        "runtime": {"sumo_version": "SUMO 1.27.1"},
        "scenario": {
            "runtime_input_sha256": runtime_inputs,
            "selection_seed": 3101,
            "demand_scale": 1.0,
        },
        "contract": {"horizon_seconds": 24_000.0},
        "episode": {
            "configured_seconds": 24_000,
            "observed_end_time_seconds": 24_000,
            "stopped_by_event_limit": False,
            "stopped_by_time_limit": False,
            "terminal_transitions": 1,
            "control_decisions": 100,
            "transitions": 100,
        },
        "passenger_welfare": {
            "accrued_generalized_seconds_per_departed_passenger": 100.0,
            "chronological_coverage_errors": {
                "wait_passenger_seconds": 0.0,
                "in_vehicle_passenger_seconds": 0.0,
                "reward": 0.0,
            },
        },
        "passengers": {
            "accrued_wait_seconds_per_departed_passenger": 20.0,
            "accrued_in_vehicle_seconds_per_departed_passenger": 60.0,
            "completion_rate": 0.99,
            "active_state_count_balance_error": 0,
        },
        "operations": {
            "headway": {"normalized_absolute_error": {"mean": 0.1}},
            "hold_seconds": {"mean": 0.0},
            "nonzero_hold_rate": 0.0,
        },
        "ledger_audit": {
            "count_source_audit": {
                "max_waiting_count_error": 0,
                "max_in_vehicle_count_error": 0,
                "subscription_fallback_queries": 0,
            }
        },
        "bus_service_semantics_audit": {"passed": True},
        "service": {
            "active_trips_at_horizon": 0,
            "departed_trips": 389,
            "completed_trips": 389,
        },
    }
    output.write_text(json.dumps(payload))
    job = {
        "output": str(output),
        "signature": "signature",
        "dataset_sha256": "dataset",
        "dataset_size_bytes": 123,
        "dataset_mtime_ns": 456,
        "source_sha256": source,
        "sumo_version": "SUMO 1.27.1",
        "sumo_seed": 2101,
        "environment_block": "environment-01",
        "method": {
            "name": "zero",
            "algorithm": "zero",
            "training_seed": None,
            "type": "zero",
        },
        "scenario": {
            "tag": "scenario",
            "runtime_input_sha256": runtime_inputs,
        },
    }
    (tmp_path / "matrix_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "transit-evaluation-matrix-v7",
                "jobs": [job],
            }
        )
    )

    rows, evidence = load_rows(tmp_path)
    assert len(rows) == 1
    assert evidence["mdp_version"] == "v7"
    assert len(evidence["evaluation_outputs_aggregate_sha256"]) == 64

    payload["source_sha256"] = {"evaluate_transit_v7.py": "changed"}
    output.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="source hash mismatch"):
        load_rows(tmp_path)
