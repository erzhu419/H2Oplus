import json

import numpy as np

from reproducibility.analyze_evaluation_matrix import (
    _holm_adjust,
    _paired_permutation_p,
    analyze,
    load_rows,
    write_flat_csv,
    write_json,
)


def _evaluation(method, seed, demand, sumo_seed, wait, headway):
    return {
        "schema_version": "transit-evaluation-v3",
        "method": method,
        "sumo_seed": sumo_seed,
        "dataset": {"sha256": "dataset-hash"},
        "contract": {"episode_seconds": 24_000.0},
        "episode": {"stopped_by_event_limit": False},
        "scenario": {"demand_scale": demand},
        "policy": {
            "training_metadata": {
                "method": method,
                "training_seed": seed,
            }
        },
        "passengers": {
            "accrued_wait_seconds_per_departed_passenger": wait,
            "completed_journey_seconds": None,
            "completion_rate": 1.0,
        },
        "operations": {
            "headway": {
                "normalized_absolute_error": {"mean": headway},
                "bunching_rate": 0.1,
                "large_gap_rate": 0.2,
            },
            "cumulative_smdp_reward": -headway,
            "total_hold_seconds": 0.0,
        },
    }


def test_analysis_uses_training_metadata_and_paired_cells(tmp_path):
    for method, seed, offset in (("reference", 1, 0.0), ("candidate", 2, -10.0)):
        for index, demand in enumerate((0.75, 1.25)):
            payload = _evaluation(
                method,
                seed,
                demand,
                100 + index,
                wait=300.0 + offset + index,
                headway=0.4 + offset / 1000.0,
            )
            write_json(tmp_path / f"{method}_{index}.json", payload)

    rows, evidence = load_rows(tmp_path, allow_partial=False)
    report = analyze(rows, reference="reference", bootstrap_replicates=100)
    comparison = next(
        row
        for row in report["comparisons"]
        if row["algorithm"] == "candidate"
        and row["metric"] == "passenger_wait_seconds"
    )
    assert evidence["dataset_sha256"] == "dataset-hash"
    assert comparison["mean_difference_algorithm_minus_reference"] == -10.0
    assert comparison["n_common_scenario_cells"] == 2

    csv_path = tmp_path / "flat.csv"
    write_flat_csv(csv_path, rows)
    assert csv_path.is_file()
    assert not (tmp_path / "flat.csv.partial").exists()


def test_exact_sign_flip_and_holm_adjustment():
    assert _paired_permutation_p(np.array([1.0, 1.0])) == 0.5
    np.testing.assert_allclose(_holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
