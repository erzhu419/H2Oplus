import copy

import pytest

from reproducibility.analyze_h2oplus_v8_stage_a2 import (
    COMPARATORS,
    EXPECTED_BLOCKS,
    EXPECTED_SCALES,
    _evaluation_identity,
    analyze,
    holm_adjust,
)


def _registration_and_freeze():
    gate = {
        "directional_mean_improvement_greater_than": 0.0,
        "mean_relative_improvement_at_least": 0.02,
        "training_seeds_with_lower_paired_mean_at_least": 4,
        "holm_adjusted_one_sided_p_less_than": 0.05,
        "completion_rate_noninferiority_margin": 0.005,
        "no_demand_scale_mean_relative_degradation_greater_than": 0.01,
    }
    registration = {
        "status": "registered-not-submitted",
        "execution_authorized": False,
        "formal_held_out_outcomes_read": False,
        "comparison_gate": gate,
    }
    freeze = {
        "status": "frozen-before-stage-A2-SAC-outcomes",
        "training": False,
        "formal_held_out_outcomes_read": False,
        "comparison_gate": copy.deepcopy(gate),
    }
    return registration, freeze


def _rows(h2oplus_multiplier=0.95, completion=0.999):
    rows = []
    for algorithm in ("h2oplus", *COMPARATORS):
        for seed in range(5101, 5106):
            for block_index, block in enumerate(EXPECTED_BLOCKS):
                for scale in EXPECTED_SCALES:
                    baseline = 100.0 + block_index + scale
                    welfare = (
                        baseline * h2oplus_multiplier
                        if algorithm == "h2oplus"
                        else baseline
                    )
                    rows.append(
                        {
                            "algorithm": algorithm,
                            "method": f"{algorithm}-seed{seed}",
                            "training_seed": seed,
                            "replicate": f"seed-{seed}",
                            "scenario": f"{block}-scale-{scale}",
                            "demand_scale": scale,
                            "sumo_seed": 4101 + block_index,
                            "demand_realization_seed": 4201 + block_index,
                            "environment_block": block,
                            "cell": (f"{block}-scale-{scale}", 4101 + block_index),
                            "metrics": {
                                "generalized_seconds_per_passenger": welfare,
                                "passenger_completion_rate": (
                                    completion if algorithm == "h2oplus" else 1.0
                                ),
                            },
                            "output": "synthetic",
                        }
                    )
    return rows


def test_holm_adjusts_exact_two_comparison_family():
    adjusted = holm_adjust({"sac-mixed": 0.01, "sac-online": 0.04})

    assert adjusted["sac-mixed"]["holm_adjusted_p"] == 0.02
    assert adjusted["sac-online"]["holm_adjusted_p"] == 0.04
    assert all(item["passed"] for item in adjusted.values())


def test_evaluation_identity_ignores_only_the_deployment_root():
    common = {
        "mdp_version": "v7",
        "dataset_sha256": "a" * 64,
        "contract": {"horizon_seconds": 24000.0},
    }
    stage_a = {
        **common,
        "source_sha256": {
            "/deployment/stage-a/workspace/H2Oplus/SimpleSAC/evaluate_transit_v7.py": "b"
            * 64
        },
    }
    stage_a2 = {
        **common,
        "source_sha256": {
            "/deployment/stage-a2/workspace/H2Oplus/SimpleSAC/evaluate_transit_v7.py": "b"
            * 64
        },
    }

    assert _evaluation_identity(stage_a) == _evaluation_identity(stage_a2)

    stage_a2["source_sha256"] = {
        "/deployment/stage-a2/workspace/H2Oplus/SimpleSAC/evaluate_transit_v7.py": "c"
        * 64
    }
    assert _evaluation_identity(stage_a) != _evaluation_identity(stage_a2)

    stage_a2["source_sha256"] = {"relative.py": "b" * 64}
    with pytest.raises(ValueError, match="deployment-portable"):
        _evaluation_identity(stage_a2)


def test_stage_a2_confirms_both_sac_comparisons_when_all_gates_pass():
    registration, freeze = _registration_and_freeze()

    report = analyze(_rows(), registration, freeze)

    assert report["all_SAC_comparisons_confirmed"] is True
    assert report["stage_b_authorized_or_submitted"] is False
    for comparator in COMPARATORS:
        comparison = report["comparisons"][comparator]
        assert comparison["h2oplus_superiority_confirmed"] is True
        assert len(comparison["paired_units"]) == 25
        assert all(item["passed"] for item in comparison["checks"].values())


def test_stage_a2_fails_closed_when_h2oplus_is_worse():
    registration, freeze = _registration_and_freeze()

    report = analyze(_rows(h2oplus_multiplier=1.02), registration, freeze)

    assert report["all_SAC_comparisons_confirmed"] is False
    for comparator in COMPARATORS:
        checks = report["comparisons"][comparator]["checks"]
        assert checks["directional_mean_superiority"]["passed"] is False
        assert checks["practical_mean_improvement"]["passed"] is False
