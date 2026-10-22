import copy
from pathlib import Path

import numpy as np

from reproducibility import analyze_h2oplus_v8_stage_a as stage_a
from reproducibility.analyze_h2oplus_v8_stage_a import (
    CANDIDATE,
    EXPECTED_BLOCKS,
    EXPECTED_SCALES,
    REFERENCE,
    analyze_gate,
    exact_one_sided_sign_flip_p,
    load_complete_rows,
)


def _registration_and_freeze():
    gate = {
        "completion_rate_noninferiority_margin": 0.005,
        "mean_relative_improvement_at_least": 0.02,
        "no_demand_scale_mean_relative_degradation_greater_than": 0.01,
        "one_sided_paired_permutation_p_less_than": 0.05,
        "training_seeds_with_lower_paired_mean_at_least": 4,
    }
    registration = {
        "status": "registered-not-submitted",
        "execution_authorized": False,
        "scientific_scope": {"supports_current_held_out_model_effect_claim": False},
        "stages": [
            {
                "name": "A-existing-policy-development-screen",
                "paired_unit": "training seed x development environment block",
                "advance_gate": gate,
            }
        ],
        "development_environment": {
            "blocks": [
                {
                    "name": block,
                    "sumo_seed": 4101 + index,
                    "demand_realization_seed": 4201 + index,
                }
                for index, block in enumerate(EXPECTED_BLOCKS)
            ]
        },
    }
    freeze = {
        "status": "frozen-before-stage-A-outcomes",
        "formal_held_out_outcomes_read": False,
        "training": False,
        "advance_gate": copy.deepcopy(gate),
    }
    return registration, freeze


def _rows(candidate_multiplier=0.95, candidate_completion=0.998):
    rows = []
    for algorithm in (REFERENCE, CANDIDATE):
        for seed in range(5101, 5106):
            for block_index, block in enumerate(EXPECTED_BLOCKS):
                for scale in EXPECTED_SCALES:
                    baseline = 100.0 + block_index + scale
                    welfare = (
                        baseline
                        if algorithm == REFERENCE
                        else baseline * candidate_multiplier
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
                                    1.0
                                    if algorithm == REFERENCE
                                    else candidate_completion
                                ),
                            },
                            "output": "synthetic",
                        }
                    )
    return rows


def test_exact_one_sided_sign_flip():
    assert exact_one_sided_sign_flip_p(np.asarray([1.0, 1.0])) == 0.25
    assert exact_one_sided_sign_flip_p(np.ones(25)) == 1.0 / (2**25)


def test_stage_a_gate_passes_all_registered_checks():
    registration, freeze = _registration_and_freeze()
    report = analyze_gate(_rows(), registration, freeze)

    assert report["stage_a_gate_passed"] is True
    assert report["stage_b_gate_eligible"] is True
    assert report["stage_b_authorized_or_submitted"] is False
    assert len(report["paired_units"]) == 25
    assert all(check["passed"] for check in report["checks"].values())


def test_stage_a_gate_fails_closed_on_primary_and_completion():
    registration, freeze = _registration_and_freeze()
    report = analyze_gate(
        _rows(candidate_multiplier=1.02, candidate_completion=0.99),
        registration,
        freeze,
    )

    assert report["stage_a_gate_passed"] is False
    assert report["checks"]["mean_relative_improvement"]["passed"] is False
    assert report["checks"]["completion_rate_noninferiority"]["passed"] is False
    assert report["checks"]["no_demand_scale_relative_degradation"]["passed"] is False


def test_load_complete_rows_maps_registered_sumo_seed_blocks(
    monkeypatch, tmp_path: Path
):
    registration, _ = _registration_and_freeze()
    roots = [tmp_path / block for block in EXPECTED_BLOCKS]
    rows = _rows()

    def fake_load_rows(root):
        block_index = EXPECTED_BLOCKS.index(root.name)
        sumo_seed = 4101 + block_index
        block_rows = [
            copy.deepcopy(row) for row in rows if row["sumo_seed"] == sumo_seed
        ]
        for row in block_rows:
            row["environment_block"] = f"sumo-seed-{sumo_seed}"
        return block_rows, {
            "mdp_version": "v7",
            "dataset_sha256": "dataset",
            "contract": {"schema": "contract"},
            "source_sha256": {"evaluator": "source"},
        }

    monkeypatch.setattr(stage_a, "load_rows", fake_load_rows)

    loaded, evidence = load_complete_rows(roots, registration)

    assert len(loaded) == 150
    assert len(evidence) == 5
    assert sorted({row["environment_block"] for row in loaded}) == EXPECTED_BLOCKS
