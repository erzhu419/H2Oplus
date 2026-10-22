import json

from reproducibility import run_h2oplus_v8_stage_a_dispatch as dispatch


def test_verify_design_returns_json_serializable_evidence(tmp_path, monkeypatch):
    manifests = []
    scales = (0.85, 1.0, 1.15)
    for block_index in range(5):
        block_manifests = []
        for scale_index, scale in enumerate(scales):
            path = tmp_path / f"block-{block_index}-scale-{scale_index}.json"
            path.write_text("{}\n")
            manifests.append(
                (
                    path.resolve(),
                    {
                        "demand_scale": scale,
                        "selection_seed": 4201 + block_index,
                    },
                )
            )
            block_manifests.append(path.name)

    evidence_by_path = dict(manifests)
    monkeypatch.setattr(
        dispatch,
        "load_validated_scenario",
        lambda path: (path, evidence_by_path[path]),
    )
    design_path = tmp_path / "design.json"
    design = {
        "status": "frozen-before-stage-A-outcomes",
        "study_stage": "development",
        "role": "candidate screen; not formal held-out evaluation",
        "formal_held_out_outcomes_read": False,
        "disjointness": {
            "verified_disjoint": True,
            "formal_held_out_outcomes_read": False,
            "held_out_demand_seed_overlap": [],
            "held_out_manifest_hash_overlap": [],
            "held_out_runtime_hash_overlap": [],
            "held_out_sumo_seed_overlap": [],
            "training_demand_seed_overlap": [],
            "training_manifest_hash_overlap": [],
            "training_runtime_hash_overlap": [],
            "training_sumo_seed_overlap": [],
        },
        "environment_cells": 15,
        "blocks": [
            {
                "name": f"block-{index}",
                "sumo_seed": 4101 + index,
                "demand_realization_seed": 4201 + index,
                "scenario_manifests": [
                    f"block-{index}-scale-{scale_index}.json"
                    for scale_index in range(3)
                ],
            }
            for index in range(5)
        ],
    }

    descriptors = dispatch.verify_design(design_path, design)

    assert all(
        isinstance(path, str)
        for block in descriptors
        for path in block["manifests"]
    )
    json.dumps({"blocks": descriptors})
