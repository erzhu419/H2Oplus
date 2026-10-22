from __future__ import annotations

import copy
from pathlib import Path

import pytest

from reproducibility.compose_collection_manifests_v7 import compose_payloads
from reproducibility.freeze_behavior_collection_matrix_v7 import (
    validate_formal_behavior_manifest,
)
from reproducibility.run_collection_matrix_v7 import MATRIX_SCHEMA, POLICIES


def _component(tmp_path: Path, policy_indices: tuple[int, ...]) -> dict:
    jobs = []
    for policy_index in policy_indices:
        policy = POLICIES[policy_index]
        for block_id, demand_scale in enumerate((0.75, 1.0, 1.25)):
            jobs.append(
                {
                    "block_id": block_id,
                    "realization_block_id": 0,
                    "policy": policy,
                    "policy_seed": 100 + policy_index * 10 + block_id,
                    "sumo_seed": 11,
                    "split": "train",
                    "diagnostic": False,
                    "episode_seconds": 24000,
                    "max_events": 0,
                    "scenario": {
                        "sha256": f"scenario-{block_id}",
                        "selection_seed": 21,
                        "demand_scale": demand_scale,
                    },
                    "output": str(
                        tmp_path / policy["policy_name"] / f"{block_id}.npz"
                    ),
                    "signature": f"signature-{policy_index}-{block_id}",
                    "source_sha256": {"source": "hash"},
                    "sumo_version": "SUMO test",
                }
            )
    selected = [POLICIES[index]["policy_name"] for index in policy_indices]
    return {
        "schema_version": MATRIX_SCHEMA,
        "study_stage": "formal",
        "environment_design": {"demand_scales": [0.75, 1.0, 1.25]},
        "independent_realization_blocks": 1,
        "environment_cells": 3,
        "behavior_mixture": {
            "policies": list(POLICIES),
            "selected_policy_names": selected,
        },
        "collector": "collector.py",
        "collector_sha256": "collector",
        "runner_sha256": "runner",
        "source_sha256": {"source": "hash"},
        "sumo_version": "SUMO test",
        "jobs": jobs,
    }


def test_compose_v7_manifests_keeps_cross_root_paths(tmp_path):
    zero = _component(tmp_path / "zero-root", (0,))
    behavior = _component(tmp_path / "behavior-root", tuple(range(1, 8)))
    payload = compose_payloads(
        [(zero, {"path": "zero"}), (behavior, {"path": "behavior"})]
    )

    assert payload["independent_realization_blocks"] == 1
    assert payload["environment_cells"] == 3
    assert len(payload["jobs"]) == 24
    assert payload["behavior_mixture"]["selected_policy_names"] == [
        policy["policy_name"] for policy in POLICIES
    ]
    roots = {Path(job["output"]).parents[1].name for job in payload["jobs"]}
    assert roots == {"zero-root", "behavior-root"}


def test_compose_v7_rejects_partial_realization_selection(tmp_path):
    zero = _component(tmp_path / "zero-root", (0,))
    behavior = _component(tmp_path / "behavior-root", tuple(range(1, 8)))
    with pytest.raises(ValueError, match="complete demand-scale realizations"):
        compose_payloads(
            [(zero, {"path": "zero"}), (behavior, {"path": "behavior"})],
            block_ids={0, 1},
        )


def test_compose_v7_rejects_environment_disagreement(tmp_path):
    zero = _component(tmp_path / "zero-root", (0,))
    behavior = _component(tmp_path / "behavior-root", tuple(range(1, 8)))
    behavior = copy.deepcopy(behavior)
    behavior["jobs"][0]["scenario"]["sha256"] = "different"
    with pytest.raises(ValueError, match="disagree"):
        compose_payloads(
            [(zero, {"path": "zero"}), (behavior, {"path": "behavior"})]
        )


def test_behavior_freeze_topology_requires_all_nonzero_policies(tmp_path):
    behavior = _component(tmp_path / "behavior-root", tuple(range(1, 8)))
    selected, topology = validate_formal_behavior_manifest(behavior)
    assert len(selected["jobs"]) == 21
    assert topology["policies"] == [policy["policy_name"] for policy in POLICIES[1:]]

    missing_policy = _component(tmp_path / "incomplete", tuple(range(1, 7)))
    with pytest.raises(ValueError, match="seven nonzero policies"):
        validate_formal_behavior_manifest(missing_policy)
