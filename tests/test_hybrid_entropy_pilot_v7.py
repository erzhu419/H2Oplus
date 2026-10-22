import json

import h5py
import pytest

from bus_h2o.mdp_contract_v7 import SCHEMA_VERSION as DATASET_SCHEMA
from bus_h2o.sumo_provenance import sha256
from reproducibility.audit_transit_v7_dataset import (
    AUDIT_SCHEMA as DATASET_AUDIT_SCHEMA,
    dataset_audit_source_hashes,
)
from reproducibility.run_hybrid_entropy_pilot_v7 import (
    PILOT_SEEDS,
    TARGET_ENTROPY_CANDIDATES,
    _load_inputs,
    entropy_target_gates,
)
from reproducibility.run_reward_loss_pilot_v7 import (
    SCHEMA_VERSION as REWARD_LOSS_SCHEMA,
    reward_loss_source_hashes,
)


def _runs():
    return [
        {
            "target_entropy": target,
            "seed": seed,
            "numerical_gate": {"eligible": True},
        }
        for target in TARGET_ENTROPY_CANDIDATES
        for seed in PILOT_SEEDS
    ]


def test_entropy_target_gate_prefers_zero_when_all_candidates_pass():
    result = entropy_target_gates(_runs())

    assert result["passed"] is True
    assert result["selected_target_entropy"] == 0.0


def test_entropy_target_gate_requires_all_seeds_and_uses_registered_fallback():
    runs = _runs()
    next(
        row
        for row in runs
        if row["target_entropy"] == 0.0 and row["seed"] == PILOT_SEEDS[0]
    )["numerical_gate"]["eligible"] = False

    result = entropy_target_gates(runs)

    assert result["gates"]["0"]["eligible"] is False
    assert result["selected_target_entropy"] == -0.5


def test_entropy_gate_authenticates_relocated_dataset_by_sha256(tmp_path):
    dataset = tmp_path / "relocated" / "pilot.h5"
    dataset.parent.mkdir()
    with h5py.File(dataset, "w") as handle:
        handle.attrs["schema_version"] = DATASET_SCHEMA
    digest = sha256(dataset)
    dataset_audit = tmp_path / "dataset.audit.json"
    dataset_audit.write_text(
        json.dumps(
            {
                "schema_version": DATASET_AUDIT_SCHEMA,
                "passed": True,
                "dataset": "/original/host/pilot.h5",
                "dataset_sha256": digest,
                "source_sha256": dataset_audit_source_hashes(),
            }
        )
    )
    reward_loss = tmp_path / "reward_loss.json"
    reward_loss.write_text(
        json.dumps(
            {
                "schema_version": REWARD_LOSS_SCHEMA,
                "dataset": {"sha256": digest},
                "source_sha256": reward_loss_source_hashes(),
                "configuration_gates": {
                    "reward_scale=0.0001/critic_loss=huber": {
                        "eligible": True
                    }
                },
                "configuration_selection": {
                    "selected": {"reward_scale": 1e-4, "critic_loss": "huber"}
                },
            }
        )
    )

    loaded_dataset_audit, _ = _load_inputs(
        dataset.resolve(),
        dataset_audit,
        reward_loss,
        reward_scale=1e-4,
        critic_loss="huber",
    )
    assert loaded_dataset_audit["dataset"] == "/original/host/pilot.h5"

    payload = json.loads(dataset_audit.read_text())
    payload["dataset_sha256"] = "0" * 64
    dataset_audit.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="matching passed audit"):
        _load_inputs(
            dataset.resolve(),
            dataset_audit,
            reward_loss,
            reward_scale=1e-4,
            critic_loss="huber",
        )
