from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from bus_h2o.mdp_contract_v7 import SCHEMA_VERSION
from bus_h2o.sumo_provenance import sha256
from reproducibility.audit_reward_conditioning_v7 import (
    AUDIT_SCHEMA,
    audit,
    reward_audit_source_hashes,
)
from reproducibility.audit_transit_v7_dataset import (
    AUDIT_SCHEMA as DATASET_AUDIT_SCHEMA,
    dataset_audit_source_hashes,
)
from reproducibility.run_reward_loss_pilot_v7 import (
    CRITIC_LOSSES,
    DEFAULT_REWARD_SCALES,
    PILOT_SEEDS,
    _load_gate_inputs,
    configuration_gates,
    select_reward_loss_configuration,
)


def test_reward_conditioning_reports_terminal_loss_contributions(tmp_path):
    dataset = tmp_path / "v7.h5"
    with h5py.File(dataset, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["content_signature"] = "content"
        handle.attrs["policy_names"] = json.dumps({"zero": 0, "random": 1})
        handle.create_dataset("rewards", data=[0.0, -3.0, -4.0, -5.0])
        handle.create_dataset("terminals", data=[0, 1, 0, 1])
        handle.create_dataset("episode_id", data=[0, 0, 1, 1])
        handle.create_dataset("policy_id", data=[0, 0, 1, 1])
        handle.create_dataset("demand_scale", data=[0.75, 0.75, 1.25, 1.25])
        handle.create_dataset("delta_t_seconds", data=[0.0, 1.0, 2.0, 3.0])

    report = audit(dataset, reward_scales=[1.0], huber_deltas=[1.0])

    assert report["rows"] == 4
    assert report["episodes"] == 2
    assert report["groups"]["by_policy"]["zero"]["episodes"] == 1
    proxy = report["reward_scale_candidates"]["1"][
        "reward_only_initial_td_proxy"
    ]
    assert np.isclose(proxy["mse"]["terminal_fraction"], 34.0 / 50.0)
    assert np.isclose(
        proxy["huber"]["delta_1"]["terminal_fraction"], 7.0 / 10.5
    )


def test_reward_loss_configuration_requires_every_registered_seed():
    runs = [
        {
            "reward_scale": scale,
            "critic_loss": loss,
            "seed": seed,
            "numerical_gate": {"eligible": True},
        }
        for scale in DEFAULT_REWARD_SCALES
        for loss in CRITIC_LOSSES
        for seed in PILOT_SEEDS
    ]
    failing = next(
        run
        for run in runs
        if run["reward_scale"] == DEFAULT_REWARD_SCALES[0]
        and run["critic_loss"] == CRITIC_LOSSES[0]
        and run["seed"] == PILOT_SEEDS[0]
    )
    failing["numerical_gate"]["eligible"] = False

    gates = configuration_gates(runs)

    failed_key = (
        f"reward_scale={DEFAULT_REWARD_SCALES[0]:.12g}/"
        f"critic_loss={CRITIC_LOSSES[0]}"
    )
    passed_key = (
        f"reward_scale={DEFAULT_REWARD_SCALES[-1]:.12g}/"
        f"critic_loss={CRITIC_LOSSES[-1]}"
    )
    assert gates[failed_key]["complete"] is True
    assert gates[failed_key]["eligible"] is False
    assert gates[passed_key]["eligible"] is True


def test_reward_loss_selection_uses_conditioning_then_prefers_mse():
    runs = [
        {
            "reward_scale": scale,
            "critic_loss": loss,
            "seed": seed,
            "numerical_gate": {"eligible": True},
        }
        for scale in DEFAULT_REWARD_SCALES
        for loss in CRITIC_LOSSES
        for seed in PILOT_SEEDS
    ]
    gates = configuration_gates(runs)
    reward_audit = {"reward_scale_candidates": {}}
    for index, scale in enumerate(DEFAULT_REWARD_SCALES):
        q95 = 10.0 ** (index - 2)
        reward_audit["reward_scale_candidates"][f"{scale:.12g}"] = {
            "absolute_scaled_reward": {
                "maximum": q95 * 4.0,
                "quantiles": {"q0.95": q95, "q0.99": q95 * 2.0},
            },
            "reward_only_initial_td_proxy": {
                "mse": {"terminal_fraction": 0.1},
                "huber": {"delta_1": {"terminal_fraction": 0.1}},
            },
        }

    selection = select_reward_loss_configuration(reward_audit, gates)

    assert selection["passed"] is True
    assert selection["selected"] == {
        "reward_scale": DEFAULT_REWARD_SCALES[2],
        "critic_loss": "mse",
        "huber_delta": 1.0,
    }


def test_reward_loss_selection_rejects_terminal_dominated_mse():
    runs = [
        {
            "reward_scale": scale,
            "critic_loss": loss,
            "seed": seed,
            "numerical_gate": {"eligible": True},
        }
        for scale in DEFAULT_REWARD_SCALES
        for loss in CRITIC_LOSSES
        for seed in PILOT_SEEDS
    ]
    gates = configuration_gates(runs)
    reward_audit = {"reward_scale_candidates": {}}
    for scale in DEFAULT_REWARD_SCALES:
        q95 = scale / 1e-4
        reward_audit["reward_scale_candidates"][f"{scale:.12g}"] = {
            "absolute_scaled_reward": {
                "maximum": q95 * 800.0,
                "quantiles": {"q0.95": q95, "q0.99": q95 * 2.0},
            },
            "reward_only_initial_td_proxy": {
                "mse": {"terminal_fraction": 0.98},
                "huber": {"delta_1": {"terminal_fraction": 0.2}},
            },
        }

    selection = select_reward_loss_configuration(reward_audit, gates)

    assert selection["selected"] == {
        "reward_scale": 1e-4,
        "critic_loss": "huber",
        "huber_delta": 1.0,
    }


def test_reward_loss_gate_authenticates_relocated_dataset_by_sha256(tmp_path):
    dataset = tmp_path / "relocated" / "pilot.h5"
    dataset.parent.mkdir()
    with h5py.File(dataset, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
    digest = sha256(dataset)
    dataset_audit = tmp_path / "dataset.audit.json"
    dataset_audit.write_text(
        json.dumps(
            {
                "schema_version": DATASET_AUDIT_SCHEMA,
                "passed": True,
                "dataset": str(Path("/original/host/pilot.h5")),
                "dataset_sha256": digest,
                "source_sha256": dataset_audit_source_hashes(),
            }
        )
    )
    reward_audit = tmp_path / "reward.json"
    reward_audit.write_text(
        json.dumps(
            {
                "schema_version": AUDIT_SCHEMA,
                "dataset_sha256": digest,
                "source_sha256": reward_audit_source_hashes(),
                "reward_scale_candidates": {
                    str(index): {"reward_scale": scale}
                    for index, scale in enumerate(DEFAULT_REWARD_SCALES)
                },
            }
        )
    )

    loaded_dataset_audit, _ = _load_gate_inputs(
        dataset.resolve(), dataset_audit, reward_audit
    )
    assert loaded_dataset_audit["dataset"] == "/original/host/pilot.h5"

    loaded = json.loads(dataset_audit.read_text())
    loaded["dataset_sha256"] = "0" * 64
    dataset_audit.write_text(json.dumps(loaded))
    with pytest.raises(ValueError, match="matching passed audit"):
        _load_gate_inputs(dataset.resolve(), dataset_audit, reward_audit)
