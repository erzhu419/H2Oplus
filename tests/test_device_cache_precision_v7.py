import json

import h5py
import numpy as np
import pytest

from bus_h2o.sumo_provenance import sha256
from reproducibility.audit_device_cache_precision_v6 import (
    MINIMUM_PAIRED_UPDATES,
    cache_precision_decision,
    full_float16_storage_error,
    validate_v7_selection_evidence,
)
from reproducibility.run_hybrid_entropy_pilot_v7 import (
    SCHEMA_VERSION as ENTROPY_PILOT_SCHEMA,
    entropy_pilot_source_hashes,
)
from reproducibility.run_reward_loss_pilot_v7 import (
    SCHEMA_VERSION as REWARD_LOSS_PILOT_SCHEMA,
    reward_loss_source_hashes,
)


def _stats(maximum=1e-4, nrmse=1e-4):
    return {
        "maximum_absolute_error": maximum,
        "normalized_root_mean_square_error": nrmse,
    }


def test_precision_decision_admits_float16_only_after_full_paired_audit():
    decision = cache_precision_decision(
        input_error={"observations": _stats(), "next_observations": _stats()},
        representation_error={
            "actor_encoded_state": _stats(),
            "deterministic_action_from_quantized_input": _stats(),
        },
        parameter_drift={"actor": _stats(), "critic": _stats()},
        policy_drift=_stats(),
        paired_updates=MINIMUM_PAIRED_UPDATES,
    )

    assert decision["float16_eligible"] is True
    assert decision["selected_observation_storage_dtype"] == "float16"


def test_precision_decision_falls_back_to_float32_on_any_failed_bound():
    decision = cache_precision_decision(
        input_error={"observations": _stats(maximum=0.01)},
        representation_error={
            "actor_encoded_state": _stats(),
            "deterministic_action_from_quantized_input": _stats(),
        },
        parameter_drift={"actor": _stats()},
        policy_drift=_stats(),
        paired_updates=MINIMUM_PAIRED_UPDATES,
    )

    assert (
        decision["checks"]["full_and_sampled_input_maximum_absolute_error"]
        is False
    )
    assert decision["float16_eligible"] is False
    assert decision["selected_observation_storage_dtype"] == "float32"


def test_full_float16_scan_covers_every_training_state(tmp_path):
    dataset = tmp_path / "states.h5"
    observations = np.asarray(
        [[0.0, 0.1], [0.3333, 11.0], [0.7777, -0.2]], dtype=np.float32
    )
    next_observations = observations + np.float32(0.0123)
    with h5py.File(dataset, "w") as handle:
        handle.create_dataset("split_id", data=np.asarray([0, 1, 0], dtype=np.int8))
        handle.create_dataset("observations", data=observations)
        handle.create_dataset("next_observations", data=next_observations)

    report = full_float16_storage_error(dataset, chunk_rows=1)

    selected = observations[[0, 2]]
    expected = np.max(np.abs(selected.astype(np.float16).astype(np.float32) - selected))
    assert report["training_rows"] == 2
    assert report["arrays"]["observations"]["elements"] == 4
    assert np.isclose(
        report["arrays"]["observations"]["maximum_absolute_error"], expected
    )


def test_v7_precision_evidence_binds_exact_upstream_artifacts(tmp_path):
    reward_loss = tmp_path / "reward-loss.json"
    reward_loss.write_text(
        json.dumps(
            {
                "schema_version": REWARD_LOSS_PILOT_SCHEMA,
                "study_stage": "numerical-pilot",
                "configuration_selection": {
                    "passed": True,
                    "selected": {
                        "reward_scale": 1e-4,
                        "critic_loss": "huber",
                        "huber_delta": 1.0,
                    },
                },
                "source_sha256": reward_loss_source_hashes(),
            }
        )
    )
    entropy = tmp_path / "entropy.json"
    entropy_payload = {
        "schema_version": ENTROPY_PILOT_SCHEMA,
        "study_stage": "numerical-pilot",
        "reward_loss_configuration": {
            "reward_scale": 1e-4,
            "critic_loss": "huber",
            "huber_delta": 1.0,
            "pilot_sha256": sha256(reward_loss),
        },
        "target_entropy_gate": {
            "passed": True,
            "selected_target_entropy": 0.0,
        },
        "source_sha256": entropy_pilot_source_hashes(),
    }
    entropy.write_text(json.dumps(entropy_payload))

    evidence = validate_v7_selection_evidence(
        reward_loss,
        entropy,
        reward_scale=1e-4,
        critic_loss="huber",
        huber_delta=1.0,
        target_entropy=0.0,
    )

    assert evidence["reward_loss_pilot"]["sha256"] == sha256(reward_loss)
    assert evidence["entropy_pilot"]["sha256"] == sha256(entropy)

    entropy_payload["reward_loss_configuration"]["pilot_sha256"] = "0" * 64
    entropy.write_text(json.dumps(entropy_payload))
    with pytest.raises(ValueError, match="current selected"):
        validate_v7_selection_evidence(
            reward_loss,
            entropy,
            reward_scale=1e-4,
            critic_loss="huber",
            huber_delta=1.0,
            target_entropy=0.0,
        )
