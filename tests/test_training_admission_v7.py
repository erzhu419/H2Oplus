import json

import h5py
import pytest

from bus_h2o.mdp_contract_v7 import SCHEMA_VERSION
from bus_h2o.sumo_provenance import sha256
from reproducibility.build_training_admission_v7 import (
    _validate_entropy_execution,
    _validate_reward_loss_execution,
    portable_simulator_identity,
    validated_training_runtime_lock,
    validated_pilot_dataset,
    validate_pilot_subset,
)
from reproducibility.run_hybrid_entropy_pilot_v7 import (
    MINIMUM_OFFLINE_UPDATES,
    MINIMUM_ONLINE_ROWS,
    MINIMUM_ONLINE_UPDATES,
    PILOT_SEEDS as ENTROPY_SEEDS,
    TARGET_ENTROPY_CANDIDATES,
    entropy_target_gates,
)
from reproducibility.run_reward_loss_pilot_v7 import (
    CRITIC_LOSSES,
    DEFAULT_REWARD_SCALES,
    MINIMUM_UPDATES,
    PILOT_SEEDS as REWARD_SEEDS,
    configuration_gates,
)


def _dataset(path, signatures, *, split="train"):
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["collection_matrix"] = json.dumps(
            {"jobs": len(signatures)}
        )
        handle.attrs["source_shards"] = json.dumps(
            [
                {"job_signature": signature}
                for signature in signatures
            ]
        )
        handle.attrs["split_names"] = json.dumps(
            {"train": 0, "validation": 1, "test": 2}
        )
        split_id = {"train": 0, "validation": 1, "test": 2}[split]
        handle.create_dataset("split_id", data=[split_id] * len(signatures))


def test_training_admission_requires_pilot_collection_to_be_strict_subset(tmp_path):
    pilot = tmp_path / "pilot.h5"
    formal = tmp_path / "formal.h5"
    shared = [f"{value:064x}" for value in range(1, 4)]
    _dataset(pilot, shared)
    _dataset(formal, [*shared, f"{4:064x}"])

    report = validate_pilot_subset(pilot, formal)

    assert report == {
        "passed": True,
        "pilot_jobs": 3,
        "formal_jobs": 4,
        "shared_jobs": 3,
        "pilot_is_strict_subset": True,
        "job_signature_source": "HDF5 source_shards[].job_signature",
        "split_source": "HDF5 split_id mapped by split_names",
    }


def test_training_admission_rejects_equal_or_nontraining_pilot(tmp_path):
    pilot = tmp_path / "pilot.h5"
    formal = tmp_path / "formal.h5"
    signatures = [f"{value:064x}" for value in range(1, 3)]
    _dataset(pilot, signatures)
    _dataset(formal, signatures)
    with pytest.raises(ValueError, match="strict subset"):
        validate_pilot_subset(pilot, formal)

    _dataset(pilot, [signatures[0]], split="validation")
    with pytest.raises(ValueError, match="non-training split"):
        validate_pilot_subset(pilot, formal)


def test_training_admission_rejects_collection_count_or_signature_drift(tmp_path):
    pilot = tmp_path / "pilot.h5"
    formal = tmp_path / "formal.h5"
    signatures = [f"{value:064x}" for value in range(1, 4)]
    _dataset(pilot, signatures[:2])
    _dataset(formal, signatures)

    with h5py.File(pilot, "r+") as handle:
        handle.attrs["collection_matrix"] = json.dumps({"jobs": 3})
    with pytest.raises(ValueError, match="collection job count"):
        validate_pilot_subset(pilot, formal)

    _dataset(pilot, signatures[:2])
    with h5py.File(pilot, "r+") as handle:
        handle.attrs["source_shards"] = json.dumps(
            [{"job_signature": signatures[0]}, {"job_signature": "invalid"}]
        )
    with pytest.raises(ValueError, match="invalid collection job signatures"):
        validate_pilot_subset(pilot, formal)


def test_training_admission_accepts_relocated_same_hash_pilot(tmp_path):
    pilot = tmp_path / "pilot.h5"
    pilot.write_bytes(b"portable pilot artifact")
    reward_loss = {
        "dataset": {
            "path": "/remote/deployment/that/is/not/locally/mounted/pilot.h5",
            "sha256": sha256(pilot),
        }
    }

    assert validated_pilot_dataset(pilot, reward_loss) == pilot.resolve()

    reward_loss["dataset"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="missing or stale"):
        validated_pilot_dataset(pilot, reward_loss)


def _simulator_inputs(path: str) -> dict:
    return {
        "schema_version": "transit-simulator-inputs-v1",
        "path": path,
        "line_ids": ["A", "B"],
        "runtime_files_sha256": {
            "config.json": "1" * 64,
            "data/A.xlsx": "2" * 64,
        },
        "runtime_sha256": "3" * 64,
        "calibration_report": {
            "path": f"{path}/calibration.json",
            "sha256": "4" * 64,
            "source_manifest_sha256": "5" * 64,
            "source_bus_objects_sha256": "6" * 64,
        },
        "aggregate_sha256": "7" * 64,
    }


def test_training_admission_simulator_identity_is_relocatable_but_hash_strict():
    local = _simulator_inputs("/local/simulator")
    remote = _simulator_inputs("/remote/deployment/simulator")

    assert portable_simulator_identity(local) == portable_simulator_identity(remote)

    remote["runtime_files_sha256"]["config.json"] = "8" * 64
    assert portable_simulator_identity(local) != portable_simulator_identity(remote)

    remote["runtime_files_sha256"]["config.json"] = "invalid"
    with pytest.raises(ValueError, match="runtime file hashes"):
        portable_simulator_identity(remote)


def _reward_loss_payload():
    runs = [
        {
            "reward_scale": scale,
            "critic_loss": loss,
            "seed": seed,
            "updates": MINIMUM_UPDATES,
            "trainer_config": {
                "reward_scale": scale,
                "critic_loss": loss,
            },
            "numerical_gate": {"eligible": True},
        }
        for scale in DEFAULT_REWARD_SCALES
        for loss in CRITIC_LOSSES
        for seed in REWARD_SEEDS
    ]
    return {
        "study_stage": "numerical-pilot",
        "design": {
            "reward_scales": list(DEFAULT_REWARD_SCALES),
            "critic_losses": list(CRITIC_LOSSES),
            "seeds": list(REWARD_SEEDS),
            "updates_per_run": MINIMUM_UPDATES,
            "minimum_updates_per_run": MINIMUM_UPDATES,
        },
        "runs": runs,
        "paired_initialization_checks": {
            f"seed{seed}/{module}": True
            for seed in REWARD_SEEDS
            for module in ("actor", "critic", "target_critic", "value", "ratio")
        },
        "configuration_gates": configuration_gates(runs),
    }


def _entropy_payload():
    runs = [
        {
            "target_entropy": target,
            "seed": seed,
            "offline_updates": MINIMUM_OFFLINE_UPDATES,
            "online_updates": MINIMUM_ONLINE_UPDATES,
            "trainer_config": {"target_entropy": target},
            "numerical_gate": {"eligible": True},
        }
        for target in TARGET_ENTROPY_CANDIDATES
        for seed in ENTROPY_SEEDS
    ]
    paired = {
        f"seed{seed}/{module}": True
        for seed in ENTROPY_SEEDS
        for module in ("actor", "critic", "target_critic", "value", "ratio")
    }
    return {
        "study_stage": "numerical-pilot",
        "design": {
            "target_entropy_candidates": list(TARGET_ENTROPY_CANDIDATES),
            "seeds": list(ENTROPY_SEEDS),
            "offline_updates": MINIMUM_OFFLINE_UPDATES,
            "online_updates": MINIMUM_ONLINE_UPDATES,
            "online_rows": MINIMUM_ONLINE_ROWS,
            "minimum_offline_updates": MINIMUM_OFFLINE_UPDATES,
            "minimum_online_updates": MINIMUM_ONLINE_UPDATES,
            "minimum_online_rows": MINIMUM_ONLINE_ROWS,
        },
        "runs": runs,
        "paired_initialization_checks": dict(paired),
        "paired_online_start_checks": dict(paired),
        "target_entropy_gate": entropy_target_gates(runs),
    }


def test_training_admission_recomputes_full_numerical_pilot_gates():
    reward = _reward_loss_payload()
    entropy = _entropy_payload()
    _validate_reward_loss_execution(reward)
    _validate_entropy_execution(entropy)

    reward["design"]["updates_per_run"] = MINIMUM_UPDATES - 1
    with pytest.raises(ValueError, match="minimum horizon"):
        _validate_reward_loss_execution(reward)

    entropy["paired_online_start_checks"][
        f"seed{ENTROPY_SEEDS[0]}/actor"
    ] = False
    with pytest.raises(ValueError, match="not fully paired"):
        _validate_entropy_execution(entropy)


def test_training_admission_requires_cryptographic_gpu_runtime_lock():
    required = {
        "python_version": "3.8.20",
        "python_implementation": "CPython",
        "python_cache_tag": "cpython-38",
        "python_executable_sha256": "1" * 64,
        "packages": {"torch": "2.4.1"},
        "torch_cuda_version": "12.4",
        "torch_binary_sha256": {
            "python_extension": "2" * 64,
            "cuda_library": "3" * 64,
        },
        "compiler": {"sha256": "4" * 64, "version": "gcc test"},
    }
    precision = {
        "runtime_fingerprint": {
            "required": required,
            "host": {"device": "cuda:0", "gpu": "unit GPU"},
        }
    }
    assert validated_training_runtime_lock(precision) == required

    precision["runtime_fingerprint"]["required"][
        "python_executable_sha256"
    ] = "invalid"
    with pytest.raises(ValueError, match="invalid identities"):
        validated_training_runtime_lock(precision)
