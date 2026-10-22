from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SimpleSAC.transit_v11.trainer import (
    ACTOR_HOLD_CRITIC_UPDATES,
    ACTOR_RELEASE_EVENTS,
    TransitH2OActorHoldTrainerV11,
    V11_TRAINER_VARIANTS,
    actor_hold_is_active,
)
from reproducibility.prepare_h2oplus_v11_actor_hold import _source_map
from reproducibility.build_h2oplus_v11_actor_hold_closeout_v2 import (
    audit_history,
    event_cap_evidence,
    held_state_hashes_identical,
)
from reproducibility.submit_h2oplus_v11_actor_hold import (
    require_remote_run_root_absent,
)


def test_launcher_uses_actor_hold_schemas_and_candidate_only_stage() -> None:
    source = (ROOT / "SimpleSAC/train_transit_v11_actor_hold.py").read_text()
    launcher = ast.parse(source)
    schema_names = {
        "PROTOCOL_SCHEMA",
        "ADMISSION_SCHEMA",
        "POLICY_MANIFEST_SCHEMA",
        "EXPORT_SUMMARY_SCHEMA",
    }
    assignments = {
        target.id: ast.literal_eval(node.value)
        for node in launcher.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in schema_names
    }
    assert assignments == {
        "PROTOCOL_SCHEMA": "transit-h2oplus-v11-actor-hold-protocol-v2",
        "ADMISSION_SCHEMA": (
            "transit-training-admission-v11-actor-hold-development-v2"
        ),
        "POLICY_MANIFEST_SCHEMA": (
            "transit-h2oplus-v11-actor-hold-policy-manifest-v2"
        ),
        "EXPORT_SUMMARY_SCHEMA": (
            "transit-h2oplus-v11-actor-hold-export-summary-v2"
        ),
    }
    builder = ast.parse(
        (ROOT / "reproducibility/prepare_h2oplus_v11_actor_hold.py").read_text()
    )
    builder_assignments = {
        target.id: ast.literal_eval(node.value)
        for node in builder.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id in {"PROTOCOL_SCHEMA", "ADMISSION_SCHEMA"}
    }
    assert builder_assignments == {
        key: assignments[key] for key in ("PROTOCOL_SCHEMA", "ADMISSION_SCHEMA")
    }
    assert 'STAGE != "candidate"' in source
    assert "def _materialize_branch" not in source
    assert "def _create_prefix_handoff" not in source


def test_actor_hold_releases_strictly_after_10k() -> None:
    assert ACTOR_RELEASE_EVENTS == 10_000
    assert ACTOR_HOLD_CRITIC_UPDATES == 5_000
    assert actor_hold_is_active(0)
    assert actor_hold_is_active(9_999)
    assert actor_hold_is_active(10_000)
    assert not actor_hold_is_active(10_001)


def test_only_registered_v11_variant_is_actor_hold_online_only() -> None:
    assert V11_TRAINER_VARIANTS == {
        "actor-hold-online-only": TransitH2OActorHoldTrainerV11
    }


def test_held_actor_schedule_advances_counter_without_retaining_states() -> None:
    trainer = object.__new__(TransitH2OActorHoldTrainerV11)
    trainer.replay_buffer = SimpleNamespace(online_size=10_050)
    trainer.config = SimpleNamespace(
        online_actor_batch_mode="latest",
        online_actor_update_interval=1,
    )
    trainer.online_critic_updates = 4_999
    trainer._online_actor_state_batches = [torch.tensor([[-1.0]])]
    trainer._online_temperature_state_batches = [torch.tensor([[-2.0]])]
    actor_states = torch.tensor([[1.0, 2.0]])
    temperature_states = torch.tensor([[3.0, 4.0]])

    scheduled = trainer._scheduled_online_actor_states(
        actor_states, temperature_states
    )

    assert scheduled is None
    assert trainer.online_critic_updates == 5_000
    assert trainer._online_actor_state_batches == []
    assert trainer._online_temperature_state_batches == []

    scheduled = trainer._scheduled_online_actor_states(
        actor_states, temperature_states
    )

    assert scheduled is not None
    scheduled_actor, scheduled_temperature = scheduled
    assert trainer.online_critic_updates == 5_001
    assert torch.equal(scheduled_actor, actor_states)
    assert torch.equal(scheduled_temperature, temperature_states)


def test_actor_state_source_switches_only_after_release() -> None:
    trainer = object.__new__(TransitH2OActorHoldTrainerV11)
    trainer.replay_buffer = SimpleNamespace(online_size=10_050)
    trainer.config = SimpleNamespace(temperature_state_source="actor")
    trainer.online_critic_updates = 4_999
    offline = {
        "observations": torch.tensor([[1.0, 2.0]]),
        "next_observations": torch.tensor([[1.5, 2.5]]),
    }
    online = {
        "observations": torch.tensor([[3.0, 4.0]]),
        "next_observations": torch.tensor([[3.5, 4.5]]),
    }

    held_states = trainer._actor_states(offline, online)
    held_temperature_states = trainer._temperature_states(offline, online)
    trainer.online_critic_updates = 5_000
    released_states = trainer._actor_states(offline, online)
    released_temperature_states = trainer._temperature_states(offline, online)

    assert torch.equal(
        held_states,
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    assert torch.equal(released_states, online["observations"])
    assert torch.equal(held_temperature_states, held_states)
    assert torch.equal(released_temperature_states, online["observations"])


def test_hold_prevents_actor_and_temperature_optimizer_call() -> None:
    trainer = object.__new__(TransitH2OActorHoldTrainerV11)
    trainer.replay_buffer = SimpleNamespace(online_size=10_000)
    trainer.config = SimpleNamespace(
        online_actor_batch_mode="latest",
        online_actor_update_interval=1,
        temperature_state_source="actor",
        offline_objective="sac",
    )
    trainer.device = torch.device("cpu")
    trainer.online_critic_updates = 0
    trainer._online_actor_state_batches = []
    trainer._online_temperature_state_batches = []
    trainer._last_actor_diagnostics = {}
    trainer._update_sac_actor = Mock(
        return_value=(torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0))
    )
    offline = {
        "observations": torch.tensor([[1.0, 2.0]]),
        "next_observations": torch.tensor([[1.5, 2.5]]),
    }
    online = {
        "observations": torch.tensor([[3.0, 4.0]]),
        "next_observations": torch.tensor([[3.5, 4.5]]),
    }

    result = trainer._update_actor_for_batch(
        online=True,
        offline_batch=offline,
        online_batch=online,
        iql_actor_weights=None,
        collect_metrics=True,
    )

    trainer._update_sac_actor.assert_not_called()
    assert result[3] == 0.0


def test_runtime_and_deployment_source_contracts_include_v10_dependencies() -> None:
    expected = {
        "SimpleSAC/train_transit_v11_actor_hold.py",
        "SimpleSAC/transit_v10/__init__.py",
        "SimpleSAC/transit_v10/provenance.py",
        "SimpleSAC/transit_v10/trainer.py",
        "SimpleSAC/transit_v11/__init__.py",
        "SimpleSAC/transit_v11/provenance.py",
        "SimpleSAC/transit_v11/trainer.py",
    }
    assert set(_source_map()) == expected
    launcher = (ROOT / "SimpleSAC/train_transit_v11_actor_hold.py").read_text()
    verifier = (
        ROOT / "reproducibility/verify_h2oplus_v11_actor_hold_deployment.py"
    ).read_text()
    for path in expected:
        assert path in launcher or path == "SimpleSAC/train_transit_v11_actor_hold.py"
        assert path in verifier


def test_v11_intervention_is_single_factor_and_fail_closed() -> None:
    trainer = object.__new__(TransitH2OActorHoldTrainerV11)
    intervention = trainer.v11_intervention

    assert intervention["variant"] == "actor-hold-online-only"
    assert intervention["actor_and_temperature_hold_through_online_events"] == 10_000
    assert intervention["actor_and_temperature_hold_online_critic_updates"] == 5_000
    assert intervention["actor_state_source_after_release"] == "online-only"
    assert intervention["offline_replay_schedule"] == [[0, 0.5], [25_000, 0.5]]
    assert intervention["online_update_order"] == "v7-actor-before-critic"
    assert intervention["unit_ratio_weights"] is True
    assert intervention["checkpoint_selection_by_behavior"] is False


def test_remote_absence_check_distinguishes_existing_root_from_transport_error() -> None:
    require_remote_run_root_absent(
        node="jtl110gpu",
        run_name="candidate",
        returncode=0,
        stdout="",
        stderr="",
    )
    with pytest.raises(FileExistsError):
        require_remote_run_root_absent(
            node="jtl110gpu",
            run_name="candidate",
            returncode=1,
            stdout="",
            stderr="",
        )
    with pytest.raises(ConnectionError, match="could not be verified"):
        require_remote_run_root_absent(
            node="jtl110gpu",
            run_name="candidate",
            returncode=255,
            stdout="",
            stderr="Connection timed out during banner exchange",
        )


def test_training_submission_batches_absence_checks_and_bounds_transport_retries() -> None:
    source = (
        ROOT / "reproducibility/submit_h2oplus_v11_actor_hold.py"
    ).read_text()
    assert "jobs_by_node" in source
    assert "for attempt in range(1, 4):" in source
    assert "if rc in {0, 1}:" in source


def test_training_sync_separates_declared_resources_from_live_calibration() -> None:
    source = (
        ROOT / "reproducibility/sync_h2oplus_v11_actor_hold_training_v2.py"
    ).read_text()
    assert '"declared_ram_mb": 32_768' in source
    assert '"declared_vram_mb": 8_192' in source
    assert '"live_ram_estimate_mb": task.get("ram_mb")' in source


def _closeout_row(epoch: int, hold_probability: float = 0.5) -> dict:
    held = epoch <= 10
    online_updates = epoch * 500
    train = {
        "train/critic_grad_norm": 10.0,
        "train/online_critic_update": float(online_updates),
        "train/update": float(50_000 + online_updates),
        "train/actor_updated": float(not held),
        "train/actor_grad_norm": 0.0 if held else 1.0,
        "train/actor_loss": 0.0 if held else -1.0,
        "train/alpha": 1.0 if held else 0.9,
        "v11/actor_hold_active": float(held),
        "v11/online_only_actor_active": float(not held),
        "v11/actor_hold_critic_updates": 5_000.0,
        "v11/actor_release_events": 10_000.0,
        "v11/unit_ratio_weights": 1.0,
        "ratio/v10_unit_weights": 1.0,
        "v10/unit_ratio_weights": 1.0,
        "v11/offline_replay_fraction": 0.5,
        "v10/offline_replay_fraction": 0.5,
        "v10/online_replay_fraction": 0.5,
        "v10/delayed_replay_active": 0.0,
        "v10/delayed_update_order_active": 0.0,
        "v10/delayed_actor_states_active": float(not held),
    }
    if not held:
        train["train/hold_probability"] = hold_probability
        train["train/positive_action_mean"] = 0.0
    cap = 5_000 + online_updates
    return {
        "epoch": epoch,
        "train": train,
        "actual_cumulative_simulator_decision_events": cap,
        "cumulative_simulator_decision_event_cap": cap,
        "rollout": {
            "maximum_simultaneous_decisions": 4,
            "blocked_simultaneous_batch_size": 0,
        },
    }


def test_closeout_history_separates_hold_and_released_actor_stability() -> None:
    rows = [_closeout_row(epoch) for epoch in range(1, 41)]

    result = audit_history(rows)

    assert result["passed"] is True
    assert result["critic_metric_rows"] == 40
    assert result["released_actor_metric_rows"] == 30
    assert result["violations"] == []

    rows[10] = _closeout_row(11, hold_probability=0.0)
    result = audit_history(rows)
    assert result["passed"] is False
    assert result["violations"] == [
        {"epoch": 11, "metric": "hold_probability", "value": 0.0}
    ]


def test_closeout_requires_identical_zero_5k_10k_actor_and_temperature() -> None:
    entries = [
        {
            "online_events_nominal": event,
            "actor_tensor_sha256": "actor-held" if event <= 10_000 else "actor-final",
            "temperature_tensor_sha256": (
                "temperature-held" if event <= 10_000 else "temperature-final"
            ),
        }
        for event in (0, 5_000, 10_000, 25_000)
    ]
    assert held_state_hashes_identical(entries)

    entries[2]["actor_tensor_sha256"] = "actor-drifted"
    assert not held_state_hashes_identical(entries)


def test_closeout_preserves_bounded_atomic_event_deviation() -> None:
    rows = [_closeout_row(epoch) for epoch in range(1, 41)]
    phase_metrics = {
        "warmup": {
            "events": 5_002,
            "maximum_simultaneous_decisions": 6,
            "hard_event_cap_reached": False,
            "blocked_simultaneous_batch_size": 0,
        }
    }

    evidence = event_cap_evidence(rows, phase_metrics, 5_000)

    assert evidence == {
        "actual": 5_002,
        "deviation": 2,
        "maximum_simultaneous_decisions": 6,
        "hard_event_cap_reached": False,
        "blocked_simultaneous_batch_size": 0,
    }


def test_closeout_accepts_only_evidenced_final_atomic_underfill() -> None:
    rows = [_closeout_row(epoch) for epoch in range(1, 41)]
    rows[-1]["actual_cumulative_simulator_decision_events"] = 24_999
    rows[-1]["rollout"]["hard_event_cap_reached"] = True
    rows[-1]["rollout"]["blocked_simultaneous_batch_size"] = 3
    phase_metrics = {
        "warmup": {
            "events": 5_000,
            "maximum_simultaneous_decisions": 4,
        }
    }

    evidence = event_cap_evidence(rows, phase_metrics, 25_000)
    assert evidence["actual"] == 24_999
    assert evidence["deviation"] == -1

    rows[-1]["rollout"]["hard_event_cap_reached"] = False
    with pytest.raises(ValueError, match="underfill"):
        event_cap_evidence(rows, phase_metrics, 25_000)
