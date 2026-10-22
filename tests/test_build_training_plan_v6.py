import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from reproducibility.build_training_dispatch_v6 import build_dispatch_payload
from reproducibility.build_training_plan_v6 import (
    ABLATION_METHODS,
    CORE_METHODS,
    FORMAL_EVENTS_PER_EPOCH,
    FORMAL_ONLINE_EVENTS,
    FORMAL_SEEDS,
    FORMAL_SERVER_GPU_CONCURRENCY,
    FIDELITY_AUDIT,
    GATE_EVENTS_PER_EPOCH,
    GATE_ONLINE_EVENTS,
    LOCAL_CORE_METHODS,
    LOCAL_H2OPLUS_METHODS,
    LOCAL_SENSITIVITY_METHODS,
    SERVER_CORE_METHODS,
    SERVER_NO_SNAPSHOT_METHODS,
    SERVER_SENSITIVITY_METHODS,
    SENSITIVITY_METHODS,
    build_jobs,
    build_plan,
    validate_fidelity_audit,
)
from SimpleSAC.transit_v3.protocols import METHOD_PROFILES
from SimpleSAC.transit_v6.trainer import TransitH2OTrainerV6


def test_dispatch_builder_runs_directly_outside_repository_cwd(tmp_path):
    script = (
        Path(__file__).resolve().parents[1]
        / "reproducibility"
        / "build_training_dispatch_v6.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--plan" in completed.stdout and "--output" in completed.stdout


def test_formal_jobs_close_equal_online_interaction_budgets():
    methods = CORE_METHODS + ABLATION_METHODS + SENSITIVITY_METHODS
    jobs, budgets = build_jobs(
        methods,
        FORMAL_SEEDS,
        formal=True,
        total_online_events=FORMAL_ONLINE_EVENTS,
        events_per_epoch=FORMAL_EVENTS_PER_EPOCH,
    )
    assert len(jobs) == len(methods) * len(FORMAL_SEEDS)
    assert len({job["name"] for job in jobs}) == len(jobs)
    for method in methods:
        profile = METHOD_PROFILES[method]
        budget = budgets[method]
        expected_events = (
            FORMAL_ONLINE_EVENTS if profile.online_update_mode != "none" else 0
        )
        assert budget["total_simulator_events"] == expected_events
        assert budget["simulator_decision_event_cap"] == expected_events
        assert budget["online_critic_updates"] == (
            budget["training_simulator_events"] * profile.utd_ratio
        )
        assert budget["online_actor_updates"] == (
            budget["online_critic_updates"] // profile.actor_update_interval
        )


def test_gate_covers_every_profile_and_preserves_update_structure():
    methods = tuple(METHOD_PROFILES)
    jobs, budgets = build_jobs(
        methods,
        (9151,),
        formal=False,
        total_online_events=GATE_ONLINE_EVENTS,
        events_per_epoch=GATE_EVENTS_PER_EPOCH,
    )
    assert {job["method"] for job in jobs} == set(METHOD_PROFILES)
    for job in jobs:
        method = job["method"]
        profile = METHOD_PROFILES[method]
        expected_events = GATE_ONLINE_EVENTS if profile.online_update_mode != "none" else 0
        assert budgets[method]["total_simulator_events"] == expected_events
        assert budgets[method]["simulator_decision_event_cap"] == expected_events
        assert job["arguments"]["utd_ratio"] == profile.utd_ratio
        assert job["arguments"]["batch_size"] == profile.batch_size


def test_local_and_server_execution_classes_are_disjoint_and_complete():
    assert set(LOCAL_H2OPLUS_METHODS).isdisjoint(SERVER_NO_SNAPSHOT_METHODS)
    assert set(LOCAL_H2OPLUS_METHODS) | set(SERVER_NO_SNAPSHOT_METHODS) == set(
        METHOD_PROFILES
    )
    assert set(LOCAL_CORE_METHODS).isdisjoint(SERVER_CORE_METHODS)
    assert set(LOCAL_CORE_METHODS) | set(SERVER_CORE_METHODS) == set(CORE_METHODS)
    assert set(LOCAL_SENSITIVITY_METHODS).isdisjoint(
        SERVER_SENSITIVITY_METHODS
    )
    assert set(LOCAL_SENSITIVITY_METHODS) | set(
        SERVER_SENSITIVITY_METHODS
    ) == set(SENSITIVITY_METHODS)
    assert SERVER_SENSITIVITY_METHODS == ("ratio-sac",)


def test_formal_server_plan_limits_only_rlpd_to_two_processes_per_gpu(tmp_path):
    generator = tmp_path / "generator.py"
    generator.write_text("# frozen generator\n")
    compiler = tmp_path / "compiler"
    compiler.write_text("compiler identity\n")
    plan = build_plan(
        plan_id="unit-formal-server",
        role="unit test",
        methods=("rlpd", "wsrl", "sac-online"),
        seeds=(5101, 5102),
        output_dir=tmp_path,
        output_root=tmp_path / "runs",
        device="cuda:0",
        formal=True,
        created_utc="2026-08-08T00:00:00+00:00",
        generator_snapshot=generator,
        compiler=compiler,
        execution_class="server-no-snapshot",
        allow_shared_gpu=False,
        allow_external_gpu_processes=False,
        max_canonical_processes_per_gpu_by_method=FORMAL_SERVER_GPU_CONCURRENCY,
    )
    quotas = {
        job["method"]: job["max_canonical_processes_per_gpu"]
        for job in plan["jobs"]
    }
    assert quotas == {"rlpd": 2, "wsrl": 1, "sac-online": 1}
    compute = plan["registration"]["compute_policy"]
    assert plan["registration"]["checkpoint_version"] == (
        TransitH2OTrainerV6.CHECKPOINT_VERSION
    )
    assert compute["shared_gpu_allowed_methods"] == ["rlpd"]
    assert compute["canonical_jobs_per_physical_gpu"] == "method-specific"
    assert compute["canonical_jobs_per_physical_gpu_by_method"] == quotas
    assert compute["runtime_timing_claimable_by_method"] == {
        "rlpd": False,
        "wsrl": True,
        "sac-online": True,
    }
    assert compute["runtime_timing_claimable"] is False


def test_formal_server_dispatch_assigns_every_job_once(tmp_path):
    generator = tmp_path / "generator.py"
    generator.write_text("# frozen generator\n")
    compiler = tmp_path / "compiler"
    compiler.write_text("#!/bin/sh\nexit 0\n")
    compiler.chmod(0o755)
    plan_payload = build_plan(
        plan_id="unit-formal-server-dispatch",
        role="unit test dispatch",
        methods=SERVER_CORE_METHODS,
        seeds=FORMAL_SEEDS,
        output_dir=tmp_path,
        output_root=tmp_path / "runs",
        device="cuda:0",
        formal=True,
        created_utc="2026-08-08T00:00:00+00:00",
        generator_snapshot=generator,
        compiler=compiler,
        execution_class="server-no-snapshot",
        allow_shared_gpu=False,
        allow_external_gpu_processes=False,
        max_canonical_processes_per_gpu_by_method=FORMAL_SERVER_GPU_CONCURRENCY,
    )
    plan = tmp_path / "formal-server.json"
    plan.write_text(json.dumps(plan_payload))

    dispatch = build_dispatch_payload(plan)
    assert {
        host["hostname"]: host["hostname_aliases"] for host in dispatch["hosts"]
    } == {
        "jtl110gpu": ["huiwei-Super-Server"],
        "jtl110gpu2": ["erzhu419-Super-Server"],
    }
    assigned = [
        job
        for host in dispatch["hosts"]
        for gpu in host["gpus"]
        for stage in gpu["stages"]
        for worker in stage["workers"]
        for job in worker["jobs"]
    ]
    assert len(assigned) == len(SERVER_CORE_METHODS) * len(FORMAL_SEEDS)
    assert len(assigned) == len(set(assigned))
    shared_stages = [
        stage
        for host in dispatch["hosts"]
        for gpu in host["gpus"]
        for stage in gpu["stages"]
        if len(stage["workers"]) == 2
    ]
    assert len(shared_stages) == 2
    assert all(stage["name"] == "rlpd-shared" for stage in shared_stages)


def _validate_recorded_fidelity(audit):
    validate_fidelity_audit(
        audit,
        current_simulator_inputs=audit["simulator_inputs"],
        current_source_sha256=audit["source_sha256"],
        current_calibration_sha256=audit["demand_calibration"]["sha256"],
    )


def test_formal_fidelity_audit_is_complete_and_locked():
    audit = json.loads(FIDELITY_AUDIT.read_text())
    _validate_recorded_fidelity(audit)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda audit: audit.update(passed=False), "has not passed"),
        (
            lambda audit: audit["design"].update(bootstrap_samples=999),
            "design no longer matches",
        ),
        (
            lambda audit: audit["simulator_episodes"].pop(),
            "episode cells are incomplete",
        ),
        (
            lambda audit: audit["strata"]["1"]["comparison"]["checks"].update(
                wait_time=False
            ),
            "checks failed",
        ),
    ),
)
def test_formal_fidelity_lock_rejects_partial_or_failed_evidence(mutation, message):
    audit = copy.deepcopy(json.loads(FIDELITY_AUDIT.read_text()))
    mutation(audit)
    with pytest.raises(RuntimeError, match=message):
        _validate_recorded_fidelity(audit)
