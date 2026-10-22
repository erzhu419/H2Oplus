from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from reproducibility.build_training_admission_v7 import SCHEMA_VERSION
from reproducibility.build_training_plan_v7 import (
    EXPECTED_EVIDENCE,
    FORMAL_SEEDS,
    FORMAL_SERVER_CPU_THREADS,
    FORMAL_SERVER_GPU_CONCURRENCY,
    HOST_CONCURRENCY_AUDIT_SCHEMA,
    H2OPLUS_CORE_METHODS,
    H2OPLUS_METHODS,
    PLAN_SCHEMA,
    SERVER_CORE_METHODS,
    SERVER_EXECUTION_CLASS,
    SERVER_H2OPLUS_EXECUTION_CLASS,
    SERVER_METHODS,
    SERVER_SENSITIVITY_METHODS,
    _resolve_relative,
    _validate_execution_boundary,
    build_offline_initializer_dependencies,
    build_plan,
    load_admission,
)
from reproducibility.build_training_dispatch_v7 import (
    build_dispatch_payload as build_dispatch_payload_v7,
)
from reproducibility.run_training_matrix_v4 import job_command, load_plan
from SimpleSAC.train_transit_v3 import sha256, simulator_input_hashes, source_hashes
from SimpleSAC.transit_v3.protocols import METHOD_PROFILES
from SimpleSAC.transit_v7.trainer import TransitH2OTrainerV7


def _runtime_lock() -> dict:
    return {
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


def _simulator_tree(tmp_path: Path) -> Path:
    sim_env = tmp_path / "sim"
    line = sim_env / "data" / "7X"
    line.mkdir(parents=True)
    (sim_env / "config.json").write_text('{"time_step": 1}')
    for name in (
        "stop_news.xlsx",
        "route_news.xlsx",
        "time_table.xlsx",
        "passenger_OD.xlsx",
    ):
        (line / name).write_bytes(name.encode())
    (sim_env / "simulator_demand_calibration.json").write_text(
        json.dumps(
            {
                "schema_version": "transit-simulator-demand-calibration-v1",
                "source_manifest_sha256": "manifest",
                "source_bus_objects_sha256": "buses",
                "output_sha256": {},
            }
        )
    )
    profile = sim_env / "observation_speed_profiles_v7_v1.npz"
    profile.write_bytes(b"unit-speed-profile")
    (sim_env / "segment_runtime_calibration_v7_v4.json").write_text(
        json.dumps(
            {
                "schema_version": "transit-simulator-runtime-calibration-v7-v4",
                "coverage_gate": {"passed": True},
                "backward_headway_observation": {
                    "dynamic_speed_profiles": {
                        "schema_version": (
                            "transit-simulator-observation-speed-profiles-v7-v1"
                        ),
                        "artifact": profile.name,
                        "sha256": sha256(profile),
                        "coverage_gate": {"passed": True},
                    }
                },
            }
        )
    )
    return sim_env


def _admission(tmp_path: Path) -> Path:
    admission = tmp_path / "training_admission_v7.json"
    dataset = tmp_path / "formal-v7.h5"
    dataset.write_bytes(b"formal-v7")
    sim_env = _simulator_tree(tmp_path)
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    evidence = {}
    for name in sorted(EXPECTED_EVIDENCE):
        path = evidence_dir / f"{name}.json"
        schema = f"unit-{name}-v1"
        path.write_text(json.dumps({"schema_version": schema}))
        evidence[name] = {
            "path": os.path.relpath(path, admission.parent),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "schema_version": schema,
        }
    simulator = simulator_input_hashes(
        sim_env, runtime_calibration_version="v7_v4"
    )
    builder = (
        Path(__file__).resolve().parents[1]
        / "reproducibility"
        / "build_training_admission_v7.py"
    )
    admission.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "study_stage": "formal",
                "passed": True,
                "dataset": {
                    "path": os.path.relpath(dataset, admission.parent),
                    "sha256": sha256(dataset),
                    "bytes": dataset.stat().st_size,
                },
                "simulator": {
                    "path": os.path.relpath(sim_env, admission.parent),
                    "aggregate_sha256": simulator["aggregate_sha256"],
                    "runtime_sha256": simulator["runtime_sha256"],
                },
                "selected_arguments": {
                    "reward_scale": 1e-4,
                    "critic_loss": "huber",
                    "huber_delta": 1.0,
                    "target_entropy": 0.0,
                    "device_cache_observation_dtype": "float32",
                    "sim_demand_scales": [0.75, 1.0, 1.25],
                },
                "evidence": evidence,
                "training_source_sha256": source_hashes("v7"),
                "training_runtime_lock": _runtime_lock(),
                "builder_sha256": sha256(builder),
            }
        )
    )
    return admission


def _host_concurrency_audits(
    tmp_path: Path, admission: dict
) -> tuple[Path, Path]:
    audit_source = (
        Path(__file__).resolve().parents[1]
        / "reproducibility"
        / "build_host_gpu_concurrency_audit_v7.py"
    )
    benchmark_source = (
        Path(__file__).resolve().parents[1]
        / "reproducibility"
        / "benchmark_training_kernel_v4.py"
    )
    paths = []
    for index, host in enumerate(("jtl110gpu", "jtl110gpu2")):
        path = tmp_path / f"{host}-host-concurrency.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": HOST_CONCURRENCY_AUDIT_SCHEMA,
                    "status": "passed",
                    "passed": True,
                    "host_label": host,
                    "method": "rlpd",
                    "phase": "online",
                    "dataset_sha256": sha256(admission["dataset"]),
                    "device_cache_mode": "float32",
                    "compile_networks": True,
                    "compile_mode": "default",
                    "max_canonical_processes_per_gpu": 1,
                    "canonical_processes_per_host": 2,
                    "cpu_threads_per_process": FORMAL_SERVER_CPU_THREADS,
                    "physical_gpus": ["0", "1"],
                    "aggregate_updates_per_second": 180.0 - 20.0 * index,
                    "aggregate_speedup": 1.9,
                    "minimum_observed_worker_rate_fraction": 0.94,
                    "decision": {"passed": True, "checks": {}},
                    "source_sha256": {
                        "reproducibility/build_host_gpu_concurrency_audit_v7.py": sha256(
                            audit_source
                        ),
                        "reproducibility/benchmark_training_kernel_v4.py": sha256(
                            benchmark_source
                        ),
                    },
                }
            )
        )
        paths.append(path)
    return tuple(paths)


def test_v7_resource_boundary_is_exhaustive_and_disjoint():
    h2oplus = set(H2OPLUS_METHODS)
    baselines = set(SERVER_METHODS)

    assert h2oplus.isdisjoint(baselines)
    assert h2oplus | baselines == set(METHOD_PROFILES)
    assert all(method.startswith("h2oplus") for method in h2oplus)
    assert baselines == {
        "rlpd",
        "wsrl",
        "sac-mixed",
        "sac-online",
        "cql",
        "iql",
        "bc",
        "ratio-sac",
    }


def test_v7_relative_artifact_paths_allow_siblings_but_reject_escape(tmp_path):
    root = tmp_path / "bundle"
    base = root / "reproducibility"
    target = root / "bus_h2o" / "dataset.h5"
    base.mkdir(parents=True)
    target.parent.mkdir()
    target.write_bytes(b"dataset")

    assert _resolve_relative(
        base,
        "../bus_h2o/dataset.h5",
        label="unit artifact",
        allowed_root=root,
    ) == target.resolve()

    with pytest.raises(ValueError, match="escapes the artifact root"):
        _resolve_relative(
            base,
            "../../outside.json",
            label="unit artifact",
            allowed_root=root,
        )


@pytest.mark.parametrize(
    ("methods", "device", "execution_class", "message"),
    (
        (
            ("rlpd",),
            "cuda:0",
            SERVER_H2OPLUS_EXECUTION_CLASS,
            "only H2O\\+ family",
        ),
        (
            ("h2oplus",),
            "cuda:0",
            SERVER_EXECUTION_CLASS,
            "require server-h2oplus",
        ),
        (
            ("h2oplus",),
            "cpu",
            SERVER_H2OPLUS_EXECUTION_CLASS,
            "must use a CUDA device",
        ),
        (
            ("h2oplus",),
            "cuda:0",
            "local-h2oplus",
            "local H2O\\+ execution is disabled",
        ),
        (("rlpd",), "cuda:0", "cpu-validation", "must use the CPU device"),
    ),
)
def test_v7_resource_boundary_rejects_misplaced_methods(
    methods, device, execution_class, message
):
    with pytest.raises(ValueError, match=message):
        _validate_execution_boundary(methods, device, execution_class)


def test_v7_server_plan_freezes_admission_and_resource_boundary(tmp_path):
    admission = load_admission(_admission(tmp_path))
    host_audits = _host_concurrency_audits(tmp_path, admission)
    generator = tmp_path / "generator.py"
    generator.write_text("# frozen generator\n")
    plan_payload = build_plan(
        plan_id="v7-unit-server",
        role="unit server baseline plan",
        methods=("rlpd", "wsrl", "sac-online"),
        seeds=(5101, 5102),
        output_dir=tmp_path,
        output_root=tmp_path / "runs",
        device="cuda:0",
        formal=True,
        created_utc="2026-08-09T00:00:00+00:00",
        generator_snapshot=generator,
        compiler=Path("/bin/true"),
        execution_class=SERVER_EXECUTION_CLASS,
        admission=admission,
        max_canonical_processes_per_gpu_by_method=FORMAL_SERVER_GPU_CONCURRENCY,
        host_concurrency_audits=host_audits,
    )
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(plan_payload))
    jobs, evidence = load_plan(plan)

    assert plan_payload["schema_version"] == PLAN_SCHEMA
    assert plan_payload["mdp_version"] == "v7"
    assert evidence["training_admission"]["sha256"] == sha256(
        admission["path"]
    )
    assert plan_payload["registration"]["checkpoint_version"] == (
        TransitH2OTrainerV7.CHECKPOINT_VERSION
    )
    assert plan_payload["registration"]["compute_policy"]["optimizer"] == {
        "name": "Adam",
        "cuda_backend": "fused",
        "capturable_device_step_counters": True,
    }
    assert plan_payload["registration"]["compute_policy"][
        "training_runtime_lock"
    ] == _runtime_lock()
    if plan_payload["execution_class"] == SERVER_EXECUTION_CLASS:
        assert plan_payload["required_environment"]["LD_PRELOAD"] == (
            "/usr/lib/x86_64-linux-gnu/libcurl.so.4"
        )
    else:
        assert "LD_PRELOAD" not in plan_payload["required_environment"]
    assert plan_payload["registration"]["legacy_snapshot_dataset_required"] is False
    assert plan_payload["registration"]["primary_outcome"] == (
        "accrued generalized passenger seconds divided by departed passengers "
        "over the fixed 24000-second horizon"
    )
    assert plan_payload["registration"]["formal_policy_artifact"] == {
        "schema_version": "transit-policy-artifact-v7-v1",
        "relative_path": "checkpoints/final_actor.pt",
        "selection_rule": (
            "the completed final actor from every registered training seed; "
            "no held-out evaluation outcome may select a checkpoint"
        ),
    }
    evaluation_design = plan_payload["registration"]["evidence"][
        "evaluation_environment_design"
    ]
    assert evaluation_design["sha256"] == sha256(
        Path(__file__).resolve().parents[1]
        / "reproducibility"
        / "evaluation_environment_design_v7_formal_v1.json"
    )
    assert "no formal v7 method reads the legacy snapshot corpus" in (
        plan_payload["registration"]["placement_disclosure"]
    )
    assert plan_payload["registration"]["compute_policy"][
        "canonical_jobs_per_physical_gpu_by_method"
    ] == {"rlpd": 1, "wsrl": 1, "sac-online": 1}
    assert plan_payload["registration"]["compute_policy"][
        "canonical_jobs_per_host"
    ] == 2
    assert plan_payload["registration"]["compute_policy"][
        "cpu_threads_per_canonical_job"
    ] == 6
    assert plan_payload["shared_arguments"]["cpu_threads"] == 6
    host_evidence = plan_payload["registration"]["compute_policy"][
        "host_gpu_concurrency"
    ]["evidence"]
    assert [item["host_label"] for item in host_evidence] == [
        "jtl110gpu",
        "jtl110gpu2",
    ]
    assert plan_payload["registration"]["compute_policy"][
        "runtime_timing_claimable_by_method"
    ] == {"rlpd": False, "wsrl": False, "sac-online": False}
    capacity_gate = plan_payload["registration"]["compute_policy"][
        "same_gpu_capacity_gate"
    ]
    assert capacity_gate["required_for_methods"] == []
    assert capacity_gate["full_registered_dataset_required"] is True
    assert capacity_gate["minimum_free_memory_bytes"] == 2**30
    assert capacity_gate["minimum_warmup_updates_per_process"] == 100
    assert capacity_gate["minimum_measured_updates_per_process"] == 5_000
    assert capacity_gate["single_process_reference_required"] is True
    assert capacity_gate["minimum_aggregate_speedup"] == 1.2
    assert capacity_gate["minimum_worker_rate_fraction"] == 0.55
    assert capacity_gate["protocol"]["sha256"] == sha256(
        Path(__file__).resolve().parents[1]
        / "reproducibility"
        / "audit_gpu_capacity_v7.py"
    )
    assert capacity_gate["benchmark_protocol"]["sha256"] == sha256(
        Path(__file__).resolve().parents[1]
        / "reproducibility"
        / "benchmark_training_kernel_v4.py"
    )

    expected = admission["selected_arguments"]
    for job in jobs:
        arguments = job["arguments"]
        for name, value in expected.items():
            assert arguments[name] == value
        assert job["max_canonical_processes_per_gpu"] == 1


def test_v7_plan_rejects_stale_admission_source_map(tmp_path):
    admission_path = _admission(tmp_path)
    payload = json.loads(admission_path.read_text())
    payload["training_source_sha256"] = {}
    admission_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="source map is stale"):
        load_admission(admission_path)


def test_v7_plan_rejects_unregistered_demand_schedule(tmp_path):
    admission_path = _admission(tmp_path)
    payload = json.loads(admission_path.read_text())
    payload["selected_arguments"]["sim_demand_scales"] = [1.0]
    admission_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="formal demand schedule"):
        load_admission(admission_path)


def test_v7_formal_branches_bind_same_seed_h2oplus_initializer(tmp_path):
    admission = load_admission(_admission(tmp_path))
    host_audits = _host_concurrency_audits(tmp_path, admission)
    generator = tmp_path / "generator.py"
    generator.write_text("# frozen generator\n")
    seeds = (5101, 5102)
    core_payload = build_plan(
        plan_id="v7-unit-server-h2oplus-core",
        role="unit server H2O+ core",
        methods=("h2oplus", "h2oplus-source"),
        seeds=seeds,
        output_dir=tmp_path,
        output_root=tmp_path / "runs" / "core",
        device="cuda:0",
        formal=True,
        created_utc="2026-08-09T00:00:00+00:00",
        generator_snapshot=generator,
        compiler=Path("/bin/true"),
        execution_class=SERVER_H2OPLUS_EXECUTION_CLASS,
        admission=admission,
        host_concurrency_audits=host_audits,
    )
    core_plan = tmp_path / "core-plan.json"
    core_plan.write_text(json.dumps(core_payload))
    dependencies = build_offline_initializer_dependencies(
        core_plan,
        output_dir=tmp_path,
        seeds=seeds,
    )
    branch_payload = build_plan(
        plan_id="v7-unit-server-h2oplus-branches",
        role="unit server H2O+ branches",
        methods=("h2oplus-no-ratio", "h2oplus-lambda0p2"),
        seeds=seeds,
        output_dir=tmp_path,
        output_root=tmp_path / "runs" / "branches",
        device="cuda:0",
        formal=True,
        created_utc="2026-08-09T00:00:00+00:00",
        generator_snapshot=generator,
        compiler=Path("/bin/true"),
        execution_class=SERVER_H2OPLUS_EXECUTION_CLASS,
        admission=admission,
        offline_initializer_dependencies=dependencies,
        host_concurrency_audits=host_audits,
    )
    branch_plan = tmp_path / "branch-plan.json"
    branch_plan.write_text(json.dumps(branch_payload))
    jobs, evidence = load_plan(branch_plan)

    assert len(evidence["offline_initializer_source_plans"]) == 1
    for job in jobs:
        dependency = job["offline_initializer_dependency"]
        assert dependency["source_job_name"] == f"h2oplus_seed{job['seed']}"
        assert job["offline_initializer"].name == "offline_initializer.pt"
        command = job_command(job)
        assert command[command.index("--offline-initializer") + 1] == str(
            job["offline_initializer"]
        )
        assert command[
            command.index("--offline-initializer-source-signature") + 1
        ] == dependency["source_run_signature"]

    branch_payload["jobs"][0]["offline_initializer"][
        "source_run_signature"
    ] = "0" * 64
    branch_plan.write_text(json.dumps(branch_payload))
    with pytest.raises(ValueError, match=r"same-seed H2O\+ source"):
        load_plan(branch_plan)


def _formal_server_plan(tmp_path: Path, methods: tuple[str, ...]) -> Path:
    admission = load_admission(_admission(tmp_path))
    host_audits = _host_concurrency_audits(tmp_path, admission)
    generator = tmp_path / "generator.py"
    generator.write_text("# frozen generator\n")
    payload = build_plan(
        plan_id="v7-unit-dispatch",
        role="unit formal server dispatch",
        methods=methods,
        seeds=FORMAL_SEEDS,
        output_dir=tmp_path,
        output_root=tmp_path / "runs",
        device="cuda:0",
        formal=True,
        created_utc="2026-08-09T00:00:00+00:00",
        generator_snapshot=generator,
        compiler=Path("/bin/true"),
        execution_class=SERVER_EXECUTION_CLASS,
        admission=admission,
        max_canonical_processes_per_gpu_by_method=(
            FORMAL_SERVER_GPU_CONCURRENCY
            if set(methods) == set(SERVER_CORE_METHODS)
            else {}
        ),
        host_concurrency_audits=host_audits,
    )
    plan = tmp_path / "formal-server-plan.json"
    plan.write_text(json.dumps(payload))
    return plan


def test_v7_core_dispatch_uses_all_four_gpus_without_same_gpu_sharing(
    tmp_path, monkeypatch
):
    plan = _formal_server_plan(tmp_path, SERVER_CORE_METHODS)
    payload = build_dispatch_payload_v7(plan, [])
    rlpd_counts = []
    for host in payload["hosts"]:
        for gpu in host["gpus"]:
            stages = [
                stage for stage in gpu["stages"] if stage["name"].startswith("rlpd")
            ]
            assert len(stages) == 1
            assert len(stages[0]["workers"]) == 1
            rlpd_counts.append(len(stages[0]["workers"][0]["jobs"]))

    assert rlpd_counts == [2, 1, 1, 1]
    assert payload["capacity_audit_lanes"] == []
    assert payload["policy"]["capacity_admission"] == (
        "no method in this plan shares a physical GPU"
    )

    import reproducibility.run_training_dispatch_v6 as dispatch_runner

    payload["plan"] = {"path": str(plan), "sha256": sha256(plan)}
    dispatch_path = tmp_path / "dispatch.json"
    dispatch_path.write_text(json.dumps(payload))
    monkeypatch.setattr(
        dispatch_runner,
        "_validate_v7_capacity_audit",
        lambda **kwargs: {"passed": True, "physical_gpu": kwargs["physical_gpu"]},
    )
    normalized, _, _ = dispatch_runner.load_dispatch(dispatch_path)
    admitted_lanes = [
        (host["hostname"], gpu["physical_gpu"])
        for host in normalized["hosts"]
        for gpu in host["gpus"]
        if "capacity_audit" in gpu
    ]
    assert admitted_lanes == []


def test_v7_ratio_sac_dispatch_stays_on_servers_without_shared_gpu(tmp_path):
    plan = _formal_server_plan(tmp_path, SERVER_SENSITIVITY_METHODS)
    payload = build_dispatch_payload_v7(plan, [])
    lane_job_counts = []
    assigned = []
    for host in payload["hosts"]:
        for gpu in host["gpus"]:
            names = [
                name
                for stage in gpu["stages"]
                for worker in stage["workers"]
                for name in worker["jobs"]
            ]
            lane_job_counts.append(len(names))
            assigned.extend(names)

    assert lane_job_counts == [2, 1, 1, 1]
    assert len(assigned) == len(FORMAL_SEEDS)
    assert len(assigned) == len(set(assigned))
    assert payload["capacity_audit_lanes"] == []
    assert payload["policy"]["capacity_admission"] == (
        "no method in this plan shares a physical GPU"
    )


def test_v7_h2oplus_dispatch_uses_all_server_gpus_without_sharing(tmp_path):
    admission = load_admission(_admission(tmp_path))
    host_audits = _host_concurrency_audits(tmp_path, admission)
    generator = tmp_path / "generator.py"
    generator.write_text("# frozen generator\n")
    payload = build_plan(
        plan_id="v7-unit-h2oplus-dispatch",
        role="unit formal server H2O+ dispatch",
        methods=H2OPLUS_CORE_METHODS,
        seeds=FORMAL_SEEDS,
        output_dir=tmp_path,
        output_root=tmp_path / "runs-h2oplus",
        device="cuda:0",
        formal=True,
        created_utc="2026-08-10T00:00:00+00:00",
        generator_snapshot=generator,
        compiler=Path("/bin/true"),
        execution_class=SERVER_H2OPLUS_EXECUTION_CLASS,
        admission=admission,
        host_concurrency_audits=host_audits,
    )
    plan = tmp_path / "formal-server-h2oplus-plan.json"
    plan.write_text(json.dumps(payload))

    dispatch = build_dispatch_payload_v7(plan, [])
    lane_job_counts = []
    assigned = []
    for host in dispatch["hosts"]:
        for gpu in host["gpus"]:
            workers = [
                worker
                for stage in gpu["stages"]
                for worker in stage["workers"]
            ]
            assert len(workers) == 1
            lane_job_counts.append(len(workers[0]["jobs"]))
            assigned.extend(workers[0]["jobs"])

    assert lane_job_counts == [3, 3, 2, 2]
    assert len(assigned) == len(H2OPLUS_CORE_METHODS) * len(FORMAL_SEEDS)
    assert len(assigned) == len(set(assigned))
    assert dispatch["capacity_audit_lanes"] == []
