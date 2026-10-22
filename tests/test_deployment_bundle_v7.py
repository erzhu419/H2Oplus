import json
from pathlib import Path

import pytest

import reproducibility.deployment_bundle_v7 as deployment_bundle_v7
from bus_h2o.sumo_provenance import sha256
from reproducibility.deployment_bundle_v7 import (
    _exclude_tree,
    _host_concurrency_artifacts,
    _offline_initializer_source_plans,
    _payload_policy,
    _python_sources,
    _registered_calibrated_npz,
    _safe_relative,
    _validate_plan_concurrency,
)


def _entry(path: str, size: int = 1):
    return {"path": path, "type": "file", "bytes": size, "sha256": "0" * 64}


def test_v7_deployment_source_inventory_excludes_generated_deployments():
    paths = list(_python_sources())
    assert paths
    assert len(paths) == len(set(paths))
    assert all(path.is_file() for path in paths)
    assert not any("deployments" in path.parts for path in paths)
    assert not any(
        part in deployment_bundle_v7.LEGACY_DATA_PARTS
        for path in paths
        for part in path.parts
    )
    assert not any(path.suffix in {".h5", ".npz", ".pt", ".pkl"} for path in paths)


def test_v7_deployment_payload_policy_allows_only_v7_hdf5_data():
    policy = _payload_policy(
        [
            _entry("SimpleSAC/snapshot_store.py"),
            _entry("bus_h2o/datasets_v7/formal/transit_smdp_v7.h5", 100),
            _entry("reproducibility/evidence.json"),
        ]
    )
    assert policy["legacy_dataset_files"] == 0
    assert policy["snapshot_or_checkpoint_payload_files"] == 0
    assert policy["v7_hdf5_files"] == [
        "bus_h2o/datasets_v7/formal/transit_smdp_v7.h5"
    ]


def test_v7_deployment_allows_only_registered_calibrated_npz():
    path = "bus_h2o/calibrated_env/observation_speed_profiles_v7_v1.npz"
    simulator = {
        "runtime_files_sha256": {
            "observation_speed_profiles_v7_v1.npz": "0" * 64,
            "simulator_journeys_v1.json": "1" * 64,
        }
    }
    allowed = _registered_calibrated_npz(simulator)

    policy = _payload_policy([_entry(path)], allowed_calibrated_npz=allowed)
    assert policy["registered_calibrated_npz_files"] == [path]

    with pytest.raises(ValueError, match="forbidden"):
        _payload_policy([_entry(path)])


@pytest.mark.parametrize(
    "path",
    (
        "bus_h2o/datasets_v6/transit_smdp_v6.h5",
        "bus_h2o/snapshots/block.pkl",
        "experiment_output/model_final.pt",
        "bus_h2o/datasets_v7/formal/shard.npz",
    ),
)
def test_v7_deployment_payload_policy_rejects_legacy_or_snapshot_data(path):
    with pytest.raises(ValueError, match="legacy/snapshot/checkpoint"):
        _payload_policy([_entry(path)])


def test_v7_deployment_authenticates_offline_initializer_source_plan(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(deployment_bundle_v7, "ROOT", tmp_path)
    source = tmp_path / "plans" / "core.json"
    source.parent.mkdir()
    source.write_text(json.dumps({"plan_id": "core"}))
    branch = source.parent / "branch.json"
    payload = {
        "jobs": [
            {
                "name": "h2oplus-no-ratio_seed5101",
                "offline_initializer": {
                    "source_plan": source.name,
                    "source_plan_sha256": sha256(source),
                },
            }
        ]
    }

    assert _offline_initializer_source_plans(branch, payload) == {
        source.resolve()
    }

    payload["jobs"][0]["offline_initializer"]["source_plan_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source plan is stale"):
        _offline_initializer_source_plans(branch, payload)


def test_v7_deployment_relative_paths_allow_workspace_siblings_only(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(deployment_bundle_v7, "ROOT", tmp_path)
    base = tmp_path / "reproducibility" / "plans"
    target = tmp_path / "bus_h2o" / "dataset.h5"
    base.mkdir(parents=True)
    target.parent.mkdir()
    target.write_bytes(b"dataset")

    assert _safe_relative(
        base,
        "../../bus_h2o/dataset.h5",
        label="unit artifact",
    ) == target.resolve()

    with pytest.raises(ValueError, match="escapes the H2Oplus workspace"):
        _safe_relative(
            base,
            "../../../outside.json",
            label="unit artifact",
        )


def test_v7_deployment_concurrency_distinguishes_gate_and_formal_rlpd():
    gate = {
        "registration": {
            "status": "execution-gate",
            "compute_policy": {
                "canonical_jobs_per_physical_gpu_by_method": {"rlpd": 1}
            },
        }
    }
    formal = {
        "registration": {
            "status": "frozen-before-formal-outcomes",
            "compute_policy": {
                "canonical_jobs_per_physical_gpu_by_method": {"rlpd": 1}
            },
        }
    }

    _validate_plan_concurrency(
        gate,
        [
            {
                "name": "rlpd_gate",
                "method": "rlpd",
                "max_canonical_processes_per_gpu": 1,
            }
        ],
        {"rlpd"},
    )
    _validate_plan_concurrency(
        formal,
        [
                {
                    "name": "rlpd_formal",
                    "method": "rlpd",
                    "max_canonical_processes_per_gpu": 1,
                }
        ],
        {"rlpd"},
    )

    with pytest.raises(ValueError, match="concurrency registration drifted"):
        _validate_plan_concurrency(gate, [], {"rlpd", "wsrl"})

    with pytest.raises(ValueError, match="GPU quota drifted"):
        _validate_plan_concurrency(
            formal,
            [
                {
                    "name": "rlpd_formal",
                    "method": "rlpd",
                    "max_canonical_processes_per_gpu": 2,
                }
            ],
            {"rlpd"},
        )


def test_v7_deployment_excludes_files_copied_with_calibrated_tree(tmp_path):
    calibrated = tmp_path / "bus_h2o" / "calibrated_env"
    evidence = calibrated / "fidelity.json"
    sibling = tmp_path / "bus_h2o" / "dataset.h5"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("evidence")
    sibling.write_text("dataset")

    assert _exclude_tree([evidence, sibling], calibrated) == [sibling]


def test_v7_deployment_closes_host_concurrency_raw_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(deployment_bundle_v7, "ROOT", tmp_path)
    reports = tmp_path / "evidence" / "reports"
    audits = tmp_path / "evidence" / "audits"
    plans = tmp_path / "plans"
    reports.mkdir(parents=True)
    audits.mkdir(parents=True)
    plans.mkdir()
    reference = reports / "reference.json"
    worker0 = reports / "worker0.json"
    worker1 = reports / "worker1.json"
    for path, text in (
        (reference, "reference"),
        (worker0, "worker0"),
        (worker1, "worker1"),
    ):
        path.write_text(text)
    audit = audits / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "single_process_reference": {
                    "path": "../reports/reference.json",
                    "sha256": sha256(reference),
                    "bytes": reference.stat().st_size,
                },
                "workers": [
                    {
                        "path": "../reports/worker0.json",
                        "sha256": sha256(worker0),
                        "bytes": worker0.stat().st_size,
                    },
                    {
                        "path": "../reports/worker1.json",
                        "sha256": sha256(worker1),
                        "bytes": worker1.stat().st_size,
                    },
                ],
            }
        )
    )
    plan = plans / "plan.json"
    payload = {
        "registration": {
            "compute_policy": {
                "host_gpu_concurrency": {
                    "evidence": [
                        {
                            "path": "../evidence/audits/audit.json",
                            "sha256": sha256(audit),
                            "bytes": audit.stat().st_size,
                        }
                    ]
                }
            }
        }
    }

    assert _host_concurrency_artifacts(plan, payload) == {
        audit.resolve(),
        reference.resolve(),
        worker0.resolve(),
        worker1.resolve(),
    }
