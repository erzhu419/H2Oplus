import json
from pathlib import Path

from reproducibility import run_evaluation_matrix as matrix


def test_auto_workers_respects_cpu_memory_and_safety_cap(monkeypatch):
    monkeypatch.setattr(matrix, "_physical_core_count", lambda: 16)
    monkeypatch.setattr(matrix, "_available_memory_bytes", lambda: 40 * 1024**3)
    assert matrix.auto_worker_count() == 8

    monkeypatch.setattr(matrix, "_available_memory_bytes", lambda: 2 * 1024**3)
    assert matrix.auto_worker_count() == 1


def test_resume_requires_exact_job_signature_and_complete_formal_episode(tmp_path):
    policy = matrix.PolicySpec("method", ("--zero-hold",), {"type": "zero-hold"})
    output = tmp_path / "result.json"
    job = matrix.EvaluationJob(
        policy=policy,
        sumo_seed=7,
        scenario_manifest=tmp_path / "manifest.json",
        output=output,
        event_log=None,
        signature="exact-signature",
        identity={},
    )
    payload = {
        "schema_version": "transit-evaluation-v3",
        "evaluation_job": {
            "signature": "exact-signature",
            "max_events": 0,
            "max_sim_time_seconds": 0,
        },
        "method": "method",
        "sumo_seed": 7,
        "episode": {
            "stopped_by_event_limit": False,
            "stopped_by_time_limit": False,
        },
    }
    output.write_text(json.dumps(payload))
    valid, reasons = matrix._valid_output(
        job, max_events=0, max_sim_time_seconds=0
    )
    assert valid == payload
    assert reasons == []

    payload["evaluation_job"]["signature"] = "stale-signature"
    output.write_text(json.dumps(payload))
    valid, reasons = matrix._valid_output(
        job, max_events=0, max_sim_time_seconds=0
    )
    assert valid is None
    assert "job signature mismatch" in reasons

    payload["evaluation_job"]["signature"] = "exact-signature"
    payload["episode"]["stopped_by_event_limit"] = True
    output.write_text(json.dumps(payload))
    valid, reasons = matrix._valid_output(
        job, max_events=0, max_sim_time_seconds=0
    )
    assert valid is None
    assert "formal episode is partial" in reasons

    payload["episode"]["stopped_by_event_limit"] = False
    payload["episode"]["stopped_by_time_limit"] = True
    payload["evaluation_job"]["max_sim_time_seconds"] = 6_000
    output.write_text(json.dumps(payload))
    valid, reasons = matrix._valid_output(
        job, max_events=0, max_sim_time_seconds=6_000
    )
    assert valid == payload
    assert reasons == []


def test_scenario_provenance_hashes_every_runtime_input(tmp_path):
    sumo_dir = tmp_path / "sumo" / "online_control"
    initialize = sumo_dir / "initialize_obj"
    initialize.mkdir(parents=True)
    for name in (
        "save_obj_lane.add.xml",
        "save_obj_stop.add.xml",
        "save_obj_signal.add.xml",
        "save_obj_line.add.xml",
        "save_obj_bus.add.xml",
    ):
        (initialize / name).write_text(f"<{name}/>")
    bridge_edge = sumo_dir / "intersection_delay" / "a_sorted_busline_edge.xml"
    bridge_edge.parent.mkdir()
    bridge_edge.write_text("<edges/>")

    scenario = tmp_path / "scenario"
    scenario.mkdir()
    network = scenario / "network.xml"
    routes = scenario / "routes.xml"
    additional = scenario / "stops.xml"
    passenger_objects = scenario / "passenger_objects.add.xml"
    observation_edge = scenario / "observation_edges.xml"
    for path in (network, routes, additional, passenger_objects, observation_edge):
        path.write_text(f"<{path.stem}/>")
    config = scenario / "scenario.sumocfg"
    config.write_text(
        "<configuration><input>"
        '<net-file value="network.xml"/>'
        '<route-files value="routes.xml"/>'
        '<additional-files value="stops.xml"/>'
        "</input></configuration>"
    )
    manifest = scenario / "manifest.json"
    scenario_files = (config, routes, passenger_objects)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "transit-passenger-demand-v1",
                "sumo_config": str(config),
                "passenger_objects": str(passenger_objects),
                "scenario_sha256": {
                    str(path): matrix.sha256(path) for path in scenario_files
                },
            }
        )
    )

    hashes = matrix.scenario_input_hashes(manifest, sumo_dir, observation_edge)
    expected = {
        manifest.resolve(),
        config.resolve(),
        network.resolve(),
        routes.resolve(),
        additional.resolve(),
        passenger_objects.resolve(),
        observation_edge.resolve(),
        bridge_edge.resolve(),
        *(path.resolve() for path in initialize.iterdir()),
    }
    assert set(hashes) == {str(path) for path in expected}
