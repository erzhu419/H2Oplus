import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import bus_h2o.sumo_provenance as sumo_provenance
from bus_h2o.sumo_provenance import (
    collection_source_paths,
    load_sumo_runtime_lock,
    load_validated_scenario,
    sha256,
    sumo_input_paths,
    sumo_runtime_fingerprint,
)


ROOT = Path(__file__).resolve().parents[1]


def _scenario(tmp_path: Path):
    scenario = tmp_path / "scenario"
    scenario.mkdir()
    network = scenario / "network.xml"
    routes_a = scenario / "routes-a.xml"
    routes_b = scenario / "routes-b.xml"
    stops = scenario / "stops.xml"
    objects = scenario / "passenger-objects.xml"
    for path in (network, routes_a, routes_b, stops, objects):
        path.write_text(f"<{path.stem}/>")
    config = scenario / "scenario.sumocfg"
    config.write_text(
        "<configuration><input>"
        '<net-file value="network.xml"/>'
        '<route-files value="routes-a.xml, routes-b.xml"/>'
        '<additional-files value="stops.xml"/>'
        "</input></configuration>"
    )
    manifest = scenario / "manifest.json"
    declared = (config, routes_a, routes_b, objects)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "transit-passenger-demand-v1",
                "demand_scale": 1.0,
                "sumo_config": str(config),
                "passenger_objects": str(objects),
                "scenario_sha256": {
                    str(path): sha256(path) for path in declared
                },
            }
        )
    )
    return manifest, config, {config, network, routes_a, routes_b, stops, objects}


def test_sumo_provenance_resolves_and_hashes_every_runtime_input(tmp_path):
    manifest, config, expected = _scenario(tmp_path)
    assert set(sumo_input_paths(config)) == {
        path for path in expected if path.name != "passenger-objects.xml"
    }
    runtime, evidence = load_validated_scenario(manifest)
    assert runtime["sumo_cfg"] == str(config.resolve())
    assert set(evidence["runtime_input_sha256"]) == {
        str(path.resolve()) for path in expected
    }


def test_sumo_provenance_rejects_a_tampered_declared_file(tmp_path):
    manifest, _, _ = _scenario(tmp_path)
    declared = next(
        path
        for path in Path(manifest.parent).glob("routes-*.xml")
        if path.name == "routes-a.xml"
    )
    declared.write_text("<tampered/>")
    with pytest.raises(ValueError, match="scenario hash mismatch"):
        load_validated_scenario(manifest)


@pytest.mark.parametrize("contract_version", ("v5", "v6", "v7"))
def test_exact_welfare_module_is_part_of_chronological_provenance(contract_version):
    paths = set(
        collection_source_paths(
            ROOT,
            ROOT.parent / "SUMO_ruiguang" / "online_control",
            ROOT / "bus_h2o" / "network_data" / "a_sorted_busline_edge.xml",
            contract_version=contract_version,
        )
    )
    assert ROOT / "SimpleSAC" / "transit_v5" / "evaluation.py" in paths


def test_formal_libsumo_runtime_is_content_locked_and_fail_closed(tmp_path):
    fingerprint = sumo_runtime_fingerprint(
        "libsumo", expected_version="1.25.0"
    )
    assert fingerprint["backend"] == "libsumo"
    assert fingerprint["runtime_version"] == "1.25.0"
    assert fingerprint["distributions"]["libsumo"][
        "native_extension_sha256"
    ]
    assert fingerprint["external_executable"]["used_by_backend"] is False

    lock = tmp_path / "runtime_lock.json"
    lock.write_text(
        json.dumps(
            {
                "schema_version": "transit-sumo-runtime-lock-v7-v1",
                "runtime_fingerprint": fingerprint,
                "runtime_signature": fingerprint["portable_signature"],
            }
        )
    )
    _, identity = load_sumo_runtime_lock(lock, expected_sha256=sha256(lock))
    assert identity["runtime_signature"] == fingerprint["portable_signature"]

    payload = json.loads(lock.read_text())
    payload["runtime_signature"] = "0" * 64
    lock.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="internally inconsistent"):
        load_sumo_runtime_lock(lock)

    with pytest.raises(RuntimeError, match="version mismatch"):
        sumo_runtime_fingerprint("libsumo", expected_version="0.0.0")


def test_runtime_fingerprint_binds_traci_to_libsumo_before_import(
    tmp_path, monkeypatch
):
    binary = tmp_path / "sumo"
    binary.write_bytes(b"sumo-runtime")
    events = []

    monkeypatch.setattr(
        sumo_provenance,
        "configure_sumo_backend_environment",
        lambda backend: events.append(("configure", backend)),
    )
    monkeypatch.setattr(sumo_provenance, "_sumo_executable", lambda: binary)
    monkeypatch.setattr(
        sumo_provenance, "_external_runtime_binary", lambda launcher: launcher
    )
    monkeypatch.setattr(
        sumo_provenance, "sumo_version", lambda executable: "SUMO 1.25.0"
    )
    monkeypatch.setattr(
        sumo_provenance,
        "_distribution_fingerprint",
        lambda name, roots: {
            "distribution": name,
            "version": "1.25.0",
            "content_signature": name,
            "files": [],
            "native_extension_sha256": {},
        },
    )

    modules = {
        "traci": SimpleNamespace(isLibsumo=lambda: True),
        "libsumo": SimpleNamespace(getVersion=lambda: (22, "SUMO 1.25.0")),
    }

    def import_module(name):
        events.append(("import", name))
        return modules[name]

    monkeypatch.setattr(sumo_provenance.importlib, "import_module", import_module)
    fingerprint = sumo_runtime_fingerprint(
        "libsumo", expected_version="1.25.0"
    )

    assert fingerprint["runtime_version"] == "1.25.0"
    assert events == [
        ("configure", "libsumo"),
        ("import", "traci"),
        ("import", "libsumo"),
    ]
